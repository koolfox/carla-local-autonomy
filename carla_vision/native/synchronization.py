"""Exact-frame helpers shared by the native CARLA worker and its tests."""

from __future__ import annotations

import math
import queue
import time
from dataclasses import dataclass
from typing import Any

from ..bridge import CarlaImageFrame
from ..scenarios.contracts import TransformRecipe


class SensorFrameError(RuntimeError):
    """Raised when a required synchronous sensor frame is absent."""


@dataclass(frozen=True)
class QueuedSensorFrame:
    sequence: int
    received_monotonic: float
    image: Any


class NativeSensorQueue:
    """Thread-safe CARLA callback queue with strict target-frame retrieval."""

    def __init__(self, name: str, *, max_frames: int = 0) -> None:
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 0:
            raise ValueError("max_frames must be a non-negative integer")
        self.name = name
        self._queue: queue.Queue[QueuedSensorFrame] = queue.Queue(maxsize=max_frames)
        self._received = 0
        self._discarded = 0

    @property
    def received(self) -> int:
        return self._received

    @property
    def discarded(self) -> int:
        return self._discarded

    def callback(self, image: Any) -> None:
        self._received += 1
        item = QueuedSensorFrame(
            sequence=self._received,
            received_monotonic=time.monotonic(),
            image=image,
        )
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._discarded += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                self._discarded += 1

    def get_next(self, timeout: float) -> QueuedSensorFrame:
        """Return the next callback item, preserving its receive sequence."""

        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty as error:
            raise SensorFrameError(
                f"{self.name} timed out waiting for its next CARLA frame"
            ) from error

    def get_exact(self, target_frame: int, timeout: float) -> QueuedSensorFrame:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise SensorFrameError(
                    f"{self.name} timed out waiting for CARLA frame {target_frame}"
                )
            queued = self.get_next(remaining)
            frame = int(queued.image.frame)
            if frame < target_frame:
                self._discarded += 1
                continue
            if frame > target_frame:
                raise SensorFrameError(
                    f"{self.name} skipped required CARLA frame {target_frame}; "
                    f"next frame is {frame}"
                )
            return queued


def compose_relative_transform(
    origin: TransformRecipe,
    relative: TransformRecipe,
) -> TransformRecipe:
    """Compose a local offset with a world transform using CARLA Euler order.

    The location is rotated by the origin's yaw, pitch, and roll using the
    standard Z-Y-X rotation matrix. Euler angles are then added, matching the
    bounded prop-placement contract used by the scenario recipes.
    """

    yaw = math.radians(origin.yaw)
    pitch = math.radians(origin.pitch)
    roll = math.radians(origin.roll)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rotation = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    local = (relative.x, relative.y, relative.z)
    translated = tuple(
        origin_value
        + sum(coefficient * component for coefficient, component in zip(row, local, strict=True))
        for origin_value, row in zip(
            (origin.x, origin.y, origin.z),
            rotation,
            strict=True,
        )
    )
    return TransformRecipe(
        x=translated[0],
        y=translated[1],
        z=translated[2],
        pitch=origin.pitch + relative.pitch,
        yaw=origin.yaw + relative.yaw,
        roll=origin.roll + relative.roll,
    )


def image_to_bridge_frame(
    queued: QueuedSensorFrame,
    *,
    fov_degrees: float,
    sensor_type: int,
) -> CarlaImageFrame:
    """Convert an official ``carla.Image`` into the common dataset contract."""

    image = queued.image
    transform = image.transform
    location = transform.location
    rotation = transform.rotation
    bgra = bytes(image.raw_data)
    width = int(image.width)
    height = int(image.height)
    expected = width * height * 4
    if len(bgra) != expected:
        raise SensorFrameError(
            f"CARLA image frame {image.frame} contains {len(bgra)} bytes; expected {expected}"
        )
    return CarlaImageFrame(
        sequence=queued.sequence,
        sensor_type=sensor_type,
        frame=int(image.frame),
        timestamp=float(image.timestamp),
        transform=(
            float(location.x),
            float(location.y),
            float(location.z),
            float(rotation.pitch),
            float(rotation.yaw),
            float(rotation.roll),
        ),
        width=width,
        height=height,
        fov=float(fov_degrees),
        bgra=bgra,
        received_monotonic=queued.received_monotonic,
    )


__all__ = [
    "NativeSensorQueue",
    "QueuedSensorFrame",
    "SensorFrameError",
    "compose_relative_transform",
    "image_to_bridge_frame",
]
