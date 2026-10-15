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

    def __init__(self, name: str, *, max_frames: int = 4) -> None:
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

        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("sensor timeout must be finite and positive")
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty as error:
            raise SensorFrameError(
                f"{self.name} timed out waiting for its next CARLA frame"
            ) from error

    def get_exact(self, target_frame: int, timeout: float) -> QueuedSensorFrame:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("sensor timeout must be finite and positive")
        if isinstance(target_frame, bool) or not isinstance(target_frame, int) or target_frame < 0:
            raise ValueError("target frame must be a non-negative integer")
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


def carla_transform_matrix(transform: TransformRecipe) -> tuple[tuple[float, ...], ...]:
    """CARLA 0.9.16 GetMatrix convention, including pitch/roll signs.

    Reference: carla-simulator/carla, LibCarla/source/carla/geom/Transform.h.
    Matrices compose rigid poses; Euler-angle addition is not a general composition.
    """
    values = (transform.x, transform.y, transform.z, transform.pitch, transform.yaw, transform.roll)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("transform must contain six finite values")
    pitch, yaw, roll = (math.radians(value) for value in values[3:])
    cp, sp, cy, sy, cr, sr = (
        math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw),
        math.sin(yaw),
        math.cos(roll),
        math.sin(roll),
    )
    return (
        (cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, transform.x),
        (cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, transform.y),
        (sp, -cp * sr, cp * cr, transform.z),
        (0.0, 0.0, 0.0, 1.0),
    )


def compose_relative_transform(
    origin: TransformRecipe, relative: TransformRecipe
) -> TransformRecipe:
    """Compose full SE(3) transforms in the CARLA axis/angle convention."""
    left, right = carla_transform_matrix(origin), carla_transform_matrix(relative)
    matrix = tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(4)) for j in range(4)) for i in range(4)
    )
    pitch = math.asin(max(-1.0, min(1.0, matrix[2][0])))
    if abs(math.cos(pitch)) > 1e-8:
        yaw = math.atan2(matrix[1][0], matrix[0][0])
        roll = math.atan2(-matrix[2][1], matrix[2][2])
    else:
        yaw, roll = math.atan2(-matrix[0][1], matrix[1][1]), 0.0
    return TransformRecipe(
        x=matrix[0][3],
        y=matrix[1][3],
        z=matrix[2][3],
        pitch=math.degrees(pitch),
        yaw=math.degrees(yaw),
        roll=math.degrees(roll),
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
