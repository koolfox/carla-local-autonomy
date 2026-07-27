"""Exact CARLA-frame synchronization for RGB and privileged teacher cameras."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol

from ..bridge import CarlaImageFrame


class CameraFrameSource(Protocol):
    def wait_for_frame(
        self,
        after_sequence: int = -1,
        timeout: float = 3.0,
    ) -> CarlaImageFrame: ...


@dataclass(frozen=True)
class SynchronizedFramePair:
    """One RGB/teacher pair rendered by CARLA on the same simulator frame."""

    rgb: CarlaImageFrame
    teacher: CarlaImageFrame
    rgb_skipped: int
    teacher_skipped: int

    @property
    def carla_frame(self) -> int:
        return self.rgb.frame


class ExactFramePairer:
    """Advance two streams until their CARLA frame identifiers match exactly.

    Dataset collection must never label an RGB image with a mask from a
    different simulation state.  This pairer therefore rejects timestamp,
    geometry, or camera-pose disagreement even if the integer frame happens to
    match.
    """

    def __init__(
        self,
        *,
        timestamp_tolerance: float = 1e-6,
        transform_tolerance: float = 1e-4,
        fov_tolerance: float = 1e-4,
    ) -> None:
        if timestamp_tolerance < 0.0:
            raise ValueError("timestamp_tolerance must be non-negative")
        if transform_tolerance < 0.0:
            raise ValueError("transform_tolerance must be non-negative")
        if fov_tolerance < 0.0:
            raise ValueError("fov_tolerance must be non-negative")
        self.timestamp_tolerance = float(timestamp_tolerance)
        self.transform_tolerance = float(transform_tolerance)
        self.fov_tolerance = float(fov_tolerance)
        self.last_rgb_sequence = -1
        self.last_teacher_sequence = -1
        self.total_rgb_skipped = 0
        self.total_teacher_skipped = 0

    def next_pair(
        self,
        rgb_stream: CameraFrameSource,
        teacher_stream: CameraFrameSource,
        *,
        timeout: float = 5.0,
    ) -> SynchronizedFramePair:
        if timeout <= 0.0:
            raise ValueError("timeout must be positive")
        deadline = time.monotonic() + timeout
        rgb = self._wait(
            rgb_stream,
            after_sequence=self.last_rgb_sequence,
            deadline=deadline,
            label="RGB",
        )
        teacher = self._wait(
            teacher_stream,
            after_sequence=self.last_teacher_sequence,
            deadline=deadline,
            label="teacher",
        )
        rgb_skipped = 0
        teacher_skipped = 0

        while rgb.frame != teacher.frame:
            if rgb.frame < teacher.frame:
                rgb_skipped += 1
                rgb = self._wait(
                    rgb_stream,
                    after_sequence=rgb.sequence,
                    deadline=deadline,
                    label="RGB",
                )
            else:
                teacher_skipped += 1
                teacher = self._wait(
                    teacher_stream,
                    after_sequence=teacher.sequence,
                    deadline=deadline,
                    label="teacher",
                )

        self._validate_pair(rgb, teacher)
        self.last_rgb_sequence = rgb.sequence
        self.last_teacher_sequence = teacher.sequence
        self.total_rgb_skipped += rgb_skipped
        self.total_teacher_skipped += teacher_skipped
        return SynchronizedFramePair(
            rgb=rgb,
            teacher=teacher,
            rgb_skipped=rgb_skipped,
            teacher_skipped=teacher_skipped,
        )

    @staticmethod
    def _wait(
        stream: CameraFrameSource,
        *,
        after_sequence: int,
        deadline: float,
        label: str,
    ) -> CarlaImageFrame:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError(f"timed out synchronizing the {label} camera")
        try:
            return stream.wait_for_frame(
                after_sequence=after_sequence,
                timeout=remaining,
            )
        except TimeoutError as error:
            raise TimeoutError(f"timed out synchronizing the {label} camera") from error

    def _validate_pair(
        self,
        rgb: CarlaImageFrame,
        teacher: CarlaImageFrame,
    ) -> None:
        if rgb.frame != teacher.frame:
            raise ValueError("camera pair does not share a CARLA frame")
        if not math.isclose(
            rgb.timestamp,
            teacher.timestamp,
            rel_tol=0.0,
            abs_tol=self.timestamp_tolerance,
        ):
            raise ValueError(
                "same-frame cameras disagree on simulation timestamp: "
                f"{rgb.timestamp} vs {teacher.timestamp}"
            )
        if (rgb.width, rgb.height) != (teacher.width, teacher.height):
            raise ValueError(
                "RGB and teacher camera resolutions differ: "
                f"{rgb.width}x{rgb.height} vs {teacher.width}x{teacher.height}"
            )
        if not math.isclose(
            rgb.fov,
            teacher.fov,
            rel_tol=0.0,
            abs_tol=self.fov_tolerance,
        ):
            raise ValueError(f"RGB and teacher camera FOV differ: {rgb.fov} vs {teacher.fov}")
        transform_error = max(
            abs(left - right) for left, right in zip(rgb.transform, teacher.transform, strict=True)
        )
        if transform_error > self.transform_tolerance:
            raise ValueError(
                "RGB and teacher camera poses differ by "
                f"{transform_error:.6f}, above {self.transform_tolerance:.6f}"
            )


__all__ = [
    "CameraFrameSource",
    "ExactFramePairer",
    "SynchronizedFramePair",
]
