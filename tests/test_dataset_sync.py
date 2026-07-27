from __future__ import annotations

import time
import unittest

import numpy as np

from carla_vision.bridge import CarlaImageFrame
from carla_vision.dataset.sync import ExactFramePairer


def frame(
    sequence: int,
    carla_frame: int,
    *,
    timestamp: float | None = None,
    transform: tuple[float, float, float, float, float, float] = (
        1.0,
        2.0,
        3.0,
        0.0,
        90.0,
        0.0,
    ),
    width: int = 4,
    height: int = 3,
    fov: float = 90.0,
) -> CarlaImageFrame:
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    return CarlaImageFrame(
        sequence=sequence,
        sensor_type=1,
        frame=carla_frame,
        timestamp=(float(carla_frame) / 20.0 if timestamp is None else timestamp),
        transform=transform,
        width=width,
        height=height,
        fov=fov,
        bgra=pixels.tobytes(),
        received_monotonic=time.monotonic(),
    )


class FakeStream:
    def __init__(self, frames: list[CarlaImageFrame]) -> None:
        self.frames = frames

    def wait_for_frame(
        self,
        after_sequence: int = -1,
        timeout: float = 3.0,
    ) -> CarlaImageFrame:
        del timeout
        for item in self.frames:
            if item.sequence > after_sequence:
                return item
        raise TimeoutError


class ExactFramePairerTests(unittest.TestCase):
    def test_advances_lagging_stream_and_tracks_skips_across_pairs(self) -> None:
        rgb = FakeStream([frame(1, 100), frame(2, 102), frame(3, 104)])
        teacher = FakeStream([frame(1, 101), frame(2, 102), frame(3, 104)])
        pairer = ExactFramePairer()

        first = pairer.next_pair(rgb, teacher)
        second = pairer.next_pair(rgb, teacher)

        self.assertEqual(first.carla_frame, 102)
        self.assertEqual(first.rgb_skipped, 1)
        self.assertEqual(first.teacher_skipped, 1)
        self.assertEqual(second.carla_frame, 104)
        self.assertEqual(pairer.total_rgb_skipped, 1)
        self.assertEqual(pairer.total_teacher_skipped, 1)

    def test_rejects_same_frame_with_different_sensor_geometry(self) -> None:
        pairer = ExactFramePairer()
        with self.assertRaisesRegex(ValueError, "resolutions differ"):
            pairer.next_pair(
                FakeStream([frame(1, 100)]),
                FakeStream([frame(1, 100, width=5)]),
            )

        shifted = (1.0, 2.0, 3.01, 0.0, 90.0, 0.0)
        with self.assertRaisesRegex(ValueError, "poses differ"):
            ExactFramePairer().next_pair(
                FakeStream([frame(1, 100)]),
                FakeStream([frame(1, 100, transform=shifted)]),
            )

    def test_timeout_names_the_stream_that_cannot_reach_a_match(self) -> None:
        pairer = ExactFramePairer()
        with self.assertRaisesRegex(TimeoutError, "RGB camera"):
            pairer.next_pair(
                FakeStream([frame(1, 100)]),
                FakeStream([frame(1, 101)]),
                timeout=0.1,
            )

    def test_invalid_tolerances_are_rejected(self) -> None:
        for keyword in (
            "timestamp_tolerance",
            "transform_tolerance",
            "fov_tolerance",
        ):
            with self.subTest(keyword=keyword):
                with self.assertRaises(ValueError):
                    ExactFramePairer(**{keyword: -1.0})


if __name__ == "__main__":
    unittest.main()
