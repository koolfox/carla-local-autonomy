from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from carla_vision.bridge import CarlaImageFrame
from carla_vision.contracts import Detection, DetectorMetadata
from carla_vision.perception import PerceptionWorker


def make_frame(sequence: int) -> CarlaImageFrame:
    image = np.full((4, 6, 4), sequence, dtype=np.uint8)
    image[:, :, 3] = 255
    return CarlaImageFrame(
        sequence=sequence,
        sensor_type=1,
        frame=100 + sequence,
        timestamp=float(sequence),
        transform=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        width=6,
        height=4,
        fov=90.0,
        bgra=image.tobytes(),
        received_monotonic=time.monotonic(),
    )


class BlockingDetector:
    name = "blocking-fake"
    metadata = DetectorMetadata(
        name=name,
        backend="test",
        weights=None,
        device="cpu",
        image_size=6,
        confidence=0.0,
    )

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = False

    def infer(self, image_bgr: np.ndarray) -> tuple[Detection, ...]:
        self.started.set()
        self.release.wait(timeout=2.0)
        return (
            Detection(
                class_id=0,
                label="pixel",
                confidence=1.0,
                xyxy=(0.0, 0.0, 1.0, 1.0),
                attributes={"value": int(image_bgr[0, 0, 0])},
            ),
        )

    def close(self) -> None:
        self.closed = True


class PerceptionWorkerTests(unittest.TestCase):
    def test_worker_drops_superseded_pending_frame(self) -> None:
        detector = BlockingDetector()
        worker = PerceptionWorker(detector)
        try:
            worker.submit(make_frame(1))
            self.assertTrue(detector.started.wait(timeout=1.0))
            worker.submit(make_frame(2))
            worker.submit(make_frame(3))
            detector.release.set()

            latest = worker.wait_for_result(after_sequence=2, timeout=2.0)
            stats = worker.stats()

            self.assertEqual(latest.sequence, 3)
            self.assertEqual(latest.detections[0].attributes["value"], 3)
            self.assertEqual(stats.submitted, 3)
            self.assertEqual(stats.processed, 2)
            self.assertEqual(stats.dropped_before_inference, 1)
        finally:
            detector.release.set()
            worker.close()
        self.assertTrue(detector.closed)


if __name__ == "__main__":
    unittest.main()
