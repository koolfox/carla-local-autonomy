from __future__ import annotations

import unittest

import numpy as np

from carla_vision.contracts import Detection, DetectorConfig, PerceptionResult


class DetectionContractTests(unittest.TestCase):
    def test_valid_detection_uses_original_pixel_coordinates(self) -> None:
        detection = Detection(
            class_id=2,
            source_class_id=7,
            label="car",
            confidence=0.75,
            xyxy=(1.0, 2.0, 31.0, 42.0),
        )
        self.assertEqual(detection.xyxy, (1.0, 2.0, 31.0, 42.0))

    def test_invalid_detection_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Detection(class_id=-1, label="car", confidence=0.5, xyxy=(0, 0, 1, 1))
        with self.assertRaises(ValueError):
            Detection(class_id=0, label="car", confidence=1.1, xyxy=(0, 0, 1, 1))
        with self.assertRaises(ValueError):
            Detection(class_id=0, label="car", confidence=0.5, xyxy=(2, 0, 1, 1))

    def test_custom_detector_requires_factory(self) -> None:
        with self.assertRaisesRegex(ValueError, "factory"):
            DetectorConfig(backend="custom")


class PerceptionResultContractTests(unittest.TestCase):
    def test_result_latency_and_age(self) -> None:
        result = PerceptionResult(
            sequence=1,
            carla_frame=10,
            source_timestamp=2.0,
            source_received_monotonic=20.0,
            completed_monotonic=20.25,
            detections=(),
            source_bgr=np.zeros((8, 12, 3), dtype=np.uint8),
            detector_name="fake",
        )
        self.assertAlmostEqual(result.inference_seconds, 0.25)
        self.assertAlmostEqual(result.age_seconds(20.5), 0.5)

    def test_non_bgr_result_source_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "HxWx3"):
            PerceptionResult(
                sequence=1,
                carla_frame=10,
                source_timestamp=2.0,
                source_received_monotonic=20.0,
                completed_monotonic=20.1,
                detections=(),
                source_bgr=np.zeros((8, 12), dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main()
