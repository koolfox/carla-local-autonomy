from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from carla_vision.contracts import (
    Detection,
    DetectorConfig,
    DetectorMetadata,
)
from carla_vision.detectors import create_detector
from carla_vision.detectors.ultralytics import UltralyticsDetector


class FakeDetector:
    name = "fake"
    metadata = DetectorMetadata(
        name="fake",
        backend="test",
        weights=None,
        device="cpu",
        image_size=8,
        confidence=0.0,
    )

    def infer(self, image_bgr: np.ndarray) -> tuple[Detection, ...]:
        return ()

    def close(self) -> None:
        pass


class _Scalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class _Vector:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


class _Box:
    cls = _Scalar(2)
    conf = _Scalar(0.875)
    xyxy = [_Vector([10.0, 20.0, 30.0, 45.0])]


class _Result:
    boxes = [_Box()]
    names = {2: "car"}


class _Model:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def __call__(self, _image: np.ndarray, **kwargs: object) -> list[_Result]:
        self.kwargs = kwargs
        return [_Result()]


class DetectorFactoryTests(unittest.TestCase):
    def test_custom_factory_output_is_accepted(self) -> None:
        with patch(
            "carla_vision.detectors.factory._load_custom_factory",
            return_value=lambda _config: FakeDetector(),
        ):
            detector = create_detector(
                DetectorConfig(backend="custom", factory="some.module:create")
            )
        self.assertEqual(detector.name, "fake")

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            create_detector(DetectorConfig(backend="not-a-model"))


class UltralyticsAdapterTests(unittest.TestCase):
    def test_backend_specific_result_is_normalized(self) -> None:
        config = DetectorConfig(
            backend="yolo",
            weights=None,
            image_size=960,
            confidence=0.3,
            device="cpu",
        )
        adapter = UltralyticsDetector.__new__(UltralyticsDetector)
        adapter._architecture = "yolo"
        adapter._config = config
        adapter._model = _Model()

        detections = adapter.infer(np.zeros((64, 96, 3), dtype=np.uint8))

        self.assertEqual(
            detections,
            (
                Detection(
                    class_id=2,
                    source_class_id=2,
                    label="car",
                    confidence=0.875,
                    xyxy=(10.0, 20.0, 30.0, 45.0),
                ),
            ),
        )
        self.assertEqual(adapter._model.kwargs["imgsz"], 960)
        self.assertEqual(adapter._model.kwargs["conf"], 0.3)


if __name__ == "__main__":
    unittest.main()
