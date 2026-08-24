from __future__ import annotations

import unittest
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np

from carla_vision.segmentation import (
    CANONICAL_CLASS_NAMES,
    RoadClass,
    SegmentationConfig,
    SegmentationMetadata,
    SegmentationResult,
    canonical_class_for_label,
    create_segmenter,
)
from carla_vision.segmentation.segformer import SegFormerSegmenter


class _FakeTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class _FakeBatch(dict[str, object]):
    def __init__(self) -> None:
        super().__init__({"pixel_values": object()})
        self.device: str | None = None

    def to(self, device: str) -> _FakeBatch:
        self.device = device
        return self


class _FakeProcessor:
    def __init__(self) -> None:
        self.image_rgb: np.ndarray | None = None
        self.batch = _FakeBatch()

    def __call__(self, *, images: np.ndarray, return_tensors: str) -> _FakeBatch:
        if return_tensors != "pt":
            raise AssertionError("adapter did not request PyTorch tensors")
        self.image_rgb = images.copy()
        return self.batch


class _FakeModelConfig:
    num_labels = 5
    id2label = {
        "0": "road-line",
        "1": "other",
        "2": "non drivable ground",
        "3": "road",
        "4": "sidewalk",
    }


class _FakeOutput:
    def __init__(self, logits: np.ndarray) -> None:
        self.logits = _FakeTensor(logits)


class _FakeModel:
    config = _FakeModelConfig()

    def __init__(self, logits: np.ndarray) -> None:
        self.logits = logits
        self.called = False

    def __call__(self, **inputs: object) -> _FakeOutput:
        self.called = "pixel_values" in inputs
        return _FakeOutput(self.logits)


class _FakeTorch:
    @staticmethod
    def inference_mode() -> nullcontext[None]:
        return nullcontext()


class _FakeSegmenter:
    name = "custom-fake"
    metadata = SegmentationMetadata(
        name=name,
        backend="test",
        checkpoint=None,
        device="cpu",
    )

    def infer(self, image_bgr: np.ndarray) -> SegmentationResult:
        height, width = image_bgr.shape[:2]
        return SegmentationResult(
            class_ids=np.zeros((height, width), dtype=np.uint8),
            confidence=np.ones((height, width), dtype=np.float32),
            model_name=self.name,
        )

    def close(self) -> None:
        pass


class SegmentationContractTests(unittest.TestCase):
    def test_canonical_class_ids_are_stable(self) -> None:
        self.assertEqual(tuple(int(item) for item in RoadClass), tuple(range(5)))
        self.assertEqual(
            CANONICAL_CLASS_NAMES,
            ("other", "road", "road_line", "sidewalk", "non_drivable_ground"),
        )

    def test_config_rejects_invalid_custom_factory_combinations(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires"):
            SegmentationConfig(backend="custom", checkpoint=None)
        with self.assertRaisesRegex(ValueError, "module:callable"):
            SegmentationConfig(backend="custom", checkpoint=None, factory="not-a-reference")
        with self.assertRaisesRegex(ValueError, "only"):
            SegmentationConfig(backend="segformer", factory="some.module:create")

    def test_result_is_copied_validated_and_read_only(self) -> None:
        source_ids = np.array([[RoadClass.ROAD, RoadClass.ROAD_LINE]], dtype=np.int64)
        source_confidence = np.array([[0.9, 0.8]], dtype=np.float64)
        result = SegmentationResult(source_ids, source_confidence, "fake")
        source_ids[:] = RoadClass.OTHER
        source_confidence[:] = 0.0

        self.assertEqual(result.class_ids.dtype, np.uint8)
        self.assertEqual(result.confidence.dtype, np.float32)
        self.assertEqual(result.class_ids.tolist(), [[1, 2]])
        self.assertFalse(result.class_ids.flags.writeable)
        self.assertFalse(result.confidence.flags.writeable)
        self.assertEqual(result.mask_for(RoadClass.ROAD).tolist(), [[True, False]])

    def test_label_aliases_map_to_canonical_classes(self) -> None:
        self.assertIs(canonical_class_for_label("Lane Markings"), RoadClass.ROAD_LINE)
        self.assertIs(canonical_class_for_label("terrain"), RoadClass.NON_DRIVABLE_GROUND)
        self.assertIs(canonical_class_for_label("car"), RoadClass.OTHER)


class SegFormerAdapterTests(unittest.TestCase):
    def test_custom_five_class_checkpoint_is_mapped_by_label_name(self) -> None:
        logits = np.full((1, 5, 1, 2), -8.0, dtype=np.float32)
        logits[0, 3, 0, 0] = 8.0  # source id 3 is road
        logits[0, 0, 0, 1] = 8.0  # source id 0 is road_line
        processor = _FakeProcessor()
        model = _FakeModel(logits)
        with patch(
            "carla_vision.segmentation.segformer._load_huggingface_runtime",
            return_value=(_FakeTorch(), processor, model),
        ):
            segmenter = SegFormerSegmenter(
                SegmentationConfig(checkpoint="local/custom-segformer-b0", device="mps")
            )

        image_bgr = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
        result = segmenter.infer(image_bgr)

        self.assertEqual(result.class_ids.tolist(), [[RoadClass.ROAD, RoadClass.ROAD_LINE]])
        self.assertTrue(model.called)
        self.assertEqual(processor.batch.device, "mps")
        self.assertEqual(processor.image_rgb.tolist(), [[[3, 2, 1], [6, 5, 4]]])
        self.assertEqual(segmenter.metadata.source_to_canonical[0], "road_line")
        self.assertEqual(segmenter.metadata.source_to_canonical[3], "road")

    def test_generic_checkpoint_labels_are_rejected(self) -> None:
        model = _FakeModel(np.zeros((1, 5, 1, 1), dtype=np.float32))
        model.config = type(
            "GenericConfig",
            (),
            {"num_labels": 5, "id2label": {index: f"LABEL_{index}" for index in range(5)}},
        )()
        with patch(
            "carla_vision.segmentation.segformer._load_huggingface_runtime",
            return_value=(_FakeTorch(), _FakeProcessor(), model),
        ):
            with self.assertRaisesRegex(ValueError, "generic LABEL_n"):
                SegFormerSegmenter(SegmentationConfig(checkpoint="local/missing-labels"))

    def test_cityscapes_labels_map_without_claiming_road_lines(self) -> None:
        labels = (
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic light",
            "traffic sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        )
        model = _FakeModel(np.zeros((1, len(labels), 1, 1), dtype=np.float32))
        model.config = type(
            "CityscapesConfig",
            (),
            {"num_labels": len(labels), "id2label": dict(enumerate(labels))},
        )()
        with patch(
            "carla_vision.segmentation.segformer._load_huggingface_runtime",
            return_value=(_FakeTorch(), _FakeProcessor(), model),
        ):
            segmenter = SegFormerSegmenter(SegmentationConfig())

        mapping = segmenter.metadata.source_to_canonical
        self.assertEqual(mapping[0], "road")
        self.assertEqual(mapping[1], "sidewalk")
        self.assertEqual(mapping[9], "non_drivable_ground")
        self.assertNotIn("road_line", mapping.values())

    def test_infer_rejects_non_rgb_uint8_input(self) -> None:
        model = _FakeModel(np.zeros((1, 5, 1, 1), dtype=np.float32))
        with patch(
            "carla_vision.segmentation.segformer._load_huggingface_runtime",
            return_value=(_FakeTorch(), _FakeProcessor(), model),
        ):
            segmenter = SegFormerSegmenter(SegmentationConfig(checkpoint="local/custom"))
        with self.assertRaisesRegex(ValueError, "uint8 HxWx3"):
            segmenter.infer(np.zeros((4, 4), dtype=np.uint8))


class SegmentationFactoryTests(unittest.TestCase):
    def test_custom_factory_output_is_accepted(self) -> None:
        with patch(
            "carla_vision.segmentation.factory._load_custom_factory",
            return_value=lambda _config: _FakeSegmenter(),
        ):
            segmenter = create_segmenter(
                SegmentationConfig(
                    backend="custom",
                    checkpoint=None,
                    factory="some.module:create",
                )
            )
        self.assertEqual(segmenter.name, "custom-fake")

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            create_segmenter(SegmentationConfig(backend="not-a-model"))


if __name__ == "__main__":
    unittest.main()
