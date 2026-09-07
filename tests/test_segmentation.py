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
    _commit_hash = "0123456789abcdef"
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

    def test_common_lane_checkpoint_aliases_map_without_checkpoint_name_logic(self) -> None:
        aliases = {
            "construction--flat--road": RoadClass.ROAD,
            "construction--flat--sidewalk": RoadClass.SIDEWALK,
            "marking--general": RoadClass.ROAD_LINE,
            "marking--crosswalk-zebra": RoadClass.ROAD_LINE,
            "lane marking - general": RoadClass.ROAD_LINE,
            "lane marking - crosswalk": RoadClass.ROAD_LINE,
            "lane marking - general/crosswalk": RoadClass.ROAD_LINE,
            "solid line": RoadClass.ROAD_LINE,
            "dashed line": RoadClass.ROAD_LINE,
            "double solid line": RoadClass.ROAD_LINE,
            "zebra": RoadClass.ROAD_LINE,
            "nature--terrain": RoadClass.NON_DRIVABLE_GROUND,
        }
        for label, expected in aliases.items():
            with self.subTest(label=label):
                self.assertIs(canonical_class_for_label(label), expected)


class SegFormerAdapterTests(unittest.TestCase):
    def test_collapsed_float32_confidence_is_clamped_after_softmax(self) -> None:
        # This distribution reproduces the MPS Cityscapes accumulation case:
        # individually valid source probabilities collapse to OTHER at
        # 1.0000001 in float32.
        logits = np.array(
            [
                -21.3489017,
                -19.1082840,
                -3.8926716,
                -13.8998241,
                -24.4031677,
                -21.1792698,
                -13.9456282,
                0.0,
                -22.0470829,
                -21.7864666,
                -9.3277435,
                -28.2154789,
                -22.2680817,
                -10.5254898,
                -13.5473795,
                -18.4357128,
                -12.6498356,
                -47.6325035,
                -9.1378117,
            ],
            dtype=np.float32,
        ).reshape(1, 19, 1, 1)
        cityscapes_labels = (
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
        model = _FakeModel(logits)
        model.config = type(
            "CityscapesConfig",
            (),
            {
                "num_labels": len(cityscapes_labels),
                "id2label": dict(enumerate(cityscapes_labels)),
            },
        )()
        with patch(
            "carla_vision.segmentation.segformer._load_huggingface_runtime",
            return_value=(_FakeTorch(), _FakeProcessor(), model),
        ):
            segmenter = SegFormerSegmenter(SegmentationConfig())

        result = segmenter.infer(np.zeros((1, 1, 3), dtype=np.uint8))

        self.assertEqual(result.class_ids.tolist(), [[RoadClass.OTHER]])
        self.assertEqual(result.confidence.tolist(), [[1.0]])

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
                SegmentationConfig(
                    checkpoint="local/custom-segformer-b0",
                    device="mps",
                    options={"revision": "research-release-v1"},
                )
            )

        image_bgr = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
        result = segmenter.infer(image_bgr)

        self.assertEqual(result.class_ids.tolist(), [[RoadClass.ROAD, RoadClass.ROAD_LINE]])
        self.assertTrue(model.called)
        self.assertEqual(processor.batch.device, "mps")
        self.assertEqual(processor.image_rgb.tolist(), [[[3, 2, 1], [6, 5, 4]]])
        self.assertEqual(segmenter.metadata.source_to_canonical[0], "road_line")
        self.assertEqual(segmenter.metadata.source_to_canonical[3], "road")
        self.assertEqual(segmenter.metadata.revision, "research-release-v1")
        self.assertEqual(segmenter.metadata.resolved_revision, "0123456789abcdef")
        self.assertTrue(segmenter.metadata.as_dict()["supports_road_line"])

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
        self.assertFalse(segmenter.metadata.as_dict()["supports_road_line"])

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
