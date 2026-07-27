from __future__ import annotations

import unittest

import numpy as np

from carla_vision.dataset import (
    CARLA_SEMANTIC_TAGS,
    DETECTOR_CATEGORIES,
    THING_TAGS,
    CarlaSemanticTag,
    decode_instance_bgra,
    detector_category_for_tag,
    extract_instance_labels,
)


def empty_frame(height: int = 6, width: int = 8) -> np.ndarray:
    return np.zeros((height, width, 4), dtype=np.uint8)


def paint(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    semantic_tag: CarlaSemanticTag,
    actor_id: int,
    alpha: int = 255,
) -> None:
    image[mask, 0] = (actor_id >> 8) & 0xFF
    image[mask, 1] = actor_id & 0xFF
    image[mask, 2] = int(semantic_tag)
    image[mask, 3] = alpha


class CarlaOntologyTests(unittest.TestCase):
    def test_semantic_tags_cover_carla_0916_values_zero_through_28(self) -> None:
        self.assertEqual([int(tag) for tag in CARLA_SEMANTIC_TAGS], list(range(29)))
        self.assertEqual(CarlaSemanticTag.TRAFFIC_LIGHT, 7)
        self.assertEqual(CarlaSemanticTag.TRAFFIC_SIGN, 8)
        self.assertEqual(CarlaSemanticTag.PEDESTRIAN, 12)
        self.assertEqual(CarlaSemanticTag.GUARD_RAIL, 28)

    def test_detector_category_ids_and_names_are_stable(self) -> None:
        self.assertEqual(
            [
                (category.id, category.name, int(category.semantic_tag))
                for category in DETECTOR_CATEGORIES
            ],
            [
                (0, "traffic_light", 7),
                (1, "traffic_sign", 8),
                (2, "pedestrian", 12),
                (3, "rider", 13),
                (4, "car", 14),
                (5, "truck", 15),
                (6, "bus", 16),
                (7, "train", 17),
                (8, "motorcycle", 18),
                (9, "bicycle", 19),
            ],
        )
        self.assertEqual(
            THING_TAGS, tuple(category.semantic_tag for category in DETECTOR_CATEGORIES)
        )
        self.assertEqual(detector_category_for_tag(14), DETECTOR_CATEGORIES[4])
        self.assertIsNone(detector_category_for_tag(CarlaSemanticTag.ROADS))
        self.assertIsNone(detector_category_for_tag(255))


class InstanceFrameDecodingTests(unittest.TestCase):
    def test_bgra_channel_order_decodes_red_tag_and_green_blue_actor_id(self) -> None:
        image = np.array([[[0x12, 0x34, 0x0E, 0x7F]]], dtype=np.uint8)

        semantic_tags, actor_ids = decode_instance_bgra(image)

        self.assertEqual(semantic_tags.dtype, np.uint8)
        self.assertEqual(actor_ids.dtype, np.uint16)
        self.assertEqual(int(semantic_tags[0, 0]), int(CarlaSemanticTag.CAR))
        self.assertEqual(int(actor_ids[0, 0]), 0x1234)
        self.assertEqual(int(image[0, 0, 3]), 0x7F, "alpha must be ignored")

    def test_decode_rejects_wrong_shape_type_and_dtype(self) -> None:
        with self.assertRaises(TypeError):
            decode_instance_bgra([[[0, 0, 0, 0]]])  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "HxWx4"):
            decode_instance_bgra(np.zeros((2, 3, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "uint8"):
            decode_instance_bgra(np.zeros((2, 3, 4), dtype=np.uint16))
        with self.assertRaisesRegex(ValueError, "positive"):
            decode_instance_bgra(np.zeros((0, 3, 4), dtype=np.uint8))


class InstanceLabelExtractionTests(unittest.TestCase):
    def test_extracts_auditable_half_open_box_and_visible_mask_metrics(self) -> None:
        image = empty_frame()
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[1, 2:5] = True
        mask[2, 2] = True
        mask[3, 4] = True
        paint(image, mask, semantic_tag=CarlaSemanticTag.CAR, actor_id=0x1234)
        original = image.copy()

        labels = extract_instance_labels(image)

        self.assertTrue(np.array_equal(image, original))
        self.assertEqual(len(labels), 1)
        label = labels[0]
        self.assertEqual(label.actor_id, 0x1234)
        self.assertIs(label.semantic_tag, CarlaSemanticTag.CAR)
        self.assertEqual(label.category_id, 4)
        self.assertEqual(label.category_name, "car")
        self.assertEqual(label.bbox_xyxy, (2, 1, 5, 4))
        self.assertEqual(label.box_width, 3)
        self.assertEqual(label.box_height, 3)
        self.assertEqual(label.visible_area_pixels, 5)
        self.assertEqual(label.bbox_area_pixels, 9)
        self.assertAlmostEqual(label.fill_ratio, 5 / 9)
        self.assertFalse(label.truncated)
        self.assertEqual(
            label.as_dict(),
            {
                "actor_id": 0x1234,
                "semantic_tag": 14,
                "category_id": 4,
                "category_name": "car",
                "bbox_xyxy": [2, 1, 5, 4],
                "visible_area_pixels": 5,
                "bbox_area_pixels": 9,
                "fill_ratio": 5 / 9,
                "truncated": False,
            },
        )

    def test_marks_any_mask_touching_an_image_edge_as_truncated(self) -> None:
        image = empty_frame(height=5, width=7)
        left_edge = np.zeros(image.shape[:2], dtype=bool)
        left_edge[1:4, 0:2] = True
        paint(
            image,
            left_edge,
            semantic_tag=CarlaSemanticTag.PEDESTRIAN,
            actor_id=513,
        )

        label = extract_instance_labels(image)[0]

        self.assertEqual(label.bbox_xyxy, (0, 1, 2, 4))
        self.assertTrue(label.truncated)
        self.assertEqual(label.actor_id, 513)
        self.assertIs(label.semantic_tag, CarlaSemanticTag.PEDESTRIAN)

    def test_filters_by_allowlist_visible_pixels_and_box_dimensions(self) -> None:
        image = empty_frame(height=8, width=12)

        car = np.zeros(image.shape[:2], dtype=bool)
        car[1:3, 1:3] = True
        paint(image, car, semantic_tag=CarlaSemanticTag.CAR, actor_id=20)

        truck = np.zeros(image.shape[:2], dtype=bool)
        truck[1:3, 5:8] = True
        paint(image, truck, semantic_tag=CarlaSemanticTag.TRUCK, actor_id=10)

        too_small = np.zeros(image.shape[:2], dtype=bool)
        too_small[5, 1] = True
        paint(image, too_small, semantic_tag=CarlaSemanticTag.BUS, actor_id=30)

        too_narrow = np.zeros(image.shape[:2], dtype=bool)
        too_narrow[4:7, 4] = True
        paint(image, too_narrow, semantic_tag=CarlaSemanticTag.RIDER, actor_id=40)

        too_short = np.zeros(image.shape[:2], dtype=bool)
        too_short[5, 7:10] = True
        paint(
            image,
            too_short,
            semantic_tag=CarlaSemanticTag.MOTORCYCLE,
            actor_id=50,
        )

        labels = extract_instance_labels(
            image,
            minimum_pixels=2,
            minimum_box_width=2,
            minimum_box_height=2,
            allowed_tags=(
                CarlaSemanticTag.CAR,
                CarlaSemanticTag.BUS,
                CarlaSemanticTag.RIDER,
                CarlaSemanticTag.MOTORCYCLE,
            ),
        )

        self.assertEqual([(item.category_name, item.actor_id) for item in labels], [("car", 20)])

    def test_output_order_is_category_then_actor_id_not_pixel_discovery_order(self) -> None:
        image = empty_frame()
        for x, tag, actor_id in (
            (1, CarlaSemanticTag.TRUCK, 90),
            (3, CarlaSemanticTag.CAR, 50),
            (5, CarlaSemanticTag.CAR, 10),
        ):
            mask = np.zeros(image.shape[:2], dtype=bool)
            mask[2:4, x : x + 2] = True
            paint(image, mask, semantic_tag=tag, actor_id=actor_id)

        labels = extract_instance_labels(image)

        self.assertEqual(
            [(item.category_name, item.actor_id) for item in labels],
            [("car", 10), ("car", 50), ("truck", 90)],
        )

    def test_nonthing_tags_are_ignored_by_default_and_rejected_in_allowlist(self) -> None:
        image = empty_frame()
        road = np.ones(image.shape[:2], dtype=bool)
        paint(image, road, semantic_tag=CarlaSemanticTag.ROADS, actor_id=0)

        self.assertEqual(extract_instance_labels(image), ())
        with self.assertRaisesRegex(ValueError, "non-detector"):
            extract_instance_labels(image, allowed_tags=(CarlaSemanticTag.ROADS,))

    def test_zero_actor_id_is_not_emitted_as_a_training_instance(self) -> None:
        image = empty_frame()
        mask = np.ones(image.shape[:2], dtype=bool)
        paint(image, mask, semantic_tag=CarlaSemanticTag.CAR, actor_id=0)

        self.assertEqual(extract_instance_labels(image), ())

    def test_filter_thresholds_must_be_positive_integers(self) -> None:
        image = empty_frame()
        for keyword, value in (
            ("minimum_pixels", 0),
            ("minimum_box_width", -1),
            ("minimum_box_height", 1.5),
            ("minimum_pixels", True),
        ):
            with self.subTest(keyword=keyword, value=value):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    extract_instance_labels(image, **{keyword: value})  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
