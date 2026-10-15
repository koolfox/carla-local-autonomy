"""Cross-format label consistency. Independent human/golden QA is still required."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2

from .instance_labels import InstanceLabel, extract_instance_labels


def verify_coco_labels(
    labels: Sequence[InstanceLabel], annotations: Sequence[Mapping[str, Any]], *, image_id: int
) -> None:
    """Compare every box/class/area/identity, not merely annotation counts."""
    if len(labels) != len(annotations):
        raise ValueError(f"COCO annotation count mismatch for image {image_id}")
    expected = {}
    for label in labels:
        x1, y1, x2, y2 = label.bbox_xyxy
        expected[(label.category_id + 1, label.actor_id)] = (
            [x1, y1, x2 - x1, y2 - y1],
            label.visible_area_pixels,
            {
                "carla_actor_id": label.actor_id,
                "carla_semantic_tag": int(label.semantic_tag),
                "visible_area_pixels": label.visible_area_pixels,
                "bbox_area_pixels": label.bbox_area_pixels,
                "mask_fill_ratio": label.fill_ratio,
                "truncated": label.truncated,
                "teacher_method": "instance_segmentation_visible_mask",
            },
        )
    seen = set()
    for annotation in annotations:
        attributes = annotation.get("attributes", {})
        if not isinstance(attributes, Mapping):
            raise ValueError("COCO attributes must be an object")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (annotation.get("category_id"), attributes.get("carla_actor_id"))
        ):
            raise ValueError("COCO category/render instance identities must be integers")
        key = (annotation.get("category_id"), attributes.get("carla_actor_id"))
        if key not in expected or key in seen or annotation.get("image_id") != image_id:
            raise ValueError("COCO label identity disagrees with instance mask")
        seen.add(key)
        bbox, area, expected_attributes = expected[key]
        actual_box = annotation.get("bbox")
        if (
            not isinstance(actual_box, list)
            or len(actual_box) != 4
            or any(
                isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                for x in actual_box
            )
            or actual_box != bbox
            or annotation.get("area") != area
            or annotation.get("iscrowd") != 0
        ):
            raise ValueError("COCO bbox/area disagrees with instance mask")
        if any(attributes.get(k) != v for k, v in expected_attributes.items()):
            raise ValueError("COCO label attributes disagree with instance mask")


def verify_sample_semantics(
    root: Path,
    sample: Mapping[str, Any],
    image: Mapping[str, Any],
    annotations: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    # Paths have already passed containment and hash checks in the base verifier.
    metadata = json.loads((root / sample["metadata"]["path"]).read_text(encoding="utf-8"))
    for field in (
        "dataset_id",
        "sample_id",
        "image_id",
        "split",
        "scenario_id",
        "episode_id",
        "carla_frame",
    ):
        if field in sample and metadata.get(field) != sample[field]:
            raise ValueError(f"metadata {field} disagrees with dataset sample")
    if image.get("id") != sample["image_id"] or image.get("file_name") != sample["rgb"]["path"]:
        raise ValueError("COCO image identity/path disagrees with dataset sample")
    timestamp = metadata.get("simulation_timestamp_seconds")
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not math.isfinite(timestamp)
        or not math.isclose(timestamp, sample["source_timestamp"], rel_tol=0, abs_tol=1e-6)
    ):
        raise ValueError("metadata timestamp differs from source sample")
    synchronization = metadata.get("synchronization", {})
    if synchronization.get("exact_carla_frame_match") is not True:
        raise ValueError("sample lacks exact frame synchronization evidence")
    camera = metadata.get("camera", {})
    transform = camera.get("world_transform")
    if (
        not isinstance(transform, list)
        or len(transform) != 6
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in transform
        )
    ):
        raise ValueError("camera world transform must contain six finite coordinates")
    fov = camera.get("fov_degrees")
    if (
        isinstance(fov, bool)
        or not isinstance(fov, (int, float))
        or not math.isfinite(fov)
        or not 0 < fov < 180
    ):
        raise ValueError("camera FOV must be finite and in (0,180)")
    height, width = camera.get("height"), camera.get("width")
    if (height, width) != (image.get("height"), image.get("width")):
        raise ValueError("COCO dimensions disagree with sample metadata")
    rgb = cv2.imread(str(root / sample["rgb"]["path"]), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(root / sample["instance_mask"]["path"]), cv2.IMREAD_UNCHANGED)
    if rgb is None or rgb.shape != (height, width, 3):
        raise ValueError("RGB image cannot be decoded at its declared dimensions")
    if mask is None or mask.shape != (height, width, 4):
        raise ValueError("instance mask cannot be decoded at its declared dimensions")
    labels = extract_instance_labels(
        mask,
        minimum_pixels=config.get("minimum_pixels", 16),
        minimum_box_width=config.get("minimum_box_width", 2),
        minimum_box_height=config.get("minimum_box_height", 2),
    )
    if [label.as_dict() for label in labels] != metadata.get("annotations"):
        raise ValueError("instance mask does not reproduce metadata annotations")
    if sample.get("annotation_count") != len(labels):
        raise ValueError("sample annotation count disagrees with instance mask")
    verify_coco_labels(labels, annotations, image_id=sample["image_id"])
    # Export consistency; independent golden tests must still check the extractor.
    from .writer import DatasetWriter

    expected_yolo = DatasetWriter._yolo_text(labels, width=width, height=height)
    if (root / sample["yolo_label"]["path"]).read_text(encoding="utf-8") != expected_yolo:
        raise ValueError("YOLO export disagrees with canonical instance labels")
