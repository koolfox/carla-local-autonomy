from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ontology import (
    DETECTOR_CATEGORY_BY_TAG,
    THING_TAGS,
    CarlaSemanticTag,
    DetectorCategory,
)


@dataclass(frozen=True, slots=True)
class InstanceLabel:
    """One visible CARLA actor instance measured in image pixel coordinates.

    ``bbox_xyxy`` uses half-open coordinates: ``x2`` and ``y2`` are one pixel
    beyond the box. ``truncated`` means the visible mask touches an image edge.
    """

    actor_id: int
    semantic_tag: CarlaSemanticTag
    category_id: int
    category_name: str
    bbox_xyxy: tuple[int, int, int, int]
    visible_area_pixels: int
    bbox_area_pixels: int
    fill_ratio: float
    truncated: bool

    @property
    def box_width(self) -> int:
        return self.bbox_xyxy[2] - self.bbox_xyxy[0]

    @property
    def box_height(self) -> int:
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]

    def as_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "semantic_tag": int(self.semantic_tag),
            "category_id": self.category_id,
            "category_name": self.category_name,
            "bbox_xyxy": list(self.bbox_xyxy),
            "visible_area_pixels": self.visible_area_pixels,
            "bbox_area_pixels": self.bbox_area_pixels,
            "fill_ratio": self.fill_ratio,
            "truncated": self.truncated,
        }


def _validate_bgra(image_bgra: np.ndarray) -> np.ndarray:
    if not isinstance(image_bgra, np.ndarray):
        raise TypeError("image_bgra must be a numpy array")
    if image_bgra.dtype != np.uint8:
        raise ValueError("image_bgra must have dtype uint8")
    if image_bgra.ndim != 3 or image_bgra.shape[2] != 4:
        raise ValueError("image_bgra must have shape HxWx4")
    if image_bgra.shape[0] <= 0 or image_bgra.shape[1] <= 0:
        raise ValueError("image_bgra width and height must be positive")
    return image_bgra


def decode_instance_bgra(image_bgra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode a CARLA instance-segmentation BGRA image.

    CARLA writes the semantic tag to the red channel. Its 16-bit actor id uses
    green as the low byte and blue as the high byte. Alpha is ignored.

    Returns:
        A ``(semantic_tags, actor_ids)`` pair shaped ``HxW``. Semantic tags are
        ``uint8`` and actor ids are ``uint16``.
    """

    image = _validate_bgra(image_bgra)
    semantic_tags = image[:, :, 2].copy()
    low = image[:, :, 1].astype(np.uint16)
    high = image[:, :, 0].astype(np.uint16)
    actor_ids = low | (high << 8)
    return semantic_tags, actor_ids


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalize_allowed_tags(
    allowed_tags: Iterable[CarlaSemanticTag | int],
) -> frozenset[CarlaSemanticTag]:
    try:
        tags = frozenset(CarlaSemanticTag(int(tag)) for tag in allowed_tags)
    except (TypeError, ValueError) as exc:
        raise ValueError("allowed_tags must contain CARLA semantic tags 0 through 28") from exc
    unsupported = tags.difference(DETECTOR_CATEGORY_BY_TAG)
    if unsupported:
        values = ", ".join(str(int(tag)) for tag in sorted(unsupported))
        raise ValueError(f"allowed_tags contains non-detector semantic tags: {values}")
    return tags


def _label_for_mask(
    mask: np.ndarray,
    *,
    actor_id: int,
    semantic_tag: CarlaSemanticTag,
    category: DetectorCategory,
    image_width: int,
    image_height: int,
) -> InstanceLabel:
    ys, xs = np.nonzero(mask)
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    visible_area = int(xs.size)
    bbox_area = (x2 - x1) * (y2 - y1)
    return InstanceLabel(
        actor_id=actor_id,
        semantic_tag=semantic_tag,
        category_id=category.id,
        category_name=category.name,
        bbox_xyxy=(x1, y1, x2, y2),
        visible_area_pixels=visible_area,
        bbox_area_pixels=bbox_area,
        fill_ratio=visible_area / bbox_area,
        truncated=x1 == 0 or y1 == 0 or x2 == image_width or y2 == image_height,
    )


def extract_instance_labels(
    image_bgra: np.ndarray,
    *,
    minimum_pixels: int = 1,
    minimum_box_width: int = 1,
    minimum_box_height: int = 1,
    allowed_tags: Iterable[CarlaSemanticTag | int] = THING_TAGS,
) -> tuple[InstanceLabel, ...]:
    """Extract deterministic detector labels from a CARLA instance frame.

    Pixels are grouped by the pair ``(semantic_tag, actor_id)``. Filters apply
    to the visible mask and its half-open bounding box. Only the ten detector
    thing tags in :mod:`carla_vision.dataset.ontology` are accepted.
    """

    minimum_pixels = _positive_integer("minimum_pixels", minimum_pixels)
    minimum_box_width = _positive_integer("minimum_box_width", minimum_box_width)
    minimum_box_height = _positive_integer("minimum_box_height", minimum_box_height)
    normalized_tags = _normalize_allowed_tags(allowed_tags)
    if not normalized_tags:
        return ()

    semantic_tags, actor_ids = decode_instance_bgra(image_bgra)
    height, width = semantic_tags.shape
    allowed_values = np.fromiter(
        (int(tag) for tag in normalized_tags),
        dtype=np.uint8,
        count=len(normalized_tags),
    )
    allowed_mask = np.isin(semantic_tags, allowed_values)
    if not bool(np.any(allowed_mask)):
        return ()

    # A 24-bit key keeps all uint8 semantic and uint16 actor-id bits intact.
    keys = (semantic_tags.astype(np.uint32) << 16) | actor_ids.astype(np.uint32)
    labels: list[InstanceLabel] = []
    for raw_key in np.unique(keys[allowed_mask]):
        key = int(raw_key)
        semantic_tag = CarlaSemanticTag(key >> 16)
        actor_id = key & 0xFFFF
        # CARLA actors use positive IDs.  Zero denotes missing/invalid instance
        # identity and must not merge unrelated pixels into one training box.
        if actor_id == 0:
            continue
        mask = allowed_mask & (keys == raw_key)
        visible_area = int(np.count_nonzero(mask))
        if visible_area < minimum_pixels:
            continue
        label = _label_for_mask(
            mask,
            actor_id=actor_id,
            semantic_tag=semantic_tag,
            category=DETECTOR_CATEGORY_BY_TAG[semantic_tag],
            image_width=width,
            image_height=height,
        )
        if label.box_width < minimum_box_width or label.box_height < minimum_box_height:
            continue
        labels.append(label)

    labels.sort(key=lambda item: (item.category_id, item.actor_id))
    return tuple(labels)
