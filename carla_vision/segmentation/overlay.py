"""Canonical, framework-independent road-segmentation visualization."""

from __future__ import annotations

import cv2
import numpy as np

from .contracts import CANONICAL_CLASS_NAMES, RoadClass, SegmentationResult

ROAD_CLASS_COLORS_BGR: dict[RoadClass, tuple[int, int, int]] = {
    RoadClass.ROAD: (220, 145, 45),
    RoadClass.ROAD_LINE: (35, 225, 255),
    RoadClass.SIDEWALK: (225, 105, 190),
    RoadClass.NON_DRIVABLE_GROUND: (75, 175, 75),
}

_CLASS_ALPHA = {
    RoadClass.ROAD: 0.34,
    RoadClass.ROAD_LINE: 0.82,
    RoadClass.SIDEWALK: 0.40,
    RoadClass.NON_DRIVABLE_GROUND: 0.36,
}


def render_segmentation_overlay(
    image_bgr: np.ndarray,
    result: SegmentationResult,
    *,
    draw_legend: bool = True,
) -> np.ndarray:
    """Color an exact source frame without mutating the raw image or mask."""

    if (
        not isinstance(image_bgr, np.ndarray)
        or image_bgr.dtype != np.uint8
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
    ):
        raise ValueError("image_bgr must be a uint8 HxWx3 array")
    if result.class_ids.shape != image_bgr.shape[:2]:
        raise ValueError("segmentation overlay requires an exact-frame mask")

    output = image_bgr.copy()
    present: list[RoadClass] = []
    for road_class, color in ROAD_CLASS_COLORS_BGR.items():
        mask = result.class_ids == int(road_class)
        if not np.any(mask):
            continue
        present.append(road_class)
        confidence = result.confidence[mask].astype(np.float32, copy=False)
        alpha = _CLASS_ALPHA[road_class] * (0.55 + 0.45 * confidence)
        source = output[mask].astype(np.float32)
        target = np.asarray(color, dtype=np.float32)
        output[mask] = np.clip(
            source * (1.0 - alpha[:, None]) + target * alpha[:, None],
            0.0,
            255.0,
        ).astype(np.uint8)

    if draw_legend and present:
        _draw_legend(output, present)
    return output


def _draw_legend(image: np.ndarray, classes: list[RoadClass]) -> None:
    height, width = image.shape[:2]
    line_height = 20
    panel_width = min(width, 220)
    panel_height = min(height, 10 + line_height * len(classes))
    top = max(0, height - panel_height)
    panel = image.copy()
    cv2.rectangle(panel, (0, top), (max(0, panel_width - 1), height - 1), (18, 18, 18), -1)
    cv2.addWeighted(panel, 0.68, image, 0.32, 0.0, image)
    for index, road_class in enumerate(classes):
        y = top + 7 + line_height * index
        color = ROAD_CLASS_COLORS_BGR[road_class]
        cv2.rectangle(image, (8, y), (20, min(height - 1, y + 12)), color, -1)
        cv2.putText(
            image,
            CANONICAL_CLASS_NAMES[int(road_class)].replace("_", " ").upper(),
            (27, min(height - 3, y + 11)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )


__all__ = ["ROAD_CLASS_COLORS_BGR", "render_segmentation_overlay"]
