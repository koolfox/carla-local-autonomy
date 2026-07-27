from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .contracts import Detection

DEFAULT_DYNAMIC_ROAD_USERS = frozenset(
    {"person", "pedestrian", "bicycle", "car", "motorcycle", "bus", "truck"}
)


@dataclass(frozen=True)
class DetectionAssessment:
    detection_index: int
    in_driving_corridor: bool
    visually_close: bool
    hazard: bool
    confidence_threshold: float


@dataclass(frozen=True)
class RiskAssessment:
    items: tuple[DetectionAssessment, ...]
    corridor_xyxy: tuple[int, int, int, int]

    @property
    def hazard(self) -> bool:
        return any(item.hazard for item in self.items)

    @property
    def hazard_indices(self) -> frozenset[int]:
        return frozenset(item.detection_index for item in self.items if item.hazard)


class HazardPolicy:
    """Conservative image-space gate, deliberately separate from the detector."""

    def __init__(
        self,
        dynamic_labels: Iterable[str] = DEFAULT_DYNAMIC_ROAD_USERS,
        *,
        default_confidence: float = 0.45,
        vulnerable_confidence: float = 0.35,
    ) -> None:
        self.dynamic_labels = frozenset(label.lower() for label in dynamic_labels)
        self.default_confidence = default_confidence
        self.vulnerable_confidence = vulnerable_confidence

    def assess(
        self,
        detections: tuple[Detection, ...],
        image_width: int,
        image_height: int,
    ) -> RiskAssessment:
        left = int(0.20 * image_width)
        right = int(0.80 * image_width)
        top = int(0.55 * image_height)
        items: list[DetectionAssessment] = []
        for index, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.xyxy
            center_x = (x1 + x2) * 0.5
            box_height = max(0.0, y2 - y1)
            box_area = max(0.0, x2 - x1) * box_height
            in_corridor = left <= center_x <= right and y2 >= top
            visually_close = (
                box_height >= 0.18 * image_height or box_area >= 0.04 * image_width * image_height
            )
            label = detection.label.lower()
            confidence_threshold = (
                self.vulnerable_confidence
                if label in {"person", "pedestrian", "bicycle"}
                else self.default_confidence
            )
            hazard = (
                label in self.dynamic_labels
                and detection.confidence >= confidence_threshold
                and in_corridor
                and visually_close
            )
            items.append(
                DetectionAssessment(
                    detection_index=index,
                    in_driving_corridor=in_corridor,
                    visually_close=visually_close,
                    hazard=hazard,
                    confidence_threshold=confidence_threshold,
                )
            )
        return RiskAssessment(
            items=tuple(items),
            corridor_xyxy=(left, top, right, image_height - 1),
        )
