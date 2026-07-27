"""Compatibility wrapper for the former YOLO-specific perception worker."""

from __future__ import annotations

from pathlib import Path

from carla_vision.contracts import Detection, DetectorConfig, PerceptionResult
from carla_vision.detectors import create_detector
from carla_vision.perception import PerceptionWorker


class YoloPerceptionWorker(PerceptionWorker):
    """Legacy constructor backed by the model-neutral perception runtime."""

    def __init__(
        self,
        weights: str | Path = "yolo26n.pt",
        image_size: int = 640,
        confidence: float = 0.2,
        device: str | None = None,
    ) -> None:
        self.device = device or "cpu"
        self.image_size = image_size
        self.confidence = confidence
        detector = create_detector(
            DetectorConfig(
                backend="yolo",
                weights=Path(weights),
                image_size=image_size,
                confidence=confidence,
                device=self.device,
            )
        )
        super().__init__(detector)


__all__ = ["Detection", "PerceptionResult", "YoloPerceptionWorker"]
