from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Detection:
    """Model-independent object detection in original-image pixel coordinates."""

    class_id: int
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    source_class_id: int | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.class_id < 0:
            raise ValueError("class_id must be non-negative")
        if not self.label:
            raise ValueError("label must not be empty")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        if len(self.xyxy) != 4 or not all(math.isfinite(value) for value in self.xyxy):
            raise ValueError("xyxy must contain four finite values")
        x1, y1, x2, y2 = self.xyxy
        if x2 < x1 or y2 < y1:
            raise ValueError("xyxy must have non-negative width and height")


@dataclass(frozen=True)
class DetectorMetadata:
    """Auditable identity and runtime configuration for a detector."""

    name: str
    backend: str
    weights: str | None
    device: str
    image_size: int
    confidence: float
    extra: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "weights": self.weights,
            "device": self.device,
            "image_size": self.image_size,
            "confidence": self.confidence,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class DetectorConfig:
    """Serializable detector selection used by the CLI and run manifests."""

    backend: str
    weights: Path | None = None
    device: str = "cpu"
    image_size: int = 640
    confidence: float = 0.25
    factory: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend must not be empty")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.backend == "custom" and not self.factory:
            raise ValueError("custom detector requires a factory in module:callable form")

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "weights": str(self.weights) if self.weights is not None else None,
            "device": self.device,
            "image_size": self.image_size,
            "confidence": self.confidence,
            "factory": self.factory,
            "options": dict(self.options),
        }


@runtime_checkable
class Detector(Protocol):
    """Runtime contract implemented by every model backend."""

    @property
    def name(self) -> str: ...

    @property
    def metadata(self) -> DetectorMetadata: ...

    def infer(self, image_bgr: np.ndarray) -> tuple[Detection, ...]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class PerceptionResult:
    """Detections tied to the exact RGB frame used for inference."""

    sequence: int
    carla_frame: int
    source_timestamp: float
    source_received_monotonic: float
    completed_monotonic: float
    detections: tuple[Detection, ...]
    source_bgr: np.ndarray
    detector_name: str = "unknown"
    inference_started_monotonic: float | None = None
    source_transform: tuple[float, float, float, float, float, float] | None = None
    source_fov: float | None = None

    def __post_init__(self) -> None:
        if self.sequence < 0 or self.carla_frame < 0:
            raise ValueError("frame identifiers must be non-negative")
        if self.completed_monotonic < self.source_received_monotonic:
            raise ValueError("completion time cannot precede frame receipt")
        if self.inference_started_monotonic is not None and not (
            self.source_received_monotonic
            <= self.inference_started_monotonic
            <= self.completed_monotonic
        ):
            raise ValueError("inference start must be between frame receipt and completion")
        if (
            not isinstance(self.source_bgr, np.ndarray)
            or self.source_bgr.dtype != np.uint8
            or self.source_bgr.ndim != 3
            or self.source_bgr.shape[2] != 3
        ):
            raise ValueError("source_bgr must be a uint8 HxWx3 array")

    @property
    def inference_seconds(self) -> float:
        """End-to-end age from client frame receipt through model completion."""

        return self.completed_monotonic - self.source_received_monotonic

    @property
    def model_inference_seconds(self) -> float | None:
        if self.inference_started_monotonic is None:
            return None
        return self.completed_monotonic - self.inference_started_monotonic

    def age_seconds(self, now_monotonic: float) -> float:
        return max(0.0, now_monotonic - self.source_received_monotonic)
