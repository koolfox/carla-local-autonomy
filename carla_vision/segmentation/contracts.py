"""Model-neutral contracts for RGB road-scene semantic segmentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

DEFAULT_SEGFORMER_B0_CHECKPOINT = (
    "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
)


class RoadClass(IntEnum):
    """Stable class identifiers shared by every road-segmentation backend."""

    OTHER = 0
    ROAD = 1
    ROAD_LINE = 2
    SIDEWALK = 3
    NON_DRIVABLE_GROUND = 4


CANONICAL_CLASS_NAMES = (
    "other",
    "road",
    "road_line",
    "sidewalk",
    "non_drivable_ground",
)


@dataclass(frozen=True)
class SegmentationConfig:
    """Serializable selection for an RGB semantic-segmentation adapter."""

    backend: str = "segformer-b0"
    checkpoint: str | Path | None = DEFAULT_SEGFORMER_B0_CHECKPOINT
    device: str = "cpu"
    factory: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        backend = self.backend.strip().lower().replace("_", "-")
        if not backend:
            raise ValueError("segmentation backend must not be empty")
        if not self.device.strip():
            raise ValueError("segmentation device must not be empty")
        if backend == "custom":
            if not self.factory:
                raise ValueError("custom segmenter requires a module:callable factory")
            module_name, separator, callable_name = self.factory.partition(":")
            if not separator or not module_name or not callable_name:
                raise ValueError("custom segmenter factory must use module:callable syntax")
        elif self.factory is not None:
            raise ValueError("factory is allowed only for the custom segmentation backend")
        if backend != "custom" and (
            self.checkpoint is None or not str(self.checkpoint).strip()
        ):
            raise ValueError("built-in segmenter requires a checkpoint")
        if not isinstance(self.options, Mapping):
            raise TypeError("segmentation options must be a mapping")
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "checkpoint": str(self.checkpoint) if self.checkpoint is not None else None,
            "device": self.device,
            "factory": self.factory,
            "options": dict(self.options),
        }


@dataclass(frozen=True)
class SegmentationMetadata:
    """Auditable model identity and source-to-canonical label mapping."""

    name: str
    backend: str
    checkpoint: str | None
    device: str
    revision: str | None = None
    resolved_revision: str | None = None
    source_labels: Mapping[int, str] = field(default_factory=dict)
    source_to_canonical: Mapping[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.backend.strip() or not self.device.strip():
            raise ValueError("segmentation metadata name, backend, and device must not be empty")
        source_labels = {int(key): str(value) for key, value in self.source_labels.items()}
        source_to_canonical = {
            int(key): str(value) for key, value in self.source_to_canonical.items()
        }
        if any(key < 0 or not value.strip() for key, value in source_labels.items()):
            raise ValueError("source labels require non-negative ids and non-empty names")
        if set(source_to_canonical) != set(source_labels):
            raise ValueError("every source label must have one canonical mapping")
        if any(value not in CANONICAL_CLASS_NAMES for value in source_to_canonical.values()):
            raise ValueError("source labels must map to canonical road classes")
        object.__setattr__(self, "source_labels", MappingProxyType(source_labels))
        object.__setattr__(
            self,
            "source_to_canonical",
            MappingProxyType(source_to_canonical),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "checkpoint": self.checkpoint,
            "device": self.device,
            "revision": self.revision,
            "resolved_revision": self.resolved_revision,
            "supports_road_line": "road_line" in self.source_to_canonical.values(),
            "canonical_classes": list(CANONICAL_CLASS_NAMES),
            "source_labels": dict(self.source_labels),
            "source_to_canonical": dict(self.source_to_canonical),
        }


@dataclass(frozen=True)
class SegmentationResult:
    """Canonical class and confidence for every pixel in the input RGB frame."""

    class_ids: np.ndarray
    confidence: np.ndarray
    model_name: str

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("segmentation result model_name must not be empty")
        if not isinstance(self.class_ids, np.ndarray) or self.class_ids.ndim != 2:
            raise ValueError("class_ids must be a 2D numpy array")
        if not np.issubdtype(self.class_ids.dtype, np.integer):
            raise ValueError("class_ids must contain integers")
        if self.class_ids.size == 0:
            raise ValueError("class_ids must not be empty")
        if int(self.class_ids.min()) < 0 or int(self.class_ids.max()) >= len(
            CANONICAL_CLASS_NAMES
        ):
            raise ValueError("class_ids contain an unknown canonical class")
        if not isinstance(self.confidence, np.ndarray) or self.confidence.shape != (
            self.class_ids.shape
        ):
            raise ValueError("confidence must be a numpy array matching class_ids")
        if not np.issubdtype(self.confidence.dtype, np.floating):
            raise ValueError("confidence must contain floating-point values")
        if not np.all(np.isfinite(self.confidence)) or np.any(self.confidence < 0.0) or np.any(
            self.confidence > 1.0
        ):
            raise ValueError("confidence values must be finite and in [0, 1]")

        class_ids = np.ascontiguousarray(self.class_ids, dtype=np.uint8).copy()
        confidence = np.ascontiguousarray(self.confidence, dtype=np.float32).copy()
        class_ids.setflags(write=False)
        confidence.setflags(write=False)
        object.__setattr__(self, "class_ids", class_ids)
        object.__setattr__(self, "confidence", confidence)

    def mask_for(self, road_class: RoadClass) -> np.ndarray:
        """Return an immutable boolean mask for one canonical class."""

        if not isinstance(road_class, RoadClass):
            raise TypeError("road_class must be a RoadClass")
        mask = self.class_ids == int(road_class)
        mask.setflags(write=False)
        return mask


@runtime_checkable
class RoadSegmenter(Protocol):
    """Backend-independent RGB semantic-segmentation interface."""

    @property
    def name(self) -> str: ...

    @property
    def metadata(self) -> SegmentationMetadata: ...

    def infer(self, image_bgr: np.ndarray) -> SegmentationResult: ...

    def close(self) -> None: ...


__all__ = [
    "CANONICAL_CLASS_NAMES",
    "DEFAULT_SEGFORMER_B0_CHECKPOINT",
    "RoadClass",
    "RoadSegmenter",
    "SegmentationConfig",
    "SegmentationMetadata",
    "SegmentationResult",
]
