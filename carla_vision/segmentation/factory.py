"""Factory for model-neutral road-segmentation adapters."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, cast

from .contracts import RoadSegmenter, SegmentationConfig, SegmentationMetadata
from .segformer import SegFormerSegmenter


def _load_custom_factory(reference: str) -> Callable[[SegmentationConfig], RoadSegmenter]:
    module_name, separator, attribute_name = reference.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("custom segmenter factory must use module:callable syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise TypeError(f"custom segmenter factory {reference!r} is not callable")
    return cast(Callable[[SegmentationConfig], RoadSegmenter], factory)


def _validate_segmenter(candidate: Any) -> RoadSegmenter:
    missing = [
        name for name in ("name", "metadata", "infer", "close") if not hasattr(candidate, name)
    ]
    if missing:
        raise TypeError(f"segmenter is missing required members: {', '.join(missing)}")
    if not isinstance(candidate.name, str) or not candidate.name.strip():
        raise TypeError("segmenter name must be a non-empty string")
    if not isinstance(candidate.metadata, SegmentationMetadata):
        raise TypeError("segmenter metadata must be SegmentationMetadata")
    if not callable(candidate.infer) or not callable(candidate.close):
        raise TypeError("segmenter infer and close members must be callable")
    return cast(RoadSegmenter, candidate)


def create_segmenter(config: SegmentationConfig) -> RoadSegmenter:
    """Create a segmenter without exposing its framework-specific API."""

    backend = config.backend.strip().lower().replace("_", "-")
    if backend in {"segformer", "segformer-b0", "hf-segformer", "huggingface-segformer"}:
        return SegFormerSegmenter(config)
    if backend == "custom":
        if config.factory is None:
            raise ValueError("custom segmenter requires factory")
        return _validate_segmenter(_load_custom_factory(config.factory)(config))
    raise ValueError(f"unsupported segmentation backend {config.backend!r}")


__all__ = ["create_segmenter"]
