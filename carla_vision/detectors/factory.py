from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, cast

from ..contracts import Detector, DetectorConfig
from .ultralytics import UltralyticsDetector


def _load_custom_factory(reference: str) -> Callable[[DetectorConfig], Detector]:
    module_name, separator, attribute_name = reference.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("custom factory must use module:callable syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise TypeError(f"custom detector factory {reference!r} is not callable")
    return cast(Callable[[DetectorConfig], Detector], factory)


def _validate_detector(candidate: Any) -> Detector:
    missing = [
        name for name in ("name", "metadata", "infer", "close") if not hasattr(candidate, name)
    ]
    if missing:
        raise TypeError(f"detector is missing required members: {', '.join(missing)}")
    if not callable(candidate.infer) or not callable(candidate.close):
        raise TypeError("detector infer and close members must be callable")
    return cast(Detector, candidate)


def create_detector(config: DetectorConfig) -> Detector:
    """Create a detector without exposing backend-specific APIs to the runtime."""

    backend = config.backend.strip().lower().replace("_", "-")
    sign_settings = config.options.get("sign_classifier")
    if sign_settings is not None and backend not in {"m9-hierarchical", "m9-hierarchical-rtdetr"}:
        raise ValueError("DeiT sign recognition currently requires M9 Hierarchical RT-DETR")
    if backend in {"yolo", "ultralytics-yolo"}:
        return UltralyticsDetector(config, architecture="yolo")
    if backend in {"rtdetr", "rt-detr", "ultralytics-rtdetr"}:
        return UltralyticsDetector(config, architecture="rtdetr")
    if backend in {"m9-hierarchical", "m9-hierarchical-rtdetr"}:
        from .m9_hierarchical import M9HierarchicalDetector

        if sign_settings is None:
            return M9HierarchicalDetector(config)
        from .deit64 import DeiT64Classifier, SignRecognitionDetector
        from .sign_config import SignClassifierConfig

        sign_config = SignClassifierConfig.from_mapping(sign_settings)
        classifier = DeiT64Classifier(sign_config, device=config.device)
        try:
            detector = M9HierarchicalDetector(config)
        except BaseException:
            classifier.close()
            raise
        return SignRecognitionDetector(detector, classifier)
    if backend == "custom":
        if config.factory is None:
            raise ValueError("custom detector requires factory")
        return _validate_detector(_load_custom_factory(config.factory)(config))
    raise ValueError(f"unsupported detector backend {config.backend!r}")
