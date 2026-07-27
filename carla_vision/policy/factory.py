"""Vision-policy factory with strict custom-adapter validation."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, cast

from .builtin import HazardStopShadowPolicy
from .contracts import VisionPolicy, VisionPolicyConfig


def _load_factory(reference: str) -> Callable[[VisionPolicyConfig], VisionPolicy]:
    module_name, separator, attribute_name = reference.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("custom vision policy factory must use module:callable syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise TypeError(f"custom vision policy factory {reference!r} is not callable")
    return cast(Callable[[VisionPolicyConfig], VisionPolicy], factory)


def _validate(candidate: Any) -> VisionPolicy:
    missing = [
        name for name in ("metadata", "reset", "propose", "close") if not hasattr(candidate, name)
    ]
    if missing:
        raise TypeError(f"vision policy is missing required members: {', '.join(missing)}")
    if not all(callable(getattr(candidate, name)) for name in ("reset", "propose", "close")):
        raise TypeError("vision policy reset, propose, and close members must be callable")
    return cast(VisionPolicy, candidate)


def create_vision_policy(config: VisionPolicyConfig) -> VisionPolicy:
    backend = config.backend.strip().lower().replace("_", "-")
    if backend == "hazard-stop":
        return HazardStopShadowPolicy(config)
    if backend == "custom":
        if config.factory is None:
            raise ValueError("custom vision policy requires factory")
        return _validate(_load_factory(config.factory)(config))
    raise ValueError(f"unsupported vision policy backend {config.backend!r}")


__all__ = ["create_vision_policy"]
