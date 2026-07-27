"""Pluggable RGB-only voxel predictor loading."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import CameraVoxelPredictor


@dataclass(frozen=True, slots=True)
class CameraVoxelModelConfig:
    checkpoint: Path | None = None
    device: str = "cpu"
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.device.strip():
            raise ValueError("voxel model device must not be empty")


def _load_factory(reference: str) -> Callable[[CameraVoxelModelConfig], Any]:
    module_name, separator, attribute_name = reference.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("voxel model factory must use module:callable syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise TypeError(f"voxel model factory {reference!r} is not callable")
    return factory


def create_camera_voxel_predictor(
    reference: str,
    config: CameraVoxelModelConfig,
) -> CameraVoxelPredictor:
    candidate = _load_factory(reference)(config)
    if not callable(getattr(candidate, "predict", None)):
        raise TypeError("camera voxel predictor must define predict(rgb_history, spec)")
    return candidate


__all__ = [
    "CameraVoxelModelConfig",
    "create_camera_voxel_predictor",
]
