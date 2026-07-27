"""A diagnostic predictor that marks every voxel free.

This validates the RGB-only predictor interface and artifact path. It is not a
perception model and must not be used for driving decisions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import CameraVoxelPrediction, VoxelGridSpec
from ..models import CameraVoxelModelConfig


class EmptySpacePredictor:
    def __init__(self, horizons_s: tuple[float, ...]) -> None:
        self.horizons_s = horizons_s

    def predict(
        self,
        rgb_history: tuple[np.ndarray, ...],
        spec: VoxelGridSpec,
    ) -> CameraVoxelPrediction:
        if not rgb_history:
            raise ValueError("rgb_history must not be empty")
        for image in rgb_history:
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("RGB history must contain uint8 HxWx3 images")
        occupancy = np.zeros((len(self.horizons_s), *spec.shape), dtype=np.float32)
        return CameraVoxelPrediction(
            occupancy_probability=occupancy,
            horizons_s=self.horizons_s,
            metadata={"diagnostic_only": True, "uses_privileged_input": False},
        )


def create_predictor(config: CameraVoxelModelConfig) -> EmptySpacePredictor:
    raw_horizons: Any = config.options.get("horizons_s", (0.0, 0.5, 1.0, 2.0))
    horizons = tuple(float(value) for value in raw_horizons)
    return EmptySpacePredictor(horizons)
