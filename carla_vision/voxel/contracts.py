"""Typed contracts for camera-centric voxel occupancy and forecasting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

UNKNOWN = np.int8(-1)
FREE = np.int8(0)
OCCUPIED = np.int8(1)
SEMANTIC_UNKNOWN = np.uint8(255)


@dataclass(frozen=True, slots=True)
class VoxelGridSpec:
    """Ego-centric metric bounds and resolution for a dense voxel grid.

    Arrays use ``(z, y, x)`` order. Metric coordinates follow CARLA/Unreal:
    x forward, y right, z up.
    """

    x_min: float = 0.0
    x_max: float = 50.0
    y_min: float = -25.0
    y_max: float = 25.0
    z_min: float = -2.0
    z_max: float = 5.0
    resolution: float = 0.5

    def __post_init__(self) -> None:
        values = (
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.z_min,
            self.z_max,
            self.resolution,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("voxel-grid values must be finite")
        if self.resolution <= 0:
            raise ValueError("voxel resolution must be positive")
        for axis, minimum, maximum in (
            ("x", self.x_min, self.x_max),
            ("y", self.y_min, self.y_max),
            ("z", self.z_min, self.z_max),
        ):
            if maximum <= minimum:
                raise ValueError(f"{axis}_max must be greater than {axis}_min")

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return array shape in ``(z, y, x)`` order."""

        return (
            math.ceil((self.z_max - self.z_min) / self.resolution),
            math.ceil((self.y_max - self.y_min) / self.resolution),
            math.ceil((self.x_max - self.x_min) / self.resolution),
        )

    def as_dict(self) -> dict[str, float | list[int]]:
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "z_min": self.z_min,
            "z_max": self.z_max,
            "resolution": self.resolution,
            "shape_zyx": list(self.shape),
        }

    def metric_to_indices(
        self,
        points_xyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert ``N x 3`` metric points to ``N x 3`` ``(z, y, x)`` indices."""

        points = np.asarray(points_xyz, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_xyz must have shape (N, 3)")
        ix = np.floor((points[:, 0] - self.x_min) / self.resolution).astype(np.int64)
        iy = np.floor((points[:, 1] - self.y_min) / self.resolution).astype(np.int64)
        iz = np.floor((points[:, 2] - self.z_min) / self.resolution).astype(np.int64)
        nz, ny, nx = self.shape
        valid = (
            (ix >= 0)
            & (ix < nx)
            & (iy >= 0)
            & (iy < ny)
            & (iz >= 0)
            & (iz < nz)
        )
        return np.stack((iz, iy, ix), axis=1), valid


@dataclass(frozen=True, slots=True)
class CameraVoxelPrediction:
    """Camera-only model output for present and future occupancy.

    ``occupancy_probability`` has shape ``(T, Z, Y, X)`` and values in [0, 1].
    The first horizon may be 0.0 for the current frame.
    """

    occupancy_probability: np.ndarray
    horizons_s: tuple[float, ...]
    semantic_logits: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


class CameraVoxelPredictor(Protocol):
    """A deployable predictor that receives RGB history only."""

    def predict(
        self,
        rgb_history: tuple[np.ndarray, ...],
        spec: VoxelGridSpec,
    ) -> CameraVoxelPrediction: ...


def validate_camera_voxel_prediction(
    prediction: CameraVoxelPrediction,
    spec: VoxelGridSpec,
) -> CameraVoxelPrediction:
    occupancy = np.asarray(prediction.occupancy_probability, dtype=np.float32)
    expected_tail = spec.shape
    if occupancy.ndim != 4 or occupancy.shape[1:] != expected_tail:
        raise ValueError(
            "occupancy_probability must have shape "
            f"(T, {expected_tail[0]}, {expected_tail[1]}, {expected_tail[2]})"
        )
    if occupancy.shape[0] != len(prediction.horizons_s):
        raise ValueError("horizon count must match occupancy time dimension")
    if not np.isfinite(occupancy).all():
        raise ValueError("occupancy probabilities must be finite")
    if np.any((occupancy < 0.0) | (occupancy > 1.0)):
        raise ValueError("occupancy probabilities must be in [0, 1]")
    horizons = tuple(float(value) for value in prediction.horizons_s)
    if not horizons or any(not math.isfinite(value) or value < 0 for value in horizons):
        raise ValueError("forecast horizons must be finite and non-negative")
    if tuple(sorted(horizons)) != horizons or len(set(horizons)) != len(horizons):
        raise ValueError("forecast horizons must be unique and increasing")
    semantics = prediction.semantic_logits
    if semantics is not None:
        semantic_array = np.asarray(semantics, dtype=np.float32)
        if semantic_array.ndim != 5:
            raise ValueError("semantic_logits must have shape (T, C, Z, Y, X)")
        if semantic_array.shape[0] != occupancy.shape[0] or semantic_array.shape[2:] != expected_tail:
            raise ValueError("semantic_logits grid and time dimensions must match occupancy")
        if not np.isfinite(semantic_array).all():
            raise ValueError("semantic logits must be finite")
    return CameraVoxelPrediction(
        occupancy_probability=occupancy,
        horizons_s=horizons,
        semantic_logits=None if semantics is None else np.asarray(semantics, dtype=np.float32),
        metadata=dict(prediction.metadata or {}),
    )
