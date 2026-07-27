"""Occupancy-aware trajectory generation and scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from .contracts import VoxelGridSpec


@dataclass(frozen=True, slots=True)
class TrajectoryCandidate:
    """Ego-frame trajectory sampled at fixed future steps."""

    steering: float
    points_xy: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points_xy, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
            raise ValueError("points_xy must have shape (T, 2) with at least one point")
        if not np.isfinite(points).all() or not math.isfinite(self.steering):
            raise ValueError("trajectory values must be finite")
        object.__setattr__(self, "points_xy", points)


@dataclass(frozen=True, slots=True)
class TrajectoryScore:
    total: float
    collision: float
    unknown: float
    curvature: float
    progress: float


def generate_constant_curvature_trajectories(
    steering_values: Sequence[float],
    *,
    speed_mps: float,
    wheelbase_m: float = 2.8,
    dt_s: float = 0.25,
    steps: int = 12,
) -> tuple[TrajectoryCandidate, ...]:
    """Generate simple bicycle-model candidates in the current ego frame."""

    if not math.isfinite(speed_mps) or speed_mps < 0:
        raise ValueError("speed_mps must be finite and non-negative")
    if not math.isfinite(wheelbase_m) or wheelbase_m <= 0:
        raise ValueError("wheelbase_m must be finite and positive")
    if not math.isfinite(dt_s) or dt_s <= 0 or steps <= 0:
        raise ValueError("dt_s and steps must be positive")
    candidates: list[TrajectoryCandidate] = []
    for steering in steering_values:
        if not math.isfinite(steering) or not -1.0 <= steering <= 1.0:
            raise ValueError("steering values must be finite and in [-1, 1]")
        x = 0.0
        y = 0.0
        yaw = 0.0
        points: list[tuple[float, float]] = []
        steering_angle = float(steering) * math.radians(35.0)
        for _ in range(steps):
            yaw += speed_mps * math.tan(steering_angle) / wheelbase_m * dt_s
            x += speed_mps * math.cos(yaw) * dt_s
            y += speed_mps * math.sin(yaw) * dt_s
            points.append((x, y))
        candidates.append(
            TrajectoryCandidate(
                steering=float(steering),
                points_xy=np.asarray(points, dtype=np.float32),
            )
        )
    if not candidates:
        raise ValueError("at least one steering value is required")
    return tuple(candidates)


def _grid_time_slice(future_occupancy: np.ndarray, step: int, steps: int) -> np.ndarray:
    occupancy = np.asarray(future_occupancy, dtype=np.float32)
    if occupancy.ndim == 3:
        return occupancy
    if occupancy.ndim != 4 or occupancy.shape[0] == 0:
        raise ValueError("future_occupancy must have shape (Z,Y,X) or (T,Z,Y,X)")
    if steps <= 1:
        index = 0
    else:
        index = round(step * (occupancy.shape[0] - 1) / (steps - 1))
    return occupancy[index]


def _footprint_values(
    occupancy_zyx: np.ndarray,
    *,
    spec: VoxelGridSpec,
    x: float,
    y: float,
    radius_m: float,
    z_min_m: float,
    z_max_m: float,
) -> np.ndarray:
    if radius_m < 0:
        raise ValueError("radius_m must be non-negative")
    center = np.asarray([[x, y, (z_min_m + z_max_m) / 2.0]], dtype=np.float32)
    center_indices, valid = spec.metric_to_indices(center)
    if not bool(valid[0]):
        return np.asarray([], dtype=np.float32)
    iz, iy, ix = center_indices[0]
    radius_cells = math.ceil(radius_m / spec.resolution)
    z_start = max(0, math.floor((z_min_m - spec.z_min) / spec.resolution))
    z_end = min(spec.shape[0], math.ceil((z_max_m - spec.z_min) / spec.resolution))
    y_start = max(0, iy - radius_cells)
    y_end = min(spec.shape[1], iy + radius_cells + 1)
    x_start = max(0, ix - radius_cells)
    x_end = min(spec.shape[2], ix + radius_cells + 1)
    if z_end <= z_start or y_end <= y_start or x_end <= x_start:
        return np.asarray([], dtype=np.float32)
    return occupancy_zyx[z_start:z_end, y_start:y_end, x_start:x_end].reshape(-1)


def score_trajectory(
    candidate: TrajectoryCandidate,
    future_occupancy: np.ndarray,
    *,
    spec: VoxelGridSpec,
    ego_radius_m: float = 1.2,
    collision_weight: float = 100.0,
    unknown_weight: float = 4.0,
    curvature_weight: float = 1.0,
    progress_weight: float = 1.0,
    z_min_m: float = -1.0,
    z_max_m: float = 2.5,
) -> TrajectoryScore:
    """Score one path against present or future occupancy probabilities.

    Negative occupancy values are treated as unknown. Values in [0, 1] are
    interpreted as occupied probabilities.
    """

    collision_cost = 0.0
    unknown_cost = 0.0
    steps = candidate.points_xy.shape[0]
    for step, (x, y) in enumerate(candidate.points_xy):
        grid = _grid_time_slice(future_occupancy, step, steps)
        values = _footprint_values(
            grid,
            spec=spec,
            x=float(x),
            y=float(y),
            radius_m=ego_radius_m,
            z_min_m=z_min_m,
            z_max_m=z_max_m,
        )
        if values.size == 0:
            unknown_cost += 1.0
            continue
        known = values >= 0.0
        unknown_cost += float(np.mean(~known))
        if np.any(known):
            collision_cost += float(np.max(np.clip(values[known], 0.0, 1.0)))
    curvature_cost = abs(candidate.steering)
    progress = float(candidate.points_xy[-1, 0])
    total = (
        collision_weight * collision_cost
        + unknown_weight * unknown_cost
        + curvature_weight * curvature_cost
        - progress_weight * progress
    )
    return TrajectoryScore(
        total=float(total),
        collision=float(collision_cost),
        unknown=float(unknown_cost),
        curvature=float(curvature_cost),
        progress=progress,
    )


def select_best_trajectory(
    candidates: Sequence[TrajectoryCandidate],
    future_occupancy: np.ndarray,
    *,
    spec: VoxelGridSpec,
    **score_options: float,
) -> tuple[TrajectoryCandidate, tuple[TrajectoryScore, ...]]:
    if not candidates:
        raise ValueError("at least one candidate trajectory is required")
    scores = tuple(
        score_trajectory(candidate, future_occupancy, spec=spec, **score_options)
        for candidate in candidates
    )
    best_index = min(range(len(scores)), key=lambda index: scores[index].total)
    return candidates[best_index], scores


def persistence_forecast(
    current_occupancy: np.ndarray,
    horizons_s: Sequence[float],
) -> np.ndarray:
    """Repeat the current occupancy as an explicit non-learned forecast baseline."""

    occupancy = np.asarray(current_occupancy, dtype=np.float32)
    if occupancy.ndim != 3:
        raise ValueError("current_occupancy must have shape (Z, Y, X)")
    horizons = tuple(float(value) for value in horizons_s)
    if not horizons or any(not math.isfinite(value) or value < 0 for value in horizons):
        raise ValueError("horizons must be finite and non-negative")
    return np.repeat(occupancy[None, ...], len(horizons), axis=0)
