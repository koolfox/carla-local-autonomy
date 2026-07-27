from __future__ import annotations

import numpy as np

from carla_vision.voxel.contracts import VoxelGridSpec
from carla_vision.voxel.planner import (
    TrajectoryCandidate,
    generate_constant_curvature_trajectories,
    persistence_forecast,
    score_trajectory,
    select_best_trajectory,
)


def _spec() -> VoxelGridSpec:
    return VoxelGridSpec(
        x_min=0.0,
        x_max=12.0,
        y_min=-6.0,
        y_max=6.0,
        z_min=-1.0,
        z_max=3.0,
        resolution=1.0,
    )


def test_generate_candidates_returns_forward_paths() -> None:
    candidates = generate_constant_curvature_trajectories(
        (-0.4, 0.0, 0.4),
        speed_mps=3.0,
        dt_s=0.25,
        steps=8,
    )
    assert len(candidates) == 3
    assert all(candidate.points_xy[-1, 0] > 0 for candidate in candidates)
    assert candidates[0].points_xy[-1, 1] < 0
    assert candidates[2].points_xy[-1, 1] > 0


def test_planner_avoids_occupied_straight_corridor() -> None:
    spec = _spec()
    occupancy = np.zeros(spec.shape, dtype=np.float32)
    for x in (3, 4, 5, 6):
        point = np.asarray([[float(x), 0.0, 0.5]], dtype=np.float32)
        indices, valid = spec.metric_to_indices(point)
        assert bool(valid[0])
        iz, iy, ix = indices[0]
        occupancy[:, max(0, iy - 1) : iy + 2, ix] = 1.0
    candidates = generate_constant_curvature_trajectories(
        (-0.65, 0.0, 0.65),
        speed_mps=3.0,
        dt_s=0.25,
        steps=12,
    )
    best, scores = select_best_trajectory(
        candidates,
        occupancy,
        spec=spec,
        ego_radius_m=0.4,
        collision_weight=100.0,
        unknown_weight=0.0,
        curvature_weight=0.1,
        progress_weight=0.2,
    )
    straight_index = 1
    assert best.steering != 0.0
    assert scores[straight_index].collision > min(scores[0].collision, scores[2].collision)


def test_unknown_space_is_penalized() -> None:
    spec = _spec()
    candidate = TrajectoryCandidate(
        steering=0.0,
        points_xy=np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
    )
    unknown = np.full(spec.shape, -1.0, dtype=np.float32)
    free = np.zeros(spec.shape, dtype=np.float32)
    unknown_score = score_trajectory(candidate, unknown, spec=spec)
    free_score = score_trajectory(candidate, free, spec=spec)
    assert unknown_score.total > free_score.total


def test_persistence_forecast_repeats_current_grid() -> None:
    current = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    forecast = persistence_forecast(current, (0.0, 0.5, 1.0))
    assert forecast.shape == (3, 2, 3, 4)
    assert np.array_equal(forecast[2], current)
