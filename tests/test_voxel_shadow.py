from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

import carla_vision.voxel.shadow as shadow_module
from carla_vision.voxel.contracts import CameraVoxelPrediction, VoxelGridSpec
from carla_vision.voxel.shadow import ShadowPlannerConfig, VoxelPlannerShadow, build_parser, run


class FakePredictor:
    def predict(self, rgb_history, spec):
        occupancy = np.zeros((2, *spec.shape), dtype=np.float32)
        indices, valid = spec.metric_to_indices(
            np.asarray([[4.0, 0.0, 0.5]], dtype=np.float32)
        )
        assert bool(valid[0])
        _iz, iy, ix = indices[0]
        occupancy[:, :, max(0, iy - 1) : iy + 2, ix] = 1.0
        return CameraVoxelPrediction(occupancy, (0.0, 1.0))


def _spec():
    return VoxelGridSpec(
        x_min=0,
        x_max=12,
        y_min=-6,
        y_max=6,
        z_min=-1,
        z_max=3,
        resolution=1,
    )


def test_shadow_warms_history_then_selects_without_actuation() -> None:
    planner = VoxelPlannerShadow(
        FakePredictor(),
        _spec(),
        ShadowPlannerConfig(
            history_frames=2,
            steering_values=(-0.7, 0.0, 0.7),
            trajectory_steps=12,
            ego_radius_m=0.4,
            uncertainty_band=0.0,
        ),
    )
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    assert planner.observe(frame=1, timestamp=0.0, rgb=image, speed_mps=3.0) is None
    record = planner.observe(frame=2, timestamp=0.05, rgb=image, speed_mps=3.0)
    assert record is not None
    assert record["actuation_applied"] is False
    assert record["selected_steering"] != 0.0
    assert len(record["candidates"]) == 3


def test_shadow_cli_dry_run_declares_zero_control_calls() -> None:
    args = build_parser().parse_args(["--dry-run", "--steering-values=-0.5,0,0.5"])
    summary = run(args)
    assert summary["actuation_enabled"] is False
    assert summary["control_calls"] == 0


def test_shadow_module_has_no_apply_control_call() -> None:
    assert shadow_module.__file__ is not None
    source = Path(shadow_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    attributes = [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)]
    assert "apply_control" not in attributes
