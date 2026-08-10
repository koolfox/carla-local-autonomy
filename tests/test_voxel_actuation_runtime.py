from __future__ import annotations

import ast
import inspect

import numpy as np

from carla_vision.voxel import actuation_runtime
from carla_vision.voxel.actuation_runtime import VoxelActuationRuntime
from carla_vision.voxel.actuation_supervisor import VoxelActuationSupervisorPolicy
from carla_vision.voxel.contracts import CameraVoxelPrediction, VoxelGridSpec
from carla_vision.voxel.shadow import ShadowPlannerConfig


class ConstantPredictor:
    def __init__(self, occupied_probability: float) -> None:
        self.occupied_probability = occupied_probability
        self.reset_calls = 0
        self.close_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        self.close_calls += 1

    def predict(
        self,
        rgb_history: tuple[np.ndarray, ...],
        spec: VoxelGridSpec,
    ) -> CameraVoxelPrediction:
        assert len(rgb_history) == 2
        occupancy = np.full(
            (2, *spec.shape),
            self.occupied_probability,
            dtype=np.float32,
        )
        return CameraVoxelPrediction(
            occupancy_probability=occupancy,
            horizons_s=(0.0, 1.0),
        )


def _spec() -> VoxelGridSpec:
    return VoxelGridSpec(
        x_min=0.0,
        x_max=8.0,
        y_min=-4.0,
        y_max=4.0,
        z_min=-1.0,
        z_max=3.0,
        resolution=1.0,
    )


def _runtime(probability: float) -> VoxelActuationRuntime:
    return VoxelActuationRuntime(
        ConstantPredictor(probability),
        _spec(),
        planner_config=ShadowPlannerConfig(
            history_frames=2,
            steering_values=(0.0,),
            trajectory_steps=4,
            trajectory_dt_s=0.25,
            ego_radius_m=0.5,
            uncertainty_band=0.05,
        ),
        supervisor_policy=VoxelActuationSupervisorPolicy(
            maximum_prediction_age_s=1.0,
            maximum_prediction_latency_ms=5000.0,
            maximum_uncertain_voxel_fraction=0.45,
            maximum_collision_risk=0.20,
            maximum_speed_mps=10.0,
            maximum_absolute_steering=0.65,
            maximum_steering_rate_per_s=2.0,
            minimum_horizon_s=1.0,
        ),
    )


def _rgb() -> np.ndarray:
    return np.zeros((8, 12, 3), dtype=np.uint8)


def test_runtime_brakes_during_rgb_history_warmup() -> None:
    runtime = _runtime(0.0)
    result = runtime.step(
        frame=1,
        timestamp=0.05,
        rgb=_rgb(),
        speed_mps=1.0,
        previous_steering=0.0,
        dt_s=0.05,
    )
    assert result.decision.actuation_authorized is False
    assert result.decision.emergency_brake is True
    assert result.decision.reason == "rgb_history_warmup"


def test_free_space_authorizes_voxel_steering() -> None:
    runtime = _runtime(0.0)
    runtime.step(
        frame=1,
        timestamp=0.05,
        rgb=_rgb(),
        speed_mps=1.0,
        previous_steering=0.0,
        dt_s=0.05,
    )
    result = runtime.step(
        frame=2,
        timestamp=0.10,
        rgb=_rgb(),
        speed_mps=1.0,
        previous_steering=0.0,
        dt_s=0.05,
    )
    assert result.decision.actuation_authorized is True
    assert result.decision.emergency_brake is False
    assert result.decision.steering == 0.0
    assert result.collision_risk == 0.0


def test_occupied_path_requests_real_runtime_emergency_brake() -> None:
    runtime = _runtime(1.0)
    runtime.step(
        frame=1,
        timestamp=0.05,
        rgb=_rgb(),
        speed_mps=1.0,
        previous_steering=0.0,
        dt_s=0.05,
    )
    result = runtime.step(
        frame=2,
        timestamp=0.10,
        rgb=_rgb(),
        speed_mps=1.0,
        previous_steering=0.0,
        dt_s=0.05,
    )
    assert result.decision.actuation_authorized is False
    assert result.decision.emergency_brake is True
    assert result.decision.reason == "collision_risk_limit"
    assert result.collision_risk is not None and result.collision_risk > 0.2


def test_actuation_runtime_itself_has_no_carla_control_call() -> None:
    tree = ast.parse(inspect.getsource(actuation_runtime))
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "apply_control" not in attributes
