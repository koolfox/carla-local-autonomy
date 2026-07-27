from __future__ import annotations

import ast
import inspect

import numpy as np

from carla_vision.voxel import actuation_gate, actuation_supervisor
from carla_vision.voxel.actuation_supervisor import (
    VoxelActuationSupervisorPolicy,
    VoxelPlannerProposal,
    evaluate_voxel_planner_proposal,
)
from carla_vision.voxel.contracts import CameraVoxelPrediction, VoxelGridSpec


def _spec() -> VoxelGridSpec:
    return VoxelGridSpec(
        x_min=0.0,
        x_max=4.0,
        y_min=-2.0,
        y_max=2.0,
        z_min=-1.0,
        z_max=1.0,
        resolution=1.0,
    )


def _prediction(spec: VoxelGridSpec) -> CameraVoxelPrediction:
    occupancy = np.zeros((2, *spec.shape), dtype=np.float32)
    return CameraVoxelPrediction(
        occupancy_probability=occupancy,
        horizons_s=(0.0, 1.0),
    )


def _readiness() -> dict:
    return {
        "status": "passed",
        "actuation_readiness": True,
        "actuation_enabled_by_this_report": False,
        "failed_checks": [],
    }


def _proposal(spec: VoxelGridSpec, **overrides) -> VoxelPlannerProposal:
    values = {
        "frame": 10,
        "generated_at_s": 9.9,
        "prediction_latency_ms": 20.0,
        "selected_steering": 0.2,
        "collision_risk": 0.05,
        "uncertain_voxel_fraction": 0.10,
        "prediction": _prediction(spec),
    }
    values.update(overrides)
    return VoxelPlannerProposal(**values)


def test_valid_proposal_is_authorized_with_rate_limit() -> None:
    spec = _spec()
    decision = evaluate_voxel_planner_proposal(
        _proposal(spec, selected_steering=0.5),
        spec=spec,
        readiness_report=_readiness(),
        now_s=10.0,
        speed_mps=3.0,
        previous_steering=0.0,
        dt_s=0.1,
        policy=VoxelActuationSupervisorPolicy(maximum_steering_rate_per_s=1.0),
    )
    assert decision.actuation_authorized is True
    assert decision.emergency_brake is False
    assert decision.steering == 0.1
    assert decision.reason == "authorized_with_steering_rate_limit"


def test_failed_readiness_always_requests_safe_stop() -> None:
    spec = _spec()
    readiness = _readiness()
    readiness["status"] = "blocked"
    decision = evaluate_voxel_planner_proposal(
        _proposal(spec),
        spec=spec,
        readiness_report=readiness,
        now_s=10.0,
        speed_mps=2.0,
        previous_steering=0.0,
        dt_s=0.05,
    )
    assert decision.actuation_authorized is False
    assert decision.emergency_brake is True
    assert decision.steering == 0.0
    assert decision.reason == "readiness_not_passed"


def test_stale_high_risk_and_invalid_predictions_fail_closed() -> None:
    spec = _spec()
    stale = evaluate_voxel_planner_proposal(
        _proposal(spec, generated_at_s=8.0),
        spec=spec,
        readiness_report=_readiness(),
        now_s=10.0,
        speed_mps=2.0,
        previous_steering=0.0,
        dt_s=0.05,
    )
    assert stale.reason == "stale_prediction"
    assert stale.emergency_brake

    high_risk = evaluate_voxel_planner_proposal(
        _proposal(spec, collision_risk=0.9),
        spec=spec,
        readiness_report=_readiness(),
        now_s=10.0,
        speed_mps=2.0,
        previous_steering=0.0,
        dt_s=0.05,
    )
    assert high_risk.reason == "collision_risk_limit"
    assert high_risk.emergency_brake

    bad_prediction = CameraVoxelPrediction(
        occupancy_probability=np.zeros((1, 1, 1, 1), dtype=np.float32),
        horizons_s=(0.0,),
    )
    invalid = evaluate_voxel_planner_proposal(
        _proposal(spec, prediction=bad_prediction),
        spec=spec,
        readiness_report=_readiness(),
        now_s=10.0,
        speed_mps=2.0,
        previous_steering=0.0,
        dt_s=0.05,
    )
    assert invalid.reason == "invalid_prediction"
    assert invalid.emergency_brake


def test_supervisor_modules_have_no_carla_actuation_call() -> None:
    for module in (actuation_gate, actuation_supervisor):
        tree = ast.parse(inspect.getsource(module))
        attributes = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        assert "apply_control" not in attributes
