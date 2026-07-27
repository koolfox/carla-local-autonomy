"""Pure fail-closed safety supervisor for future voxel-planner actuation.

The supervisor validates a planner proposal and emits an abstract decision. It
has no CARLA dependency and never applies vehicle control. A rejected proposal
always requests a full emergency stop.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .contracts import CameraVoxelPrediction, VoxelGridSpec, validate_camera_voxel_prediction

VOXEL_ACTUATION_SUPERVISOR_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class VoxelActuationSupervisorPolicy:
    maximum_prediction_age_s: float = 0.20
    maximum_prediction_latency_ms: float = 120.0
    maximum_uncertain_voxel_fraction: float = 0.45
    maximum_collision_risk: float = 0.20
    maximum_speed_mps: float = 8.0
    maximum_absolute_steering: float = 0.65
    maximum_steering_rate_per_s: float = 1.5
    minimum_horizon_s: float = 1.0

    def __post_init__(self) -> None:
        positive = (
            "maximum_prediction_age_s",
            "maximum_prediction_latency_ms",
            "maximum_speed_mps",
            "maximum_absolute_steering",
            "maximum_steering_rate_per_s",
            "minimum_horizon_s",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("maximum_uncertain_voxel_fraction", "maximum_collision_risk"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.maximum_absolute_steering > 1.0:
            raise ValueError("maximum_absolute_steering must not exceed 1")


@dataclass(frozen=True, slots=True)
class VoxelPlannerProposal:
    frame: int
    generated_at_s: float
    prediction_latency_ms: float
    selected_steering: float
    collision_risk: float
    uncertain_voxel_fraction: float
    prediction: CameraVoxelPrediction


@dataclass(frozen=True, slots=True)
class VoxelSupervisorDecision:
    schema_version: str
    actuation_authorized: bool
    emergency_brake: bool
    steering: float
    reason: str
    checks: tuple[dict[str, Any], ...]

    @classmethod
    def safe_stop(
        cls,
        reason: str,
        checks: list[dict[str, Any]],
    ) -> "VoxelSupervisorDecision":
        return cls(
            schema_version=VOXEL_ACTUATION_SUPERVISOR_SCHEMA_VERSION,
            actuation_authorized=False,
            emergency_brake=True,
            steering=0.0,
            reason=reason,
            checks=tuple(checks),
        )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checks"] = list(self.checks)
        return payload


def _record(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    observed: Any,
    required: Any,
) -> bool:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "observed": observed,
            "required": required,
        }
    )
    return bool(passed)


def _readiness_passed(report: dict[str, Any]) -> bool:
    return (
        report.get("status") == "passed"
        and report.get("actuation_readiness") is True
        and report.get("actuation_enabled_by_this_report") is False
        and not report.get("failed_checks")
    )


def evaluate_voxel_planner_proposal(
    proposal: VoxelPlannerProposal,
    *,
    spec: VoxelGridSpec,
    readiness_report: dict[str, Any],
    now_s: float,
    speed_mps: float,
    previous_steering: float,
    dt_s: float,
    policy: VoxelActuationSupervisorPolicy | None = None,
) -> VoxelSupervisorDecision:
    """Validate one proposal and return authorization or a full safe stop."""

    policy = policy or VoxelActuationSupervisorPolicy()
    checks: list[dict[str, Any]] = []

    if not _record(
        checks,
        "readiness_report",
        _readiness_passed(readiness_report),
        readiness_report.get("status"),
        "passed",
    ):
        return VoxelSupervisorDecision.safe_stop("readiness_not_passed", checks)

    if proposal.frame < 0:
        _record(checks, "frame", False, proposal.frame, ">= 0")
        return VoxelSupervisorDecision.safe_stop("invalid_frame", checks)
    _record(checks, "frame", True, proposal.frame, ">= 0")

    scalar_values = {
        "now_s": now_s,
        "generated_at_s": proposal.generated_at_s,
        "prediction_latency_ms": proposal.prediction_latency_ms,
        "selected_steering": proposal.selected_steering,
        "collision_risk": proposal.collision_risk,
        "uncertain_voxel_fraction": proposal.uncertain_voxel_fraction,
        "speed_mps": speed_mps,
        "previous_steering": previous_steering,
        "dt_s": dt_s,
    }
    non_finite = [name for name, value in scalar_values.items() if not math.isfinite(value)]
    if non_finite:
        _record(checks, "finite_scalars", False, non_finite, "all finite")
        return VoxelSupervisorDecision.safe_stop("non_finite_proposal", checks)
    _record(checks, "finite_scalars", True, "all finite", "all finite")

    if dt_s <= 0.0 or speed_mps < 0.0:
        _record(
            checks,
            "runtime_state",
            False,
            {"dt_s": dt_s, "speed_mps": speed_mps},
            {"dt_s": "> 0", "speed_mps": ">= 0"},
        )
        return VoxelSupervisorDecision.safe_stop("invalid_runtime_state", checks)

    age_s = now_s - proposal.generated_at_s
    if not _record(
        checks,
        "prediction_age_s",
        0.0 <= age_s <= policy.maximum_prediction_age_s,
        age_s,
        f"0 <= age <= {policy.maximum_prediction_age_s}",
    ):
        return VoxelSupervisorDecision.safe_stop("stale_prediction", checks)

    if not _record(
        checks,
        "prediction_latency_ms",
        0.0 <= proposal.prediction_latency_ms <= policy.maximum_prediction_latency_ms,
        proposal.prediction_latency_ms,
        f"<= {policy.maximum_prediction_latency_ms}",
    ):
        return VoxelSupervisorDecision.safe_stop("prediction_latency_limit", checks)

    try:
        prediction = validate_camera_voxel_prediction(proposal.prediction, spec)
    except (TypeError, ValueError) as error:
        _record(checks, "prediction_contract", False, str(error), "valid prediction contract")
        return VoxelSupervisorDecision.safe_stop("invalid_prediction", checks)
    _record(checks, "prediction_contract", True, "valid", "valid prediction contract")

    maximum_horizon = max(prediction.horizons_s)
    if not _record(
        checks,
        "prediction_horizon_s",
        maximum_horizon >= policy.minimum_horizon_s,
        maximum_horizon,
        f">= {policy.minimum_horizon_s}",
    ):
        return VoxelSupervisorDecision.safe_stop("insufficient_prediction_horizon", checks)

    if not _record(
        checks,
        "uncertain_voxel_fraction",
        0.0 <= proposal.uncertain_voxel_fraction
        <= policy.maximum_uncertain_voxel_fraction,
        proposal.uncertain_voxel_fraction,
        f"<= {policy.maximum_uncertain_voxel_fraction}",
    ):
        return VoxelSupervisorDecision.safe_stop("uncertainty_limit", checks)

    if not _record(
        checks,
        "collision_risk",
        0.0 <= proposal.collision_risk <= policy.maximum_collision_risk,
        proposal.collision_risk,
        f"<= {policy.maximum_collision_risk}",
    ):
        return VoxelSupervisorDecision.safe_stop("collision_risk_limit", checks)

    if not _record(
        checks,
        "speed_mps",
        speed_mps <= policy.maximum_speed_mps,
        speed_mps,
        f"<= {policy.maximum_speed_mps}",
    ):
        return VoxelSupervisorDecision.safe_stop("speed_limit", checks)

    if not _record(
        checks,
        "absolute_steering",
        abs(proposal.selected_steering) <= policy.maximum_absolute_steering,
        proposal.selected_steering,
        f"abs <= {policy.maximum_absolute_steering}",
    ):
        return VoxelSupervisorDecision.safe_stop("steering_limit", checks)

    maximum_delta = policy.maximum_steering_rate_per_s * dt_s
    steering_delta = proposal.selected_steering - previous_steering
    limited_steering = float(
        np.clip(
            proposal.selected_steering,
            previous_steering - maximum_delta,
            previous_steering + maximum_delta,
        )
    )
    _record(
        checks,
        "steering_rate",
        abs(steering_delta) <= maximum_delta,
        steering_delta,
        f"abs delta <= {maximum_delta}",
    )

    return VoxelSupervisorDecision(
        schema_version=VOXEL_ACTUATION_SUPERVISOR_SCHEMA_VERSION,
        actuation_authorized=True,
        emergency_brake=False,
        steering=limited_steering,
        reason=(
            "authorized"
            if math.isclose(limited_steering, proposal.selected_steering, abs_tol=1e-12)
            else "authorized_with_steering_rate_limit"
        ),
        checks=tuple(checks),
    )


__all__ = [
    "VOXEL_ACTUATION_SUPERVISOR_SCHEMA_VERSION",
    "VoxelActuationSupervisorPolicy",
    "VoxelPlannerProposal",
    "VoxelSupervisorDecision",
    "evaluate_voxel_planner_proposal",
]
