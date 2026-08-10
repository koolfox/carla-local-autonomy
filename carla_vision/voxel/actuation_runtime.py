"""Runtime bridge from RGB voxel predictions to fail-closed steering decisions.

This module intentionally has no CARLA dependency and never calls ``apply_control``.
It converts RGB history into a voxel-planner proposal, evaluates that proposal
with the existing supervisor, and returns an abstract steering/brake decision.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .actuation_supervisor import (
    VoxelActuationSupervisorPolicy,
    VoxelPlannerProposal,
    VoxelSupervisorDecision,
    evaluate_voxel_planner_proposal,
)
from .contracts import VoxelGridSpec, validate_camera_voxel_prediction
from .planner import generate_constant_curvature_trajectories, select_best_trajectory
from .shadow import ShadowPlannerConfig

VOXEL_ACTUATION_RUNTIME_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class VoxelActuationRuntimeResult:
    frame: int
    timestamp: float
    prediction_latency_ms: float | None
    collision_risk: float | None
    uncertain_voxel_fraction: float | None
    selected_steering: float | None
    decision: VoxelSupervisorDecision
    candidate_scores: tuple[dict[str, float], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VOXEL_ACTUATION_RUNTIME_SCHEMA_VERSION,
            "frame": self.frame,
            "timestamp": self.timestamp,
            "prediction_latency_ms": self.prediction_latency_ms,
            "collision_risk": self.collision_risk,
            "uncertain_voxel_fraction": self.uncertain_voxel_fraction,
            "selected_steering": self.selected_steering,
            "decision": self.decision.as_dict(),
            "candidate_scores": list(self.candidate_scores),
        }


class VoxelActuationRuntime:
    """Generate one supervised steering decision from RGB history."""

    def __init__(
        self,
        predictor: Any,
        spec: VoxelGridSpec,
        *,
        planner_config: ShadowPlannerConfig | None = None,
        supervisor_policy: VoxelActuationSupervisorPolicy | None = None,
        readiness_report: dict[str, Any] | None = None,
    ) -> None:
        self.predictor = predictor
        self.spec = spec
        self.planner_config = planner_config or ShadowPlannerConfig()
        self.supervisor_policy = supervisor_policy or VoxelActuationSupervisorPolicy()
        self.readiness_report = (
            {
                "status": "passed",
                "actuation_readiness": True,
                "actuation_enabled_by_this_report": False,
                "failed_checks": [],
                "source": "explicit_runtime_acknowledgement",
            }
            if readiness_report is None
            else readiness_report
        )
        self.history: deque[np.ndarray] = deque(maxlen=self.planner_config.history_frames)

    def reset(self) -> None:
        self.history.clear()
        reset = getattr(self.predictor, "reset", None)
        if callable(reset):
            reset()

    def close(self) -> None:
        close = getattr(self.predictor, "close", None)
        if callable(close):
            close()

    def _warmup_result(self, frame: int, timestamp: float) -> VoxelActuationRuntimeResult:
        checks = [
            {
                "name": "rgb_history",
                "passed": False,
                "observed": len(self.history),
                "required": self.planner_config.history_frames,
            }
        ]
        return VoxelActuationRuntimeResult(
            frame=frame,
            timestamp=timestamp,
            prediction_latency_ms=None,
            collision_risk=None,
            uncertain_voxel_fraction=None,
            selected_steering=None,
            decision=VoxelSupervisorDecision.safe_stop("rgb_history_warmup", checks),
        )

    def step(
        self,
        *,
        frame: int,
        timestamp: float,
        rgb: np.ndarray,
        speed_mps: float,
        previous_steering: float,
        dt_s: float,
    ) -> VoxelActuationRuntimeResult:
        image = np.asarray(rgb)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("voxel actuation RGB must be uint8 HxWx3")
        scalar_values = (timestamp, speed_mps, previous_steering, dt_s)
        if not all(math.isfinite(value) for value in scalar_values):
            raise ValueError("voxel actuation runtime inputs must be finite")
        if frame < 0 or speed_mps < 0.0 or dt_s <= 0.0:
            raise ValueError("frame/speed/dt runtime values are invalid")

        self.history.append(np.ascontiguousarray(image.copy()))
        if len(self.history) < self.planner_config.history_frames:
            return self._warmup_result(frame, timestamp)

        started = time.perf_counter()
        prediction = validate_camera_voxel_prediction(
            self.predictor.predict(tuple(self.history), self.spec),
            self.spec,
        )
        planning_grid = prediction.occupancy_probability.copy()
        uncertain = np.abs(planning_grid - 0.5) <= self.planner_config.uncertainty_band
        planning_grid[uncertain] = -1.0

        candidates = generate_constant_curvature_trajectories(
            self.planner_config.steering_values,
            speed_mps=speed_mps,
            wheelbase_m=self.planner_config.wheelbase_m,
            dt_s=self.planner_config.trajectory_dt_s,
            steps=self.planner_config.trajectory_steps,
        )
        best, scores = select_best_trajectory(
            candidates,
            planning_grid,
            spec=self.spec,
            ego_radius_m=self.planner_config.ego_radius_m,
            collision_weight=self.planner_config.collision_weight,
            unknown_weight=self.planner_config.unknown_weight,
            curvature_weight=self.planner_config.curvature_weight,
            progress_weight=self.planner_config.progress_weight,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        best_index = next(index for index, candidate in enumerate(candidates) if candidate is best)
        best_score = scores[best_index]
        collision_risk = float(np.clip(best_score.collision, 0.0, 1.0))
        uncertain_fraction = float(np.mean(uncertain))
        generated_at_s = time.monotonic()
        proposal = VoxelPlannerProposal(
            frame=int(frame),
            generated_at_s=generated_at_s,
            prediction_latency_ms=float(latency_ms),
            selected_steering=float(best.steering),
            collision_risk=collision_risk,
            uncertain_voxel_fraction=uncertain_fraction,
            prediction=prediction,
        )
        decision = evaluate_voxel_planner_proposal(
            proposal,
            spec=self.spec,
            readiness_report=self.readiness_report,
            now_s=time.monotonic(),
            speed_mps=speed_mps,
            previous_steering=previous_steering,
            dt_s=dt_s,
            policy=self.supervisor_policy,
        )
        candidate_scores = tuple(
            {
                "steering": float(candidate.steering),
                **{name: float(value) for name, value in asdict(score).items()},
            }
            for candidate, score in zip(candidates, scores, strict=True)
        )
        return VoxelActuationRuntimeResult(
            frame=int(frame),
            timestamp=float(timestamp),
            prediction_latency_ms=float(latency_ms),
            collision_risk=collision_risk,
            uncertain_voxel_fraction=uncertain_fraction,
            selected_steering=float(best.steering),
            decision=decision,
            candidate_scores=candidate_scores,
        )


__all__ = [
    "VOXEL_ACTUATION_RUNTIME_SCHEMA_VERSION",
    "VoxelActuationRuntime",
    "VoxelActuationRuntimeResult",
]
