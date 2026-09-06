"""Conservative planning view of temporal voxel occupancy predictions.

The camera model predicts occupancy probabilities. Planning additionally needs a
single, explicit policy for treating low-confidence occupancy as unknown space.
This module owns that interpretation so shadow and actuation paths cannot drift.

``unknown_mask`` means *unknown to the planner under the configured confidence
policy*. It is not presented as calibrated epistemic uncertainty or sensor
observability. A future model may expose those quantities explicitly without
changing the planner's unknown-space contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .contracts import (
    UNKNOWN,
    CameraVoxelPrediction,
    VoxelGridSpec,
    validate_camera_voxel_prediction,
)


@dataclass(frozen=True, slots=True)
class PlanningOccupancyEvidence:
    """Validated temporal occupancy plus a fail-conservative planning view.

    ``planning_grid`` has the same ``(T, Z, Y, X)`` shape as the model output.
    Known cells retain their occupancy probability in ``[0, 1]``. Cells inside
    the configured ambiguity band are encoded as ``UNKNOWN`` (``-1``), matching
    the existing planner contract.

    Arrays owned by this value are independent copies and read-only.
    """

    prediction: CameraVoxelPrediction
    planning_grid: np.ndarray
    unknown_mask: np.ndarray
    uncertainty_band: float

    @property
    def horizons_s(self) -> tuple[float, ...]:
        return self.prediction.horizons_s

    @property
    def unknown_fraction(self) -> float:
        return float(np.mean(self.unknown_mask))


def _read_only_copy(value: np.ndarray, *, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def build_planning_occupancy_evidence(
    prediction: CameraVoxelPrediction,
    spec: VoxelGridSpec,
    *,
    uncertainty_band: float,
) -> PlanningOccupancyEvidence:
    """Validate a model prediction and derive the canonical planning evidence.

    The current deployed models expose occupancy probability but no calibrated
    observability/epistemic-uncertainty head. Until such a contract exists, a
    symmetric band around 0.5 is deliberately treated as planner-unknown. The
    policy is explicit and centralized here rather than reimplemented by each
    runtime.
    """

    band = float(uncertainty_band)
    if not math.isfinite(band) or not 0.0 <= band < 0.5:
        raise ValueError("uncertainty_band must be finite and in [0, 0.5)")

    validated = validate_camera_voxel_prediction(prediction, spec)
    occupancy = _read_only_copy(
        validated.occupancy_probability,
        dtype=np.float32,
    )
    semantics = (
        None
        if validated.semantic_logits is None
        else _read_only_copy(validated.semantic_logits, dtype=np.float32)
    )
    stable_prediction = CameraVoxelPrediction(
        occupancy_probability=occupancy,
        horizons_s=validated.horizons_s,
        semantic_logits=semantics,
        metadata=dict(validated.metadata or {}),
    )

    unknown_mask = np.ascontiguousarray(np.abs(occupancy - 0.5) <= band)
    unknown_mask.setflags(write=False)

    planning_grid = np.array(occupancy, dtype=np.float32, copy=True, order="C")
    planning_grid[unknown_mask] = float(UNKNOWN)
    planning_grid.setflags(write=False)

    return PlanningOccupancyEvidence(
        prediction=stable_prediction,
        planning_grid=planning_grid,
        unknown_mask=unknown_mask,
        uncertainty_band=band,
    )


__all__ = [
    "PlanningOccupancyEvidence",
    "build_planning_occupancy_evidence",
]
