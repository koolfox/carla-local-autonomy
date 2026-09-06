from __future__ import annotations

import numpy as np
import pytest

from carla_vision.voxel.contracts import UNKNOWN, CameraVoxelPrediction, VoxelGridSpec
from carla_vision.voxel.planning_occupancy import build_planning_occupancy_evidence


def _spec() -> VoxelGridSpec:
    return VoxelGridSpec(
        x_min=0.0,
        x_max=3.0,
        y_min=0.0,
        y_max=1.0,
        z_min=0.0,
        z_max=1.0,
        resolution=1.0,
    )


def test_build_planning_evidence_marks_only_ambiguous_cells_unknown() -> None:
    spec = _spec()
    probability = np.asarray([[[[0.10, 0.50, 0.90]]]], dtype=np.float32)

    evidence = build_planning_occupancy_evidence(
        CameraVoxelPrediction(probability, horizons_s=(0.0,)),
        spec,
        uncertainty_band=0.08,
    )

    assert evidence.horizons_s == (0.0,)
    assert evidence.unknown_mask.tolist() == [[[[False, True, False]]]]
    assert evidence.planning_grid[0, 0, 0, 0] == pytest.approx(0.10)
    assert evidence.planning_grid[0, 0, 0, 1] == pytest.approx(float(UNKNOWN))
    assert evidence.planning_grid[0, 0, 0, 2] == pytest.approx(0.90)
    assert evidence.unknown_fraction == pytest.approx(1.0 / 3.0)


def test_build_planning_evidence_owns_read_only_arrays() -> None:
    spec = _spec()
    probability = np.asarray([[[[0.25, 0.50, 0.75]]]], dtype=np.float32)
    semantics = np.zeros((1, 2, *spec.shape), dtype=np.float32)
    metadata = {"model": "fixture"}

    evidence = build_planning_occupancy_evidence(
        CameraVoxelPrediction(
            probability,
            horizons_s=(0.0,),
            semantic_logits=semantics,
            metadata=metadata,
        ),
        spec,
        uncertainty_band=0.05,
    )

    probability[...] = 1.0
    semantics[...] = 7.0
    metadata["model"] = "mutated"

    assert evidence.prediction.occupancy_probability[0, 0, 0, 0] == pytest.approx(0.25)
    assert evidence.prediction.semantic_logits is not None
    assert evidence.prediction.semantic_logits[0, 0, 0, 0, 0] == pytest.approx(0.0)
    assert evidence.prediction.metadata == {"model": "fixture"}
    assert not evidence.prediction.occupancy_probability.flags.writeable
    assert not evidence.prediction.semantic_logits.flags.writeable
    assert not evidence.planning_grid.flags.writeable
    assert not evidence.unknown_mask.flags.writeable

    with pytest.raises(ValueError):
        evidence.planning_grid[0, 0, 0, 0] = 0.0
    with pytest.raises(ValueError):
        evidence.unknown_mask[0, 0, 0, 0] = True


@pytest.mark.parametrize("uncertainty_band", [-0.01, 0.5, float("inf"), float("nan")])
def test_build_planning_evidence_rejects_invalid_uncertainty_band(
    uncertainty_band: float,
) -> None:
    spec = _spec()
    probability = np.zeros((1, *spec.shape), dtype=np.float32)

    with pytest.raises(ValueError, match="uncertainty_band"):
        build_planning_occupancy_evidence(
            CameraVoxelPrediction(probability, horizons_s=(0.0,)),
            spec,
            uncertainty_band=uncertainty_band,
        )


def test_build_planning_evidence_reuses_prediction_validation() -> None:
    spec = _spec()
    wrong_shape = np.zeros((1, 1, 1, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="shape"):
        build_planning_occupancy_evidence(
            CameraVoxelPrediction(wrong_shape, horizons_s=(0.0,)),
            spec,
            uncertainty_band=0.08,
        )
