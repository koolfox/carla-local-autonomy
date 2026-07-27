from __future__ import annotations

import json

import numpy as np
import pytest

from carla_vision.voxel.capture import build_parser, run
from carla_vision.voxel.contracts import VoxelGridSpec, validate_camera_voxel_prediction
from carla_vision.voxel.models import CameraVoxelModelConfig, create_camera_voxel_predictor


def test_voxel_capture_dry_run_does_not_import_carla(capsys: pytest.CaptureFixture[str]) -> None:
    args = build_parser().parse_args(
        [
            "--dry-run",
            "--frames",
            "5",
            "--voxel-resolution",
            "1.0",
            "--forecast-horizons",
            "0,1,2",
        ]
    )
    result = run(args)
    printed = json.loads(capsys.readouterr().out)
    assert result["frames"] == 5
    assert result["grid"]["resolution"] == 1.0
    assert printed["forecast_horizons_s"] == [0.0, 1.0, 2.0]


def test_rgb_only_requires_predictor() -> None:
    args = build_parser().parse_args(["--dry-run", "--mode", "rgb-only"])
    with pytest.raises(ValueError, match="requires --predictor-factory"):
        run(args)


def test_diagnostic_predictor_is_rgb_only() -> None:
    predictor = create_camera_voxel_predictor(
        "carla_vision.voxel.model_examples.empty_space:create_predictor",
        CameraVoxelModelConfig(options={"horizons_s": [0.0, 1.0]}),
    )
    spec = VoxelGridSpec(
        x_min=0.0,
        x_max=2.0,
        y_min=-1.0,
        y_max=1.0,
        z_min=0.0,
        z_max=1.0,
        resolution=1.0,
    )
    rgb_history = (np.zeros((4, 4, 3), dtype=np.uint8),)
    prediction = validate_camera_voxel_prediction(predictor.predict(rgb_history, spec), spec)
    assert prediction.occupancy_probability.shape == (2, *spec.shape)
    assert not np.any(prediction.occupancy_probability)
    assert prediction.metadata == {"diagnostic_only": True, "uses_privileged_input": False}
