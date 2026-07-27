from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from carla_vision.imitation.model import ImitationControlNet, ImitationModelConfig
from carla_vision.imitation.predictor import ImitationDrivingPredictor
from carla_vision.model_driver import ModelDriverConfig, ModelObservation


def _checkpoint(path: Path) -> Path:
    config = ImitationModelConfig(
        image_size_hw=(32, 48),
        encoder_channels=8,
        hidden_dim=16,
        dropout=0.0,
    )
    model = ImitationControlNet(config)
    torch.save(
        {
            "schema_version": "1.0",
            "task": "behavior_agent_control_imitation",
            "model_config": config.as_dict(),
            "state_dict": model.state_dict(),
        },
        path,
    )
    return path


def test_checkpoint_predictor_satisfies_model_driver_contract(tmp_path: Path) -> None:
    predictor = ImitationDrivingPredictor(
        ModelDriverConfig(
            checkpoint=_checkpoint(tmp_path / "model.pt"),
            device="cpu",
            options={"longitudinal_deadband": 0.02},
        )
    )
    observation = ModelObservation(
        frame=5,
        timestamp=1.0,
        image_bgr=np.zeros((40, 60, 3), dtype=np.uint8),
        speed_mps=3.5,
        dt_seconds=0.05,
    )
    control = predictor.predict(observation)
    assert 0.0 <= control.throttle <= 1.0
    assert -1.0 <= control.steer <= 1.0
    assert 0.0 <= control.brake <= 1.0
    assert not (control.throttle > 0.05 and control.brake > 0.05)
    predictor.reset()
    predictor.close()


def test_predictor_rejects_wrong_checkpoint_task(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "model.pt")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["task"] = "not-driving"
    torch.save(payload, path)
    try:
        ImitationDrivingPredictor(
            ModelDriverConfig(checkpoint=path, device="cpu", options={})
        )
    except ValueError as error:
        assert "task" in str(error)
    else:
        raise AssertionError("wrong checkpoint task was accepted")
