from __future__ import annotations

import torch

from carla_vision.imitation.model import ImitationControlNet, ImitationModelConfig
from carla_vision.imitation.objectives import compute_imitation_losses, control_metrics


def test_model_outputs_bounded_controls_and_backpropagates() -> None:
    config = ImitationModelConfig(
        image_size_hw=(32, 48),
        encoder_channels=8,
        hidden_dim=16,
        dropout=0.0,
    )
    model = ImitationControlNet(config)
    image = torch.rand(3, 3, 32, 48)
    speed = torch.tensor([[0.1], [0.5], [0.8]], dtype=torch.float32)
    target = torch.tensor(
        [[0.2, 0.5], [-0.7, -0.6], [0.0, 0.1]],
        dtype=torch.float32,
    )
    outputs = model(image, speed)
    assert outputs["steer"].shape == (3,)
    assert outputs["longitudinal"].shape == (3,)
    assert torch.all(outputs["steer"].abs() <= 1.0)
    assert torch.all(outputs["longitudinal"].abs() <= 1.0)
    losses = compute_imitation_losses(outputs, target)
    losses.total.backward()
    assert losses.total.item() > 0.0
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_metrics_include_braking_and_steering_quality() -> None:
    outputs = {
        "steer": torch.tensor([0.2, -0.4, 0.0]),
        "longitudinal": torch.tensor([0.5, -0.7, 0.1]),
    }
    target = torch.tensor([[0.1, 0.4], [-0.5, -0.6], [0.0, 0.2]])
    metrics = control_metrics(outputs, target)
    assert metrics["steer_mae"] >= 0.0
    assert metrics["longitudinal_mae"] >= 0.0
    assert metrics["brake_precision"] == 1.0
    assert metrics["brake_recall"] == 1.0
