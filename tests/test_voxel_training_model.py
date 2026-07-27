from __future__ import annotations

import torch

from carla_vision.voxel.training.losses import compute_voxel_losses
from carla_vision.voxel.training.metrics import occupancy_metrics, semantic_mean_iou
from carla_vision.voxel.training.model import TemporalVoxelModelConfig, TemporalVoxelNet


def _model() -> TemporalVoxelNet:
    return TemporalVoxelNet(
        TemporalVoxelModelConfig(
            horizons_s=(0.0, 1.0),
            grid_shape_zyx=(4, 8, 8),
            semantic_classes=7,
            encoder_channels=32,
            recurrent_channels=48,
            seed_channels=32,
            decoder_channels=16,
        )
    )


def test_temporal_model_outputs_present_and_future_voxels() -> None:
    model = _model()
    rgb = torch.rand(2, 4, 3, 64, 96)
    outputs = model(rgb)
    assert tuple(outputs["occupancy_logits"].shape) == (2, 2, 4, 8, 8)
    assert tuple(outputs["semantic_logits"].shape) == (2, 2, 7, 4, 8, 8)
    assert model.parameter_count > 0


def test_masked_losses_and_metrics_are_finite() -> None:
    model = _model()
    rgb = torch.rand(1, 3, 3, 48, 64)
    outputs = model(rgb)
    occupancy = torch.zeros(1, 2, 4, 8, 8, dtype=torch.int8)
    occupancy[..., 1:3, 3:5, 3:5] = 1
    occupancy[..., 0, :, :] = -1
    semantics = torch.full_like(occupancy, 255, dtype=torch.uint8)
    semantics[occupancy == 1] = 3
    losses = compute_voxel_losses(outputs, occupancy, semantics)
    losses.total.backward()
    assert torch.isfinite(losses.total)
    metrics = occupancy_metrics(outputs["occupancy_logits"], occupancy, (0.0, 1.0))
    assert 0.0 <= metrics["mean_occupied_iou"] <= 1.0
    miou = semantic_mean_iou(
        outputs["semantic_logits"], semantics, occupancy, class_count=7
    )
    assert 0.0 <= miou <= 1.0
