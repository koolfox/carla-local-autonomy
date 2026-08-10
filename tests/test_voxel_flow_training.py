from __future__ import annotations

import torch

from carla_vision.voxel.training.flow_losses import compute_flow_voxel_losses
from carla_vision.voxel.training.flow_metrics import (
    dynamic_occupancy_iou,
    flow_endpoint_metrics,
)
from carla_vision.voxel.training.flow_model import TemporalVoxelFlowNet
from carla_vision.voxel.training.model import TemporalVoxelModelConfig


def _model() -> TemporalVoxelFlowNet:
    return TemporalVoxelFlowNet(
        TemporalVoxelModelConfig(
            horizons_s=(0.0, 1.0),
            grid_shape_zyx=(4, 8, 8),
            semantic_classes=7,
            encoder_channels=16,
            recurrent_channels=16,
            seed_channels=16,
            decoder_channels=8,
        )
    )


def test_flow_model_forward_and_loss_backward() -> None:
    model = _model()
    rgb = torch.rand(2, 4, 3, 32, 48)
    outputs = model(rgb)
    assert outputs["occupancy_logits"].shape == (2, 2, 4, 8, 8)
    assert outputs["semantic_logits"].shape == (2, 2, 7, 4, 8, 8)
    assert outputs["flow_mps"].shape == (2, 3, 4, 8, 8)

    occupancy = torch.zeros(2, 2, 4, 8, 8, dtype=torch.int8)
    occupancy[:, :, 1:3, 3:5, 3:5] = 1
    semantics = torch.full_like(occupancy, 255, dtype=torch.uint8)
    semantics[occupancy == 1] = 3
    flow = torch.zeros(2, 3, 4, 8, 8)
    flow[:, 0, 1:3, 3:5, 3:5] = 3.0
    valid = torch.zeros(2, 4, 8, 8, dtype=torch.bool)
    valid[:, 1:3, 3:5, 3:5] = True

    losses = compute_flow_voxel_losses(outputs, occupancy, semantics, flow, valid)
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_dynamic_iou_and_flow_endpoint_metrics_perfect_case() -> None:
    occupancy_logits = torch.full((1, 1, 2, 2, 2), -10.0)
    occupancy_logits[..., 0, 0, 0] = 10.0
    semantic_logits = torch.full((1, 1, 7, 2, 2, 2), -10.0)
    semantic_logits[:, :, 3, 0, 0, 0] = 10.0
    occupancy = torch.zeros((1, 1, 2, 2, 2), dtype=torch.int8)
    occupancy[..., 0, 0, 0] = 1
    semantics = torch.full_like(occupancy, 255, dtype=torch.uint8)
    semantics[..., 0, 0, 0] = 3
    metrics = dynamic_occupancy_iou(
        occupancy_logits,
        semantic_logits,
        occupancy,
        semantics,
        (0.0,),
    )
    assert metrics["mean_dynamic_iou"] == 1.0

    flow = torch.zeros((1, 3, 2, 2, 2))
    valid = torch.zeros((1, 2, 2, 2), dtype=torch.bool)
    valid[..., 0, 0, 0] = True
    flow_metrics = flow_endpoint_metrics(flow, flow.clone(), valid)
    assert flow_metrics["flow_epe_mps"] == 0.0
    assert flow_metrics["flow_valid_voxels"] == 1.0
