"""Loss composition for temporal occupancy plus sparse dynamic voxel flow."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .losses import VoxelLosses, compute_voxel_losses


@dataclass(frozen=True, slots=True)
class FlowVoxelLosses:
    total: torch.Tensor
    base: VoxelLosses
    flow: torch.Tensor

    def detached(self) -> dict[str, float]:
        values = self.base.detached()
        values["base_loss"] = values.pop("loss")
        values["flow_loss"] = float(self.flow.detach().cpu())
        values["loss"] = float(self.total.detach().cpu())
        return values


def sparse_flow_smooth_l1_loss(
    prediction_mps: torch.Tensor,
    target_mps: torch.Tensor,
    valid: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    if prediction_mps.shape != target_mps.shape or prediction_mps.ndim != 5:
        raise ValueError("flow prediction and target must have shape (B,3,Z,Y,X)")
    if prediction_mps.shape[1] != 3:
        raise ValueError("flow prediction must contain xyz channels")
    if valid.shape != prediction_mps.shape[:1] + prediction_mps.shape[2:]:
        raise ValueError("flow valid mask must have shape (B,Z,Y,X)")
    if beta <= 0:
        raise ValueError("flow Smooth-L1 beta must be positive")
    mask = valid.bool().unsqueeze(1).expand_as(prediction_mps)
    if not bool(mask.any()):
        return prediction_mps.sum() * 0.0
    return F.smooth_l1_loss(
        prediction_mps[mask],
        target_mps[mask],
        reduction="mean",
        beta=beta,
    )


def compute_flow_voxel_losses(
    outputs: dict[str, torch.Tensor],
    occupancy_target: torch.Tensor,
    semantic_target: torch.Tensor,
    flow_target_mps: torch.Tensor,
    flow_valid: torch.Tensor,
    *,
    flow_weight: float = 0.20,
    flow_beta: float = 1.0,
    semantic_weight: float = 0.25,
    temporal_weight: float = 0.05,
    focal_gamma: float = 2.0,
    positive_weight: float = 4.0,
) -> FlowVoxelLosses:
    if flow_weight < 0:
        raise ValueError("flow_weight must be non-negative")
    base = compute_voxel_losses(
        outputs,
        occupancy_target,
        semantic_target,
        semantic_weight=semantic_weight,
        temporal_weight=temporal_weight,
        focal_gamma=focal_gamma,
        positive_weight=positive_weight,
    )
    flow = sparse_flow_smooth_l1_loss(
        outputs["flow_mps"],
        flow_target_mps,
        flow_valid,
        beta=flow_beta,
    )
    return FlowVoxelLosses(total=base.total + flow_weight * flow, base=base, flow=flow)
