"""Visibility-aware losses for temporal voxel occupancy training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .dataset import COMPACT_SEMANTIC_IGNORE


@dataclass(frozen=True, slots=True)
class VoxelLosses:
    total: torch.Tensor
    occupancy: torch.Tensor
    semantics: torch.Tensor
    temporal: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach().cpu()),
            "occupancy_loss": float(self.occupancy.detach().cpu()),
            "semantic_loss": float(self.semantics.detach().cpu()),
            "temporal_loss": float(self.temporal.detach().cpu()),
        }


def masked_occupancy_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    gamma: float = 2.0,
    positive_weight: float = 4.0,
) -> torch.Tensor:
    if logits.shape != target.shape:
        raise ValueError("occupancy logits and target must have the same shape")
    if gamma < 0 or positive_weight <= 0:
        raise ValueError("gamma must be non-negative and positive_weight must be positive")
    known = target >= 0
    if not bool(known.any()):
        return logits.sum() * 0.0
    binary = (target == 1).to(dtype=logits.dtype)
    raw = F.binary_cross_entropy_with_logits(logits, binary, reduction="none")
    probability = torch.sigmoid(logits)
    pt = torch.where(binary > 0.5, probability, 1.0 - probability)
    class_weight = torch.where(binary > 0.5, positive_weight, 1.0)
    loss = class_weight * (1.0 - pt).pow(gamma) * raw
    return loss[known].mean()


def occupied_semantic_loss(
    semantic_logits: torch.Tensor,
    semantic_target: torch.Tensor,
    occupancy_target: torch.Tensor,
) -> torch.Tensor:
    if semantic_logits.ndim != 6:
        raise ValueError("semantic_logits must have shape (B,T,C,Z,Y,X)")
    if semantic_target.shape != occupancy_target.shape:
        raise ValueError("semantic and occupancy targets must have the same shape")
    expected = (
        semantic_logits.shape[0],
        semantic_logits.shape[1],
        *semantic_logits.shape[3:],
    )
    if semantic_target.shape != expected:
        raise ValueError("semantic target dimensions must match semantic logits")
    supervised = (occupancy_target == 1) & (semantic_target != COMPACT_SEMANTIC_IGNORE)
    if not bool(supervised.any()):
        return semantic_logits.sum() * 0.0
    logits = semantic_logits.permute(0, 1, 3, 4, 5, 2)[supervised]
    labels = semantic_target[supervised].long()
    return F.cross_entropy(logits, labels)


def temporal_probability_consistency(
    occupancy_logits: torch.Tensor,
    occupancy_target: torch.Tensor,
) -> torch.Tensor:
    if occupancy_logits.shape != occupancy_target.shape:
        raise ValueError("occupancy logits and targets must have the same shape")
    if occupancy_logits.shape[1] < 2:
        return occupancy_logits.sum() * 0.0
    left_known = occupancy_target[:, :-1] >= 0
    right_known = occupancy_target[:, 1:] >= 0
    visible = left_known & right_known
    if not bool(visible.any()):
        return occupancy_logits.sum() * 0.0
    probabilities = torch.sigmoid(occupancy_logits)
    difference = torch.abs(probabilities[:, 1:] - probabilities[:, :-1])
    return difference[visible].mean()


def compute_voxel_losses(
    outputs: dict[str, torch.Tensor],
    occupancy_target: torch.Tensor,
    semantic_target: torch.Tensor,
    *,
    semantic_weight: float = 0.25,
    temporal_weight: float = 0.05,
    focal_gamma: float = 2.0,
    positive_weight: float = 4.0,
) -> VoxelLosses:
    if semantic_weight < 0 or temporal_weight < 0:
        raise ValueError("loss weights must be non-negative")
    occupancy_logits = outputs["occupancy_logits"]
    semantic_logits = outputs["semantic_logits"]
    occupancy = masked_occupancy_focal_loss(
        occupancy_logits,
        occupancy_target,
        gamma=focal_gamma,
        positive_weight=positive_weight,
    )
    semantics = occupied_semantic_loss(
        semantic_logits,
        semantic_target,
        occupancy_target,
    )
    temporal = temporal_probability_consistency(occupancy_logits, occupancy_target)
    total = occupancy + semantic_weight * semantics + temporal_weight * temporal
    return VoxelLosses(
        total=total,
        occupancy=occupancy,
        semantics=semantics,
        temporal=temporal,
    )
