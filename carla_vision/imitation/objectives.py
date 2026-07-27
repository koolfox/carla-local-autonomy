"""Losses and offline control metrics for imitation driving."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class ImitationLosses:
    total: torch.Tensor
    steer: torch.Tensor
    longitudinal: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach().cpu()),
            "steer_loss": float(self.steer.detach().cpu()),
            "longitudinal_loss": float(self.longitudinal.detach().cpu()),
        }


def compute_imitation_losses(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    *,
    steer_weight: float = 2.0,
    longitudinal_weight: float = 1.0,
    high_steer_boost: float = 1.5,
    brake_boost: float = 2.0,
) -> ImitationLosses:
    if target.ndim != 2 or target.shape[1] != 2:
        raise ValueError("target must have shape (B, 2): steer and longitudinal")
    predicted_steer = outputs["steer"]
    predicted_longitudinal = outputs["longitudinal"]
    if predicted_steer.shape != target[:, 0].shape:
        raise ValueError("predicted steer shape must match target batch")
    if predicted_longitudinal.shape != target[:, 1].shape:
        raise ValueError("predicted longitudinal shape must match target batch")
    for name, value in (
        ("steer_weight", steer_weight),
        ("longitudinal_weight", longitudinal_weight),
        ("high_steer_boost", high_steer_boost),
        ("brake_boost", brake_boost),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    steer_target = target[:, 0]
    longitudinal_target = target[:, 1]
    steer_per_sample = F.smooth_l1_loss(
        predicted_steer,
        steer_target,
        reduction="none",
        beta=0.1,
    )
    longitudinal_per_sample = F.smooth_l1_loss(
        predicted_longitudinal,
        longitudinal_target,
        reduction="none",
        beta=0.1,
    )
    steer_scale = 1.0 + high_steer_boost * steer_target.abs()
    brake_scale = 1.0 + brake_boost * (longitudinal_target < -0.05).to(torch.float32)
    steer_loss = (steer_per_sample * steer_scale).mean()
    longitudinal_loss = (longitudinal_per_sample * brake_scale).mean()
    total = steer_weight * steer_loss + longitudinal_weight * longitudinal_loss
    return ImitationLosses(
        total=total,
        steer=steer_loss,
        longitudinal=longitudinal_loss,
    )


def control_metrics(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
) -> dict[str, float]:
    steer = outputs["steer"].detach()
    longitudinal = outputs["longitudinal"].detach()
    target_steer = target[:, 0]
    target_longitudinal = target[:, 1]
    steer_error = steer - target_steer
    longitudinal_error = longitudinal - target_longitudinal
    target_brake = target_longitudinal < -0.05
    predicted_brake = longitudinal < -0.05
    true_positive = torch.sum(target_brake & predicted_brake).item()
    false_positive = torch.sum(~target_brake & predicted_brake).item()
    false_negative = torch.sum(target_brake & ~predicted_brake).item()
    brake_precision = true_positive / max(true_positive + false_positive, 1)
    brake_recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "steer_mae": float(steer_error.abs().mean().cpu()),
        "steer_rmse": float(torch.sqrt(torch.mean(steer_error.square())).cpu()),
        "longitudinal_mae": float(longitudinal_error.abs().mean().cpu()),
        "longitudinal_rmse": float(
            torch.sqrt(torch.mean(longitudinal_error.square())).cpu()
        ),
        "brake_precision": float(brake_precision),
        "brake_recall": float(brake_recall),
        "steer_direction_accuracy": float(
            torch.mean((torch.sign(steer) == torch.sign(target_steer)).to(torch.float32)).cpu()
        ),
    }


__all__ = ["ImitationLosses", "compute_imitation_losses", "control_metrics"]
