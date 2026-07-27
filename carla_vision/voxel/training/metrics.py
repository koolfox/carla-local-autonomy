"""Metrics for present and future voxel occupancy predictions."""

from __future__ import annotations

from typing import Sequence

import torch

from .dataset import COMPACT_SEMANTIC_IGNORE


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    value = denominator.item()
    if value == 0:
        return 1.0 if numerator.item() == 0 else 0.0
    return float((numerator / denominator).item())


def occupancy_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    horizons_s: Sequence[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    if logits.shape != target.shape or logits.ndim != 5:
        raise ValueError("logits and target must have shape (B,T,Z,Y,X)")
    if logits.shape[1] != len(horizons_s):
        raise ValueError("horizon count must match time dimension")
    probability = torch.sigmoid(logits)
    prediction = probability >= threshold
    result: dict[str, float] = {}
    ious: list[float] = []
    briers: list[float] = []
    for index, horizon in enumerate(horizons_s):
        known = target[:, index] >= 0
        occupied = target[:, index] == 1
        predicted = prediction[:, index]
        intersection = (predicted & occupied & known).sum().float()
        union = ((predicted | occupied) & known).sum().float()
        iou = _safe_ratio(intersection, union)
        true_free = (~predicted & ~occupied & known).sum().float()
        predicted_free = (~predicted & known).sum().float()
        actual_free = (~occupied & known).sum().float()
        free_precision = _safe_ratio(true_free, predicted_free)
        free_recall = _safe_ratio(true_free, actual_free)
        if bool(known.any()):
            binary = occupied.to(dtype=probability.dtype)
            brier = float(((probability[:, index] - binary).square()[known]).mean().item())
        else:
            brier = 0.0
        prefix = f"h{float(horizon):.3f}"
        result[f"{prefix}_occupied_iou"] = iou
        result[f"{prefix}_free_precision"] = free_precision
        result[f"{prefix}_free_recall"] = free_recall
        result[f"{prefix}_brier"] = brier
        ious.append(iou)
        briers.append(brier)
    result["mean_occupied_iou"] = sum(ious) / len(ious)
    result["mean_brier"] = sum(briers) / len(briers)
    return result


def semantic_mean_iou(
    semantic_logits: torch.Tensor,
    semantic_target: torch.Tensor,
    occupancy_target: torch.Tensor,
    *,
    class_count: int,
) -> float:
    if semantic_logits.ndim != 6:
        raise ValueError("semantic_logits must have shape (B,T,C,Z,Y,X)")
    prediction = semantic_logits.argmax(dim=2)
    valid = (occupancy_target == 1) & (semantic_target != COMPACT_SEMANTIC_IGNORE)
    class_ious: list[float] = []
    for class_index in range(class_count):
        predicted = prediction == class_index
        target = semantic_target == class_index
        intersection = (predicted & target & valid).sum().float()
        union = ((predicted | target) & valid).sum().float()
        if union.item() > 0:
            class_ious.append(float((intersection / union).item()))
    return sum(class_ious) / len(class_ious) if class_ious else 0.0
