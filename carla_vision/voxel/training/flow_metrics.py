"""Metrics for dynamic occupancy and sparse voxel flow."""

from __future__ import annotations

from collections.abc import Sequence

import torch

DYNAMIC_COMPACT_CLASSES = (3, 4)  # vehicle, pedestrian


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    if denominator.item() == 0:
        return 1.0 if numerator.item() == 0 else 0.0
    return float((numerator / denominator).item())


def dynamic_occupancy_iou(
    occupancy_logits: torch.Tensor,
    semantic_logits: torch.Tensor,
    occupancy_target: torch.Tensor,
    semantic_target: torch.Tensor,
    horizons_s: Sequence[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    if occupancy_logits.shape != occupancy_target.shape or occupancy_logits.ndim != 5:
        raise ValueError("occupancy tensors must have shape (B,T,Z,Y,X)")
    if semantic_logits.ndim != 6:
        raise ValueError("semantic_logits must have shape (B,T,C,Z,Y,X)")
    if semantic_logits.shape[:2] != occupancy_logits.shape[:2]:
        raise ValueError("semantic and occupancy time dimensions must match")
    if semantic_logits.shape[3:] != occupancy_logits.shape[2:]:
        raise ValueError("semantic and occupancy grid dimensions must match")
    if semantic_target.shape != occupancy_target.shape:
        raise ValueError("semantic and occupancy targets must have the same shape")
    if occupancy_logits.shape[1] != len(horizons_s):
        raise ValueError("horizon count must match time dimension")

    occupied_prediction = torch.sigmoid(occupancy_logits) >= threshold
    semantic_prediction = semantic_logits.argmax(dim=2)
    result: dict[str, float] = {}
    values: list[float] = []
    for index, horizon in enumerate(horizons_s):
        known = occupancy_target[:, index] >= 0
        target_dynamic = (occupancy_target[:, index] == 1) & (
            (semantic_target[:, index] == DYNAMIC_COMPACT_CLASSES[0])
            | (semantic_target[:, index] == DYNAMIC_COMPACT_CLASSES[1])
        )
        predicted_dynamic_semantic = (
            (semantic_prediction[:, index] == DYNAMIC_COMPACT_CLASSES[0])
            | (semantic_prediction[:, index] == DYNAMIC_COMPACT_CLASSES[1])
        )
        predicted_dynamic = occupied_prediction[:, index] & predicted_dynamic_semantic
        intersection = (known & target_dynamic & predicted_dynamic).sum().float()
        union = (known & (target_dynamic | predicted_dynamic)).sum().float()
        score = _safe_ratio(intersection, union)
        result[f"h{float(horizon):.3f}_dynamic_iou"] = score
        values.append(score)
    result["mean_dynamic_iou"] = sum(values) / len(values)
    return result


def flow_endpoint_metrics(
    prediction_mps: torch.Tensor,
    target_mps: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, float]:
    if prediction_mps.shape != target_mps.shape or prediction_mps.ndim != 5:
        raise ValueError("flow prediction and target must have shape (B,3,Z,Y,X)")
    if prediction_mps.shape[1] != 3:
        raise ValueError("flow prediction must contain xyz channels")
    if valid.shape != prediction_mps.shape[:1] + prediction_mps.shape[2:]:
        raise ValueError("flow valid mask must have shape (B,Z,Y,X)")
    valid = valid.bool()
    if not bool(valid.any()):
        return {"flow_epe_mps": 0.0, "flow_mae_mps": 0.0, "flow_valid_voxels": 0.0}
    difference = prediction_mps - target_mps
    endpoint = torch.linalg.vector_norm(difference, dim=1)
    absolute = difference.abs().mean(dim=1)
    return {
        "flow_epe_mps": float(endpoint[valid].mean().item()),
        "flow_mae_mps": float(absolute[valid].mean().item()),
        "flow_valid_voxels": float(valid.sum().item()),
    }
