"""Support-aware ratios and paired group bootstrap of COCO AP.

A duplicate group draw receives NEW image/annotation IDs. Reusing the original
IDs would collapse bootstrap multiplicities in COCOeval and invalidate the CI.
Group independence must come from the study split, not this arithmetic.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .coco import evaluate_coco


def supported_ratios(tp: int, fp: int, fn: int) -> dict[str, Any]:
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (tp, fp, fn)):
        raise ValueError("counts must be non-negative integers")
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ground_truth_support": tp + fn,
        "prediction_support": tp + fp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "false_negative_rate": fn / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "undefined_policy": "null_with_explicit_denominator",
    }


def resample_coco(
    ground_truth: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    group_by_image: Mapping[int, str],
    sampled_groups: Sequence[str],
):
    images = {int(row["id"]): row for row in ground_truth["images"]}
    if set(group_by_image) - set(images) or any(
        not isinstance(g, str) or not g for g in group_by_image.values()
    ):
        raise ValueError("bootstrap group map must name valid images and non-empty group IDs")
    by_group = defaultdict(list)
    anns, preds = defaultdict(list), defaultdict(list)
    for image_id, group in group_by_image.items():
        by_group[group].append(image_id)
    for row in ground_truth["annotations"]:
        anns[int(row["image_id"])].append(row)
    for row in predictions:
        if int(row["image_id"]) not in group_by_image:
            raise ValueError("prediction lies outside bootstrap selection")
        preds[int(row["image_id"])].append(row)
    out = {
        "info": copy.deepcopy(ground_truth.get("info", {})),
        "categories": copy.deepcopy(ground_truth["categories"]),
        "images": [],
        "annotations": [],
    }
    out_predictions = []
    for group in sampled_groups:
        if group not in by_group:
            raise ValueError("bootstrap sampled an unknown group")
        for original_id in sorted(by_group[group]):
            image_id = len(out["images"]) + 1
            out["images"].append({**images[original_id], "id": image_id})
            for annotation in anns[original_id]:
                out["annotations"].append(
                    {
                        **copy.deepcopy(annotation),
                        "image_id": image_id,
                        "id": len(out["annotations"]) + 1,
                    }
                )
            out_predictions.extend(
                {**copy.deepcopy(row), "image_id": image_id} for row in preds[original_id]
            )
    return out, out_predictions


def bootstrap_coco(
    ground_truth: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    group_by_image: Mapping[int, str],
    replicates: int,
    confidence: float = 0.95,
    seed: int = 17,
    reference_predictions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates <= 0
        or not math.isfinite(confidence)
        or not 0 < confidence < 1
        or not group_by_image
    ):
        raise ValueError("invalid group-bootstrap settings")
    groups = tuple(sorted(set(group_by_image.values())))
    rng = np.random.default_rng(seed)
    draws = []
    deltas = []
    for _ in range(replicates):
        sampled = [str(g) for g in rng.choice(groups, len(groups), replace=True)]
        truth, pred = resample_coco(ground_truth, predictions, group_by_image, sampled)
        ids = [row["id"] for row in truth["images"]]
        ap = evaluate_coco(truth, pred, image_ids=ids).aggregate["ap_50_95"]
        if ap is not None:
            draws.append(ap)
        if reference_predictions is not None:
            ref_truth, ref_pred = resample_coco(
                ground_truth, reference_predictions, group_by_image, sampled
            )
            ref_ap = evaluate_coco(ref_truth, ref_pred, image_ids=ids).aggregate["ap_50_95"]
            if ap is not None and ref_ap is not None:
                deltas.append(ap - ref_ap)

    def interval(values):
        if not values or len(groups) < 2:
            return {
                "mean": float(np.mean(values)) if values else None,
                "lower": None,
                "upper": None,
            }
        alpha = (1 - confidence) / 2
        return {
            "mean": float(np.mean(values)),
            "lower": float(np.quantile(values, alpha)),
            "upper": float(np.quantile(values, 1 - alpha)),
        }

    return {
        "schema_version": "1.0",
        "unit": "scene_or_route_group",
        "group_count": len(groups),
        "replicates": replicates,
        "confidence": confidence,
        "seed": seed,
        "ap_50_95": interval(draws),
        "paired_ap_delta": interval(deltas) if reference_predictions is not None else None,
        "undefined_replicates": replicates - len(draws),
        "paired_defined_replicates": len(deltas),
        "degenerate_single_group": len(groups) < 2,
        "interval_conditions": "AP_defined_and_at_least_two_groups",
        "training_seed_uncertainty_included": False,
    }
