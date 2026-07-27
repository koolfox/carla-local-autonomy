"""Thin canonical wrapper around pycocotools COCOeval."""

from __future__ import annotations

import contextlib
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


@dataclass(frozen=True)
class CocoEvaluation:
    aggregate: Mapping[str, float | None]
    per_class: tuple[Mapping[str, Any], ...]
    pr_curves: Mapping[str, Mapping[str, Sequence[float]]]
    log: str


def _available(value: float) -> float | None:
    return float(value) if value >= 0.0 and np.isfinite(value) else None


def _empty_results(ground_truth: COCO) -> COCO:
    detections = COCO()
    detections.dataset = {
        "images": list(ground_truth.dataset.get("images", ())),
        "categories": list(ground_truth.dataset.get("categories", ())),
        "annotations": [],
    }
    detections.createIndex()
    return detections


def evaluate_coco(
    ground_truth_path: str | Path,
    predictions: Sequence[Mapping[str, Any]],
    *,
    image_ids: Sequence[int],
) -> CocoEvaluation:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        ground_truth = COCO(str(ground_truth_path))
        detections = (
            ground_truth.loadRes([dict(prediction) for prediction in predictions])
            if predictions
            else _empty_results(ground_truth)
        )
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluator.params.imgIds = list(image_ids)
        evaluator.params.catIds = sorted(ground_truth.getCatIds())
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    names = (
        "ap_50_95",
        "ap_50",
        "ap_75",
        "ap_small",
        "ap_medium",
        "ap_large",
        "ar_1",
        "ar_10",
        "ar_100",
        "ar_small",
        "ar_medium",
        "ar_large",
    )
    aggregate = {
        name: _available(float(value)) for name, value in zip(names, evaluator.stats, strict=True)
    }
    categories = {
        int(category["id"]): str(category["name"])
        for category in ground_truth.dataset["categories"]
    }
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    iou_thresholds = evaluator.params.iouThrs
    recall_thresholds = evaluator.params.recThrs
    iou_50_index = int(np.argmin(np.abs(iou_thresholds - 0.5)))
    iou_75_index = int(np.argmin(np.abs(iou_thresholds - 0.75)))
    per_class = []
    pr_curves: dict[str, Mapping[str, Sequence[float]]] = {}
    for category_index, category_id in enumerate(evaluator.params.catIds):
        all_precision = precision[:, :, category_index, 0, -1]
        ap_values = all_precision[all_precision > -1]
        precision_50 = all_precision[iou_50_index]
        precision_75 = all_precision[iou_75_index]
        values_50 = precision_50[precision_50 > -1]
        values_75 = precision_75[precision_75 > -1]
        all_recall = recall[:, category_index, 0, -1]
        recall_values = all_recall[all_recall > -1]
        name = categories[category_id]
        per_class.append(
            {
                "category_id": category_id,
                "category_name": name,
                "ap_50_95": float(ap_values.mean()) if ap_values.size else None,
                "ap_50": float(values_50.mean()) if values_50.size else None,
                "ap_75": float(values_75.mean()) if values_75.size else None,
                "ar_100": float(recall_values.mean()) if recall_values.size else None,
            }
        )
        curve = np.where(precision_50 > -1, precision_50, np.nan)
        pr_curves[name] = {
            "recall": [float(value) for value in recall_thresholds],
            "precision_iou_50": [
                None if not np.isfinite(value) else float(value) for value in curve
            ],
        }
    return CocoEvaluation(
        aggregate=aggregate,
        per_class=tuple(per_class),
        pr_curves=pr_curves,
        log=output.getvalue(),
    )


__all__ = ["CocoEvaluation", "evaluate_coco"]
