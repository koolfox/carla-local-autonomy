"""Operating-point, calibration, confusion, and bootstrap metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def bbox_iou_xywh(left: Sequence[float], right: Sequence[float]) -> float:
    lx, ly, lw, lh = (float(value) for value in left)
    rx, ry, rw, rh = (float(value) for value in right)
    left_x2, left_y2 = lx + lw, ly + lh
    right_x2, right_y2 = rx + rw, ry + rh
    intersection_width = max(0.0, min(left_x2, right_x2) - max(lx, rx))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(ly, ry))
    intersection = intersection_width * intersection_height
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass(frozen=True)
class OperatingMetrics:
    overall: Mapping[str, Any]
    per_class: tuple[Mapping[str, Any], ...]
    image_counts: Mapping[int, Mapping[str, int]]
    prediction_outcomes: tuple[Mapping[str, Any], ...]
    confusion_matrix: np.ndarray
    confusion_labels: tuple[str, ...]


def _ratios(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "false_negative_rate": 1.0 - recall,
        "f1": f1,
    }


def operating_metrics(
    ground_truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    image_ids: Sequence[int],
    categories: Mapping[int, str],
    confidence_threshold: float,
    iou_threshold: float,
) -> OperatingMetrics:
    gt_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    pred_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for annotation in ground_truth:
        gt_by_image[int(annotation["image_id"])].append(annotation)
    for prediction in predictions:
        if float(prediction["score"]) >= confidence_threshold:
            pred_by_image[int(prediction["image_id"])].append(prediction)

    class_counts: dict[int, Counter[str]] = {
        category_id: Counter(tp=0, fp=0, fn=0) for category_id in categories
    }
    image_counts: dict[int, Mapping[str, int]] = {}
    outcomes: list[Mapping[str, Any]] = []
    category_ids = tuple(sorted(categories))
    category_index = {category_id: index for index, category_id in enumerate(category_ids)}
    background_index = len(category_ids)
    confusion = np.zeros(
        (len(category_ids) + 1, len(category_ids) + 1),
        dtype=np.int64,
    )

    for image_id in image_ids:
        image_gt = list(gt_by_image.get(image_id, ()))
        image_predictions = sorted(
            pred_by_image.get(image_id, ()),
            key=lambda prediction: float(prediction["score"]),
            reverse=True,
        )
        unmatched_gt = set(range(len(image_gt)))
        image_tp = 0
        image_fp = 0
        for prediction in image_predictions:
            prediction_category = int(prediction["category_id"])
            candidates = [
                (
                    bbox_iou_xywh(prediction["bbox"], image_gt[index]["bbox"]),
                    index,
                )
                for index in unmatched_gt
                if int(image_gt[index]["category_id"]) == prediction_category
            ]
            best_iou, best_index = max(candidates, default=(0.0, -1))
            correct = best_index >= 0 and best_iou >= iou_threshold
            if correct:
                unmatched_gt.remove(best_index)
                class_counts[prediction_category]["tp"] += 1
                image_tp += 1
            else:
                class_counts[prediction_category]["fp"] += 1
                image_fp += 1
            outcomes.append(
                {
                    "image_id": image_id,
                    "category_id": prediction_category,
                    "score": float(prediction["score"]),
                    "correct": correct,
                    "matched_iou": best_iou if correct else 0.0,
                }
            )
        for index in unmatched_gt:
            class_counts[int(image_gt[index]["category_id"])]["fn"] += 1
        image_counts[image_id] = {
            "tp": image_tp,
            "fp": image_fp,
            "fn": len(unmatched_gt),
        }

        unmatched_gt_confusion = set(range(len(image_gt)))
        unmatched_predictions = set(range(len(image_predictions)))
        candidate_pairs = sorted(
            (
                (
                    bbox_iou_xywh(prediction["bbox"], annotation["bbox"]),
                    gt_index,
                    prediction_index,
                )
                for gt_index, annotation in enumerate(image_gt)
                for prediction_index, prediction in enumerate(image_predictions)
            ),
            reverse=True,
        )
        for iou, gt_index, prediction_index in candidate_pairs:
            if iou < iou_threshold:
                break
            if (
                gt_index not in unmatched_gt_confusion
                or prediction_index not in unmatched_predictions
            ):
                continue
            gt_category = int(image_gt[gt_index]["category_id"])
            prediction_category = int(image_predictions[prediction_index]["category_id"])
            confusion[
                category_index[gt_category],
                category_index[prediction_category],
            ] += 1
            unmatched_gt_confusion.remove(gt_index)
            unmatched_predictions.remove(prediction_index)
        for gt_index in unmatched_gt_confusion:
            gt_category = int(image_gt[gt_index]["category_id"])
            confusion[category_index[gt_category], background_index] += 1
        for prediction_index in unmatched_predictions:
            prediction_category = int(image_predictions[prediction_index]["category_id"])
            confusion[background_index, category_index[prediction_category]] += 1

    per_class = []
    total = Counter(tp=0, fp=0, fn=0)
    for category_id in category_ids:
        counts = class_counts[category_id]
        total.update(counts)
        per_class.append(
            {
                "category_id": category_id,
                "category_name": categories[category_id],
                **_ratios(counts["tp"], counts["fp"], counts["fn"]),
            }
        )
    return OperatingMetrics(
        overall=_ratios(total["tp"], total["fp"], total["fn"]),
        per_class=tuple(per_class),
        image_counts=image_counts,
        prediction_outcomes=tuple(outcomes),
        confusion_matrix=confusion,
        confusion_labels=tuple(categories[category_id] for category_id in category_ids)
        + ("background",),
    )


def calibration_metrics(
    outcomes: Sequence[Mapping[str, Any]],
    *,
    bins: int,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if not outcomes:
        return (
            {
                "count": 0,
                "expected_calibration_error": None,
                "brier_score": None,
                "mean_confidence": None,
                "empirical_accuracy": None,
            },
            tuple(),
        )
    scores = np.asarray([float(item["score"]) for item in outcomes], dtype=np.float64)
    correct = np.asarray([bool(item["correct"]) for item in outcomes], dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        mask = (
            (scores >= lower) & (scores <= upper)
            if index == bins - 1
            else (scores >= lower) & (scores < upper)
        )
        count = int(mask.sum())
        if count:
            confidence = float(scores[mask].mean())
            accuracy = float(correct[mask].mean())
            ece += count / len(scores) * abs(confidence - accuracy)
        else:
            confidence = None
            accuracy = None
        rows.append(
            {
                "bin": index,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_confidence": confidence,
                "empirical_accuracy": accuracy,
            }
        )
    return (
        {
            "count": int(len(scores)),
            "expected_calibration_error": ece,
            "brier_score": float(np.mean((scores - correct) ** 2)),
            "mean_confidence": float(scores.mean()),
            "empirical_accuracy": float(correct.mean()),
        },
        tuple(rows),
    )


def stratified_metrics(
    image_counts: Mapping[int, Mapping[str, int]],
    strata: Mapping[str, Mapping[int, str]],
) -> tuple[dict[str, Any], ...]:
    rows = []
    for stratum_name, values_by_image in sorted(strata.items()):
        grouped: dict[str, Counter[str]] = defaultdict(lambda: Counter(tp=0, fp=0, fn=0))
        for image_id, counts in image_counts.items():
            value = str(values_by_image.get(image_id, "unknown"))
            grouped[value].update(counts)
        for value, counts in sorted(grouped.items()):
            rows.append(
                {
                    "stratum": stratum_name,
                    "value": value,
                    "image_count": sum(
                        values_by_image.get(image_id, "unknown") == value
                        for image_id in image_counts
                    ),
                    **_ratios(counts["tp"], counts["fp"], counts["fn"]),
                }
            )
    return tuple(rows)


def bootstrap_episode_metrics(
    image_counts: Mapping[int, Mapping[str, int]],
    episode_by_image: Mapping[int, str],
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    episode_totals: dict[str, Counter[str]] = defaultdict(lambda: Counter(tp=0, fp=0, fn=0))
    for image_id, counts in image_counts.items():
        episode_totals[episode_by_image[image_id]].update(counts)
    episode_ids = tuple(sorted(episode_totals))
    if not episode_ids:
        raise ValueError("bootstrap requires at least one episode")
    rng = np.random.default_rng(seed)
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(episode_ids, size=len(episode_ids), replace=True)
        counts = Counter(tp=0, fp=0, fn=0)
        for episode_id in sampled:
            counts.update(episode_totals[str(episode_id)])
        ratios = _ratios(counts["tp"], counts["fp"], counts["fn"])
        precision.append(float(ratios["precision"]))
        recall.append(float(ratios["recall"]))
        f1.append(float(ratios["f1"]))
    alpha = (1.0 - confidence) / 2.0

    def interval(values: Sequence[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "lower": float(np.quantile(array, alpha)),
            "upper": float(np.quantile(array, 1.0 - alpha)),
        }

    return {
        "unit": "episode",
        "episode_count": len(episode_ids),
        "replicates": replicates,
        "confidence": confidence,
        "seed": seed,
        "precision": interval(precision),
        "recall": interval(recall),
        "f1": interval(f1),
        "degenerate_single_episode": len(episode_ids) == 1,
    }


__all__ = [
    "OperatingMetrics",
    "bbox_iou_xywh",
    "bootstrap_episode_metrics",
    "calibration_metrics",
    "operating_metrics",
    "stratified_metrics",
]
