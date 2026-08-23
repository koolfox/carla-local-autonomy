"""Canonical offline evaluation over an immutable CARLA dataset release."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402

from ..artifacts import RunArtifactTracker, fingerprint_file
from ..contracts import Detection, Detector, DetectorConfig
from ..dataset.verified import VerifiedDataset, load_verified_dataset
from ..detectors.factory import create_detector
from ..model_release.verified import load_verified_model
from ..scenarios.seeds import derive_seed_bundle
from .coco import CocoEvaluation, evaluate_coco
from .contracts import EVALUATION_RUN_SCHEMA_VERSION, EvaluationConfig, load_evaluation_config
from .metrics import (
    OperatingMetrics,
    bootstrap_episode_metrics,
    calibration_metrics,
    operating_metrics,
    stratified_metrics,
)


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fallback_fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0].keys()) if rows else tuple(fallback_fields)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _class_mapper(
    categories: Mapping[int, str],
    aliases: Mapping[str, str],
) -> tuple[dict[str, int], dict[str, str]]:
    category_by_normalized = {
        _normalize_label(name): category_id for category_id, name in categories.items()
    }
    normalized_aliases = {
        _normalize_label(source): _normalize_label(target) for source, target in aliases.items()
    }
    unknown_targets = sorted(
        {target for target in normalized_aliases.values() if target not in category_by_normalized}
    )
    if unknown_targets:
        raise ValueError(
            "class alias targets are absent from the dataset ontology: "
            + ", ".join(unknown_targets)
        )
    return category_by_normalized, normalized_aliases


def _map_detection(
    detection: Detection,
    *,
    image_width: int,
    image_height: int,
    category_by_normalized: Mapping[str, int],
    aliases: Mapping[str, str],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    normalized_source = _normalize_label(detection.label)
    normalized_target = aliases.get(normalized_source, normalized_source)
    category_id = category_by_normalized.get(normalized_target)
    x1, y1, x2, y2 = detection.xyxy
    clipped_x1 = max(0.0, min(float(image_width), x1))
    clipped_y1 = max(0.0, min(float(image_height), y1))
    clipped_x2 = max(0.0, min(float(image_width), x2))
    clipped_y2 = max(0.0, min(float(image_height), y2))
    metadata = {
        "source_class_id": detection.source_class_id,
        "source_label": detection.label,
        "source_confidence": detection.confidence,
        "source_xyxy": list(detection.xyxy),
        "normalized_source_label": normalized_source,
        "normalized_target_label": normalized_target,
        "mapped_category_id": category_id,
        "clipped": [clipped_x1, clipped_y1, clipped_x2, clipped_y2] != list(detection.xyxy),
    }
    if category_id is None or clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
        metadata["accepted"] = False
        metadata["rejection_reason"] = (
            "unmapped_class" if category_id is None else "degenerate_after_clipping"
        )
        return None, metadata
    metadata["accepted"] = True
    return (
        {
            "category_id": category_id,
            "bbox": [
                clipped_x1,
                clipped_y1,
                clipped_x2 - clipped_x1,
                clipped_y2 - clipped_y1,
            ],
            "score": detection.confidence,
        },
        metadata,
    )


def _describe(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "stddev": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "stddev": float(array.std()),
    }


def _load_sample_metadata(dataset: VerifiedDataset, sample: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata_reference = sample.get("metadata")
    if not isinstance(metadata_reference, Mapping):
        raise RuntimeError(f"sample {sample.get('sample_id')} metadata reference is invalid")
    path = (dataset.root / str(metadata_reference["path"])).resolve(strict=True)
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"sample metadata {path} must be an object")
    return payload


def _selected_samples(
    dataset: VerifiedDataset,
    config: EvaluationConfig,
) -> list[Mapping[str, Any]]:
    available = set(dataset.partitions)
    missing = sorted(set(config.partitions) - available)
    if missing:
        raise ValueError("evaluation partitions are absent from the dataset: " + ", ".join(missing))
    samples = [
        sample for sample in dataset.dataset["samples"] if str(sample["split"]) in config.partitions
    ]
    if not samples:
        raise ValueError("evaluation selection contains no samples")
    return samples


def _save_figure(figure: Any, base_path: Path) -> tuple[Path, Path]:
    png = base_path.with_suffix(".png")
    svg = base_path.with_suffix(".svg")
    figure.tight_layout()
    figure.savefig(png, dpi=180)
    figure.savefig(svg)
    plt.close(figure)
    return png, svg


def _plot_class_metrics(
    base_path: Path,
    per_class: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    labels = [str(row["category_name"]) for row in per_class]
    ap = [
        np.nan if row.get("coco_ap_50_95") is None else float(row["coco_ap_50_95"])
        for row in per_class
    ]
    recall = [float(row["recall"]) for row in per_class]
    y = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(11, max(6, len(labels) * 0.5)))
    axis.barh(y - 0.18, ap, height=0.34, label="COCO AP@[.50:.95]", color="#3a86ff")
    axis.barh(y + 0.18, recall, height=0.34, label="Recall at operating point", color="#2a9d8f")
    axis.set_yticks(y, labels)
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Metric")
    axis.set_title("Per-class detector performance")
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    return _save_figure(figure, base_path)


def _plot_pr_curves(
    base_path: Path,
    curves: Mapping[str, Mapping[str, Sequence[float | None]]],
) -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(9, 7))
    plotted = 0
    for label, values in curves.items():
        recall = np.asarray(values["recall"], dtype=np.float64)
        precision = np.asarray(
            [np.nan if value is None else value for value in values["precision_iou_50"]],
            dtype=np.float64,
        )
        if not np.isfinite(precision).any():
            continue
        axis.plot(recall, precision, label=label)
        plotted += 1
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Interpolated precision")
    axis.set_title("COCO precision-recall at IoU=0.50")
    axis.grid(alpha=0.25)
    if plotted:
        axis.legend(fontsize=8, ncol=2)
    return _save_figure(figure, base_path)


def _plot_calibration(
    base_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    centers = [(float(row["lower"]) + float(row["upper"])) / 2.0 for row in rows]
    accuracy = [
        np.nan if row["empirical_accuracy"] is None else float(row["empirical_accuracy"])
        for row in rows
    ]
    confidence = [
        np.nan if row["mean_confidence"] is None else float(row["mean_confidence"]) for row in rows
    ]
    figure, axis = plt.subplots(figsize=(8, 7))
    axis.plot([0, 1], [0, 1], "--", color="#666666", label="Perfect calibration")
    axis.plot(centers, accuracy, "o-", label="Empirical accuracy")
    axis.plot(centers, confidence, "s:", label="Mean confidence")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Confidence bin")
    axis.set_ylabel("Probability")
    axis.set_title("Detection confidence calibration")
    axis.grid(alpha=0.25)
    axis.legend()
    return _save_figure(figure, base_path)


def _plot_confusion(
    base_path: Path,
    matrix: np.ndarray,
    labels: Sequence[str],
) -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Ground truth")
    axis.set_title("Operating-point confusion matrix")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            if value:
                axis.text(column, row, str(value), ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, fraction=0.046)
    return _save_figure(figure, base_path)


def _plot_latency(
    base_path: Path,
    inference_ms: Sequence[float],
) -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(9, 6))
    if inference_ms:
        axis.hist(
            inference_ms,
            bins=min(30, max(5, round(math.sqrt(len(inference_ms))))),
            color="#e76f51",
        )
        axis.axvline(
            np.median(inference_ms),
            color="#222222",
            linestyle="--",
            label="median",
        )
        axis.legend()
    axis.set_xlabel("Detector inference latency (ms)")
    axis.set_ylabel("Images")
    axis.set_title("Offline inference latency")
    axis.grid(alpha=0.25)
    return _save_figure(figure, base_path)


def _qualitative_montage(
    path: Path,
    *,
    dataset: VerifiedDataset,
    selected_samples: Sequence[Mapping[str, Any]],
    image_counts: Mapping[int, Mapping[str, int]],
    ground_truth_by_image: Mapping[int, Sequence[Mapping[str, Any]]],
    predictions_by_image: Mapping[int, Sequence[Mapping[str, Any]]],
    categories: Mapping[int, str],
    confidence_threshold: float,
    count: int,
) -> list[int]:
    ordered = sorted(
        selected_samples,
        key=lambda sample: (
            -int(image_counts[int(sample["image_id"])]["fn"]),
            -int(image_counts[int(sample["image_id"])]["fp"]),
            int(sample["image_id"]),
        ),
    )[:count]
    tiles: list[np.ndarray] = []
    selected_ids: list[int] = []
    for sample in ordered:
        image_id = int(sample["image_id"])
        image_path = dataset.root / str(sample["rgb"]["path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not read qualitative image {image_path}")
        for annotation in ground_truth_by_image.get(image_id, ()):
            x, y, width, height = (int(round(value)) for value in annotation["bbox"])
            cv2.rectangle(image, (x, y), (x + width, y + height), (40, 220, 40), 2)
            cv2.putText(
                image,
                f"GT {categories[int(annotation['category_id'])]}",
                (x, max(15, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (40, 220, 40),
                1,
                cv2.LINE_AA,
            )
        for prediction in predictions_by_image.get(image_id, ()):
            if float(prediction["score"]) < confidence_threshold:
                continue
            x, y, width, height = (int(round(value)) for value in prediction["bbox"])
            cv2.rectangle(image, (x, y), (x + width, y + height), (30, 80, 245), 2)
            cv2.putText(
                image,
                f"P {categories[int(prediction['category_id'])]} {prediction['score']:.2f}",
                (x, min(image.shape[0] - 4, y + height + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (30, 80, 245),
                1,
                cv2.LINE_AA,
            )
        counts = image_counts[image_id]
        cv2.rectangle(image, (0, 0), (image.shape[1], 28), (18, 18, 18), -1)
        cv2.putText(
            image,
            f"image={image_id} TP={counts['tp']} FP={counts['fp']} FN={counts['fn']}",
            (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        target_width = 480
        target_height = round(image.shape[0] * target_width / image.shape[1])
        tiles.append(cv2.resize(image, (target_width, target_height)))
        selected_ids.append(image_id)
    if not tiles:
        raise RuntimeError("qualitative montage selection is empty")
    columns = 2
    rows = math.ceil(len(tiles) / columns)
    tile_height = max(tile.shape[0] for tile in tiles)
    canvas = np.full(
        (rows * tile_height, columns * 480, 3),
        24,
        dtype=np.uint8,
    )
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        canvas[
            row * tile_height : row * tile_height + tile.shape[0],
            column * 480 : (column + 1) * 480,
        ] = tile
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"could not write qualitative montage {path}")
    return selected_ids


def _merge_per_class(
    coco: CocoEvaluation,
    operating: OperatingMetrics,
) -> tuple[dict[str, Any], ...]:
    coco_by_id = {int(row["category_id"]): row for row in coco.per_class}
    rows = []
    for operating_row in operating.per_class:
        category_id = int(operating_row["category_id"])
        coco_row = coco_by_id[category_id]
        rows.append(
            {
                **operating_row,
                "coco_ap_50_95": coco_row["ap_50_95"],
                "coco_ap_50": coco_row["ap_50"],
                "coco_ap_75": coco_row["ap_75"],
                "coco_ar_100": coco_row["ar_100"],
            }
        )
    return tuple(rows)


def _model_reference(detector_config: DetectorConfig) -> Mapping[str, Any]:
    if detector_config.weights is None:
        return {
            "kind": "detector_factory",
            "backend": detector_config.backend,
            "factory": detector_config.factory,
        }
    return {
        "kind": "evaluated_weights",
        "backend": detector_config.backend,
        **fingerprint_file(detector_config.weights),
    }


def run_evaluation(
    *,
    config_path: str | Path,
    dataset_path: str | Path,
    detector_config: DetectorConfig,
    runs_root: str | Path = "runs",
    run_id: str | None = None,
    acknowledge_locked_test: bool = False,
    cli_args: Sequence[str] | Mapping[str, Any] = (),
    repository_root: str | Path | None = None,
    detector_override: Detector | None = None,
    model_reference_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config_path_resolved = Path(config_path).expanduser().resolve(strict=True)
    config = load_evaluation_config(config_path_resolved)
    dataset = load_verified_dataset(dataset_path)
    samples = _selected_samples(dataset, config)
    if config.purpose == "confirmatory" and "unassigned" in config.partitions:
        raise RuntimeError("confirmatory evaluation cannot use unassigned data")
    locked_partitions = sorted(
        partition for partition in config.partitions if partition.startswith("test")
    )
    if locked_partitions and not acknowledge_locked_test:
        raise RuntimeError(
            "locked test evaluation requires --acknowledge-locked-test: "
            + ", ".join(locked_partitions)
        )
    detector_config = replace(
        detector_config,
        confidence=config.minimum_prediction_confidence,
    )
    model_reference = (
        dict(model_reference_override)
        if model_reference_override is not None
        else _model_reference(detector_config)
    )
    config_reference = {
        "kind": "evaluation_preregistration",
        **fingerprint_file(config_path_resolved),
    }
    resolved_config = {
        "schema_version": EVALUATION_RUN_SCHEMA_VERSION,
        "object_type": "offline_detector_evaluation",
        "evaluation": config.as_dict(),
        "source_config": config_reference,
        "dataset": dict(dataset.reference),
        "detector": detector_config.as_dict(),
        "model": model_reference,
        "selected_sample_count": len(samples),
        "locked_test_acknowledged": acknowledge_locked_test,
        "runtime_sensor_contract": "front_monocular_rgb_only",
        "privileged_metadata_use": "evaluation_stratification_only",
    }
    tracker = RunArtifactTracker(
        runs_root,
        run_id=run_id,
        cli_args=cli_args,
        config=resolved_config,
        repository_root=repository_root,
        model_refs=[model_reference],
        dataset_refs=[dataset.reference],
    )
    with tracker:
        resolved_path = tracker.artifact_path("resolved_evaluation_config.json")
        predictions_path = tracker.artifact_path("predictions/predictions.coco.json")
        frame_log_path = tracker.artifact_path("predictions/frames.jsonl")
        coco_log_path = tracker.artifact_path("logs/coco_eval.log")
        summary_path = tracker.artifact_path("summary_metrics.json")
        per_class_path = tracker.artifact_path("tables/per_class.csv")
        strata_path = tracker.artifact_path("tables/strata.csv")
        calibration_path = tracker.artifact_path("tables/calibration.csv")
        bootstrap_path = tracker.artifact_path("bootstrap.json")
        _write_json(resolved_path, resolved_config)

        categories = {
            int(category["id"]): str(category["name"]) for category in dataset.coco["categories"]
        }
        category_by_normalized, normalized_aliases = _class_mapper(
            categories,
            config.class_aliases,
        )
        detector = detector_override or create_detector(detector_config)
        detector_metadata = detector.metadata.as_dict()
        predictions: list[dict[str, Any]] = []
        frame_rows: list[dict[str, Any]] = []
        inference_seconds: list[float] = []
        unmapped_labels: Counter[str] = Counter()
        accepted_count = 0
        rejected_degenerate = 0
        try:
            for position, sample in enumerate(samples, start=1):
                image_id = int(sample["image_id"])
                image_path = dataset.root / str(sample["rgb"]["path"])
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"could not read evaluation image {image_path}")
                started = time.perf_counter()
                detections = detector.infer(image)
                elapsed = time.perf_counter() - started
                inference_seconds.append(elapsed)
                mapped_rows = []
                source_rows = []
                for detection in detections:
                    mapped, source_metadata = _map_detection(
                        detection,
                        image_width=image.shape[1],
                        image_height=image.shape[0],
                        category_by_normalized=category_by_normalized,
                        aliases=normalized_aliases,
                    )
                    source_rows.append(source_metadata)
                    if mapped is None:
                        if source_metadata["rejection_reason"] == "unmapped_class":
                            unmapped_labels[detection.label] += 1
                        else:
                            rejected_degenerate += 1
                        continue
                    canonical = {
                        "image_id": image_id,
                        **mapped,
                    }
                    predictions.append(canonical)
                    mapped_rows.append(canonical)
                    accepted_count += 1
                frame_rows.append(
                    {
                        "position": position,
                        "sample_id": sample["sample_id"],
                        "image_id": image_id,
                        "partition": sample["split"],
                        "scenario_id": sample["scenario_id"],
                        "episode_id": sample["episode_id"],
                        "carla_frame": sample["carla_frame"],
                        "inference_seconds": elapsed,
                        "source_detection_count": len(detections),
                        "mapped_prediction_count": len(mapped_rows),
                        "source_detections": source_rows,
                        "mapped_predictions": mapped_rows,
                    }
                )
                print(
                    f"image={position}/{len(samples)} id={image_id} "
                    f"detections={len(detections)} mapped={len(mapped_rows)} "
                    f"inference_ms={elapsed * 1000.0:.1f}",
                    flush=True,
                )
        finally:
            detector.close()

        image_ids = [int(sample["image_id"]) for sample in samples]
        selected_image_ids = set(image_ids)
        ground_truth = [
            annotation
            for annotation in dataset.coco["annotations"]
            if int(annotation["image_id"]) in selected_image_ids
        ]
        coco_path = dataset.root / "annotations" / "instances.coco.json"
        coco = evaluate_coco(coco_path, predictions, image_ids=image_ids)
        _atomic_write_text(coco_log_path, coco.log)
        operating = operating_metrics(
            ground_truth,
            predictions,
            image_ids=image_ids,
            categories=categories,
            confidence_threshold=config.operating_confidence,
            iou_threshold=config.matching_iou,
        )
        calibration_summary, calibration_rows = calibration_metrics(
            operating.prediction_outcomes,
            bins=config.calibration_bins,
        )

        metadata_by_image = {
            int(sample["image_id"]): _load_sample_metadata(dataset, sample) for sample in samples
        }
        episode_by_image = {
            int(sample["image_id"]): (f"{sample['scenario_id']}/{sample['episode_id']}")
            for sample in samples
        }
        strata_values: dict[str, dict[int, str]] = {
            "partition": {int(sample["image_id"]): str(sample["split"]) for sample in samples},
            "scenario": {int(sample["image_id"]): str(sample["scenario_id"]) for sample in samples},
            "episode": episode_by_image,
            "map_family": {},
            "weather": {},
            "light": {},
        }
        for image_id, metadata in metadata_by_image.items():
            context = metadata.get("context", {})
            if not isinstance(context, Mapping):
                context = {}
            strata_values["map_family"][image_id] = str(
                context.get("map_family", context.get("map", "unknown"))
            )
            strata_values["weather"][image_id] = str(
                context.get(
                    "weather_recipe_id",
                    context.get("weather_recipe", "unknown"),
                )
            )
            strata_values["light"][image_id] = str(context.get("light", "unknown"))
        strata_rows = stratified_metrics(operating.image_counts, strata_values)
        seed_bundle = derive_seed_bundle(
            config.master_seed,
            namespace_prefix=f"evaluation/{config.evaluation_id}",
        )
        bootstrap = bootstrap_episode_metrics(
            operating.image_counts,
            episode_by_image,
            replicates=config.bootstrap_replicates,
            confidence=config.bootstrap_confidence,
            seed=seed_bundle.values["bootstrap"],
        )
        per_class = _merge_per_class(coco, operating)

        ground_truth_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        predictions_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for annotation in ground_truth:
            ground_truth_by_image[int(annotation["image_id"])].append(annotation)
        for prediction in predictions:
            predictions_by_image[int(prediction["image_id"])].append(prediction)
        montage_path = tracker.artifact_path("qualitative/failures_and_predictions.png")
        montage_ids = _qualitative_montage(
            montage_path,
            dataset=dataset,
            selected_samples=samples,
            image_counts=operating.image_counts,
            ground_truth_by_image=ground_truth_by_image,
            predictions_by_image=predictions_by_image,
            categories=categories,
            confidence_threshold=config.operating_confidence,
            count=config.montage_count,
        )

        plot_paths = []
        plot_paths.extend(
            _plot_class_metrics(
                tracker.artifact_path("plots/per_class", create_parent=True),
                per_class,
            )
        )
        plot_paths.extend(
            _plot_pr_curves(
                tracker.artifact_path("plots/pr_iou50", create_parent=True),
                coco.pr_curves,
            )
        )
        plot_paths.extend(
            _plot_calibration(
                tracker.artifact_path("plots/calibration", create_parent=True),
                calibration_rows,
            )
        )
        plot_paths.extend(
            _plot_confusion(
                tracker.artifact_path("plots/confusion_matrix", create_parent=True),
                operating.confusion_matrix,
                operating.confusion_labels,
            )
        )
        plot_paths.extend(
            _plot_latency(
                tracker.artifact_path("plots/inference_latency", create_parent=True),
                [value * 1000.0 for value in inference_seconds],
            )
        )

        _write_json(predictions_path, predictions)
        _write_jsonl(frame_log_path, frame_rows)
        _write_csv(
            per_class_path,
            per_class,
            fallback_fields=("category_id", "category_name"),
        )
        _write_csv(
            strata_path,
            strata_rows,
            fallback_fields=("stratum", "value"),
        )
        _write_csv(
            calibration_path,
            calibration_rows,
            fallback_fields=(
                "bin",
                "lower",
                "upper",
                "count",
                "mean_confidence",
                "empirical_accuracy",
            ),
        )
        _write_json(bootstrap_path, bootstrap)
        latency_ms = [value * 1000.0 for value in inference_seconds]
        summary = {
            "schema_version": EVALUATION_RUN_SCHEMA_VERSION,
            "status": "success",
            "evaluation_id": config.evaluation_id,
            "purpose": config.purpose,
            "dataset": dict(dataset.reference),
            "model": model_reference,
            "detector": detector_metadata,
            "selection": {
                "partitions": list(config.partitions),
                "images": len(samples),
                "ground_truth_annotations": len(ground_truth),
                "locked_test_acknowledged": acknowledge_locked_test,
            },
            "prediction_mapping": {
                "source_minimum_confidence": config.minimum_prediction_confidence,
                "operating_confidence": config.operating_confidence,
                "accepted_predictions": accepted_count,
                "unmapped_predictions": sum(unmapped_labels.values()),
                "unmapped_label_frequency": dict(sorted(unmapped_labels.items())),
                "degenerate_after_clipping": rejected_degenerate,
                "class_aliases": dict(config.class_aliases),
            },
            "coco": dict(coco.aggregate),
            "operating_point": dict(operating.overall),
            "calibration": calibration_summary,
            "bootstrap": bootstrap,
            "latency_ms": {
                "all_images": _describe(latency_ms),
                "excluding_first_image": _describe(latency_ms[1:]),
                "first_image_cold_start": latency_ms[0],
            },
            "qualitative_selection": {
                "rule": "descending false negatives, then false positives, then image ID",
                "image_ids": montage_ids,
            },
            "seed_schedule": seed_bundle.as_dict(),
            "limitations": [
                "COCO metrics describe 2D detection, not driving safety.",
                "Calibration correctness uses greedy class-aware IoU matching.",
                "Latency is measured on the declared evaluation client hardware.",
                "Privileged sample metadata is used only for post-hoc stratification.",
            ],
        }
        _write_json(summary_path, summary)

        artifacts = (
            (resolved_path, "resolved_evaluation_configuration"),
            (predictions_path, "canonical_coco_predictions"),
            (frame_log_path, "per_image_prediction_log"),
            (coco_log_path, "coco_evaluation_log"),
            (summary_path, "evaluation_summary"),
            (per_class_path, "per_class_metrics"),
            (strata_path, "stratified_metrics"),
            (calibration_path, "calibration_bins"),
            (bootstrap_path, "episode_bootstrap_metrics"),
            (montage_path, "qualitative_failure_panel"),
            *((path, "evaluation_plot") for path in plot_paths),
        )
        for path, role in artifacts:
            tracker.register_artifact(
                path,
                role=role,
                metadata={
                    "evaluation_id": config.evaluation_id,
                    "sample_count": len(samples),
                },
            )
    return {
        "run_id": tracker.run_id,
        "run_dir": str(tracker.run_dir),
        "manifest": str(tracker.manifest_path),
        "summary": summary,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run canonical COCO and operating-point evaluation for RT-DETR, "
            "YOLO, or a custom detector"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--detector",
        choices=("rtdetr", "yolo", "custom"),
        default=None,
    )
    parser.add_argument(
        "--model-package",
        help="verified immutable model package; replaces detector/weights/factory/image-size",
    )
    parser.add_argument("--weights")
    parser.add_argument("--detector-factory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run-id")
    parser.add_argument("--acknowledge-locked-test", action="store_true")
    args = parser.parse_args(argv)
    if args.model_package:
        conflicting = [
            option
            for option, value in (
                ("--detector", args.detector),
                ("--weights", args.weights),
                ("--detector-factory", args.detector_factory),
                ("--image-size", args.image_size),
            )
            if value is not None
        ]
        if conflicting:
            parser.error("--model-package cannot be combined with " + ", ".join(conflicting))
    else:
        args.detector = args.detector or "rtdetr"
        args.image_size = args.image_size or 640
        if args.detector == "custom" and not args.detector_factory:
            parser.error("--detector custom requires --detector-factory")
        if args.detector != "custom" and not args.weights:
            parser.error("built-in detectors require --weights")
    if args.image_size is not None and args.image_size <= 0:
        parser.error("--image-size must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    verified_model = load_verified_model(args.model_package) if args.model_package else None
    detector_config = (
        verified_model.detector_config(device=args.device, confidence=0.0)
        if verified_model is not None
        else DetectorConfig(
            backend=args.detector,
            weights=Path(args.weights) if args.weights else None,
            device=args.device,
            image_size=args.image_size,
            factory=args.detector_factory,
        )
    )
    result = run_evaluation(
        config_path=args.config,
        dataset_path=args.dataset,
        detector_config=detector_config,
        runs_root=args.runs_root,
        run_id=args.run_id,
        acknowledge_locked_test=args.acknowledge_locked_test,
        cli_args=vars(args),
        repository_root=Path.cwd(),
        model_reference_override=(verified_model.reference if verified_model is not None else None),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


__all__ = [
    "EVALUATION_RUN_SCHEMA_VERSION",
    "main",
    "parse_args",
    "run_evaluation",
]
