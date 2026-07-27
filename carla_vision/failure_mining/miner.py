"""Mine a deterministic, diverse human-review queue from validation failures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402

from ..artifacts import RunArtifactTracker, fingerprint_file
from ..evaluation.metrics import bbox_iou_xywh
from ..evaluation.verified import VerifiedEvaluation, load_verified_evaluation
from ..scenarios.seeds import derive_seed
from ..thresholds.verified import (
    VerifiedThresholdSelection,
    load_verified_threshold_selection,
)
from .contracts import FailureMiningConfig, load_failure_mining_config

FAILURE_MINING_RUN_SCHEMA_VERSION = "1.0"


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
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


def _write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
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
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_sample_context(
    evaluation: VerifiedEvaluation,
    sample: Mapping[str, Any],
) -> Mapping[str, Any]:
    reference = sample.get("metadata")
    if not isinstance(reference, Mapping):
        return {}
    path = evaluation.dataset.root / str(reference["path"])
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(value, Mapping):
        return {}
    context = value.get("context")
    return context if isinstance(context, Mapping) else {}


def _greedy_pairs(
    ground_truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    gt_indices: set[int],
    prediction_indices: set[int],
    *,
    predicate: Any,
    minimum_iou: float,
    maximum_iou: float | None = None,
) -> list[tuple[int, int, float]]:
    candidates = []
    for gt_index in gt_indices:
        for prediction_index in prediction_indices:
            if not predicate(
                ground_truth[gt_index],
                predictions[prediction_index],
            ):
                continue
            iou = bbox_iou_xywh(
                ground_truth[gt_index]["bbox"],
                predictions[prediction_index]["bbox"],
            )
            if iou < minimum_iou or (maximum_iou is not None and iou >= maximum_iou):
                continue
            candidates.append((iou, gt_index, prediction_index))
    candidates.sort(
        key=lambda item: (
            -item[0],
            int(ground_truth[item[1]].get("id", item[1])),
            item[2],
        )
    )
    pairs = []
    for iou, gt_index, prediction_index in candidates:
        if gt_index not in gt_indices or prediction_index not in prediction_indices:
            continue
        gt_indices.remove(gt_index)
        prediction_indices.remove(prediction_index)
        pairs.append((gt_index, prediction_index, iou))
    return pairs


def _failure_id(
    *,
    evaluation_run_id: str,
    image_id: int,
    failure_type: str,
    ground_truth_id: int | None,
    prediction_index: int | None,
    threshold: float,
) -> str:
    payload = json.dumps(
        {
            "evaluation_run_id": evaluation_run_id,
            "image_id": image_id,
            "failure_type": failure_type,
            "ground_truth_id": ground_truth_id,
            "prediction_index": prediction_index,
            "threshold": threshold,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "fail-" + hashlib.sha256(payload).hexdigest()[:20]


def _record(
    *,
    evaluation: VerifiedEvaluation,
    sample: Mapping[str, Any],
    context: Mapping[str, Any],
    categories: Mapping[int, str],
    failure_type: str,
    ground_truth: Mapping[str, Any] | None,
    prediction: Mapping[str, Any] | None,
    prediction_index: int | None,
    matched_iou: float,
    operating_confidence: float,
    width: int,
    height: int,
) -> dict[str, Any]:
    gt_category_id = int(ground_truth["category_id"]) if ground_truth is not None else None
    prediction_category_id = int(prediction["category_id"]) if prediction is not None else None
    object_category_id = gt_category_id if gt_category_id is not None else prediction_category_id
    bbox = ground_truth["bbox"] if ground_truth is not None else prediction["bbox"]
    area_ratio = float(bbox[2]) * float(bbox[3]) / float(width * height)
    center_x = (float(bbox[0]) + float(bbox[2]) / 2.0) / width
    gt_id = int(ground_truth.get("id", -1)) if ground_truth is not None else None
    return {
        "schema_version": FAILURE_MINING_RUN_SCHEMA_VERSION,
        "failure_id": _failure_id(
            evaluation_run_id=evaluation.run_id,
            image_id=int(sample["image_id"]),
            failure_type=failure_type,
            ground_truth_id=gt_id,
            prediction_index=prediction_index,
            threshold=operating_confidence,
        ),
        "source_evaluation_id": evaluation.run_id,
        "dataset_id": evaluation.dataset.dataset_id,
        "sample_id": str(sample["sample_id"]),
        "image_id": int(sample["image_id"]),
        "partition": str(sample["split"]),
        "scenario_id": str(sample["scenario_id"]),
        "episode_id": str(sample["episode_id"]),
        "episode_key": f"{sample['scenario_id']}/{sample['episode_id']}",
        "carla_frame": int(sample["carla_frame"]),
        "rgb_path": str(sample["rgb"]["path"]),
        "rgb_sha256": str(sample["rgb"]["sha256"]),
        "failure_type": failure_type,
        "ground_truth_id": gt_id,
        "ground_truth_category_id": gt_category_id,
        "ground_truth_category": (
            categories[gt_category_id] if gt_category_id is not None else None
        ),
        "ground_truth_bbox": (list(ground_truth["bbox"]) if ground_truth is not None else None),
        "prediction_index": prediction_index,
        "prediction_category_id": prediction_category_id,
        "prediction_category": (
            categories[prediction_category_id] if prediction_category_id is not None else None
        ),
        "prediction_bbox": (list(prediction["bbox"]) if prediction is not None else None),
        "prediction_score": (float(prediction["score"]) if prediction is not None else None),
        "object_category_id": object_category_id,
        "object_category": categories[int(object_category_id)],
        "matched_iou": matched_iou,
        "operating_confidence": operating_confidence,
        "area_ratio": area_ratio,
        "center_x_normalized": center_x,
        "map": str(context.get("map", "unknown")),
        "weather": str(
            context.get(
                "weather_recipe_id",
                context.get("weather_recipe", "unknown"),
            )
        ),
        "light": str(context.get("light", "unknown")),
        "base_severity": None,
        "rarity_factor": None,
        "priority_score": None,
        "tie_seed": None,
        "selected": False,
        "selected_rank": None,
        "review_status": "pending",
    }


def _failures_for_image(
    config: FailureMiningConfig,
    *,
    evaluation: VerifiedEvaluation,
    sample: Mapping[str, Any],
    ground_truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    categories: Mapping[int, str],
    context: Mapping[str, Any],
    operating_confidence: float,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    retained = [
        prediction
        for prediction in predictions
        if float(prediction["score"]) >= operating_confidence
    ]
    gt_indices = set(range(len(ground_truth)))
    prediction_indices = set(range(len(retained)))

    def same_class(
        gt: Mapping[str, Any],
        prediction: Mapping[str, Any],
    ) -> bool:
        return int(gt["category_id"]) == int(prediction["category_id"])

    def different_class(
        gt: Mapping[str, Any],
        prediction: Mapping[str, Any],
    ) -> bool:
        return int(gt["category_id"]) != int(prediction["category_id"])

    _greedy_pairs(
        ground_truth,
        retained,
        gt_indices,
        prediction_indices,
        predicate=same_class,
        minimum_iou=evaluation.config.matching_iou,
    )
    misclassifications = _greedy_pairs(
        ground_truth,
        retained,
        gt_indices,
        prediction_indices,
        predicate=different_class,
        minimum_iou=evaluation.config.matching_iou,
    )
    localizations = _greedy_pairs(
        ground_truth,
        retained,
        gt_indices,
        prediction_indices,
        predicate=same_class,
        minimum_iou=config.localization_iou_minimum,
        maximum_iou=evaluation.config.matching_iou,
    )
    failures: list[dict[str, Any]] = []

    def add(
        failure_type: str,
        gt_index: int | None,
        prediction_index: int | None,
        iou: float,
    ) -> None:
        if failure_type not in config.failure_types:
            return
        failures.append(
            _record(
                evaluation=evaluation,
                sample=sample,
                context=context,
                categories=categories,
                failure_type=failure_type,
                ground_truth=(ground_truth[gt_index] if gt_index is not None else None),
                prediction=(retained[prediction_index] if prediction_index is not None else None),
                prediction_index=prediction_index,
                matched_iou=iou,
                operating_confidence=operating_confidence,
                width=width,
                height=height,
            )
        )

    for gt_index, prediction_index, iou in misclassifications:
        add("misclassification", gt_index, prediction_index, iou)
    for gt_index, prediction_index, iou in localizations:
        add("localization", gt_index, prediction_index, iou)
    for gt_index in sorted(gt_indices):
        add("false_negative", gt_index, None, 0.0)
    for prediction_index in sorted(prediction_indices):
        add("false_positive", None, prediction_index, 0.0)
    return failures


def _prioritize(
    config: FailureMiningConfig,
    failures: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    group_counts = Counter(
        (str(row["failure_type"]), str(row["object_category"])) for row in failures
    )
    prioritized = []
    for raw in failures:
        row = dict(raw)
        failure_type = str(row["failure_type"])
        category = str(row["object_category"])
        critical = config.critical_multiplier if category in config.critical_categories else 1.0
        scale = 1.0 + min(
            1.0,
            math.sqrt(max(0.0, float(row["area_ratio"]))) * 5.0,
        )
        corridor = 1.25 if 0.25 <= float(row["center_x_normalized"]) <= 0.75 else 1.0
        confidence = (
            1.0 if row["prediction_score"] is None else 0.5 + float(row["prediction_score"])
        )
        base = (
            float(config.severity_weights[failure_type]) * critical * scale * corridor * confidence
        )
        rarity = 1.0 / math.sqrt(group_counts[(failure_type, category)])
        rarity_factor = 1.0 + config.rarity_weight * rarity
        tie_seed = derive_seed(
            config.master_seed,
            f"failure-mining/{config.mining_id}/{row['failure_id']}",
        )
        row["base_severity"] = base
        row["rarity_factor"] = rarity_factor
        row["priority_score"] = base * rarity_factor
        row["tie_seed"] = tie_seed
        prioritized.append(row)
    return prioritized


def _select_queue(
    config: FailureMiningConfig,
    failures: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(
        (dict(row) for row in failures),
        key=lambda row: (
            -float(row["priority_score"]),
            int(row["tie_seed"]),
            str(row["failure_id"]),
        ),
    )
    image_counts: Counter[int] = Counter()
    episode_counts: Counter[str] = Counter()
    selected_ids: set[str] = set()
    queue: list[dict[str, Any]] = []
    for row in ordered:
        if len(queue) >= config.max_candidates:
            break
        image_id = int(row["image_id"])
        episode_key = str(row["episode_key"])
        if (
            image_counts[image_id] >= config.per_image_limit
            or episode_counts[episode_key] >= config.per_episode_limit
        ):
            continue
        row["selected"] = True
        row["selected_rank"] = len(queue) + 1
        queue.append(row)
        selected_ids.add(str(row["failure_id"]))
        image_counts[image_id] += 1
        episode_counts[episode_key] += 1
    all_rows = []
    selected_by_id = {str(row["failure_id"]): row for row in queue}
    for row in sorted(
        failures,
        key=lambda item: (
            int(item["image_id"]),
            str(item["failure_type"]),
            str(item["failure_id"]),
        ),
    ):
        all_rows.append(dict(selected_by_id.get(str(row["failure_id"]), row)))
    return all_rows, queue


def _csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    expanded = {
        **row,
        "ground_truth_bbox_json": json.dumps(
            row["ground_truth_bbox"],
            separators=(",", ":"),
        ),
        "prediction_bbox_json": json.dumps(
            row["prediction_bbox"],
            separators=(",", ":"),
        ),
    }
    return {field: expanded.get(field) for field in _QUEUE_FIELDS}


_QUEUE_FIELDS = (
    "selected_rank",
    "failure_id",
    "failure_type",
    "priority_score",
    "base_severity",
    "rarity_factor",
    "object_category",
    "source_evaluation_id",
    "dataset_id",
    "sample_id",
    "image_id",
    "partition",
    "scenario_id",
    "episode_id",
    "carla_frame",
    "rgb_path",
    "rgb_sha256",
    "ground_truth_id",
    "ground_truth_category",
    "ground_truth_bbox_json",
    "prediction_index",
    "prediction_category",
    "prediction_bbox_json",
    "prediction_score",
    "matched_iou",
    "operating_confidence",
    "area_ratio",
    "center_x_normalized",
    "map",
    "weather",
    "light",
    "review_status",
)


def _save_figure(figure: Any, base_path: Path) -> tuple[Path, Path]:
    png = base_path.with_suffix(".png")
    svg = base_path.with_suffix(".svg")
    figure.tight_layout()
    figure.savefig(png, dpi=180)
    figure.savefig(svg)
    plt.close(figure)
    return png, svg


def _plot_type_counts(
    base_path: Path,
    queue: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    counts = Counter(str(row["failure_type"]) for row in queue)
    labels = sorted(counts)
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.bar(labels, [counts[label] for label in labels], color="#e76f51")
    axis.set_ylabel("Selected failures")
    axis.set_title("Failure review queue by type")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    return _save_figure(figure, base_path)


def _plot_category_counts(
    base_path: Path,
    queue: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    counts = Counter(str(row["object_category"]) for row in queue)
    labels = sorted(counts)
    figure, axis = plt.subplots(figsize=(10, max(5, len(labels) * 0.45)))
    y = np.arange(len(labels))
    axis.barh(y, [counts[label] for label in labels], color="#3a86ff")
    axis.set_yticks(y, labels)
    axis.set_xlabel("Selected failures")
    axis.set_title("Failure review queue by object category")
    axis.grid(axis="x", alpha=0.25)
    return _save_figure(figure, base_path)


def _plot_priority(
    base_path: Path,
    failures: Sequence[Mapping[str, Any]],
    queue: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.hist(
        [float(row["priority_score"]) for row in failures],
        bins=min(30, max(5, round(math.sqrt(len(failures))))),
        alpha=0.65,
        label="all failures",
    )
    axis.hist(
        [float(row["priority_score"]) for row in queue],
        bins=min(30, max(5, round(math.sqrt(len(queue))))),
        alpha=0.65,
        label="review queue",
    )
    axis.set_xlabel("Priority score")
    axis.set_ylabel("Failures")
    axis.set_title("Failure priority distribution")
    axis.grid(alpha=0.25)
    axis.legend()
    return _save_figure(figure, base_path)


def _montage(
    path: Path,
    *,
    evaluation: VerifiedEvaluation,
    queue: Sequence[Mapping[str, Any]],
    count: int,
) -> list[str]:
    selected = list(queue[:count])
    tile_width, tile_height = 480, 300
    columns = 2
    rows = math.ceil(len(selected) / columns)
    canvas = np.full(
        (rows * tile_height, columns * tile_width, 3),
        24,
        dtype=np.uint8,
    )
    selected_ids = []
    for index, failure in enumerate(selected):
        image = cv2.imread(
            str(evaluation.dataset.root / str(failure["rgb_path"])),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise RuntimeError(f"could not read failure image {failure['rgb_path']}")
        gt_bbox = failure["ground_truth_bbox"]
        if gt_bbox is not None:
            x, y, width, height = (int(round(float(value))) for value in gt_bbox)
            cv2.rectangle(
                image,
                (x, y),
                (x + width, y + height),
                (40, 220, 40),
                2,
            )
        prediction_bbox = failure["prediction_bbox"]
        if prediction_bbox is not None:
            x, y, width, height = (int(round(float(value))) for value in prediction_bbox)
            cv2.rectangle(
                image,
                (x, y),
                (x + width, y + height),
                (30, 80, 245),
                2,
            )
        image = cv2.resize(image, (tile_width, tile_height))
        cv2.rectangle(image, (0, 0), (tile_width, 42), (18, 18, 18), -1)
        caption = (
            f"#{failure['selected_rank']} {failure['failure_type']} "
            f"{failure['object_category']} score={failure['priority_score']:.2f}"
        )
        cv2.putText(
            image,
            caption[:74],
            (8, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        row, column = divmod(index, columns)
        canvas[
            row * tile_height : (row + 1) * tile_height,
            column * tile_width : (column + 1) * tile_width,
        ] = image
        selected_ids.append(str(failure["failure_id"]))
    if not selected_ids:
        raise RuntimeError("failure montage queue is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"could not write failure montage {path}")
    return selected_ids


def _write_checksum_index(
    root: Path,
    paths: Sequence[Path],
    output: Path,
) -> None:
    lines = []
    for path in sorted(
        paths,
        key=lambda value: value.relative_to(root).as_posix().encode("utf-8"),
    ):
        reference = fingerprint_file(path)
        lines.append(f"{reference['sha256']}  {path.relative_to(root).as_posix()}")
    _atomic_write_text(output, "\n".join(lines) + "\n")


def _threshold(
    config: FailureMiningConfig,
    evaluation: VerifiedEvaluation,
    threshold_selection: str | Path | None,
) -> tuple[float, VerifiedThresholdSelection | None]:
    if config.threshold_source == "evaluation_config":
        if threshold_selection is not None:
            raise ValueError(
                "threshold selection artifact is incompatible with evaluation_config source"
            )
        return evaluation.config.operating_confidence, None
    if threshold_selection is None:
        raise ValueError("selection_artifact threshold source requires --threshold-selection")
    selection = load_verified_threshold_selection(threshold_selection)
    if selection.descriptor["source_evaluation"] != evaluation.reference:
        raise ValueError("threshold selection was not produced from the source evaluation")
    return selection.operating_confidence, selection


def run_failure_mining(
    *,
    config_path: str | Path,
    evaluation_run: str | Path,
    threshold_selection: str | Path | None = None,
    runs_root: str | Path = "runs",
    cli_args: Sequence[str] | Mapping[str, Any] = (),
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    config_path_resolved = Path(config_path).expanduser().resolve(strict=True)
    config = load_failure_mining_config(config_path_resolved)
    evaluation = load_verified_evaluation(evaluation_run)
    if any(not partition.startswith("val") for partition in evaluation.config.partitions):
        raise RuntimeError("failure mining requires validation-only source partitions")
    if config.localization_iou_minimum >= evaluation.config.matching_iou:
        raise ValueError("localization_iou_minimum must be below evaluation matching_iou")
    category_names = {str(category["name"]) for category in evaluation.dataset.coco["categories"]}
    unknown_critical = sorted(set(config.critical_categories) - category_names)
    if unknown_critical:
        raise ValueError(
            "critical categories are absent from ontology: " + ", ".join(unknown_critical)
        )
    operating_confidence, verified_threshold = _threshold(
        config,
        evaluation,
        threshold_selection,
    )
    tracker = RunArtifactTracker(
        runs_root,
        run_id=config.mining_id,
        cli_args=cli_args,
        config={
            "schema_version": FAILURE_MINING_RUN_SCHEMA_VERSION,
            "object_type": "validation_failure_mining",
            "mining": config.as_dict(),
            "operating_confidence": operating_confidence,
        },
        repository_root=repository_root,
        model_refs=[evaluation.resolved["model"]],
        dataset_refs=[evaluation.dataset.reference],
        input_refs=[
            {
                "kind": "failure_mining_preregistration",
                **fingerprint_file(config_path_resolved),
            },
            evaluation.reference,
            *([verified_threshold.reference] if verified_threshold is not None else []),
        ],
    )
    descriptor: dict[str, Any]
    with tracker:
        resolved_path = tracker.artifact_path("resolved_failure_mining_config.json")
        all_path = tracker.artifact_path("failures/all_failures.jsonl")
        queue_path = tracker.artifact_path("failures/review_queue.jsonl")
        queue_csv_path = tracker.artifact_path("tables/review_queue.csv")
        review_template_path = tracker.artifact_path("review/review_template.csv")
        summary_path = tracker.artifact_path("mining_summary.json")
        descriptor_path = tracker.artifact_path("failure_mining.json")
        report_path = tracker.artifact_path("failure_mining.md")
        montage_path = tracker.artifact_path("qualitative/failure_review_queue.png")
        checksum_path = tracker.artifact_path("checksums.sha256")
        resolved = {
            "schema_version": FAILURE_MINING_RUN_SCHEMA_VERSION,
            "object_type": "validation_failure_mining",
            "mining": config.as_dict(),
            "source_config": fingerprint_file(config_path_resolved),
            "source_evaluation": dict(evaluation.reference),
            "threshold_selection": (
                dict(verified_threshold.reference) if verified_threshold is not None else None
            ),
            "operating_confidence": operating_confidence,
            "matching_iou": evaluation.config.matching_iou,
            "dataset": dict(evaluation.dataset.reference),
            "model": evaluation.resolved["model"],
            "runtime_sensor_contract": "front_monocular_rgb_only",
        }
        _write_json(resolved_path, resolved)

        samples = [
            sample
            for sample in evaluation.dataset.dataset["samples"]
            if str(sample["split"]) in evaluation.config.partitions
        ]
        coco_images = {int(image["id"]): image for image in evaluation.dataset.coco["images"]}
        categories = {
            int(category["id"]): str(category["name"])
            for category in evaluation.dataset.coco["categories"]
        }
        ground_truth_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        predictions_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for annotation in evaluation.dataset.coco["annotations"]:
            ground_truth_by_image[int(annotation["image_id"])].append(annotation)
        for prediction in evaluation.predictions:
            predictions_by_image[int(prediction["image_id"])].append(prediction)
        raw_failures: list[dict[str, Any]] = []
        for sample in samples:
            image_id = int(sample["image_id"])
            image_record = coco_images[image_id]
            raw_failures.extend(
                _failures_for_image(
                    config,
                    evaluation=evaluation,
                    sample=sample,
                    ground_truth=ground_truth_by_image.get(image_id, ()),
                    predictions=predictions_by_image.get(image_id, ()),
                    categories=categories,
                    context=_load_sample_context(evaluation, sample),
                    operating_confidence=operating_confidence,
                    width=int(image_record["width"]),
                    height=int(image_record["height"]),
                )
            )
        if not raw_failures:
            raise RuntimeError("source evaluation contains no configured failure candidates")
        prioritized = _prioritize(config, raw_failures)
        all_failures, queue = _select_queue(config, prioritized)
        if not queue:
            raise RuntimeError("failure diversity caps selected an empty queue")
        _write_jsonl(all_path, all_failures)
        _write_jsonl(queue_path, queue)
        _write_csv(
            queue_csv_path,
            [_csv_row(row) for row in queue],
            fields=_QUEUE_FIELDS,
        )
        review_rows = [
            {
                "failure_id": row["failure_id"],
                "selected_rank": row["selected_rank"],
                "failure_type": row["failure_type"],
                "object_category": row["object_category"],
                "decision": "",
                "corrected_failure_type": "",
                "reviewer": "",
                "review_notes": "",
                "proposed_remedy": "",
            }
            for row in queue
        ]
        _write_csv(
            review_template_path,
            review_rows,
            fields=(
                "failure_id",
                "selected_rank",
                "failure_type",
                "object_category",
                "decision",
                "corrected_failure_type",
                "reviewer",
                "review_notes",
                "proposed_remedy",
            ),
        )
        plot_paths: list[Path] = []
        plot_paths.extend(
            _plot_type_counts(
                tracker.artifact_path(
                    "plots/failure_types",
                    create_parent=True,
                ),
                queue,
            )
        )
        plot_paths.extend(
            _plot_category_counts(
                tracker.artifact_path(
                    "plots/failure_categories",
                    create_parent=True,
                ),
                queue,
            )
        )
        plot_paths.extend(
            _plot_priority(
                tracker.artifact_path(
                    "plots/priority_distribution",
                    create_parent=True,
                ),
                all_failures,
                queue,
            )
        )
        montage_ids = _montage(
            montage_path,
            evaluation=evaluation,
            queue=queue,
            count=config.montage_count,
        )
        all_type_counts = dict(sorted(Counter(row["failure_type"] for row in all_failures).items()))
        queue_type_counts = dict(sorted(Counter(row["failure_type"] for row in queue).items()))
        summary = {
            "schema_version": FAILURE_MINING_RUN_SCHEMA_VERSION,
            "mining_id": config.mining_id,
            "source_evaluation_id": evaluation.run_id,
            "validation_partitions": list(evaluation.config.partitions),
            "operating_confidence": operating_confidence,
            "all_failure_count": len(all_failures),
            "review_queue_count": len(queue),
            "all_type_counts": all_type_counts,
            "queue_type_counts": queue_type_counts,
            "unique_images_all": len({int(row["image_id"]) for row in all_failures}),
            "unique_images_queue": len({int(row["image_id"]) for row in queue}),
            "unique_episodes_queue": len({str(row["episode_key"]) for row in queue}),
            "montage_failure_ids": montage_ids,
            "review_status": "pending",
        }
        _write_json(summary_path, summary)
        descriptor = {
            "schema_version": FAILURE_MINING_RUN_SCHEMA_VERSION,
            "object_type": "validation_failure_mining_release",
            "status": "complete",
            "mining_id": config.mining_id,
            "title": config.title,
            "purpose": config.purpose,
            "source_evaluation": dict(evaluation.reference),
            "threshold_selection": (
                dict(verified_threshold.reference) if verified_threshold is not None else None
            ),
            "dataset": dict(evaluation.dataset.reference),
            "model": evaluation.resolved["model"],
            "validation_partitions": list(evaluation.config.partitions),
            "operating_confidence": operating_confidence,
            "matching_iou": evaluation.config.matching_iou,
            "all_failure_count": len(all_failures),
            "review_queue_count": len(queue),
            "review_status": "pending",
            "runtime_sensor_contract": "front_monocular_rgb_only",
            "privileged_metadata_use": (
                "ground-truth failure categorization and post-hoc strata only"
            ),
            "limitations": [
                "Failure ranking is a development heuristic, not a safety metric.",
                "Review candidates remain in validation and never migrate to training.",
                "A remedy requires a new scenario seed and a new dataset release.",
                "Human confirmation is required before root-cause claims.",
            ],
        }
        _write_json(descriptor_path, descriptor)
        _atomic_write_text(
            report_path,
            f"""# {config.title}

- Mining ID: `{config.mining_id}`
- Source evaluation: `{evaluation.run_id}`
- Validation partitions: {", ".join(evaluation.config.partitions)}
- Operating confidence: `{operating_confidence:.8f}`
- All categorized failures: `{len(all_failures)}`
- Review queue: `{len(queue)}`
- Review status: `pending`

## Queue by type

{json.dumps(queue_type_counts, sort_keys=True)}

## Boundary

The queue is selected only from validation data. Confirmed validation frames
remain validation evidence and cannot be copied into training. A training
remedy must be generated under a new scenario/episode identity and released in
a new dataset version. Complete `review/review_template.csv` before drawing
root-cause conclusions.
""",
        )
        payload_paths = [
            resolved_path,
            all_path,
            queue_path,
            queue_csv_path,
            review_template_path,
            summary_path,
            descriptor_path,
            report_path,
            montage_path,
            *plot_paths,
        ]
        _write_checksum_index(tracker.run_dir, payload_paths, checksum_path)
        artifacts = {
            resolved_path: "resolved_failure_mining_configuration",
            all_path: "all_failure_candidates",
            queue_path: "failure_review_queue",
            queue_csv_path: "failure_review_queue_table",
            review_template_path: "failure_review_template",
            summary_path: "failure_mining_summary",
            descriptor_path: "failure_mining_release_manifest",
            report_path: "failure_mining_report",
            montage_path: "failure_mining_qualitative_panel",
            checksum_path: "failure_mining_checksum_index",
            **{path: "failure_mining_plot" for path in plot_paths},
        }
        for path, role in artifacts.items():
            tracker.register_artifact(
                path,
                role=role,
                metadata={
                    "mining_id": config.mining_id,
                    "source_evaluation": evaluation.run_id,
                },
            )
    return {
        "mining_id": tracker.run_id,
        "run_dir": str(tracker.run_dir),
        "manifest": str(tracker.manifest_path),
        "descriptor": descriptor,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Categorize and rank detector failures from a verified validation-only evaluation"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluation-run", required=True)
    parser.add_argument("--threshold-selection")
    parser.add_argument("--runs-root", default="runs")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_failure_mining(
        config_path=args.config,
        evaluation_run=args.evaluation_run,
        threshold_selection=args.threshold_selection,
        runs_root=args.runs_root,
        cli_args=vars(args),
        repository_root=Path.cwd(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


__all__ = [
    "FAILURE_MINING_RUN_SCHEMA_VERSION",
    "main",
    "parse_args",
    "run_failure_mining",
]
