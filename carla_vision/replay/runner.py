"""Run paired detector comparisons over one immutable ordered RGB sample set."""

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402

from ..artifacts import RunArtifactTracker, fingerprint_file
from ..contracts import Detector, DetectorConfig
from ..dataset.verified import VerifiedDataset, load_verified_dataset
from ..evaluation.contracts import EvaluationConfig, load_evaluation_config
from ..evaluation.metrics import operating_metrics
from ..evaluation.runner import run_evaluation
from ..model_release.verified import load_verified_model
from ..scenarios.seeds import derive_seed
from ..verification import verify_research_object
from .contracts import (
    REPLAY_RUN_SCHEMA_VERSION,
    ReplayConfig,
    ReplayModelSpec,
    load_replay_config,
)


@dataclass(frozen=True)
class _ResolvedModel:
    spec: ReplayModelSpec
    detector_config: DetectorConfig
    reference: Mapping[str, Any]


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


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path, name: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read {name}: {error}") from error


def _load_jsonl(path: Path, name: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise RuntimeError(f"{name} line {line_number} is not an object")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read {name}: {error}") from error
    return rows


def _artifact_by_role(
    root: Path,
    manifest: Mapping[str, Any],
    role: str,
) -> Path:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("child evaluation manifest artifacts are invalid")
    matches = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping) and artifact.get("role") == role
    ]
    if len(matches) != 1:
        raise RuntimeError(f"child evaluation must contain exactly one {role!r} artifact")
    return (root / str(matches[0]["path"])).resolve(strict=True)


def _selected_samples(
    dataset: VerifiedDataset,
    evaluation: EvaluationConfig,
) -> tuple[Mapping[str, Any], ...]:
    missing = sorted(set(evaluation.partitions) - set(dataset.partitions))
    if missing:
        raise ValueError("evaluation partitions are absent from the dataset: " + ", ".join(missing))
    samples = tuple(
        sample
        for sample in dataset.dataset["samples"]
        if str(sample["split"]) in evaluation.partitions
    )
    if not samples:
        raise ValueError("replay selection contains no samples")
    return samples


def _sample_order_rows(
    samples: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "position": position,
            "sample_id": str(sample["sample_id"]),
            "image_id": int(sample["image_id"]),
            "partition": str(sample["split"]),
            "scenario_id": str(sample["scenario_id"]),
            "episode_id": str(sample["episode_id"]),
            "carla_frame": int(sample["carla_frame"]),
            "rgb_sha256": str(sample["rgb"]["sha256"]),
        }
        for position, sample in enumerate(samples, start=1)
    ]


def _order_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        list(rows),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _child_order_rows(
    frame_rows: Sequence[Mapping[str, Any]],
    samples_by_id: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for row in frame_rows:
        image_id = int(row["image_id"])
        sample = samples_by_id.get(image_id)
        if sample is None:
            raise RuntimeError(f"child evaluation returned unexpected image ID {image_id}")
        result.append(
            {
                "position": int(row["position"]),
                "sample_id": str(row["sample_id"]),
                "image_id": image_id,
                "partition": str(row["partition"]),
                "scenario_id": str(row["scenario_id"]),
                "episode_id": str(row["episode_id"]),
                "carla_frame": int(row["carla_frame"]),
                "rgb_sha256": str(sample["rgb"]["sha256"]),
            }
        )
    return result


def _resolve_models(config: ReplayConfig) -> tuple[_ResolvedModel, ...]:
    resolved: list[_ResolvedModel] = []
    for spec in config.models:
        if spec.model_package is not None:
            package = load_verified_model(spec.model_package)
            detector_config = package.detector_config(
                device=spec.device,
                confidence=0.0,
            )
            source_reference: Mapping[str, Any] = dict(package.reference)
        else:
            if spec.detector is None:
                raise RuntimeError(f"model {spec.model_id} has no detector specification")
            detector_config = spec.detector.detector_config(device=spec.device)
            source_reference = {
                "kind": "development_detector_specification",
                "backend": detector_config.backend,
                "image_size": detector_config.image_size,
                "factory": detector_config.factory,
                "options": dict(detector_config.options),
                "weights": (
                    fingerprint_file(detector_config.weights)
                    if detector_config.weights is not None
                    else None
                ),
            }
        reference = {
            "kind": "paired_replay_model",
            "replay_model_id": spec.model_id,
            "label": spec.label,
            "device": spec.device,
            "source": source_reference,
        }
        resolved.append(
            _ResolvedModel(
                spec=spec,
                detector_config=detector_config,
                reference=reference,
            )
        )
    return tuple(resolved)


def _safe_child_run_id(replay_id: str, model_id: str) -> str:
    candidate = f"{replay_id}--{model_id}"
    if len(candidate) <= 128:
        return candidate
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:16]
    return f"{candidate[:111]}-{digest}"


def _ratios(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _quantile_interval(
    values: Sequence[float],
    *,
    confidence: float,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(array.mean()),
        "lower": float(np.quantile(array, alpha)),
        "upper": float(np.quantile(array, 1.0 - alpha)),
    }


def _paired_bootstrap(
    *,
    reference_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    replicates: int,
    confidence: float,
    seed: int,
    latency_exclude_first_images: int,
) -> dict[str, Any]:
    reference_by_image = {int(row["image_id"]): row for row in reference_rows}
    candidate_by_image = {int(row["image_id"]): row for row in candidate_rows}
    if reference_by_image.keys() != candidate_by_image.keys():
        raise RuntimeError("paired bootstrap image sets disagree")
    episode_images: dict[str, list[int]] = defaultdict(list)
    ordered = sorted(reference_rows, key=lambda row: int(row["position"]))
    for row in ordered:
        episode_images[str(row["episode_key"])].append(int(row["image_id"]))
    episode_ids = tuple(sorted(episode_images))
    eligible_latency_images = {
        int(row["image_id"]) for row in ordered[latency_exclude_first_images:]
    }
    latency_episode_ids = tuple(
        episode_id
        for episode_id in episode_ids
        if any(image_id in eligible_latency_images for image_id in episode_images[episode_id])
    )
    if not episode_ids or not latency_episode_ids:
        raise RuntimeError("paired bootstrap has no eligible episode")

    def aggregate(
        rows_by_image: Mapping[int, Mapping[str, Any]],
        sampled_episodes: Sequence[str],
    ) -> dict[str, float]:
        counts = Counter(tp=0, fp=0, fn=0)
        latencies: list[float] = []
        for episode_id in sampled_episodes:
            for image_id in episode_images[str(episode_id)]:
                row = rows_by_image[image_id]
                counts.update(
                    {
                        "tp": int(row["tp"]),
                        "fp": int(row["fp"]),
                        "fn": int(row["fn"]),
                    }
                )
                if image_id in eligible_latency_images:
                    latencies.append(float(row["inference_ms"]))
        ratios = _ratios(counts["tp"], counts["fp"], counts["fn"])
        return {
            **ratios,
            "fp": float(counts["fp"]),
            "fn": float(counts["fn"]),
            "error_count": float(counts["fp"] + counts["fn"]),
            "latency_mean_ms": (float(np.mean(latencies)) if latencies else float("nan")),
        }

    point_reference = aggregate(reference_by_image, episode_ids)
    point_candidate = aggregate(candidate_by_image, episode_ids)
    metrics = (
        "precision",
        "recall",
        "f1",
        "fp",
        "fn",
        "error_count",
        "latency_mean_ms",
    )
    samples: dict[str, list[float]] = {metric: [] for metric in metrics}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        sampled = tuple(
            str(value)
            for value in rng.choice(
                episode_ids,
                size=len(episode_ids),
                replace=True,
            )
        )
        reference = aggregate(reference_by_image, sampled)
        candidate = aggregate(candidate_by_image, sampled)
        for metric in metrics:
            if metric == "latency_mean_ms":
                continue
            delta = candidate[metric] - reference[metric]
            if math.isfinite(delta):
                samples[metric].append(delta)
        sampled_latency = tuple(
            str(value)
            for value in rng.choice(
                latency_episode_ids,
                size=len(latency_episode_ids),
                replace=True,
            )
        )
        latency_reference = aggregate(reference_by_image, sampled_latency)
        latency_candidate = aggregate(candidate_by_image, sampled_latency)
        latency_delta = latency_candidate["latency_mean_ms"] - latency_reference["latency_mean_ms"]
        if math.isfinite(latency_delta):
            samples["latency_mean_ms"].append(latency_delta)

    result: dict[str, Any] = {
        "unit": "episode",
        "episode_count": len(episode_ids),
        "latency_episode_count": len(latency_episode_ids),
        "replicates": replicates,
        "confidence": confidence,
        "seed": seed,
        "degenerate_single_episode": len(episode_ids) == 1,
        "latency_exclude_first_images": latency_exclude_first_images,
        "metrics": {},
    }
    for metric in metrics:
        if len(samples[metric]) != replicates:
            raise RuntimeError(f"paired bootstrap produced incomplete finite values for {metric}")
        result["metrics"][metric] = {
            "reference": point_reference[metric],
            "candidate": point_candidate[metric],
            "delta_candidate_minus_reference": (point_candidate[metric] - point_reference[metric]),
            "favorable_direction": (
                "higher" if metric in {"precision", "recall", "f1"} else "lower"
            ),
            "bootstrap_delta": _quantile_interval(
                samples[metric],
                confidence=confidence,
            ),
        }
    return result


def _describe(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p95": None,
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
        "stddev": float(array.std()),
    }


def _save_figure(figure: Any, base_path: Path) -> tuple[Path, Path]:
    png = base_path.with_suffix(".png")
    svg = base_path.with_suffix(".svg")
    figure.tight_layout()
    figure.savefig(png, dpi=180)
    figure.savefig(svg)
    plt.close(figure)
    return png, svg


def _plot_accuracy(
    base_path: Path,
    aggregate_rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    labels = [str(row["label"]) for row in aggregate_rows]
    metrics = (
        ("coco_ap_50_95", "COCO AP"),
        ("operating_recall", "Recall"),
        ("operating_f1", "F1"),
    )
    x = np.arange(len(labels))
    width = 0.24
    figure, axis = plt.subplots(figsize=(max(9, len(labels) * 2.5), 6))
    for index, (field, title) in enumerate(metrics):
        values = [float(row[field]) for row in aggregate_rows]
        axis.bar(
            x + (index - 1) * width,
            values,
            width=width,
            label=title,
        )
    axis.set_xticks(x, labels, rotation=15, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Metric value")
    axis.set_title("Paired detector accuracy on identical RGB frames")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    return _save_figure(figure, base_path)


def _plot_latency(
    base_path: Path,
    model_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    labels: Mapping[str, str],
    *,
    exclude_first: int,
) -> tuple[Path, Path]:
    ordered_ids = list(model_rows)
    values = [
        [
            float(row["inference_ms"])
            for row in sorted(model_rows[model_id], key=lambda item: int(item["position"]))[
                exclude_first:
            ]
        ]
        for model_id in ordered_ids
    ]
    figure, axis = plt.subplots(figsize=(max(9, len(values) * 2.2), 6))
    axis.boxplot(values, tick_labels=[labels[model_id] for model_id in ordered_ids])
    axis.set_ylabel("Inference latency (ms)")
    axis.set_title(f"Paired latency after excluding first {exclude_first} image(s)")
    axis.grid(axis="y", alpha=0.25)
    return _save_figure(figure, base_path)


def _plot_error_counts(
    base_path: Path,
    aggregate_rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    labels = [str(row["label"]) for row in aggregate_rows]
    x = np.arange(len(labels))
    width = 0.25
    figure, axis = plt.subplots(figsize=(max(9, len(labels) * 2.5), 6))
    for offset, (field, title, color) in enumerate(
        (
            ("operating_tp", "TP", "#2a9d8f"),
            ("operating_fp", "FP", "#e9c46a"),
            ("operating_fn", "FN", "#e76f51"),
        )
    ):
        axis.bar(
            x + (offset - 1) * width,
            [int(row[field]) for row in aggregate_rows],
            width=width,
            label=title,
            color=color,
        )
    axis.set_xticks(x, labels, rotation=15, ha="right")
    axis.set_ylabel("Detections")
    axis.set_title("Operating-point outcomes")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    return _save_figure(figure, base_path)


def _plot_disagreement_heatmap(
    base_path: Path,
    model_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    labels: Mapping[str, str],
) -> tuple[Path, Path]:
    model_ids = list(model_rows)
    image_ids = [
        int(row["image_id"])
        for row in sorted(
            next(iter(model_rows.values())),
            key=lambda item: int(item["position"]),
        )
    ]
    matrix = np.asarray(
        [
            [
                int(row["fp"]) + int(row["fn"])
                for row in sorted(
                    model_rows[model_id],
                    key=lambda item: int(item["position"]),
                )
            ]
            for model_id in model_ids
        ],
        dtype=np.int64,
    )
    figure, axis = plt.subplots(
        figsize=(max(8, len(image_ids) * 0.8), max(4, len(model_ids) * 0.8))
    )
    image = axis.imshow(matrix, cmap="magma")
    axis.set_xticks(np.arange(len(image_ids)), [str(value) for value in image_ids])
    axis.set_yticks(
        np.arange(len(model_ids)),
        [labels[model_id] for model_id in model_ids],
    )
    axis.set_xlabel("Image ID")
    axis.set_ylabel("Model")
    axis.set_title("Per-image error burden (FP + FN)")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, fraction=0.046)
    return _save_figure(figure, base_path)


def _draw_boxes(
    image: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    categories: Mapping[int, str],
    color: tuple[int, int, int],
    prefix: str,
    confidence_threshold: float | None = None,
) -> None:
    for row in rows:
        if confidence_threshold is not None and float(row["score"]) < confidence_threshold:
            continue
        x, y, width, height = (int(round(float(value))) for value in row["bbox"])
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
        suffix = "" if "score" not in row else f" {float(row['score']):.2f}"
        cv2.putText(
            image,
            f"{prefix} {categories[int(row['category_id'])]}{suffix}",
            (x, max(42, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )


def _comparison_montage(
    path: Path,
    *,
    dataset: VerifiedDataset,
    samples: Sequence[Mapping[str, Any]],
    model_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    predictions_by_model_image: Mapping[
        str,
        Mapping[int, Sequence[Mapping[str, Any]]],
    ],
    labels: Mapping[str, str],
    categories: Mapping[int, str],
    ground_truth_by_image: Mapping[int, Sequence[Mapping[str, Any]]],
    operating_confidence: float,
    count: int,
) -> list[int]:
    rows_by_model_image = {
        model_id: {int(row["image_id"]): row for row in rows}
        for model_id, rows in model_rows.items()
    }
    ranked = sorted(
        samples,
        key=lambda sample: (
            -(
                max(
                    int(rows_by_model_image[model_id][int(sample["image_id"])]["fp"])
                    + int(rows_by_model_image[model_id][int(sample["image_id"])]["fn"])
                    for model_id in model_rows
                )
                - min(
                    int(rows_by_model_image[model_id][int(sample["image_id"])]["fp"])
                    + int(rows_by_model_image[model_id][int(sample["image_id"])]["fn"])
                    for model_id in model_rows
                )
            ),
            -sum(
                int(rows_by_model_image[model_id][int(sample["image_id"])]["fp"])
                + int(rows_by_model_image[model_id][int(sample["image_id"])]["fn"])
                for model_id in model_rows
            ),
            int(sample["image_id"]),
        ),
    )[:count]
    model_ids = list(model_rows)
    tile_width = 480
    tile_height = 316
    canvas = np.full(
        (len(ranked) * tile_height, (len(model_ids) + 1) * tile_width, 3),
        24,
        dtype=np.uint8,
    )
    selected: list[int] = []
    for row_index, sample in enumerate(ranked):
        image_id = int(sample["image_id"])
        source = cv2.imread(
            str(dataset.root / str(sample["rgb"]["path"])),
            cv2.IMREAD_COLOR,
        )
        if source is None:
            raise RuntimeError(f"could not read montage image {image_id}")
        panels: list[np.ndarray] = []
        ground_truth_panel = source.copy()
        _draw_boxes(
            ground_truth_panel,
            ground_truth_by_image.get(image_id, ()),
            categories=categories,
            color=(40, 220, 40),
            prefix="GT",
        )
        panels.append(ground_truth_panel)
        for model_id in model_ids:
            panel = source.copy()
            _draw_boxes(
                panel,
                predictions_by_model_image[model_id].get(image_id, ()),
                categories=categories,
                color=(30, 80, 245),
                prefix="P",
                confidence_threshold=operating_confidence,
            )
            panels.append(panel)
        titles = ["Ground truth", *(labels[model_id] for model_id in model_ids)]
        for column, (panel, title) in enumerate(zip(panels, titles, strict=True)):
            panel = cv2.resize(panel, (tile_width, tile_height))
            cv2.rectangle(panel, (0, 0), (tile_width, 34), (18, 18, 18), -1)
            if column == 0:
                caption = f"image={image_id} | {title}"
            else:
                model_id = model_ids[column - 1]
                outcome = rows_by_model_image[model_id][image_id]
                caption = f"{title} | TP={outcome['tp']} FP={outcome['fp']} FN={outcome['fn']}"
            cv2.putText(
                panel,
                caption[:76],
                (8, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            top = row_index * tile_height
            left = column * tile_width
            canvas[top : top + tile_height, left : left + tile_width] = panel
        selected.append(image_id)
    if not selected:
        raise RuntimeError("comparison montage selection is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"could not write comparison montage {path}")
    return selected


def _write_checksum_index(
    root: Path,
    paths: Sequence[Path],
    output: Path,
) -> None:
    entries = []
    for path in sorted(
        paths,
        key=lambda value: value.relative_to(root).as_posix().encode("utf-8"),
    ):
        reference = fingerprint_file(path)
        entries.append(f"{reference['sha256']}  {path.relative_to(root).as_posix()}")
    _atomic_write_text(output, "\n".join(entries) + "\n")


def _markdown_report(
    *,
    config: ReplayConfig,
    dataset: VerifiedDataset,
    evaluation: EvaluationConfig,
    aggregate_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    sample_order_digest: str,
) -> str:
    lines = [
        f"# {config.title}",
        "",
        f"- Replay ID: `{config.replay_id}`",
        f"- Purpose: `{config.purpose}`",
        f"- Dataset: `{dataset.dataset_id}`",
        f"- Evaluation: `{evaluation.evaluation_id}`",
        f"- RGB sample-order SHA-256: `{sample_order_digest}`",
        "- Runtime sensor contract: `front_monocular_rgb_only`",
        "",
        "## Aggregate results",
        "",
        "| Model | COCO AP | Recall | F1 | FP | FN | Median latency (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| "
            f"{row['label']} | {float(row['coco_ap_50_95']):.4f} | "
            f"{float(row['operating_recall']):.4f} | "
            f"{float(row['operating_f1']):.4f} | "
            f"{int(row['operating_fp'])} | {int(row['operating_fn'])} | "
            f"{float(row['latency_median_ms']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Paired differences",
            "",
            "Deltas are candidate minus reference. Confidence intervals use "
            "episode-cluster bootstrap on the exact same ordered RGB samples.",
            "",
            "| Candidate | Metric | Delta | CI lower | CI upper | Direction |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in paired_rows:
        lines.append(
            "| "
            f"{row['candidate_model_id']} | {row['metric']} | "
            f"{float(row['delta']):.6f} | {float(row['ci_lower']):.6f} | "
            f"{float(row['ci_upper']):.6f} | {row['favorable_direction']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- This replay measures 2D detection, not driving safety or policy quality.",
            "- The execution order is the preregistered model-list order.",
            "- Latency includes inference only and is specific to the recorded client hardware.",
            "- Single-episode bootstrap intervals are degenerate and are descriptive only.",
            "- Development results are not confirmatory thesis claims.",
            "",
        ]
    )
    return "\n".join(lines)


def run_replay(
    *,
    config_path: str | Path,
    dataset_path: str | Path,
    evaluation_config_path: str | Path,
    runs_root: str | Path = "runs",
    acknowledge_locked_test: bool = False,
    cli_args: Sequence[str] | Mapping[str, Any] = (),
    repository_root: str | Path | None = None,
    detector_overrides: Mapping[str, Detector] | None = None,
) -> dict[str, Any]:
    """Evaluate every model on an identical sample order and compare pairs."""

    config_path_resolved = Path(config_path).expanduser().resolve(strict=True)
    evaluation_path_resolved = Path(evaluation_config_path).expanduser().resolve(strict=True)
    config = load_replay_config(config_path_resolved)
    evaluation = load_evaluation_config(evaluation_path_resolved)
    dataset = load_verified_dataset(dataset_path)
    if config.purpose != evaluation.purpose:
        raise ValueError("replay and evaluation purposes must match")
    if evaluation.purpose == "confirmatory" and "unassigned" in evaluation.partitions:
        raise RuntimeError("confirmatory replay cannot use unassigned data")
    locked_partitions = sorted(
        partition for partition in evaluation.partitions if partition.startswith("test")
    )
    if locked_partitions and not acknowledge_locked_test:
        raise RuntimeError(
            "locked test replay requires --acknowledge-locked-test: " + ", ".join(locked_partitions)
        )
    samples = _selected_samples(dataset, evaluation)
    if config.latency_exclude_first_images >= len(samples):
        raise ValueError("latency_exclude_first_images must be smaller than selected sample count")
    expected_order = _sample_order_rows(samples)
    expected_order_digest = _order_digest(expected_order)
    resolved_models = _resolve_models(config)
    model_ids = {model.spec.model_id for model in resolved_models}
    overrides = dict(detector_overrides or {})
    unknown_overrides = sorted(set(overrides) - model_ids)
    if unknown_overrides:
        raise ValueError(
            "detector overrides reference unknown models: " + ", ".join(unknown_overrides)
        )
    runs_root_resolved = Path(runs_root).expanduser().resolve()
    prospective_ids = [
        config.replay_id,
        *(_safe_child_run_id(config.replay_id, model.spec.model_id) for model in resolved_models),
    ]
    collisions = [run_id for run_id in prospective_ids if (runs_root_resolved / run_id).exists()]
    if collisions:
        raise FileExistsError("replay output IDs already exist: " + ", ".join(collisions))

    resolved_document = {
        "schema_version": REPLAY_RUN_SCHEMA_VERSION,
        "object_type": "paired_detector_replay",
        "replay": config.as_dict(),
        "source_config": fingerprint_file(config_path_resolved),
        "evaluation": {
            "config": evaluation.as_dict(),
            "source": fingerprint_file(evaluation_path_resolved),
        },
        "dataset": dict(dataset.reference),
        "selected_sample_count": len(samples),
        "sample_order_sha256": expected_order_digest,
        "models": [dict(model.reference) for model in resolved_models],
        "runtime_sensor_contract": "front_monocular_rgb_only",
        "privileged_metadata_use": "ground_truth_scoring_and_stratification_only",
    }
    tracker = RunArtifactTracker(
        runs_root_resolved,
        run_id=config.replay_id,
        cli_args=cli_args,
        config=resolved_document,
        repository_root=repository_root,
        model_refs=[model.reference for model in resolved_models],
        dataset_refs=[dataset.reference],
        input_refs=[
            {
                "kind": "replay_preregistration",
                **fingerprint_file(config_path_resolved),
            },
            {
                "kind": "evaluation_preregistration",
                **fingerprint_file(evaluation_path_resolved),
            },
        ],
    )
    summary: dict[str, Any]
    with tracker:
        resolved_path = tracker.artifact_path("resolved_replay_config.json")
        sample_order_path = tracker.artifact_path("tables/sample_order.csv")
        child_runs_path = tracker.artifact_path("child_runs.json")
        aggregate_path = tracker.artifact_path("tables/aggregate_metrics.csv")
        per_image_path = tracker.artifact_path("tables/per_image_metrics.csv")
        paired_path = tracker.artifact_path("tables/paired_differences.csv")
        disagreement_path = tracker.artifact_path("tables/disagreements.csv")
        bootstrap_path = tracker.artifact_path("paired_bootstrap.json")
        descriptor_path = tracker.artifact_path("replay.json")
        report_path = tracker.artifact_path("replay.md")
        montage_path = tracker.artifact_path("qualitative/paired_disagreements.png")
        checksum_path = tracker.artifact_path("checksums.sha256")
        _write_json(resolved_path, resolved_document)
        _write_csv(
            sample_order_path,
            expected_order,
            fields=(
                "position",
                "sample_id",
                "image_id",
                "partition",
                "scenario_id",
                "episode_id",
                "carla_frame",
                "rgb_sha256",
            ),
        )

        child_runs: list[dict[str, Any]] = []
        child_payloads: dict[str, dict[str, Any]] = {}
        samples_by_id = {int(sample["image_id"]): sample for sample in samples}
        for model in resolved_models:
            model_id = model.spec.model_id
            child_run_id = _safe_child_run_id(config.replay_id, model_id)
            child_result = run_evaluation(
                config_path=evaluation_path_resolved,
                dataset_path=dataset.root,
                detector_config=model.detector_config,
                runs_root=runs_root_resolved,
                run_id=child_run_id,
                acknowledge_locked_test=acknowledge_locked_test,
                cli_args={
                    "parent_replay_id": config.replay_id,
                    "model_id": model_id,
                    "evaluation_config": str(evaluation_path_resolved),
                    "dataset": str(dataset.root),
                },
                repository_root=repository_root,
                detector_override=overrides.get(model_id),
                model_reference_override=model.reference,
            )
            child_root = Path(child_result["run_dir"]).resolve(strict=True)
            verification = verify_research_object(
                child_root,
                verify_references=True,
                deep=True,
                reject_unregistered=True,
                require_clean_git=config.purpose == "confirmatory",
            )
            manifest_path = child_root / "manifest.json"
            child_manifest = _load_json(
                manifest_path,
                f"{model_id} child manifest",
            )
            if not isinstance(child_manifest, Mapping):
                raise RuntimeError("child evaluation manifest must be an object")
            frame_path = _artifact_by_role(
                child_root,
                child_manifest,
                "per_image_prediction_log",
            )
            predictions_path = _artifact_by_role(
                child_root,
                child_manifest,
                "canonical_coco_predictions",
            )
            summary_path = _artifact_by_role(
                child_root,
                child_manifest,
                "evaluation_summary",
            )
            frame_rows = _load_jsonl(
                frame_path,
                f"{model_id} frame log",
            )
            child_order = _child_order_rows(frame_rows, samples_by_id)
            child_order_digest = _order_digest(child_order)
            if child_order != expected_order:
                raise RuntimeError(
                    f"model {model_id} did not evaluate the exact preregistered sample order"
                )
            predictions = _load_json(
                predictions_path,
                f"{model_id} predictions",
            )
            evaluation_summary = _load_json(
                summary_path,
                f"{model_id} evaluation summary",
            )
            if not isinstance(predictions, list) or not isinstance(
                evaluation_summary,
                Mapping,
            ):
                raise RuntimeError(f"model {model_id} child artifacts are invalid")
            child_reference = {
                "kind": "paired_child_evaluation",
                "model_id": model_id,
                "run_id": child_run_id,
                "root": str(child_root),
                "manifest": fingerprint_file(manifest_path),
                "sample_order_sha256": child_order_digest,
            }
            tracker.add_input_reference(child_reference)
            child_runs.append(
                {
                    **child_reference,
                    "artifact_count": verification.artifact_count,
                }
            )
            child_payloads[model_id] = {
                "root": child_root,
                "manifest": child_manifest,
                "frames": frame_rows,
                "predictions": predictions,
                "summary": evaluation_summary,
            }

        categories = {
            int(category["id"]): str(category["name"]) for category in dataset.coco["categories"]
        }
        image_ids = [int(sample["image_id"]) for sample in samples]
        selected_image_ids = set(image_ids)
        ground_truth = [
            annotation
            for annotation in dataset.coco["annotations"]
            if int(annotation["image_id"]) in selected_image_ids
        ]
        ground_truth_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for annotation in ground_truth:
            ground_truth_by_image[int(annotation["image_id"])].append(annotation)

        model_rows: dict[str, list[dict[str, Any]]] = {}
        aggregate_rows: list[dict[str, Any]] = []
        predictions_by_model_image: dict[
            str,
            dict[int, list[Mapping[str, Any]]],
        ] = {}
        labels = {model.spec.model_id: model.spec.label for model in resolved_models}
        for model in resolved_models:
            model_id = model.spec.model_id
            payload = child_payloads[model_id]
            predictions = payload["predictions"]
            frames = payload["frames"]
            child_summary = payload["summary"]
            operating = operating_metrics(
                ground_truth,
                predictions,
                image_ids=image_ids,
                categories=categories,
                confidence_threshold=evaluation.operating_confidence,
                iou_threshold=evaluation.matching_iou,
            )
            frames_by_image = {int(frame["image_id"]): frame for frame in frames}
            rows: list[dict[str, Any]] = []
            by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
            for prediction in predictions:
                by_image[int(prediction["image_id"])].append(prediction)
            predictions_by_model_image[model_id] = dict(by_image)
            for order_row in expected_order:
                image_id = int(order_row["image_id"])
                counts = operating.image_counts[image_id]
                ratios = _ratios(
                    int(counts["tp"]),
                    int(counts["fp"]),
                    int(counts["fn"]),
                )
                frame = frames_by_image[image_id]
                operating_prediction_count = sum(
                    float(prediction["score"]) >= evaluation.operating_confidence
                    for prediction in by_image.get(image_id, ())
                )
                rows.append(
                    {
                        "model_id": model_id,
                        "label": model.spec.label,
                        **order_row,
                        "episode_key": (f"{order_row['scenario_id']}/{order_row['episode_id']}"),
                        "tp": int(counts["tp"]),
                        "fp": int(counts["fp"]),
                        "fn": int(counts["fn"]),
                        **ratios,
                        "error_count": int(counts["fp"]) + int(counts["fn"]),
                        "inference_ms": float(frame["inference_seconds"]) * 1000.0,
                        "source_detection_count": int(frame["source_detection_count"]),
                        "mapped_prediction_count": int(frame["mapped_prediction_count"]),
                        "operating_prediction_count": operating_prediction_count,
                    }
                )
            model_rows[model_id] = rows
            eligible_latency = [
                float(row["inference_ms"]) for row in rows[config.latency_exclude_first_images :]
            ]
            coco = child_summary["coco"]
            overall = operating.overall
            aggregate_rows.append(
                {
                    "model_id": model_id,
                    "label": model.spec.label,
                    "backend": str(model.detector_config.backend),
                    "device": model.spec.device,
                    "image_size": model.detector_config.image_size,
                    "images": len(rows),
                    "ground_truth_annotations": len(ground_truth),
                    "coco_ap_50_95": float(coco["ap_50_95"]),
                    "coco_ap_50": float(coco["ap_50"]),
                    "coco_ap_75": float(coco["ap_75"]),
                    "coco_ar_100": float(coco["ar_100"]),
                    "operating_tp": int(overall["tp"]),
                    "operating_fp": int(overall["fp"]),
                    "operating_fn": int(overall["fn"]),
                    "operating_precision": float(overall["precision"]),
                    "operating_recall": float(overall["recall"]),
                    "operating_f1": float(overall["f1"]),
                    "calibration_ece": child_summary["calibration"]["expected_calibration_error"],
                    "latency_count": len(eligible_latency),
                    "latency_mean_ms": float(np.mean(eligible_latency)),
                    "latency_median_ms": float(np.median(eligible_latency)),
                    "latency_p95_ms": float(np.percentile(eligible_latency, 95)),
                    "cold_start_ms": float(rows[0]["inference_ms"]),
                    "child_run_id": str(child_runs[len(aggregate_rows)]["run_id"]),
                }
            )

        reference_rows = model_rows[config.reference_model_id]
        paired_bootstrap: dict[str, Any] = {
            "schema_version": REPLAY_RUN_SCHEMA_VERSION,
            "reference_model_id": config.reference_model_id,
            "pairs": {},
        }
        paired_rows: list[dict[str, Any]] = []
        disagreements: list[dict[str, Any]] = []
        reference_by_image = {int(row["image_id"]): row for row in reference_rows}
        for model in resolved_models:
            candidate_id = model.spec.model_id
            if candidate_id == config.reference_model_id:
                continue
            candidate_rows = model_rows[candidate_id]
            seed = derive_seed(
                config.master_seed,
                (
                    f"replay/{config.replay_id}/"
                    f"{config.reference_model_id}-vs-{candidate_id}/bootstrap"
                ),
            )
            bootstrap = _paired_bootstrap(
                reference_rows=reference_rows,
                candidate_rows=candidate_rows,
                replicates=config.bootstrap_replicates,
                confidence=config.bootstrap_confidence,
                seed=seed,
                latency_exclude_first_images=config.latency_exclude_first_images,
            )
            paired_bootstrap["pairs"][candidate_id] = bootstrap
            for metric, values in bootstrap["metrics"].items():
                interval = values["bootstrap_delta"]
                paired_rows.append(
                    {
                        "reference_model_id": config.reference_model_id,
                        "candidate_model_id": candidate_id,
                        "metric": metric,
                        "reference_value": values["reference"],
                        "candidate_value": values["candidate"],
                        "delta": values["delta_candidate_minus_reference"],
                        "favorable_direction": values["favorable_direction"],
                        "bootstrap_mean_delta": interval["mean"],
                        "ci_lower": interval["lower"],
                        "ci_upper": interval["upper"],
                        "confidence": config.bootstrap_confidence,
                        "replicates": config.bootstrap_replicates,
                        "episode_count": bootstrap["episode_count"],
                        "degenerate_single_episode": bootstrap["degenerate_single_episode"],
                        "seed": seed,
                    }
                )
            for candidate_row in candidate_rows:
                image_id = int(candidate_row["image_id"])
                reference_row = reference_by_image[image_id]
                count_disagreement = any(
                    int(candidate_row[field]) != int(reference_row[field])
                    for field in ("tp", "fp", "fn")
                )
                prediction_count_disagreement = int(
                    candidate_row["operating_prediction_count"]
                ) != int(reference_row["operating_prediction_count"])
                disagreements.append(
                    {
                        "reference_model_id": config.reference_model_id,
                        "candidate_model_id": candidate_id,
                        "position": int(candidate_row["position"]),
                        "sample_id": str(candidate_row["sample_id"]),
                        "image_id": image_id,
                        "episode_key": str(candidate_row["episode_key"]),
                        "reference_tp": int(reference_row["tp"]),
                        "reference_fp": int(reference_row["fp"]),
                        "reference_fn": int(reference_row["fn"]),
                        "candidate_tp": int(candidate_row["tp"]),
                        "candidate_fp": int(candidate_row["fp"]),
                        "candidate_fn": int(candidate_row["fn"]),
                        "delta_tp": int(candidate_row["tp"]) - int(reference_row["tp"]),
                        "delta_fp": int(candidate_row["fp"]) - int(reference_row["fp"]),
                        "delta_fn": int(candidate_row["fn"]) - int(reference_row["fn"]),
                        "delta_error_count": int(candidate_row["error_count"])
                        - int(reference_row["error_count"]),
                        "delta_inference_ms": float(candidate_row["inference_ms"])
                        - float(reference_row["inference_ms"]),
                        "outcome_disagreement": count_disagreement,
                        "prediction_count_disagreement": (prediction_count_disagreement),
                    }
                )

        per_image_rows = [
            row for model in resolved_models for row in model_rows[model.spec.model_id]
        ]
        _write_json(child_runs_path, child_runs)
        _write_csv(
            aggregate_path,
            aggregate_rows,
            fields=(
                "model_id",
                "label",
                "backend",
                "device",
                "image_size",
                "images",
                "ground_truth_annotations",
                "coco_ap_50_95",
                "coco_ap_50",
                "coco_ap_75",
                "coco_ar_100",
                "operating_tp",
                "operating_fp",
                "operating_fn",
                "operating_precision",
                "operating_recall",
                "operating_f1",
                "calibration_ece",
                "latency_count",
                "latency_mean_ms",
                "latency_median_ms",
                "latency_p95_ms",
                "cold_start_ms",
                "child_run_id",
            ),
        )
        _write_csv(
            per_image_path,
            per_image_rows,
            fields=(
                "model_id",
                "label",
                "position",
                "sample_id",
                "image_id",
                "partition",
                "scenario_id",
                "episode_id",
                "carla_frame",
                "rgb_sha256",
                "episode_key",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "error_count",
                "inference_ms",
                "source_detection_count",
                "mapped_prediction_count",
                "operating_prediction_count",
            ),
        )
        _write_csv(
            paired_path,
            paired_rows,
            fields=(
                "reference_model_id",
                "candidate_model_id",
                "metric",
                "reference_value",
                "candidate_value",
                "delta",
                "favorable_direction",
                "bootstrap_mean_delta",
                "ci_lower",
                "ci_upper",
                "confidence",
                "replicates",
                "episode_count",
                "degenerate_single_episode",
                "seed",
            ),
        )
        _write_csv(
            disagreement_path,
            disagreements,
            fields=(
                "reference_model_id",
                "candidate_model_id",
                "position",
                "sample_id",
                "image_id",
                "episode_key",
                "reference_tp",
                "reference_fp",
                "reference_fn",
                "candidate_tp",
                "candidate_fp",
                "candidate_fn",
                "delta_tp",
                "delta_fp",
                "delta_fn",
                "delta_error_count",
                "delta_inference_ms",
                "outcome_disagreement",
                "prediction_count_disagreement",
            ),
        )
        _write_json(bootstrap_path, paired_bootstrap)

        plot_paths: list[Path] = []
        plot_paths.extend(
            _plot_accuracy(
                tracker.artifact_path(
                    "plots/accuracy_comparison",
                    create_parent=True,
                ),
                aggregate_rows,
            )
        )
        plot_paths.extend(
            _plot_latency(
                tracker.artifact_path(
                    "plots/latency_comparison",
                    create_parent=True,
                ),
                model_rows,
                labels,
                exclude_first=config.latency_exclude_first_images,
            )
        )
        plot_paths.extend(
            _plot_error_counts(
                tracker.artifact_path(
                    "plots/operating_outcomes",
                    create_parent=True,
                ),
                aggregate_rows,
            )
        )
        plot_paths.extend(
            _plot_disagreement_heatmap(
                tracker.artifact_path(
                    "plots/error_heatmap",
                    create_parent=True,
                ),
                model_rows,
                labels,
            )
        )
        montage_ids = _comparison_montage(
            montage_path,
            dataset=dataset,
            samples=samples,
            model_rows=model_rows,
            predictions_by_model_image=predictions_by_model_image,
            labels=labels,
            categories=categories,
            ground_truth_by_image=ground_truth_by_image,
            operating_confidence=evaluation.operating_confidence,
            count=config.montage_count,
        )
        summary = {
            "schema_version": REPLAY_RUN_SCHEMA_VERSION,
            "object_type": "paired_detector_replay_release",
            "status": "complete",
            "replay_id": config.replay_id,
            "title": config.title,
            "purpose": config.purpose,
            "dataset": dict(dataset.reference),
            "evaluation_id": evaluation.evaluation_id,
            "reference_model_id": config.reference_model_id,
            "model_count": len(resolved_models),
            "models": [
                {
                    "model_id": model.spec.model_id,
                    "label": model.spec.label,
                    "reference": dict(model.reference),
                }
                for model in resolved_models
            ],
            "child_runs": child_runs,
            "sample_count": len(samples),
            "sample_order_sha256": expected_order_digest,
            "aggregate_metrics": aggregate_rows,
            "paired_comparison_count": len(paired_rows),
            "disagreement_row_count": len(disagreements),
            "outcome_disagreement_count": sum(
                bool(row["outcome_disagreement"]) for row in disagreements
            ),
            "qualitative_image_ids": montage_ids,
            "latency": {
                "excluded_first_images": config.latency_exclude_first_images,
                "statistics_by_model": {
                    model_id: _describe(
                        [
                            float(row["inference_ms"])
                            for row in rows[config.latency_exclude_first_images :]
                        ]
                    )
                    for model_id, rows in model_rows.items()
                },
            },
            "bootstrap": {
                "unit": "episode",
                "replicates": config.bootstrap_replicates,
                "confidence": config.bootstrap_confidence,
                "master_seed": config.master_seed,
                "pair_seeds": {
                    candidate_id: values["seed"]
                    for candidate_id, values in paired_bootstrap["pairs"].items()
                },
            },
            "runtime_sensor_contract": "front_monocular_rgb_only",
            "privileged_metadata_use": (
                "ground_truth scoring, pairing, and post-hoc grouping only"
            ),
            "limitations": [
                "2D detection metrics do not establish driving safety.",
                "Model execution is sequential in preregistered list order.",
                "Latency is client-hardware-specific and excludes model loading.",
                "Single-episode bootstrap intervals are descriptive and degenerate.",
                "Development replay results are not confirmatory thesis claims.",
            ],
        }
        _write_json(descriptor_path, summary)
        _atomic_write_text(
            report_path,
            _markdown_report(
                config=config,
                dataset=dataset,
                evaluation=evaluation,
                aggregate_rows=aggregate_rows,
                paired_rows=paired_rows,
                sample_order_digest=expected_order_digest,
            ),
        )

        payload_paths = [
            resolved_path,
            sample_order_path,
            child_runs_path,
            aggregate_path,
            per_image_path,
            paired_path,
            disagreement_path,
            bootstrap_path,
            descriptor_path,
            report_path,
            montage_path,
            *plot_paths,
        ]
        _write_checksum_index(
            tracker.run_dir,
            payload_paths,
            checksum_path,
        )
        artifacts = {
            resolved_path: "resolved_replay_configuration",
            sample_order_path: "replay_sample_order",
            child_runs_path: "replay_child_evaluations",
            aggregate_path: "replay_aggregate_metrics",
            per_image_path: "replay_per_image_metrics",
            paired_path: "replay_paired_differences",
            disagreement_path: "replay_disagreements",
            bootstrap_path: "replay_paired_bootstrap",
            descriptor_path: "replay_release_manifest",
            report_path: "replay_report",
            montage_path: "replay_qualitative_panel",
            checksum_path: "replay_checksum_index",
            **{path: "replay_plot" for path in plot_paths},
        }
        for path, role in artifacts.items():
            tracker.register_artifact(
                path,
                role=role,
                metadata={
                    "replay_id": config.replay_id,
                    "model_count": len(resolved_models),
                    "sample_count": len(samples),
                },
            )
    return {
        "replay_id": tracker.run_id,
        "run_dir": str(tracker.run_dir),
        "manifest": str(tracker.manifest_path),
        "summary": summary,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run RT-DETR, YOLO, custom, or packaged detectors on the exact same "
            "verified RGB sample order and build paired comparisons"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--evaluation-config", required=True)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--acknowledge-locked-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_replay(
        config_path=args.config,
        dataset_path=args.dataset,
        evaluation_config_path=args.evaluation_config,
        runs_root=args.runs_root,
        acknowledge_locked_test=args.acknowledge_locked_test,
        cli_args=vars(args),
        repository_root=Path.cwd(),
    )
    output = {
        "replay_id": result["replay_id"],
        "run_dir": result["run_dir"],
        "manifest": result["manifest"],
        "model_count": result["summary"]["model_count"],
        "sample_count": result["summary"]["sample_count"],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    return 0


__all__ = [
    "REPLAY_RUN_SCHEMA_VERSION",
    "main",
    "parse_args",
    "run_replay",
]
