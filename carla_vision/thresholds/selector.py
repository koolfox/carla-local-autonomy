"""Select and seal one detector operating threshold using validation only."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402

from ..artifacts import RunArtifactTracker, fingerprint_file
from ..evaluation.metrics import operating_metrics
from ..evaluation.verified import VerifiedEvaluation, load_verified_evaluation
from ..scenarios.seeds import derive_seed
from ..verification import verify_research_object
from .contracts import ThresholdSelectionConfig, load_threshold_selection_config

THRESHOLD_SELECTION_RUN_SCHEMA_VERSION = "1.0"


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


def _thresholds(config: ThresholdSelectionConfig) -> tuple[float, ...]:
    grid = config.grid
    if grid.scale == "linear":
        values = np.linspace(grid.minimum, grid.maximum, grid.steps)
    else:
        values = np.geomspace(grid.minimum, grid.maximum, grid.steps)
    canonical = [float(format(float(value), ".15g")) for value in values]
    canonical[0] = grid.minimum
    canonical[-1] = grid.maximum
    return tuple(dict.fromkeys(canonical))


def _ratios(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_negative_rate": 1.0 - recall,
    }


def _metric_row(
    config: ThresholdSelectionConfig,
    *,
    threshold: float,
    tp: int,
    fp: int,
    fn: int,
) -> dict[str, Any]:
    ratios = _ratios(tp, fp, fn)
    feasible = True
    if config.objective == "minimum_recall":
        feasible = ratios["recall"] >= float(config.target_recall)
    elif config.objective == "minimum_precision":
        feasible = ratios["precision"] >= float(config.target_precision)
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        **ratios,
        "weighted_error": (config.false_positive_cost * fp + config.false_negative_cost * fn),
        "feasible": feasible,
    }


def _selection_key(
    config: ThresholdSelectionConfig,
    row: Mapping[str, Any],
) -> tuple[float, ...]:
    tie = float(row["threshold"])
    if config.tie_breaker == "lowest_threshold":
        tie = -tie
    if config.objective == "maximize_f1":
        return (
            float(row["f1"]),
            float(row["recall"]),
            float(row["precision"]),
            tie,
        )
    if config.objective == "minimum_recall":
        return (
            -float(row["fp"]),
            float(row["f1"]),
            float(row["precision"]),
            tie,
        )
    if config.objective == "minimum_precision":
        return (
            float(row["recall"]),
            float(row["f1"]),
            -float(row["fn"]),
            tie,
        )
    return (
        -float(row["weighted_error"]),
        float(row["f1"]),
        float(row["recall"]),
        tie,
    )


def _select(
    config: ThresholdSelectionConfig,
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    feasible = [row for row in rows if bool(row["feasible"])]
    if not feasible:
        return None
    return max(feasible, key=lambda row: _selection_key(config, row))


def _validation_samples(
    evaluation: VerifiedEvaluation,
) -> list[Mapping[str, Any]]:
    partitions = tuple(evaluation.config.partitions)
    if not partitions or any(not partition.startswith("val") for partition in partitions):
        raise RuntimeError(
            "threshold selection requires validation-only partitions; got " + ", ".join(partitions)
        )
    return [
        sample
        for sample in evaluation.dataset.dataset["samples"]
        if str(sample["split"]) in partitions
    ]


def _sweep(
    config: ThresholdSelectionConfig,
    evaluation: VerifiedEvaluation,
    samples: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[Mapping[int, Mapping[str, int]]]]:
    image_ids = [int(sample["image_id"]) for sample in samples]
    selected = set(image_ids)
    ground_truth = [
        annotation
        for annotation in evaluation.dataset.coco["annotations"]
        if int(annotation["image_id"]) in selected
    ]
    categories = {
        int(category["id"]): str(category["name"])
        for category in evaluation.dataset.coco["categories"]
    }
    rows: list[dict[str, Any]] = []
    image_counts: list[Mapping[int, Mapping[str, int]]] = []
    for threshold in _thresholds(config):
        operating = operating_metrics(
            ground_truth,
            evaluation.predictions,
            image_ids=image_ids,
            categories=categories,
            confidence_threshold=threshold,
            iou_threshold=evaluation.config.matching_iou,
        )
        rows.append(
            _metric_row(
                config,
                threshold=threshold,
                tp=int(operating.overall["tp"]),
                fp=int(operating.overall["fp"]),
                fn=int(operating.overall["fn"]),
            )
        )
        image_counts.append(operating.image_counts)
    return rows, image_counts


def _interval(
    values: Sequence[float],
    *,
    confidence: float,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "lower": float(np.quantile(array, alpha)),
        "upper": float(np.quantile(array, 1.0 - alpha)),
    }


def _bootstrap(
    config: ThresholdSelectionConfig,
    *,
    sweep_rows: Sequence[Mapping[str, Any]],
    image_counts: Sequence[Mapping[int, Mapping[str, int]]],
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    episode_images: dict[str, list[int]] = defaultdict(list)
    for sample in samples:
        episode_key = f"{sample['scenario_id']}/{sample['episode_id']}"
        episode_images[episode_key].append(int(sample["image_id"]))
    episode_ids = tuple(sorted(episode_images))
    if not episode_ids:
        raise RuntimeError("threshold bootstrap requires at least one episode")
    seed = derive_seed(
        config.master_seed,
        f"threshold-selection/{config.selection_id}/bootstrap",
    )
    rng = np.random.default_rng(seed)
    selected_thresholds: list[float] = []
    selected_metrics: dict[str, list[float]] = {
        "precision": [],
        "recall": [],
        "f1": [],
        "weighted_error": [],
    }
    infeasible = 0
    for _ in range(config.bootstrap_replicates):
        sampled = tuple(
            str(value)
            for value in rng.choice(
                episode_ids,
                size=len(episode_ids),
                replace=True,
            )
        )
        replicate_rows: list[dict[str, Any]] = []
        for sweep_row, counts_by_image in zip(
            sweep_rows,
            image_counts,
            strict=True,
        ):
            counts = Counter(tp=0, fp=0, fn=0)
            for episode_id in sampled:
                for image_id in episode_images[episode_id]:
                    counts.update(counts_by_image[image_id])
            replicate_rows.append(
                _metric_row(
                    config,
                    threshold=float(sweep_row["threshold"]),
                    tp=counts["tp"],
                    fp=counts["fp"],
                    fn=counts["fn"],
                )
            )
        selected = _select(config, replicate_rows)
        if selected is None:
            infeasible += 1
            continue
        selected_thresholds.append(float(selected["threshold"]))
        for metric in selected_metrics:
            selected_metrics[metric].append(float(selected[metric]))
    if not selected_thresholds:
        raise RuntimeError("threshold objective was infeasible in every bootstrap replicate")
    return {
        "unit": "episode",
        "episode_count": len(episode_ids),
        "replicates": config.bootstrap_replicates,
        "feasible_replicates": len(selected_thresholds),
        "infeasible_replicates": infeasible,
        "feasible_fraction": len(selected_thresholds) / config.bootstrap_replicates,
        "confidence": config.bootstrap_confidence,
        "seed": seed,
        "degenerate_single_episode": len(episode_ids) == 1,
        "selected_threshold": _interval(
            selected_thresholds,
            confidence=config.bootstrap_confidence,
        ),
        "selected_metrics": {
            metric: _interval(
                values,
                confidence=config.bootstrap_confidence,
            )
            for metric, values in selected_metrics.items()
        },
    }


def _save_figure(figure: Any, base_path: Path) -> tuple[Path, Path]:
    png = base_path.with_suffix(".png")
    svg = base_path.with_suffix(".svg")
    figure.tight_layout()
    figure.savefig(png, dpi=180)
    figure.savefig(svg)
    plt.close(figure)
    return png, svg


def _plot_metrics(
    base_path: Path,
    rows: Sequence[Mapping[str, Any]],
    selected_threshold: float,
) -> tuple[Path, Path]:
    thresholds = [float(row["threshold"]) for row in rows]
    figure, axis = plt.subplots(figsize=(9, 6))
    for metric, color in (
        ("precision", "#3a86ff"),
        ("recall", "#2a9d8f"),
        ("f1", "#e76f51"),
    ):
        axis.plot(
            thresholds,
            [float(row[metric]) for row in rows],
            label=metric,
            color=color,
        )
    axis.axvline(
        selected_threshold,
        color="#222222",
        linestyle="--",
        label=f"selected={selected_threshold:.4f}",
    )
    axis.set_xlabel("Confidence threshold")
    axis.set_ylabel("Metric")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Validation operating-threshold sweep")
    axis.grid(alpha=0.25)
    axis.legend()
    return _save_figure(figure, base_path)


def _plot_errors(
    base_path: Path,
    rows: Sequence[Mapping[str, Any]],
    selected_threshold: float,
) -> tuple[Path, Path]:
    thresholds = [float(row["threshold"]) for row in rows]
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(
        thresholds,
        [int(row["fp"]) for row in rows],
        label="false positives",
        color="#e9c46a",
    )
    axis.plot(
        thresholds,
        [int(row["fn"]) for row in rows],
        label="false negatives",
        color="#e76f51",
    )
    axis.axvline(selected_threshold, color="#222222", linestyle="--")
    axis.set_xlabel("Confidence threshold")
    axis.set_ylabel("Count")
    axis.set_title("Validation errors across threshold")
    axis.grid(alpha=0.25)
    axis.legend()
    return _save_figure(figure, base_path)


def _plot_bootstrap(
    base_path: Path,
    bootstrap: Mapping[str, Any],
    selected_threshold: float,
) -> tuple[Path, Path]:
    interval = bootstrap["selected_threshold"]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(
        [0],
        [float(interval["median"])],
        yerr=[
            [float(interval["median"]) - float(interval["lower"])],
            [float(interval["upper"]) - float(interval["median"])],
        ],
        fmt="o",
        capsize=6,
        color="#3a86ff",
        label="episode bootstrap interval",
    )
    axis.scatter(
        [0.12],
        [selected_threshold],
        marker="x",
        color="#e76f51",
        s=90,
        label="full-validation selection",
    )
    axis.set_xlim(-0.4, 0.5)
    axis.set_xticks([])
    axis.set_ylabel("Selected confidence threshold")
    axis.set_title("Threshold-selection stability")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    return _save_figure(figure, base_path)


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


def _report(
    config: ThresholdSelectionConfig,
    selected: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    evaluation: VerifiedEvaluation,
) -> str:
    interval = bootstrap["selected_threshold"]
    return f"""# {config.title}

- Selection ID: `{config.selection_id}`
- Source evaluation: `{evaluation.run_id}`
- Validation partitions: {", ".join(evaluation.config.partitions)}
- Objective: `{config.objective}`
- Selected threshold: `{float(selected["threshold"]):.8f}`
- Bootstrap threshold median: `{float(interval["median"]):.8f}`
- Bootstrap {config.bootstrap_confidence:.1%} interval: \
`[{float(interval["lower"]):.8f}, {float(interval["upper"]):.8f}]`
- Episode count: `{bootstrap["episode_count"]}`
- Feasible bootstrap fraction: `{float(bootstrap["feasible_fraction"]):.4f}`

## Selected operating point

| TP | FP | FN | Precision | Recall | F1 | Weighted error |
|---:|---:|---:|---:|---:|---:|---:|
| {selected["tp"]} | {selected["fp"]} | {selected["fn"]} | \
{float(selected["precision"]):.6f} | {float(selected["recall"]):.6f} | \
{float(selected["f1"]):.6f} | {float(selected["weighted_error"]):.6f} |

## Method boundary

This threshold was selected only from the verified validation partitions
listed above. It must be frozen before locked-test inference. A threshold
selected from training, unassigned, or test data is rejected by the selector.
Single-episode bootstrap output is descriptive and marked degenerate.
"""


def run_threshold_selection(
    *,
    config_path: str | Path,
    evaluation_run: str | Path,
    runs_root: str | Path = "runs",
    cli_args: Sequence[str] | Mapping[str, Any] = (),
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    config_path_resolved = Path(config_path).expanduser().resolve(strict=True)
    config = load_threshold_selection_config(config_path_resolved)
    evaluation = load_verified_evaluation(evaluation_run)
    if config.purpose == "confirmatory":
        verify_research_object(
            evaluation.root,
            verify_references=True,
            deep=True,
            reject_unregistered=True,
            require_clean_git=True,
        )
    samples = _validation_samples(evaluation)
    if config.grid.minimum < evaluation.config.minimum_prediction_confidence:
        raise ValueError("threshold grid minimum is below the source prediction-retention floor")
    tracker = RunArtifactTracker(
        runs_root,
        run_id=config.selection_id,
        cli_args=cli_args,
        config={
            "schema_version": THRESHOLD_SELECTION_RUN_SCHEMA_VERSION,
            "object_type": "validation_threshold_selection",
            "selection": config.as_dict(),
            "source_evaluation": dict(evaluation.reference),
        },
        repository_root=repository_root,
        model_refs=[evaluation.resolved["model"]],
        dataset_refs=[evaluation.dataset.reference],
        input_refs=[
            {
                "kind": "threshold_selection_preregistration",
                **fingerprint_file(config_path_resolved),
            },
            evaluation.reference,
        ],
    )
    descriptor: dict[str, Any]
    with tracker:
        resolved_path = tracker.artifact_path("resolved_threshold_selection_config.json")
        sweep_path = tracker.artifact_path("tables/threshold_sweep.csv")
        selected_path = tracker.artifact_path("selected_threshold.json")
        bootstrap_path = tracker.artifact_path("bootstrap.json")
        report_path = tracker.artifact_path("threshold_selection.md")
        descriptor_path = tracker.artifact_path("threshold_selection.json")
        checksum_path = tracker.artifact_path("checksums.sha256")
        resolved = {
            "schema_version": THRESHOLD_SELECTION_RUN_SCHEMA_VERSION,
            "object_type": "validation_threshold_selection",
            "selection": config.as_dict(),
            "source_config": fingerprint_file(config_path_resolved),
            "source_evaluation": dict(evaluation.reference),
            "dataset": dict(evaluation.dataset.reference),
            "model": evaluation.resolved["model"],
            "validation_partitions": list(evaluation.config.partitions),
            "prediction_retention_floor": (evaluation.config.minimum_prediction_confidence),
            "matching_iou": evaluation.config.matching_iou,
            "runtime_sensor_contract": "front_monocular_rgb_only",
        }
        _write_json(resolved_path, resolved)
        sweep_rows, image_counts = _sweep(config, evaluation, samples)
        selected = _select(config, sweep_rows)
        if selected is None:
            raise RuntimeError("no threshold satisfies the preregistered selection objective")
        bootstrap = _bootstrap(
            config,
            sweep_rows=sweep_rows,
            image_counts=image_counts,
            samples=samples,
        )
        _write_csv(
            sweep_path,
            sweep_rows,
            fields=(
                "threshold",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "false_negative_rate",
                "weighted_error",
                "feasible",
            ),
        )
        selected_payload = {
            "schema_version": THRESHOLD_SELECTION_RUN_SCHEMA_VERSION,
            "selection_id": config.selection_id,
            "objective": config.objective,
            "tie_breaker": config.tie_breaker,
            "operating_confidence": float(selected["threshold"]),
            "operating_point": dict(selected),
            "validation_partitions": list(evaluation.config.partitions),
            "source_evaluation": dict(evaluation.reference),
            "model": evaluation.resolved["model"],
            "dataset": dict(evaluation.dataset.reference),
        }
        _write_json(selected_path, selected_payload)
        _write_json(bootstrap_path, bootstrap)
        plot_paths: list[Path] = []
        plot_paths.extend(
            _plot_metrics(
                tracker.artifact_path("plots/metrics", create_parent=True),
                sweep_rows,
                float(selected["threshold"]),
            )
        )
        plot_paths.extend(
            _plot_errors(
                tracker.artifact_path("plots/errors", create_parent=True),
                sweep_rows,
                float(selected["threshold"]),
            )
        )
        plot_paths.extend(
            _plot_bootstrap(
                tracker.artifact_path(
                    "plots/bootstrap_threshold",
                    create_parent=True,
                ),
                bootstrap,
                float(selected["threshold"]),
            )
        )
        _atomic_write_text(
            report_path,
            _report(config, selected, bootstrap, evaluation),
        )
        descriptor = {
            "schema_version": THRESHOLD_SELECTION_RUN_SCHEMA_VERSION,
            "object_type": "validation_threshold_selection_release",
            "status": "complete",
            "selection_id": config.selection_id,
            "title": config.title,
            "purpose": config.purpose,
            "objective": config.objective,
            "source_evaluation": dict(evaluation.reference),
            "dataset": dict(evaluation.dataset.reference),
            "model": evaluation.resolved["model"],
            "validation_partitions": list(evaluation.config.partitions),
            "sample_count": len(samples),
            "episode_count": bootstrap["episode_count"],
            "sweep_count": len(sweep_rows),
            "selected_threshold": float(selected["threshold"]),
            "selected_operating_point": dict(selected),
            "bootstrap": bootstrap,
            "runtime_sensor_contract": "front_monocular_rgb_only",
            "limitations": [
                "The selected threshold is specific to this model and validation release.",
                "Selection uncertainty does not replace independent test evaluation.",
                "A single-episode bootstrap interval is descriptive and degenerate.",
                "Detector confidence calibration can drift after retraining or export.",
            ],
        }
        _write_json(descriptor_path, descriptor)
        payload_paths = [
            resolved_path,
            sweep_path,
            selected_path,
            bootstrap_path,
            report_path,
            descriptor_path,
            *plot_paths,
        ]
        _write_checksum_index(tracker.run_dir, payload_paths, checksum_path)
        artifacts = {
            resolved_path: "resolved_threshold_selection_configuration",
            sweep_path: "threshold_sweep_table",
            selected_path: "selected_operating_threshold",
            bootstrap_path: "threshold_selection_bootstrap",
            report_path: "threshold_selection_report",
            descriptor_path: "threshold_selection_release_manifest",
            checksum_path: "threshold_selection_checksum_index",
            **{path: "threshold_selection_plot" for path in plot_paths},
        }
        for path, role in artifacts.items():
            tracker.register_artifact(
                path,
                role=role,
                metadata={
                    "selection_id": config.selection_id,
                    "source_evaluation": evaluation.run_id,
                },
            )
    return {
        "selection_id": tracker.run_id,
        "run_dir": str(tracker.run_dir),
        "manifest": str(tracker.manifest_path),
        "descriptor": descriptor,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a detector operating threshold from a verified "
            "validation-only evaluation and seal the result"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluation-run", required=True)
    parser.add_argument("--runs-root", default="runs")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_threshold_selection(
        config_path=args.config,
        evaluation_run=args.evaluation_run,
        runs_root=args.runs_root,
        cli_args=vars(args),
        repository_root=Path.cwd(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


__all__ = [
    "THRESHOLD_SELECTION_RUN_SCHEMA_VERSION",
    "main",
    "parse_args",
    "run_threshold_selection",
]
