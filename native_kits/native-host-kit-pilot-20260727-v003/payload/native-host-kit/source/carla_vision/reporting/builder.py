"""Build checksum-indexed reports exclusively from verified source manifests."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402

from ..artifacts import RunArtifactTracker, fingerprint_file
from ..verification import VerificationResult, verify_research_object
from .contracts import ReportConfig, ReportSource, load_report_config

REPORT_RELEASE_SCHEMA_VERSION = "1.0"
_SUMMARY_ROLE = {
    "dataset": "dataset_manifest",
    "dataset_qa": "dataset_qa_summary",
    "evaluation": "evaluation_summary",
    "failure_mining": "failure_mining_release_manifest",
    "failure_review": "failure_review_release_manifest",
    "model": "model_release_manifest",
    "native_host_kit": "native_host_kit_release_manifest",
    "native_preflight": "native_preflight_summary",
    "operator_session": "operator_job_status",
    "paired_replay": "replay_release_manifest",
    "reproduction_bundle": "reproduction_bundle_release_manifest",
    "report": "report_release_manifest",
    "runtime_analysis": "analysis_summary_metrics",
    "scenario_plan": "scenario_plan_summary",
    "shadow_matrix": "shadow_matrix_release_manifest",
    "threshold_selection": "threshold_selection_release_manifest",
    "training": "training_summary",
    "vision_shadow": "run_summary",
}


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_text(path, buffer.getvalue())


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read {name}: {error}") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{name} must contain a JSON object")
    return payload


def _source_path(config_path: Path, raw: str) -> Path:
    value = Path(raw).expanduser()
    if not value.is_absolute():
        value = config_path.parent / value
    return value.resolve(strict=True)


def _artifact_for_role(
    root: Path,
    manifest: Mapping[str, Any],
    role: str,
) -> tuple[Path, Mapping[str, Any]]:
    matches = [
        artifact
        for artifact in manifest["artifacts"]
        if isinstance(artifact, Mapping) and artifact.get("role") == role
    ]
    if len(matches) != 1:
        raise RuntimeError(f"source {root.name!r} must contain exactly one {role!r} artifact")
    artifact = matches[0]
    path = (root / str(artifact["path"])).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"source summary escapes its object root: {path}") from error
    return path, artifact


def _numeric(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _nested(raw: Mapping[str, Any], path: str) -> Any:
    value: Any = raw
    for part in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _add_metric(
    rows: list[dict[str, Any]],
    *,
    source_id: str,
    label: str,
    kind: str,
    metric: str,
    value: Any,
    unit: str = "",
    support: Any = "",
    note: str = "",
) -> None:
    number = _numeric(value)
    if number is None:
        return
    rows.append(
        {
            "source_id": source_id,
            "label": label,
            "kind": kind,
            "metric": metric,
            "value": number,
            "unit": unit,
            "support": support,
            "note": note,
        }
    )


def _extract_metrics(
    source_id: str,
    source: ReportSource,
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(metric: str, path: str, unit: str = "", support: Any = "", note: str = "") -> None:
        _add_metric(
            rows,
            source_id=source_id,
            label=source.label,
            kind=source.kind,
            metric=metric,
            value=_nested(summary, path),
            unit=unit,
            support=support,
            note=note,
        )

    if source.kind == "evaluation":
        image_support = _nested(summary, "selection.images") or ""
        for metric, path in (
            ("coco_ap_50_95", "coco.ap_50_95"),
            ("coco_ap_50", "coco.ap_50"),
            ("coco_ap_75", "coco.ap_75"),
            ("precision", "operating_point.precision"),
            ("recall", "operating_point.recall"),
            ("false_negative_rate", "operating_point.false_negative_rate"),
            ("f1", "operating_point.f1"),
        ):
            add(metric, path, support=image_support)
        add("images", "selection.images", "images")
        add("ground_truth_annotations", "selection.ground_truth_annotations", "objects")
        add("episodes", "bootstrap.episode_count", "episodes")
        add(
            "inference_median",
            "latency_ms.excluding_first_image.median",
            "ms",
        )
        add("inference_p95", "latency_ms.excluding_first_image.p95", "ms")
    elif source.kind == "runtime_analysis":
        for metric, path, unit in (
            ("frames", "metrics.frames.count", "frames"),
            ("effective_source_fps", "metrics.frames.effective_source_fps", "fps"),
            ("detections", "metrics.detections.total", "objects"),
            ("hazard_frame_rate", "metrics.detections.hazard_frame_rate", "fraction"),
            ("pipeline_latency_median", "metrics.timing.pipeline_latency_ms.median", "ms"),
            ("pipeline_latency_p95", "metrics.timing.pipeline_latency_ms.p95", "ms"),
            ("model_latency_median", "metrics.timing.model_inference_ms.median", "ms"),
            ("estimated_distance", "metrics.simulator_speed_mps.estimated_distance_metres", "m"),
        ):
            add(metric, path, unit)
    elif source.kind == "vision_shadow":
        for metric, path, unit in (
            ("elapsed", "elapsed", "s"),
            ("distance_travelled", "distance_travelled", "m"),
            ("max_simulator_speed", "max_simulator_speed", "m/s"),
            ("perception_submitted", "perception_stats.submitted", "frames"),
            ("perception_processed", "perception_stats.processed", "frames"),
            (
                "perception_dropped_before_inference",
                "perception_stats.dropped_before_inference",
                "frames",
            ),
            ("recording_written", "recording_stats.written", "frames"),
            ("policy_proposals", "vision_shadow_stats.proposal_count", "proposals"),
            (
                "policy_throttle_proposals",
                "vision_shadow_stats.throttle_proposal_count",
                "proposals",
            ),
            (
                "policy_braking_proposals",
                "vision_shadow_stats.braking_proposal_count",
                "proposals",
            ),
            (
                "policy_latency_median",
                "vision_shadow_stats.latency_median_ms",
                "ms",
            ),
            ("policy_latency_p95", "vision_shadow_stats.latency_p95_ms", "ms"),
        ):
            add(metric, path, unit)
        actuation_authorized = summary.get("vision_shadow_actuation_authorized")
        if isinstance(actuation_authorized, bool):
            _add_metric(
                rows,
                source_id=source_id,
                label=source.label,
                kind=source.kind,
                metric="vision_policy_actuation_authorized",
                value=int(actuation_authorized),
                unit="boolean",
                note="0=false, 1=true",
            )
    elif source.kind == "shadow_matrix":
        for metric, path, unit in (
            ("planned_runs", "planned_run_count", "runs"),
            ("successful_runs", "successful_run_count", "runs"),
            ("failed_runs", "failed_run_count", "runs"),
        ):
            add(metric, path, unit)
        actuation_authorized = summary.get("vision_policy_actuation_authorized")
        if isinstance(actuation_authorized, bool):
            _add_metric(
                rows,
                source_id=source_id,
                label=source.label,
                kind=source.kind,
                metric="vision_policy_actuation_authorized",
                value=int(actuation_authorized),
                unit="boolean",
                note="0=false, 1=true",
            )
    elif source.kind == "dataset_qa":
        add("samples", "samples", "images")
        add("annotations", "annotations", "objects")
        for name, value in dict(summary.get("class_frequency", {})).items():
            _add_metric(
                rows,
                source_id=source_id,
                label=source.label,
                kind=source.kind,
                metric=f"class_count.{name}",
                value=value,
                unit="objects",
            )
        for name, value in dict(summary.get("checks", {})).items():
            if isinstance(value, bool):
                value = int(value)
            _add_metric(
                rows,
                source_id=source_id,
                label=source.label,
                kind=source.kind,
                metric=f"check.{name}",
                value=value,
            )
    elif source.kind == "dataset":
        add("samples", "sample_count", "images")
        add("annotations", "annotation_count", "objects")
        for name, value in dict(summary.get("split_counts", {})).items():
            _add_metric(
                rows,
                source_id=source_id,
                label=source.label,
                kind=source.kind,
                metric=f"split_count.{name}",
                value=value,
                unit="images",
            )
    elif source.kind == "scenario_plan":
        add("recipes", "recipe_count", "recipes")
        add("episodes", "episode_count", "episodes")
        add("planned_captures", "planned_capture_count", "images")
        for name, value in dict(summary.get("partition_counts", {})).items():
            _add_metric(
                rows,
                source_id=source_id,
                label=source.label,
                kind=source.kind,
                metric=f"partition_count.{name}",
                value=value,
                unit="episodes",
            )
    elif source.kind == "training":
        metrics = summary.get("metrics", {})
        if isinstance(metrics, Mapping):
            for name, value in metrics.items():
                _add_metric(
                    rows,
                    source_id=source_id,
                    label=source.label,
                    kind=source.kind,
                    metric=f"training.{name}",
                    value=value,
                )
    elif source.kind == "model":
        add("selection_value", "selection.value")
        add("weight_bytes", "weights.size_bytes", "bytes")
    elif source.kind == "native_host_kit":
        for metric, path, unit in (
            ("payload_files", "payload_file_count", "files"),
            ("payload_source_files", "payload_source_file_count", "files"),
            ("payload_bytes", "payload_total_bytes", "bytes"),
            ("selected_episodes", "selection.selected_episode_count", "episodes"),
            ("planned_captures", "selection.planned_capture_count", "images"),
            ("requirements_packages", "requirements.package_count", "packages"),
            ("carla_wheel_hashes", "requirements.carla_wheel_hash_count", "hashes"),
        ):
            add(metric, path, unit)
        for metric, path in (
            ("requirements_hash_pinned", "requirements.all_packages_hash_pinned"),
            ("preflight_read_only", "safety.preflight_is_read_only"),
            (
                "ready_preflight_required",
                "safety.collection_requires_verified_ready_preflight",
            ),
            (
                "confirmation_token_required",
                "safety.collection_requires_literal_confirmation_token",
            ),
            ("build_contacted_simulator", "generation.simulator_contacted"),
            ("build_mutated_simulator", "generation.simulator_mutated"),
        ):
            value = _nested(summary, path)
            if isinstance(value, bool):
                _add_metric(
                    rows,
                    source_id=source_id,
                    label=source.label,
                    kind=source.kind,
                    metric=metric,
                    value=int(value),
                    unit="boolean",
                )
    elif source.kind == "native_preflight":
        for metric, path, unit in (
            ("selected_episodes", "selection.selected_episode_count", "episodes"),
            ("planned_captures", "selection.planned_capture_count", "images"),
            ("checks", "checks.count", "checks"),
            ("checks_passed", "checks.passed", "checks"),
            ("checks_failed", "checks.failed", "checks"),
            ("checks_pending", "checks.pending", "checks"),
            ("checks_skipped", "checks.skipped", "checks"),
            ("checks_warnings", "checks.warnings", "checks"),
            ("socket_latency", "endpoint.socket.latency_ms", "ms"),
            (
                "read_only_rpc_latency",
                "endpoint.bridge_read_only_rpc.latency_ms",
                "ms",
            ),
        ):
            add(metric, path, unit)
        for metric, path in (
            ("read_only", "read_only"),
            ("simulator_contacted", "simulator_contacted"),
            ("simulator_mutated", "simulator_mutated"),
            ("automated_ready", "automated_ready"),
            ("manual_ready", "manual_ready"),
            ("ready_for_native_execution", "ready_for_native_execution"),
            ("rpc_reachable", "endpoint.socket.reachable"),
            ("native_pythonapi_importable", "endpoint.native_pythonapi.importable"),
        ):
            value = _nested(summary, path)
            if isinstance(value, bool):
                _add_metric(
                    rows,
                    source_id=source_id,
                    label=source.label,
                    kind=source.kind,
                    metric=metric,
                    value=int(value),
                    unit="boolean",
                    note="0=false, 1=true",
                )
    elif source.kind == "operator_session":
        add("returncode", "returncode")
        for metric, field in (
            ("job_succeeded", "status"),
            ("motion_authorized", "motion_authorized"),
            ("destructive_operation", "destructive"),
            ("stop_requested", "stop_requested"),
        ):
            value = summary.get(field)
            if field == "status":
                value = value == "success"
            if isinstance(value, bool):
                _add_metric(
                    rows,
                    source_id=source_id,
                    label=source.label,
                    kind=source.kind,
                    metric=metric,
                    value=int(value),
                    unit="boolean",
                    note="0=false, 1=true",
                )
        if summary.get("expected_output") is not None:
            expected_output_exists = summary.get("expected_output_exists")
            if isinstance(expected_output_exists, bool):
                _add_metric(
                    rows,
                    source_id=source_id,
                    label=source.label,
                    kind=source.kind,
                    metric="expected_output_exists",
                    value=int(expected_output_exists),
                    unit="boolean",
                    note="0=false, 1=true",
                )
    elif source.kind == "paired_replay":
        for metric, path, unit in (
            ("models", "model_count", "models"),
            ("images", "sample_count", "images"),
            ("outcome_disagreements", "outcome_disagreement_count", "images"),
            ("disagreement_rows", "disagreement_row_count", "rows"),
            ("paired_comparisons", "paired_comparison_count", "comparisons"),
            ("bootstrap_replicates", "bootstrap.replicates", "replicates"),
        ):
            add(metric, path, unit)
        aggregate = summary.get("aggregate_metrics")
        if isinstance(aggregate, Sequence):
            for raw in aggregate:
                if not isinstance(raw, Mapping):
                    continue
                model_id = str(raw.get("model_id", "unknown"))
                for metric in (
                    "coco_ap_50_95",
                    "operating_precision",
                    "operating_recall",
                    "operating_f1",
                    "latency_median_ms",
                    "latency_p95_ms",
                ):
                    _add_metric(
                        rows,
                        source_id=source_id,
                        label=source.label,
                        kind=source.kind,
                        metric=f"{model_id}.{metric}",
                        value=raw.get(metric),
                        unit="ms" if metric.startswith("latency_") else "",
                        support=raw.get("images", ""),
                    )
    elif source.kind == "threshold_selection":
        for metric, path, unit in (
            ("selected_threshold", "selected_threshold", "confidence"),
            ("sweep_count", "sweep_count", "thresholds"),
            ("samples", "sample_count", "images"),
            ("episodes", "episode_count", "episodes"),
            ("selected_precision", "selected_operating_point.precision", ""),
            ("selected_recall", "selected_operating_point.recall", ""),
            ("selected_f1", "selected_operating_point.f1", ""),
        ):
            add(metric, path, unit)
    elif source.kind == "failure_mining":
        for metric, path, unit in (
            ("all_failures", "all_failure_count", "failures"),
            ("review_queue", "review_queue_count", "failures"),
            ("operating_confidence", "operating_confidence", "confidence"),
        ):
            add(metric, path, unit)
    elif source.kind == "failure_review":
        for metric, path, unit in (
            ("reviewed", "reviewed_count", "failures"),
            ("confirmed", "confirmed_count", "failures"),
            ("label_issues", "label_issue_count", "failures"),
        ):
            add(metric, path, unit)
    elif source.kind == "reproduction_bundle":
        for metric, path, unit in (
            ("sources", "source_count", "objects"),
            ("source_files", "source_file_count", "files"),
            ("source_bytes", "source_total_bytes", "bytes"),
            ("dependency_locks", "dependency_lock_count", "lockfiles"),
            ("commands", "command_count", "commands"),
        ):
            add(metric, path, unit)
    elif source.kind == "report":
        for metric, path, unit in (
            ("sources", "source_count", "objects"),
            ("metric_rows", "metric_row_count", "rows"),
            ("artifact_inventory_rows", "artifact_inventory_row_count", "rows"),
        ):
            add(metric, path, unit)
    return rows


def _verify_source(
    config_path: Path,
    source: ReportSource,
    *,
    confirmatory: bool,
) -> tuple[VerificationResult, Mapping[str, Any], Mapping[str, Any], Path | None]:
    path = _source_path(config_path, source.path)
    verification = verify_research_object(
        path,
        verify_references=True,
        deep=True,
        reject_unregistered=True,
        require_clean_git=confirmatory,
    )
    root = Path(verification.root)
    manifest = _load_json(root / "manifest.json", f"{source.label} manifest")
    summary_path: Path | None = None
    if source.kind == "other":
        summary: Mapping[str, Any] = {}
    else:
        summary_path, _ = _artifact_for_role(root, manifest, _SUMMARY_ROLE[source.kind])
        summary = _load_json(summary_path, f"{source.label} summary")
    return verification, manifest, summary, summary_path


def _source_inventory_plot(
    base_path: Path,
    inventory: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    labels = [str(row["label"]) for row in inventory]
    artifacts = [int(row["artifact_count"]) for row in inventory]
    bytes_mib = [int(row["artifact_bytes"]) / (1024 * 1024) for row in inventory]
    positions = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(13, max(5, len(labels) * 0.65)))
    axes[0].barh(positions, artifacts, color="#3a86ff")
    axes[0].set_yticks(positions, labels)
    axes[0].set_xlabel("Registered artifacts")
    axes[0].set_title("Source artifact counts")
    axes[0].grid(axis="x", alpha=0.25)
    axes[1].barh(positions, bytes_mib, color="#2a9d8f")
    axes[1].set_yticks(positions, labels)
    axes[1].set_xlabel("Registered payload (MiB)")
    axes[1].set_title("Source artifact payload")
    axes[1].grid(axis="x", alpha=0.25)
    figure.tight_layout()
    png = base_path.with_suffix(".png")
    svg = base_path.with_suffix(".svg")
    figure.savefig(png, dpi=180)
    figure.savefig(svg)
    plt.close(figure)
    return png, svg


def _evaluation_plot(
    base_path: Path,
    metrics: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path] | None:
    selected = [
        row
        for row in metrics
        if row["kind"] == "evaluation"
        and row["metric"] in {"coco_ap_50_95", "precision", "recall", "f1"}
    ]
    if not selected:
        return None
    labels = list(dict.fromkeys(str(row["label"]) for row in selected))
    metric_names = ("coco_ap_50_95", "precision", "recall", "f1")
    positions = np.arange(len(labels))
    width = 0.18
    figure, axis = plt.subplots(figsize=(max(10, len(labels) * 2.4), 6))
    for index, metric in enumerate(metric_names):
        lookup = {
            str(row["label"]): float(row["value"]) for row in selected if row["metric"] == metric
        }
        values = [lookup.get(label, np.nan) for label in labels]
        axis.bar(
            positions + (index - 1.5) * width,
            values,
            width=width,
            label=metric,
        )
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Metric")
    axis.set_title("Offline detector evaluation summary")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    png = base_path.with_suffix(".png")
    svg = base_path.with_suffix(".svg")
    figure.savefig(png, dpi=180)
    figure.savefig(svg)
    plt.close(figure)
    return png, svg


def _runtime_performance_plot(
    base_path: Path,
    metrics: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path] | None:
    selected = [row for row in metrics if row["kind"] == "runtime_analysis"]
    labels = list(dict.fromkeys(str(row["label"]) for row in selected))
    if not labels:
        return None

    metric_lookup = {
        (str(row["label"]), str(row["metric"])): float(row["value"]) for row in selected
    }
    positions = np.arange(len(labels))
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(max(12, len(labels) * 3.2), 5.8),
    )
    fps_values = [metric_lookup.get((label, "effective_source_fps"), np.nan) for label in labels]
    axes[0].bar(positions, fps_values, color="#2a9d8f")
    axes[0].set_xticks(positions, labels, rotation=20, ha="right")
    axes[0].set_ylabel("Effective source FPS")
    axes[0].set_title("Observed live throughput")
    axes[0].grid(axis="y", alpha=0.25)

    width = 0.24
    latency_metrics = (
        ("model_latency_median", "Model median"),
        ("pipeline_latency_median", "Pipeline median"),
        ("pipeline_latency_p95", "Pipeline p95"),
    )
    for index, (metric, display_name) in enumerate(latency_metrics):
        values = [metric_lookup.get((label, metric), np.nan) for label in labels]
        axes[1].bar(
            positions + (index - 1) * width,
            values,
            width=width,
            label=display_name,
        )
    axes[1].set_xticks(positions, labels, rotation=20, ha="right")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("Observed live latency")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    figure.suptitle("Sequential live runs: systems performance, not detector accuracy")
    figure.tight_layout()
    png = base_path.with_suffix(".png")
    svg = base_path.with_suffix(".svg")
    figure.savefig(png, dpi=180)
    figure.savefig(svg)
    plt.close(figure)
    return png, svg


def _markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    def escaped(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(fields) + " |"
    separator = "|" + "|".join("---" for _ in fields) + "|"
    body = [
        "| " + " | ".join(escaped(row.get(field, "")) for field in fields) + " |" for row in rows
    ]
    return "\n".join((header, separator, *body))


def _report_markdown(
    config: ReportConfig,
    inventory: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
) -> str:
    claims = "\n".join(f"- {claim}" for claim in config.claims)
    limitations = "\n".join(f"- {item}" for item in config.limitations)
    inventory_fields = (
        "source_id",
        "label",
        "kind",
        "status",
        "artifact_count",
        "artifact_bytes",
        "git_commit",
        "git_dirty",
    )
    metric_fields = ("label", "kind", "metric", "value", "unit", "support", "note")
    return f"""# {config.title}

Report ID: `{config.report_id}`  
Purpose: `{config.purpose}`  
Authors: {", ".join(config.authors)}

## Thesis context

{config.thesis_context}

## Registered claims

{claims}

## Verified source inventory

{_markdown_table(inventory, inventory_fields)}

## Machine-readable metrics

{_markdown_table(metrics, metric_fields)}

## Limitations

{limitations}

## Reproducibility

Every source object was verified read-only before this report was created.
The CSV tables are canonical; this Markdown document and all plots are derived
from them. Verify the sealed report with:

```bash
uv run carla-verify reports/{config.report_id} --reject-unregistered
```
"""


def _write_checksum_index(root: Path, paths: Sequence[Path], output: Path) -> None:
    entries = []
    for path in paths:
        reference = fingerprint_file(path)
        entries.append((path.relative_to(root).as_posix(), reference["sha256"]))
    entries.sort(key=lambda item: item[0].encode("utf-8"))
    _atomic_write_text(
        output,
        "".join(f"{digest}  {relative}\n" for relative, digest in entries),
    )


def build_report(
    *,
    config_path: str | Path,
    reports_root: str | Path = "reports",
    cli_args: Sequence[str] | Mapping[str, Any] = (),
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    resolved_config_path = Path(config_path).expanduser().resolve(strict=True)
    config = load_report_config(resolved_config_path)
    source_records = [
        (
            source,
            *_verify_source(
                resolved_config_path,
                source,
                confirmatory=config.purpose == "confirmatory",
            ),
        )
        for source in config.sources
    ]
    source_ids = [verification.run_id for _, verification, _, _, _ in source_records]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("report sources must have unique object IDs")

    inventory: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    source_descriptors: list[dict[str, Any]] = []
    input_references: list[dict[str, Any]] = []
    for source, verification, manifest, summary, summary_path in source_records:
        source_manifest = dict(verification.manifest)
        source_reference = {
            "kind": "report_source",
            "source_kind": source.kind,
            "label": source.label,
            "run_id": verification.run_id,
            "root": verification.root,
            **source_manifest,
        }
        input_references.append(source_reference)
        inventory.append(
            {
                "source_id": verification.run_id,
                "label": source.label,
                "kind": source.kind,
                "status": verification.status,
                "artifact_count": verification.artifact_count,
                "artifact_bytes": verification.artifact_bytes,
                "git_commit": verification.git.get("commit") or "",
                "git_dirty": verification.git.get("dirty"),
                "manifest_sha256": source_manifest["sha256"],
            }
        )
        metrics.extend(
            _extract_metrics(
                verification.run_id,
                source,
                summary,
            )
        )
        for artifact in manifest["artifacts"]:
            artifact_rows.append(
                {
                    "source_id": verification.run_id,
                    "label": source.label,
                    "kind": source.kind,
                    "role": artifact["role"],
                    "path": artifact["path"],
                    "sha256": artifact["sha256"],
                    "size_bytes": artifact["size_bytes"],
                }
            )
        source_descriptors.append(
            {
                **source_reference,
                "summary": (fingerprint_file(summary_path) if summary_path is not None else None),
                "deep_verification": dict(verification.deep_verification),
            }
        )

    tracker = RunArtifactTracker(
        reports_root,
        run_id=config.report_id,
        cli_args=cli_args,
        config={
            "schema_version": REPORT_RELEASE_SCHEMA_VERSION,
            "object_type": "research_report",
            "report": config.as_dict(),
        },
        repository_root=repository_root,
        input_refs=input_references,
    )
    with tracker:
        config_output = tracker.artifact_path("report_config.json")
        descriptor_path = tracker.artifact_path("report.json")
        markdown_path = tracker.artifact_path("report.md")
        inventory_path = tracker.artifact_path("tables/source_inventory.csv")
        metrics_path = tracker.artifact_path("tables/metrics.csv")
        artifacts_path = tracker.artifact_path("tables/source_artifacts.csv")
        checksum_path = tracker.artifact_path("checksums.sha256")
        _write_json(config_output, config.as_dict())
        _write_csv(
            inventory_path,
            inventory,
            (
                "source_id",
                "label",
                "kind",
                "status",
                "artifact_count",
                "artifact_bytes",
                "git_commit",
                "git_dirty",
                "manifest_sha256",
            ),
        )
        _write_csv(
            metrics_path,
            metrics,
            (
                "source_id",
                "label",
                "kind",
                "metric",
                "value",
                "unit",
                "support",
                "note",
            ),
        )
        _write_csv(
            artifacts_path,
            artifact_rows,
            ("source_id", "label", "kind", "role", "path", "sha256", "size_bytes"),
        )
        plot_paths: list[Path] = []
        if config.include_plots:
            plot_paths.extend(
                _source_inventory_plot(
                    tracker.artifact_path("plots/source_inventory"),
                    inventory,
                )
            )
            evaluation_plot = _evaluation_plot(
                tracker.artifact_path("plots/evaluation_summary"),
                metrics,
            )
            if evaluation_plot is not None:
                plot_paths.extend(evaluation_plot)
            runtime_plot = _runtime_performance_plot(
                tracker.artifact_path("plots/runtime_performance"),
                metrics,
            )
            if runtime_plot is not None:
                plot_paths.extend(runtime_plot)
        _atomic_write_text(markdown_path, _report_markdown(config, inventory, metrics))
        descriptor = {
            "schema_version": REPORT_RELEASE_SCHEMA_VERSION,
            "object_type": "research_report_release",
            "status": "complete",
            "report_id": config.report_id,
            "title": config.title,
            "authors": list(config.authors),
            "purpose": config.purpose,
            "thesis_context": config.thesis_context,
            "claims": list(config.claims),
            "limitations": list(config.limitations),
            "source_count": len(source_descriptors),
            "metric_row_count": len(metrics),
            "artifact_inventory_row_count": len(artifact_rows),
            "sources": source_descriptors,
            "canonical_tables": {
                "source_inventory": fingerprint_file(inventory_path),
                "metrics": fingerprint_file(metrics_path),
                "source_artifacts": fingerprint_file(artifacts_path),
            },
            "report_document": fingerprint_file(markdown_path),
            "plots": [fingerprint_file(path) for path in plot_paths],
            "generation": {
                "manual_metric_transcription": False,
                "all_sources_verified_before_generation": True,
                "confirmatory_clean_git_gate": config.purpose == "confirmatory",
            },
        }
        _write_json(descriptor_path, descriptor)
        payload_paths = [
            config_output,
            inventory_path,
            metrics_path,
            artifacts_path,
            markdown_path,
            descriptor_path,
            *plot_paths,
        ]
        _write_checksum_index(tracker.run_dir, payload_paths, checksum_path)
        role_by_path = {
            config_output: "report_configuration",
            descriptor_path: "report_release_manifest",
            markdown_path: "report_document",
            inventory_path: "report_source_inventory",
            metrics_path: "report_metric_table",
            artifacts_path: "report_source_artifact_table",
            checksum_path: "report_checksum_index",
            **{path: "report_plot" for path in plot_paths},
        }
        for path, role in role_by_path.items():
            tracker.register_artifact(
                path,
                role=role,
                metadata={
                    "report_id": config.report_id,
                    "source_count": len(source_descriptors),
                },
            )
    return {
        "report_id": config.report_id,
        "report_dir": str(tracker.run_dir),
        "manifest": str(tracker.manifest_path),
        "descriptor": descriptor,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a checksum-indexed Markdown report, canonical CSV tables, and "
            "plots from verified research manifests"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--reports-root", default="reports")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_report(
        config_path=args.config,
        reports_root=args.reports_root,
        cli_args=vars(args),
        repository_root=Path.cwd(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


__all__ = [
    "REPORT_RELEASE_SCHEMA_VERSION",
    "build_report",
    "main",
    "parse_args",
]
