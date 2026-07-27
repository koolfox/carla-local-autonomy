"""Build voxel-actuation readiness evidence from immutable training and run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .training.dataset import load_voxel_episode

VOXEL_EVIDENCE_BUILDER_SCHEMA_VERSION = "1.0"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("voxel checkpoint must contain a mapping")
    required = ("model_config", "state_dict", "grid_spec", "history_frames", "horizons_s")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError("voxel checkpoint is missing fields: " + ", ".join(missing))
    history_frames = payload.get("history_frames")
    horizons = payload.get("horizons_s")
    if not isinstance(history_frames, int) or history_frames <= 0:
        raise ValueError("voxel checkpoint history_frames must be positive")
    if not isinstance(horizons, (list, tuple)) or not horizons:
        raise ValueError("voxel checkpoint horizons_s must be a non-empty sequence")
    converted_horizons = tuple(float(value) for value in horizons)
    if any(not math.isfinite(value) or value < 0.0 for value in converted_horizons):
        raise ValueError("voxel checkpoint horizons must be finite and non-negative")
    return {
        "rgb_only_runtime_contract": True,
        "checkpoint_sha256": _sha256(path),
        "checkpoint_path": str(path),
        "history_frames": history_frames,
        "horizons_s": list(converted_horizons),
        "grid_spec": dict(payload["grid_spec"]),
    }


def _verify_training_datasets(roots: Sequence[Path]) -> dict[str, Any]:
    if not roots:
        raise ValueError("at least one training dataset is required")
    groups: set[str] = set()
    episodes: list[dict[str, Any]] = []
    artifact_count = 0
    for root in roots:
        episode = load_voxel_episode(root)
        groups.add(episode.group_key)
        missing: list[str] = []
        for record in episode.records:
            for field in ("rgb", "teacher_voxel"):
                relative = record.get(field)
                if not relative:
                    missing.append(f"frame {record.get('frame')} missing {field}")
                    continue
                path = episode.root / str(relative)
                if not path.is_file():
                    missing.append(str(path))
                else:
                    artifact_count += 1
        if missing:
            raise ValueError(
                f"voxel training dataset has missing artifacts: {episode.root}: {missing[:10]}"
            )
        episodes.append(
            {
                "root": str(episode.root),
                "episode_id": episode.episode_id,
                "group_key": episode.group_key,
                "record_count": len(episode.records),
            }
        )
    return {
        "dataset_verification_status": "passed",
        "dataset_count": len(roots),
        "route_group_count": len(groups),
        "verified_artifact_count": artifact_count,
        "datasets": episodes,
    }


def _split_leakage(training_summary: Mapping[str, Any]) -> tuple[bool, dict[str, list[str]]]:
    splits = training_summary.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("voxel training summary must contain splits")
    normalized: dict[str, set[str]] = {}
    for name in ("train", "val", "test"):
        values = splits.get(name)
        if not isinstance(values, list):
            raise ValueError(f"voxel training summary splits.{name} must be a list")
        normalized[name] = {str(value) for value in values}
    overlaps: dict[str, list[str]] = {}
    names = tuple(normalized)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            shared = sorted(normalized[left_name] & normalized[right_name])
            if shared:
                overlaps[f"{left_name}:{right_name}"] = shared
    return bool(overlaps), overlaps


def _shadow_artifacts(roots: Sequence[Path]) -> dict[str, Any]:
    if not roots:
        raise ValueError("at least one shadow run is required")
    latencies: list[float] = []
    uncertainties: list[float] = []
    total_records = 0
    total_errors = 0
    control_calls = 0
    actuation_enabled = False
    run_details: list[dict[str, Any]] = []
    for root in roots:
        summary = _read_json(root / "summary.json")
        records_path = root / "records.jsonl"
        run_records = 0
        invalid_records = 0
        if records_path.is_file():
            for line_number, raw in enumerate(
                records_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if not raw.strip():
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid shadow record {records_path}:{line_number}: {error}"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(
                        f"shadow record must be an object: {records_path}:{line_number}"
                    )
                latency = record.get("prediction_latency_ms")
                uncertainty = record.get("uncertain_voxel_fraction")
                if not isinstance(latency, (int, float)) or not math.isfinite(float(latency)):
                    invalid_records += 1
                    continue
                if not isinstance(uncertainty, (int, float)) or not math.isfinite(
                    float(uncertainty)
                ):
                    invalid_records += 1
                    continue
                if record.get("actuation_applied") is not False:
                    raise ValueError(f"shadow record reports actuation: {records_path}:{line_number}")
                latencies.append(float(latency))
                uncertainties.append(float(uncertainty))
                run_records += 1
        summary_records = int(summary.get("records", run_records))
        summary_errors = int(summary.get("errors", 0))
        total_records += max(run_records, summary_records)
        total_errors += summary_errors + invalid_records
        control_calls += int(summary.get("control_calls", 0))
        actuation_enabled = actuation_enabled or bool(summary.get("actuation_enabled", False))
        if summary.get("read_only") is not True:
            raise ValueError(f"shadow summary is not read-only: {root}")
        run_details.append(
            {
                "root": str(root),
                "records": max(run_records, summary_records),
                "errors": summary_errors + invalid_records,
                "stop_reason": summary.get("stop_reason"),
            }
        )
    denominator = total_records + total_errors
    return {
        "run_count": len(roots),
        "record_count": total_records,
        "error_count": total_errors,
        "error_rate": total_errors / max(denominator, 1),
        "p95_latency_ms": None if not latencies else float(np.percentile(latencies, 95)),
        "maximum_uncertain_voxel_fraction": (
            None if not uncertainties else float(max(uncertainties))
        ),
        "actuation_enabled": actuation_enabled,
        "control_calls": control_calls,
        "runs": run_details,
    }


def build_voxel_readiness_evidence(
    *,
    checkpoint: Path,
    training_summary_path: Path,
    training_dataset_roots: Sequence[Path],
    benchmark_report_path: Path,
    shadow_roots: Sequence[Path],
    issue_7_acceptance_complete: bool,
    operator_review_complete: bool,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    training_summary_path = training_summary_path.expanduser().resolve(strict=True)
    benchmark_report_path = benchmark_report_path.expanduser().resolve(strict=True)
    training_dataset_roots = tuple(
        path.expanduser().resolve(strict=True) for path in training_dataset_roots
    )
    shadow_roots = tuple(path.expanduser().resolve(strict=True) for path in shadow_roots)

    training_summary = _read_json(training_summary_path)
    benchmark = _read_json(benchmark_report_path)
    if benchmark.get("status") != "complete":
        raise ValueError("voxel benchmark report must have status complete")
    leakage, overlaps = _split_leakage(training_summary)
    training = _verify_training_datasets(training_dataset_roots)
    training.update(
        {
            "route_group_leakage": leakage,
            "route_group_overlap": overlaps,
            "training_summary_path": str(training_summary_path),
            "best_mean_occupied_iou": training_summary.get("best_mean_occupied_iou"),
            "flow_head": training_summary.get("flow_head"),
        }
    )
    shadow = _shadow_artifacts(shadow_roots)
    evidence = {
        "schema_version": VOXEL_EVIDENCE_BUILDER_SCHEMA_VERSION,
        "issue_7_acceptance_complete": bool(issue_7_acceptance_complete),
        "operator_review_complete": bool(operator_review_complete),
        "model": _verify_checkpoint(checkpoint),
        "training": training,
        "validation": {
            "benchmark_report_path": str(benchmark_report_path),
            "benchmark_run_count": benchmark.get("run_count"),
            "evaluated_prediction_frame_count": benchmark.get(
                "evaluated_prediction_frame_count"
            ),
            "current_occupied_iou": benchmark.get("current_occupied_iou"),
            "future_occupied_iou_1s": benchmark.get("future_occupied_iou_1s"),
            "future_persistence_iou_1s": benchmark.get("future_persistence_iou_1s"),
            "future_model_minus_persistence_iou_1s": benchmark.get(
                "future_model_minus_persistence_iou_1s"
            ),
            "brier_score": benchmark.get("mean_brier_score"),
            "maximum_uncertain_voxel_fraction": benchmark.get(
                "maximum_uncertain_voxel_fraction"
            ),
            "p95_latency_ms": shadow.get("p95_latency_ms"),
        },
        "shadow": shadow,
        "completion_note": (
            "Flags remain false unless real benchmark acceptance and human failure review "
            "have actually completed. This builder never infers completion automatically."
        ),
    }
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build voxel-actuation readiness evidence from training and run artifacts."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--training-dataset", action="append", type=Path, default=[])
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--shadow-run", action="append", type=Path, default=[])
    parser.add_argument("--issue-7-acceptance-complete", action="store_true")
    parser.add_argument("--operator-review-complete", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evidence = build_voxel_readiness_evidence(
        checkpoint=args.checkpoint,
        training_summary_path=args.training_summary,
        training_dataset_roots=args.training_dataset,
        benchmark_report_path=args.benchmark_report,
        shadow_roots=args.shadow_run,
        issue_7_acceptance_complete=args.issue_7_acceptance_complete,
        operator_review_complete=args.operator_review_complete,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


__all__ = [
    "VOXEL_EVIDENCE_BUILDER_SCHEMA_VERSION",
    "build_parser",
    "build_voxel_readiness_evidence",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
