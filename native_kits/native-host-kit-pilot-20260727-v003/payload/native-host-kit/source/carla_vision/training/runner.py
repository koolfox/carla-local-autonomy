"""Checksum-tracked training orchestration for interchangeable model backends."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, TextIO

from ..artifacts import RunArtifactTracker, fingerprint_file
from ..dataset.verified import VerifiedDataset, load_verified_dataset
from ..scenarios.seeds import derive_seed_bundle
from .backends import create_training_backend
from .contracts import (
    TrainerBackend,
    TrainingConfig,
    TrainingRequest,
    TrainingResult,
    load_training_config,
)
from .seeding import apply_training_seeds

TRAINING_RUN_SCHEMA_VERSION = "1.0"


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


def _sanitize_metrics(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _sanitize_metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_metrics(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _sanitize_metrics(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


class _Tee(io.TextIOBase):
    def __init__(self, console: TextIO, log: TextIO) -> None:
        self.console = console
        self.log = log

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        self.console.write(value)
        self.log.write(value)
        return len(value)

    def flush(self) -> None:
        if not self.console.closed:
            self.console.flush()
        if not self.log.closed:
            self.log.flush()


def _training_data_yaml(
    dataset: VerifiedDataset,
    *,
    train_partitions: Sequence[str],
    validation_partitions: Sequence[str],
) -> str:
    def paths(partitions: Sequence[str]) -> str:
        return "\n".join(f"  - images/{partition}" for partition in partitions)

    names = "\n".join(
        f"  {category_id}: {json.dumps(name, ensure_ascii=False)}"
        for category_id, name in dataset.categories.items()
    )
    return (
        f"path: {json.dumps(str(dataset.root), ensure_ascii=False)}\n"
        f"train:\n{paths(train_partitions)}\n"
        f"val:\n{paths(validation_partitions)}\n"
        "test: []\n"
        f"names:\n{names}\n"
    )


def _ensure_partition_directories(
    dataset: VerifiedDataset,
    partitions: Sequence[str],
) -> None:
    for partition in partitions:
        image_dir = dataset.root / "images" / partition
        label_dir = dataset.root / "labels" / partition
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise RuntimeError(f"dataset partition {partition!r} lacks images/labels directories")


def _path_within(path: Path, root: Path, name: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"{name} must stay inside the training run directory") from error
    if not resolved.is_file():
        raise RuntimeError(f"{name} is not a file: {resolved}")
    return resolved


def _artifact_role(relative: Path) -> str:
    normalized = relative.as_posix()
    if normalized.endswith("/weights/best.pt"):
        return "best_checkpoint"
    if normalized.endswith("/weights/last.pt"):
        return "last_checkpoint"
    if normalized.endswith("results.csv") or normalized.endswith("history.csv"):
        return "training_history"
    if normalized.endswith("args.yaml"):
        return "backend_resolved_arguments"
    if relative.suffix.lower() in {".png", ".svg", ".pdf"}:
        return "training_plot_or_panel"
    if relative.suffix.lower() in {".pt", ".pth", ".ckpt"}:
        return "training_checkpoint"
    return "backend_training_output"


def _register_backend_outputs(
    tracker: RunArtifactTracker,
    backend_root: Path,
) -> list[dict[str, Any]]:
    registered = []
    if not backend_root.exists():
        return registered
    for path in sorted(backend_root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlinked backend artifact is forbidden: {path}")
        if not path.is_file():
            continue
        registered.append(
            tracker.register_artifact(
                path,
                role=_artifact_role(path.relative_to(tracker.run_dir)),
                metadata={"producer": "trainer_backend"},
            )
        )
    return registered


def _resolved_payload(
    *,
    config: TrainingConfig,
    config_reference: Mapping[str, Any],
    dataset: VerifiedDataset,
    weights_reference: Mapping[str, Any],
    train_partitions: Sequence[str],
    validation_partitions: Sequence[str],
    seeds: Mapping[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_RUN_SCHEMA_VERSION,
        "object_type": "detector_training",
        "mode": "dry_run" if dry_run else "training",
        "experiment_id": config.experiment_id,
        "config": config.as_dict(),
        "source_config": dict(config_reference),
        "dataset": dict(dataset.reference),
        "pretrained_weights": dict(weights_reference),
        "resolved_partitions": {
            "train": list(train_partitions),
            "validation": list(validation_partitions),
            "locked_test_exposed_to_trainer": False,
        },
        "seeds": dict(seeds),
        "backend_seed": seeds["values"]["model_init"],
        "runtime_sensor_contract": "front_monocular_rgb_only",
    }


def run_training(
    *,
    config_path: str | Path,
    dataset_path: str | Path,
    pretrained_weights: str | Path,
    runs_root: str | Path = "runs",
    run_id: str | None = None,
    dry_run: bool = False,
    device_override: str | None = None,
    cli_args: Sequence[str] | Mapping[str, Any] = (),
    repository_root: str | Path | None = None,
    backend_override: TrainerBackend | None = None,
) -> dict[str, Any]:
    resolved_config_path = Path(config_path).expanduser().resolve(strict=True)
    config = load_training_config(resolved_config_path)
    if device_override is not None:
        if not device_override.strip():
            raise ValueError("device_override must not be empty")
        config = replace(config, device=device_override.strip())
    dataset = load_verified_dataset(dataset_path)
    train_partitions, validation_partitions = dataset.resolve_training_partitions(
        training_partitions=config.training_partitions,
        validation_partitions=config.validation_partitions,
    )
    _ensure_partition_directories(
        dataset,
        (*train_partitions, *validation_partitions),
    )
    weights_path = Path(pretrained_weights).expanduser().resolve(strict=True)
    if not weights_path.is_file():
        raise ValueError(f"pretrained weights are not a regular file: {weights_path}")
    weights_reference = {
        "kind": "pretrained_weights",
        "backend": config.backend,
        **fingerprint_file(weights_path),
    }
    config_reference = {
        "kind": "training_preregistration",
        **fingerprint_file(resolved_config_path),
    }
    seed_bundle = derive_seed_bundle(
        config.master_seed,
        namespace_prefix=f"training/{config.experiment_id}",
    )
    resolved = _resolved_payload(
        config=config,
        config_reference=config_reference,
        dataset=dataset,
        weights_reference=weights_reference,
        train_partitions=train_partitions,
        validation_partitions=validation_partitions,
        seeds=seed_bundle.as_dict(),
        dry_run=dry_run,
    )
    tracker = RunArtifactTracker(
        runs_root,
        run_id=run_id,
        cli_args=cli_args,
        config=resolved,
        repository_root=repository_root,
        model_refs=[weights_reference],
        dataset_refs=[dataset.reference],
    )
    with tracker:
        resolved_path = tracker.artifact_path("resolved_training_config.json")
        seeds_path = tracker.artifact_path("seeds.json")
        data_path = tracker.artifact_path("data/training_data.yaml")
        summary_path = tracker.artifact_path("training_summary.json")
        log_path = tracker.artifact_path("logs/training.log")
        _write_json(resolved_path, resolved)
        _write_json(seeds_path, seed_bundle.as_dict())
        _atomic_write_text(
            data_path,
            _training_data_yaml(
                dataset,
                train_partitions=train_partitions,
                validation_partitions=validation_partitions,
            ),
        )
        for path, role in (
            (resolved_path, "resolved_training_configuration"),
            (seeds_path, "training_seed_schedule"),
            (data_path, "trainer_dataset_configuration"),
        ):
            tracker.register_artifact(
                path,
                role=role,
                metadata={"experiment_id": config.experiment_id},
            )

        if dry_run:
            summary = {
                "schema_version": TRAINING_RUN_SCHEMA_VERSION,
                "status": "dry_run",
                "experiment_id": config.experiment_id,
                "backend": config.backend,
                "dataset_id": dataset.dataset_id,
                "sample_count": dataset.sample_count,
                "train_partitions": list(train_partitions),
                "validation_partitions": list(validation_partitions),
                "pretrained_weights": weights_reference,
                "training_executed": False,
            }
            _write_json(summary_path, summary)
            tracker.register_artifact(
                summary_path,
                role="training_plan_summary",
                metadata={"status": "dry_run"},
            )
            return {
                "run_id": tracker.run_id,
                "run_dir": str(tracker.run_dir),
                "manifest": str(tracker.manifest_path),
                "summary": summary,
            }

        seed_state = apply_training_seeds(
            seed_bundle,
            deterministic=config.deterministic,
        )
        backend = backend_override or create_training_backend(config)
        backend_root = tracker.artifact_path("backend", create_parent=True)
        backend_root.mkdir(exist_ok=True)
        request = TrainingRequest(
            config=config,
            pretrained_weights=weights_path,
            dataset_root=dataset.root,
            data_config=data_path,
            output_root=backend_root,
            backend_seed=seed_bundle.values["model_init"],
        )
        result: TrainingResult | None = None
        training_error: BaseException | None = None
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            tee_stdout = _Tee(sys.stdout, log)
            tee_stderr = _Tee(sys.stderr, log)
            try:
                with (
                    contextlib.redirect_stdout(tee_stdout),
                    contextlib.redirect_stderr(tee_stderr),
                ):
                    print(
                        f"experiment={config.experiment_id} backend={backend.name} "
                        f"dataset={dataset.dataset_id}",
                        flush=True,
                    )
                    result = backend.train(request)
            except BaseException as error:
                training_error = error
        tracker.register_artifact(
            log_path,
            role="training_log",
            metadata={"experiment_id": config.experiment_id},
        )
        _register_backend_outputs(tracker, backend_root)
        if training_error is not None:
            raise training_error
        if result is None:
            raise RuntimeError("training backend returned no result")
        best = _path_within(
            result.best_checkpoint,
            tracker.run_dir,
            "best checkpoint",
        )
        last = _path_within(
            result.last_checkpoint,
            tracker.run_dir,
            "last checkpoint",
        )
        if best == last:
            raise RuntimeError("best and last checkpoints must be distinct files")
        tracker.register_artifact(
            best,
            role="best_checkpoint",
            metadata={
                "experiment_id": config.experiment_id,
                "backend": result.backend_name,
            },
        )
        tracker.register_artifact(
            last,
            role="last_checkpoint",
            metadata={
                "experiment_id": config.experiment_id,
                "backend": result.backend_name,
            },
        )
        output_dir = result.output_dir.expanduser().resolve(strict=True)
        try:
            output_dir.relative_to(tracker.run_dir.resolve())
        except ValueError as error:
            raise RuntimeError("training backend output directory escapes the run") from error
        best_reference = {
            "kind": "trained_checkpoint",
            "checkpoint_role": "best",
            "backend": result.backend_name,
            **fingerprint_file(best),
        }
        last_reference = {
            "kind": "trained_checkpoint",
            "checkpoint_role": "last",
            "backend": result.backend_name,
            **fingerprint_file(last),
        }
        tracker.add_model_reference(best_reference)
        tracker.add_model_reference(last_reference)
        summary = {
            "schema_version": TRAINING_RUN_SCHEMA_VERSION,
            "status": "success",
            "experiment_id": config.experiment_id,
            "backend": result.backend_name,
            "backend_metadata": _sanitize_metrics(result.metadata),
            "dataset": dict(dataset.reference),
            "train_partitions": list(train_partitions),
            "validation_partitions": list(validation_partitions),
            "pretrained_weights": weights_reference,
            "best_checkpoint": best_reference,
            "last_checkpoint": last_reference,
            "metrics": _sanitize_metrics(result.metrics),
            "seed_state": seed_state,
            "training_executed": True,
        }
        _write_json(summary_path, summary)
        tracker.register_artifact(
            summary_path,
            role="training_summary",
            metadata={
                "experiment_id": config.experiment_id,
                "backend": result.backend_name,
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
            "Train RT-DETR, YOLO, or a custom detector from an integrity-verified "
            "CARLA dataset release"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run-id")
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_training(
        config_path=args.config,
        dataset_path=args.dataset,
        pretrained_weights=args.weights,
        runs_root=args.runs_root,
        run_id=args.run_id,
        dry_run=args.dry_run,
        device_override=args.device,
        cli_args=vars(args),
        repository_root=Path.cwd(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


__all__ = [
    "TRAINING_RUN_SCHEMA_VERSION",
    "main",
    "parse_args",
    "run_training",
]
