"""Promote a verified training checkpoint into an immutable model package."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..artifacts import RunArtifactTracker, fingerprint_file
from ..dataset.verified import load_verified_dataset
from ..verification import verify_research_object
from .contracts import (
    MODEL_RELEASE_SCHEMA_VERSION,
    ModelReleaseConfig,
    load_model_release_config,
)


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


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read {name}: {error}") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{name} must contain a JSON object")
    return payload


def _artifact_for_role(
    run_dir: Path,
    manifest: Mapping[str, Any],
    role: str,
) -> tuple[Path, Mapping[str, Any]]:
    matches = [
        artifact
        for artifact in manifest["artifacts"]
        if isinstance(artifact, Mapping) and artifact.get("role") == role
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"training run must contain exactly one {role!r} artifact; found {len(matches)}"
        )
    entry = matches[0]
    relative = Path(str(entry["path"]))
    path = (run_dir / relative).resolve(strict=True)
    try:
        path.relative_to(run_dir)
    except ValueError as error:
        raise RuntimeError(f"training artifact escapes its run directory: {relative}") from error
    return path, entry


def _metric_value(metrics: Mapping[str, Any], dotted_name: str) -> float:
    value: Any = metrics
    for part in dotted_name.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"selection metric is absent from training summary: {dotted_name}")
        value = value[part]
    if isinstance(value, bool):
        raise ValueError(f"selection metric is not numeric: {dotted_name}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"selection metric is not finite: {dotted_name}")
    return result


def _source_dataset_reference(
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Any]:
    references = manifest.get("references")
    if not isinstance(references, Mapping):
        raise RuntimeError("training manifest references are missing")
    datasets = references.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise RuntimeError("training run must reference exactly one dataset release")
    reference = datasets[0]
    if not isinstance(reference, Mapping):
        raise RuntimeError("training dataset reference must be an object")
    manifest_reference = reference.get("manifest")
    if not isinstance(manifest_reference, Mapping):
        raise RuntimeError("training dataset reference lacks its tracker manifest")
    dataset_manifest_path = Path(str(manifest_reference.get("path", ""))).resolve(strict=True)
    dataset = load_verified_dataset(dataset_manifest_path.parent)
    if dataset.dataset_id != reference.get("dataset_id"):
        raise RuntimeError("training dataset identity changed before model promotion")
    return reference, dataset


def _validate_against_training(
    config: ModelReleaseConfig,
    resolved: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> float:
    if resolved.get("object_type") != "detector_training" or resolved.get("mode") != "training":
        raise RuntimeError("source run is not a completed detector-training execution")
    training_config = resolved.get("config")
    if not isinstance(training_config, Mapping):
        raise RuntimeError("resolved training configuration is missing")
    if summary.get("status") != "success" or summary.get("training_executed") is not True:
        raise RuntimeError("source training summary is not a successful executed run")
    source_backend = str(training_config.get("backend", "")).lower().replace("_", "-")
    if config.backend != source_backend:
        raise ValueError(
            f"model backend {config.backend!r} disagrees with training backend {source_backend!r}"
        )
    if config.input.image_size != training_config.get("image_size"):
        raise ValueError("model input image_size disagrees with the training configuration")
    validation = tuple(str(value) for value in resolved["resolved_partitions"]["validation"])
    if not set(config.selection.partitions) <= set(validation):
        raise ValueError("model selection partitions are absent from training validation")
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise RuntimeError("training summary metrics are missing")
    return _metric_value(metrics, config.selection.metric)


def _safe_checkpoint_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    return suffix if suffix in {".pt", ".pth", ".ckpt", ".safetensors"} else ".bin"


def _copy_stable(source: Path, destination: Path) -> None:
    before = fingerprint_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    after = fingerprint_file(source)
    copied = fingerprint_file(destination)
    if before != after or (
        copied["sha256"],
        copied["size_bytes"],
    ) != (
        before["sha256"],
        before["size_bytes"],
    ):
        destination.unlink(missing_ok=True)
        raise RuntimeError("source checkpoint changed during model packaging")


def _model_card(descriptor: Mapping[str, Any]) -> str:
    limitations = "\n".join(f"- {item}" for item in descriptor["limitations"])
    categories = "\n".join(
        f"- `{category['id']}` — {category['name']}"
        for category in descriptor["ontology"]["categories"]
    )
    selection = descriptor["selection"]
    weights = descriptor["weights"]
    return f"""# {descriptor["model_id"]}

Status: immutable detector model package  
Architecture: {descriptor["architecture"]}  
Backend: {descriptor["backend"]}  
Runtime sensor contract: front monocular RGB only

## Intended use

{descriptor["intended_use"]}

## Input and output

- Color order: {descriptor["input"]["color_order"]}
- Image size: {descriptor["input"]["image_size"]}
- Resize: {descriptor["input"]["resize_method"]}
- Normalization: {descriptor["input"]["normalization"]}
- Temporal context: {descriptor["input"]["temporal_context_frames"]} frame
- Detector adapter: {descriptor["inference"]["detector_backend"]}
- Custom factory: {descriptor["inference"]["detector_factory"]}
- Output: `{descriptor["output_contract"]}`

## Checkpoint selection

- Source run: `{descriptor["source_training"]["run_id"]}`
- Role: `{weights["checkpoint_role"]}`
- Metric: `{selection["metric"]}` ({selection["mode"]})
- Value: {selection["value"]}
- Validation partitions: {", ".join(selection["partitions"])}
- SHA-256: `{weights["sha256"]}`

## Canonical categories

{categories}

## Limitations

{limitations}

## Licenses and upstream

- Packaged weights: {descriptor["license"]["weights"]}
- Framework code: {descriptor["license"]["code"]}
- Upstream implementation/assets: {descriptor["license"]["upstream"]}
- Upstream source: {descriptor["upstream_source"]}

## Security

Checkpoint formats based on Python pickle can execute code while loading.
Load this package only after verifying its manifest and SHA-256 digest and
only when its source training run is trusted. CARLA results do not establish
real-world driving safety.
"""


def _write_checksum_index(root: Path, paths: Sequence[Path], output: Path) -> None:
    entries = []
    for path in paths:
        reference = fingerprint_file(path)
        relative = path.relative_to(root).as_posix()
        entries.append((relative, str(reference["sha256"])))
    entries.sort(key=lambda item: item[0].encode("utf-8"))
    _atomic_write_text(
        output,
        "".join(f"{digest}  {relative}\n" for relative, digest in entries),
    )


def package_model(
    *,
    config_path: str | Path,
    training_run: str | Path,
    models_root: str | Path = "models",
    cli_args: Sequence[str] | Mapping[str, Any] = (),
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    resolved_config_path = Path(config_path).expanduser().resolve(strict=True)
    config = load_model_release_config(resolved_config_path)
    source_verification = verify_research_object(
        training_run,
        verify_references=True,
        deep=True,
        reject_unregistered=True,
    )
    source_run_dir = Path(source_verification.root)
    source_manifest = _load_json(source_run_dir / "manifest.json", "training manifest")
    resolved_path, _ = _artifact_for_role(
        source_run_dir,
        source_manifest,
        "resolved_training_configuration",
    )
    summary_path, _ = _artifact_for_role(
        source_run_dir,
        source_manifest,
        "training_summary",
    )
    checkpoint_path, checkpoint_artifact = _artifact_for_role(
        source_run_dir,
        source_manifest,
        f"{config.checkpoint_role}_checkpoint",
    )
    resolved = _load_json(resolved_path, "resolved training configuration")
    summary = _load_json(summary_path, "training summary")
    selected_metric = _validate_against_training(config, resolved, summary)
    dataset_reference, dataset = _source_dataset_reference(source_manifest)
    source_manifest_reference = fingerprint_file(source_run_dir / "manifest.json")
    source_checkpoint_reference = {
        "kind": "source_training_checkpoint",
        "checkpoint_role": config.checkpoint_role,
        **fingerprint_file(checkpoint_path),
    }
    tracker_config = {
        "schema_version": MODEL_RELEASE_SCHEMA_VERSION,
        "object_type": "model_release",
        "release": config.as_dict(),
        "source_training_run": {
            "run_id": source_verification.run_id,
            **source_manifest_reference,
        },
    }
    tracker = RunArtifactTracker(
        models_root,
        run_id=config.model_id,
        cli_args=cli_args,
        config=tracker_config,
        repository_root=repository_root,
        model_refs=[source_checkpoint_reference],
        dataset_refs=[dataset_reference],
        input_refs=[
            {
                "kind": "source_training_run",
                "run_id": source_verification.run_id,
                **source_manifest_reference,
            }
        ],
    )
    with tracker:
        resolved_release_path = tracker.artifact_path("release_config.json")
        weights_path = tracker.artifact_path(
            f"weights/model{_safe_checkpoint_suffix(checkpoint_path)}"
        )
        descriptor_path = tracker.artifact_path("model.json")
        card_path = tracker.artifact_path("model-card.md")
        checksum_path = tracker.artifact_path("checksums.sha256")
        _write_json(resolved_release_path, config.as_dict())
        _copy_stable(checkpoint_path, weights_path)
        packaged_weights = {
            "path": weights_path.relative_to(tracker.run_dir).as_posix(),
            "checkpoint_role": config.checkpoint_role,
            "format": weights_path.suffix.lstrip("."),
            **{
                key: value for key, value in fingerprint_file(weights_path).items() if key != "path"
            },
            "source_artifact": {
                "path": str(checkpoint_path),
                "role": checkpoint_artifact["role"],
                "sha256": checkpoint_artifact["sha256"],
                "size_bytes": checkpoint_artifact["size_bytes"],
            },
        }
        descriptor = {
            "schema_version": MODEL_RELEASE_SCHEMA_VERSION,
            "object_type": "detector_model_release",
            "status": "complete",
            "model_id": config.model_id,
            "architecture": config.architecture,
            "backend": config.backend,
            "runtime_sensor_contract": "front_monocular_rgb_only",
            "input": asdict(config.input),
            "inference": asdict(config.inference),
            "output_contract": config.output_contract,
            "weights": packaged_weights,
            "selection": {
                **asdict(config.selection),
                "value": selected_metric,
            },
            "source_training": {
                "run_id": source_verification.run_id,
                "experiment_id": resolved["experiment_id"],
                "manifest": source_manifest_reference,
                "resolved_configuration": fingerprint_file(resolved_path),
                "summary": fingerprint_file(summary_path),
                "metrics": summary["metrics"],
                "seed_state": summary.get("seed_state"),
            },
            "dataset": dataset_reference,
            "ontology": {
                "dataset_id": dataset.dataset_id,
                "categories": [
                    {"id": category_id, "name": name}
                    for category_id, name in dataset.categories.items()
                ],
            },
            "intended_use": config.intended_use,
            "limitations": list(config.limitations),
            "license": asdict(config.license),
            "upstream_source": config.upstream_source,
            "release_config": fingerprint_file(resolved_release_path),
            "security": {
                "checkpoint_is_potentially_executable": weights_path.suffix
                in {".pt", ".pth", ".ckpt", ".bin"},
                "verification_required_before_loading": True,
            },
        }
        _write_json(descriptor_path, descriptor)
        _atomic_write_text(card_path, _model_card(descriptor))
        _write_checksum_index(
            tracker.run_dir,
            (resolved_release_path, weights_path, descriptor_path, card_path),
            checksum_path,
        )
        for path, role in (
            (checksum_path, "model_checksum_index"),
            (card_path, "model_card"),
            (descriptor_path, "model_release_manifest"),
            (resolved_release_path, "model_release_configuration"),
            (weights_path, "packaged_model_weights"),
        ):
            tracker.register_artifact(
                path,
                role=role,
                metadata={
                    "model_id": config.model_id,
                    "source_training_run": source_verification.run_id,
                },
            )
    return {
        "model_id": config.model_id,
        "model_dir": str(tracker.run_dir),
        "manifest": str(tracker.manifest_path),
        "descriptor": descriptor,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Promote a verified detector-training checkpoint into a "
            "checksum-indexed immutable model package"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--models-root", default="models")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = package_model(
        config_path=args.config,
        training_run=args.training_run,
        models_root=args.models_root,
        cli_args=vars(args),
        repository_root=Path.cwd(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


__all__ = [
    "MODEL_RELEASE_SCHEMA_VERSION",
    "main",
    "package_model",
    "parse_args",
]
