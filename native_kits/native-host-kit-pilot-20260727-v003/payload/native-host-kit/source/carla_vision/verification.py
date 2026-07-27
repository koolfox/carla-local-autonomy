"""Read-only integrity verification for tracked research objects.

This module verifies the common ``RunArtifactTracker`` envelope without
trusting paths from the manifest.  It can additionally traverse fingerprinted
external references, validate release checksum indexes, reject unregistered
files, and invoke object-specific dataset or scenario-plan verification.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .artifacts import SCHEMA_VERSION, fingerprint_file
from .reproducibility import (
    REPRODUCIBILITY_SCHEMA_VERSION,
    canonical_experiment_id,
    configuration_identity,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_ALLOWED_STATUSES = frozenset({"created", "running", "success", "failed"})
_SCENARIO_ROLES = frozenset(
    {
        "resolved_scenario_suite",
        "resolved_split_plan",
        "planned_episodes_jsonl",
        "scenario_plan_summary",
    }
)
_EVALUATION_ROLES = frozenset(
    {
        "canonical_coco_predictions",
        "evaluation_summary",
        "per_image_prediction_log",
        "resolved_evaluation_configuration",
    }
)
_VISION_SHADOW_ROLES = frozenset(
    {
        "detections_jsonl",
        "run_summary",
        "vision_policy_input_audit",
        "vision_shadow_proposals",
    }
)
_SHADOW_MATRIX_ROLES = frozenset(
    {
        "resolved_shadow_matrix_configuration",
        "shadow_matrix_plan_jsonl",
        "shadow_matrix_plan_csv",
        "shadow_matrix_execution_results",
        "shadow_matrix_release_manifest",
    }
)
_NATIVE_PREFLIGHT_ROLES = frozenset(
    {
        "native_preflight_checks",
        "native_preflight_selected_episodes",
        "native_preflight_summary",
    }
)
_NATIVE_HOST_KIT_ROLES = frozenset(
    {
        "native_host_kit_configuration",
        "native_host_kit_release_manifest",
        "native_host_kit_payload_archive",
        "native_host_kit_payload_inventory_json",
        "native_host_kit_payload_inventory_csv",
        "native_host_kit_readme",
        "native_host_kit_checksum_index",
    }
)


class ArtifactIntegrityError(RuntimeError):
    """Raised when a tracked research object fails integrity validation."""


@dataclass(frozen=True)
class VerificationResult:
    root: str
    manifest: Mapping[str, Any]
    run_id: str
    status: str
    schema_version: str
    artifact_count: int
    artifact_bytes: int
    role_counts: Mapping[str, int]
    checksum_index_count: int
    checksum_index_entries: int
    external_reference_count: int
    deep_verification: Mapping[str, Any]
    git: Mapping[str, Any]
    unregistered_file_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json_object(path: Path, name: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(f"could not read {name}: {error}") from error
    if not isinstance(value, Mapping):
        raise ArtifactIntegrityError(f"{name} must contain a JSON object")
    return value


def _resolve_root_and_manifest(path: str | Path) -> tuple[Path, Path]:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except OSError as error:
        raise ArtifactIntegrityError(f"verification target does not exist: {path}") from error
    if resolved.is_dir():
        root = resolved
        manifest_path = root / "manifest.json"
    elif resolved.is_file() and resolved.name == "manifest.json":
        root = resolved.parent
        manifest_path = resolved
    else:
        raise ArtifactIntegrityError(
            "verification target must be an object directory or its manifest.json"
        )
    if not manifest_path.is_file():
        raise ArtifactIntegrityError(f"manifest.json is missing from {root}")
    return root, manifest_path


def _safe_relative_file(root: Path, relative: str, *, context: str) -> Path:
    raw = Path(relative)
    if (
        not relative
        or raw.is_absolute()
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in raw.parts)
    ):
        raise ArtifactIntegrityError(f"{context} has an unsafe relative path: {relative!r}")
    unresolved = root / raw
    current = root
    for part in raw.parts:
        current = current / part
        if current.is_symlink():
            raise ArtifactIntegrityError(f"{context} uses a symlink: {relative}")
    try:
        resolved = unresolved.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ArtifactIntegrityError(
            f"{context} escapes the object root or is missing: {relative}"
        ) from error
    if not resolved.is_file():
        raise ArtifactIntegrityError(f"{context} is not a regular file: {relative}")
    return resolved


def _fingerprint(path: Path, *, context: str) -> tuple[str, int]:
    try:
        reference = fingerprint_file(path)
    except (OSError, RuntimeError, ValueError) as error:
        raise ArtifactIntegrityError(f"could not fingerprint {context}: {error}") from error
    return str(reference["sha256"]), int(reference["size_bytes"])


def _validate_fingerprint_fields(
    raw: Mapping[str, Any],
    *,
    context: str,
) -> tuple[str, int]:
    digest = raw.get("sha256")
    size = raw.get("size_bytes")
    if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
        raise ArtifactIntegrityError(f"{context} has an invalid SHA-256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ArtifactIntegrityError(f"{context} has an invalid byte size")
    return digest, size


def _parse_utc_timestamp(raw: Any, *, field: str) -> datetime:
    if not isinstance(raw, str) or not raw.endswith("Z"):
        raise ArtifactIntegrityError(f"manifest timestamp {field} must be ISO 8601 UTC")
    try:
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00")
    except ValueError as error:
        raise ArtifactIntegrityError(f"manifest timestamp {field} is invalid") from error
    return parsed


def _verify_manifest_envelope(
    manifest: Mapping[str, Any],
    *,
    allow_non_success: bool,
    require_clean_git: bool,
) -> tuple[str, str, Mapping[str, Any]]:
    schema_version = manifest.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ArtifactIntegrityError(
            f"unsupported tracker schema {schema_version!r}; expected {SCHEMA_VERSION!r}"
        )
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ArtifactIntegrityError("manifest run_id is missing")
    status = manifest.get("status")
    if status not in _ALLOWED_STATUSES:
        raise ArtifactIntegrityError(f"manifest status is invalid: {status!r}")
    if not allow_non_success and status != "success":
        raise ArtifactIntegrityError(f"manifest status must be success, got {status!r}")
    failure = manifest.get("failure")
    if status == "failed" and not isinstance(failure, Mapping):
        raise ArtifactIntegrityError("failed manifest must retain a failure object")
    if status != "failed" and failure is not None:
        raise ArtifactIntegrityError("non-failed manifest cannot retain a failure object")

    timestamps = manifest.get("timestamps")
    if not isinstance(timestamps, Mapping):
        raise ArtifactIntegrityError("manifest timestamps must be an object")
    created = _parse_utc_timestamp(timestamps.get("created_at"), field="created_at")
    updated = _parse_utc_timestamp(timestamps.get("updated_at"), field="updated_at")
    started_raw = timestamps.get("started_at")
    finished_raw = timestamps.get("finished_at")
    started = None if started_raw is None else _parse_utc_timestamp(started_raw, field="started_at")
    finished = (
        None if finished_raw is None else _parse_utc_timestamp(finished_raw, field="finished_at")
    )
    if status in {"running", "success", "failed"} and started is None:
        raise ArtifactIntegrityError(f"{status} manifest is missing started_at")
    if status in {"success", "failed"} and finished is None:
        raise ArtifactIntegrityError(f"{status} manifest is missing finished_at")
    ordered = [value for value in (created, started, finished, updated) if value is not None]
    if ordered != sorted(ordered):
        raise ArtifactIntegrityError("manifest timestamps are not monotonic")

    git = manifest.get("git")
    if not isinstance(git, Mapping):
        raise ArtifactIntegrityError("manifest git metadata must be an object")
    if require_clean_git:
        commit = git.get("commit")
        if (
            git.get("available") is not True
            or git.get("dirty") is not False
            or not isinstance(commit, str)
            or not _GIT_COMMIT_PATTERN.fullmatch(commit)
        ):
            raise ArtifactIntegrityError(
                "clean-git gate requires an available commit and dirty=false"
            )
    return run_id, str(status), git


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ArtifactIntegrityError(
            f"{context} fields differ from schema; missing={missing}, extra={extra}"
        )


def _verify_reproducibility_envelope(
    manifest: Mapping[str, Any],
    *,
    run_id: str,
) -> None:
    """Verify the optional extension while accepting pre-extension legacy runs."""

    raw = manifest.get("reproducibility")
    experiment = manifest.get("experiment")
    if raw is None:
        if experiment is not None:
            raise ArtifactIntegrityError(
                "manifest experiment identity requires reproducibility metadata"
            )
        return
    if not isinstance(raw, Mapping):
        raise ArtifactIntegrityError("manifest reproducibility metadata must be an object")
    _require_exact_keys(
        raw,
        {"schema_version", "configuration", "dependency_locks", "hardware"},
        context="manifest reproducibility metadata",
    )
    if raw.get("schema_version") != REPRODUCIBILITY_SCHEMA_VERSION:
        raise ArtifactIntegrityError(
            "unsupported reproducibility schema "
            f"{raw.get('schema_version')!r}; expected {REPRODUCIBILITY_SCHEMA_VERSION!r}"
        )

    invocation = manifest.get("invocation")
    if not isinstance(invocation, Mapping):
        raise ArtifactIntegrityError("manifest invocation must be an object")
    config = invocation.get("config")
    if not isinstance(config, Mapping):
        raise ArtifactIntegrityError("manifest invocation config must be an object")
    configuration = raw.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ArtifactIntegrityError("manifest reproducibility configuration must be an object")
    try:
        expected_configuration = configuration_identity(config)
    except (TypeError, ValueError) as error:
        raise ArtifactIntegrityError(
            f"manifest invocation config is not canonicalizable: {error}"
        ) from error
    if dict(configuration) != expected_configuration:
        raise ArtifactIntegrityError(
            "manifest configuration identity does not reproduce from invocation config"
        )

    locks = raw.get("dependency_locks")
    if not isinstance(locks, list):
        raise ArtifactIntegrityError("manifest dependency_locks must be an array")
    prior_path: str | None = None
    seen_paths: set[str] = set()
    for index, lock in enumerate(locks):
        context = f"manifest dependency_locks[{index}]"
        if not isinstance(lock, Mapping):
            raise ArtifactIntegrityError(f"{context} must be an object")
        _require_exact_keys(
            lock,
            {"relative_path", "sha256", "size_bytes"},
            context=context,
        )
        relative = lock.get("relative_path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in Path(relative).parts)
        ):
            raise ArtifactIntegrityError(f"{context} has an unsafe relative path")
        if relative in seen_paths:
            raise ArtifactIntegrityError(f"duplicate dependency lock path: {relative}")
        if prior_path is not None and relative < prior_path:
            raise ArtifactIntegrityError("dependency lock records must be path-sorted")
        _validate_fingerprint_fields(lock, context=context)
        seen_paths.add(relative)
        prior_path = relative

    hardware = raw.get("hardware")
    if not isinstance(hardware, Mapping):
        raise ArtifactIntegrityError("manifest hardware metadata must be an object")
    _require_exact_keys(
        hardware,
        {"cpu", "memory", "accelerators", "tools", "container"},
        context="manifest hardware metadata",
    )
    cpu = hardware.get("cpu")
    memory = hardware.get("memory")
    accelerators = hardware.get("accelerators")
    tools = hardware.get("tools")
    container = hardware.get("container")
    if not all(
        isinstance(value, Mapping) for value in (cpu, memory, accelerators, tools, container)
    ):
        raise ArtifactIntegrityError("manifest hardware subsections must be objects")
    _require_exact_keys(
        cpu,
        {"architecture", "model", "logical_count"},
        context="manifest CPU metadata",
    )
    _require_exact_keys(
        memory,
        {"total_bytes"},
        context="manifest memory metadata",
    )
    _require_exact_keys(
        accelerators,
        {
            "available",
            "version",
            "cuda",
            "cudnn",
            "mps",
            "rocm_version",
            "determinism",
        },
        context="manifest accelerator metadata",
    )
    _require_exact_keys(
        tools,
        {"python_compiler", "git", "uv"},
        context="manifest tool metadata",
    )
    _require_exact_keys(
        container,
        {"detected", "image_digest"},
        context="manifest container metadata",
    )
    cuda = accelerators.get("cuda")
    cudnn = accelerators.get("cudnn")
    mps = accelerators.get("mps")
    determinism = accelerators.get("determinism")
    if not all(isinstance(value, Mapping) for value in (cuda, cudnn, mps, determinism)):
        raise ArtifactIntegrityError("manifest accelerator subsections must be objects")
    _require_exact_keys(
        cuda,
        {"available", "runtime_version", "device_count", "devices"},
        context="manifest CUDA metadata",
    )
    _require_exact_keys(
        cudnn,
        {"available", "version"},
        context="manifest cuDNN metadata",
    )
    _require_exact_keys(
        mps,
        {"available", "built"},
        context="manifest MPS metadata",
    )
    _require_exact_keys(
        determinism,
        {
            "deterministic_algorithms",
            "cudnn_benchmark",
            "cudnn_deterministic",
        },
        context="manifest deterministic backend metadata",
    )
    devices = cuda.get("devices")
    if not isinstance(devices, list):
        raise ArtifactIntegrityError("manifest CUDA devices must be an array")
    for index, device in enumerate(devices):
        if not isinstance(device, Mapping):
            raise ArtifactIntegrityError(f"manifest CUDA device {index} must be an object")
        _require_exact_keys(
            device,
            {"index", "name"},
            context=f"manifest CUDA device {index}",
        )
    logical_count = cpu.get("logical_count")
    total_memory = memory.get("total_bytes")
    if logical_count is not None and (
        isinstance(logical_count, bool) or not isinstance(logical_count, int) or logical_count <= 0
    ):
        raise ArtifactIntegrityError("manifest CPU logical_count is invalid")
    if total_memory is not None and (
        isinstance(total_memory, bool) or not isinstance(total_memory, int) or total_memory <= 0
    ):
        raise ArtifactIntegrityError("manifest memory total_bytes is invalid")

    if experiment is None:
        return
    if not isinstance(experiment, Mapping):
        raise ArtifactIntegrityError("manifest experiment identity must be an object or null")
    _require_exact_keys(
        experiment,
        {"canonical_id", "stage", "model", "master_seed"},
        context="manifest experiment identity",
    )
    stage = experiment.get("stage")
    model = experiment.get("model")
    seed = experiment.get("master_seed")
    if not isinstance(stage, str) or not isinstance(model, str):
        raise ArtifactIntegrityError("manifest experiment stage/model are invalid")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ArtifactIntegrityError("manifest experiment master_seed is invalid")
    timestamps = manifest.get("timestamps")
    if not isinstance(timestamps, Mapping):
        raise ArtifactIntegrityError("manifest timestamps must be an object")
    created_at = _parse_utc_timestamp(
        timestamps.get("created_at"),
        field="created_at",
    )
    try:
        expected_id = canonical_experiment_id(
            started_at=created_at,
            stage=stage,
            model=model,
            config=config,
            seed=seed,
        )
    except (TypeError, ValueError) as error:
        raise ArtifactIntegrityError(f"manifest experiment identity is invalid: {error}") from error
    if experiment.get("canonical_id") != expected_id or run_id != expected_id:
        raise ArtifactIntegrityError(
            "manifest canonical experiment ID does not reproduce from its inputs"
        )


def _verify_tracker_artifacts(
    root: Path,
    manifest: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], set[str], Counter[str], int]:
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise ArtifactIntegrityError("manifest artifacts must be an array")
    artifacts: list[Mapping[str, Any]] = []
    paths: set[str] = set()
    roles: Counter[str] = Counter()
    total_bytes = 0
    prior_path: str | None = None
    for index, raw in enumerate(raw_artifacts):
        context = f"artifact {index}"
        if not isinstance(raw, Mapping):
            raise ArtifactIntegrityError(f"{context} must be an object")
        relative = raw.get("path")
        role = raw.get("role")
        if not isinstance(relative, str) or not relative:
            raise ArtifactIntegrityError(f"{context} path is missing")
        if relative == "manifest.json":
            raise ArtifactIntegrityError("manifest.json cannot register itself")
        if relative in paths:
            raise ArtifactIntegrityError(f"duplicate artifact path: {relative}")
        if prior_path is not None and relative < prior_path:
            raise ArtifactIntegrityError("artifact entries must be sorted by path")
        if not isinstance(role, str) or not role.strip():
            raise ArtifactIntegrityError(f"{context} role is missing")
        expected_digest, expected_size = _validate_fingerprint_fields(raw, context=context)
        artifact_path = _safe_relative_file(root, relative, context=context)
        actual_digest, actual_size = _fingerprint(artifact_path, context=relative)
        if (actual_digest, actual_size) != (expected_digest, expected_size):
            raise ArtifactIntegrityError(f"artifact fingerprint mismatch: {relative}")
        mime_type = raw.get("mime_type")
        registered_at = raw.get("registered_at")
        metadata = raw.get("metadata")
        if not isinstance(mime_type, str) or not mime_type:
            raise ArtifactIntegrityError(f"{context} mime_type is missing")
        _parse_utc_timestamp(registered_at, field=f"artifacts[{index}].registered_at")
        if not isinstance(metadata, Mapping):
            raise ArtifactIntegrityError(f"{context} metadata must be an object")
        artifacts.append(raw)
        paths.add(relative)
        roles[role] += 1
        total_bytes += expected_size
        prior_path = relative
    return artifacts, paths, roles, total_bytes


def _verify_checksum_index(
    root: Path,
    relative: str,
) -> tuple[set[str], int]:
    path = _safe_relative_file(root, relative, context="checksum index")
    try:
        payload = path.read_bytes()
        text = payload.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise ArtifactIntegrityError(
            f"could not read checksum index {relative}: {error}"
        ) from error
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise ArtifactIntegrityError(
            f"checksum index must be non-empty UTF-8 with LF termination: {relative}"
        )
    entries: set[str] = set()
    ordered_paths: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        digest, separator, entry_relative = line.partition("  ")
        context = f"{relative} line {line_number}"
        if (
            not separator
            or not _SHA256_PATTERN.fullmatch(digest)
            or not entry_relative
            or entry_relative == relative
        ):
            raise ArtifactIntegrityError(f"invalid checksum entry at {context}")
        if entry_relative in entries:
            raise ArtifactIntegrityError(f"duplicate checksum path: {entry_relative}")
        entry_path = _safe_relative_file(root, entry_relative, context=context)
        actual_digest, _ = _fingerprint(entry_path, context=entry_relative)
        if actual_digest != digest:
            raise ArtifactIntegrityError(f"checksum mismatch: {entry_relative}")
        entries.add(entry_relative)
        ordered_paths.append(entry_relative)
    if ordered_paths != sorted(ordered_paths, key=lambda value: value.encode("utf-8")):
        raise ArtifactIntegrityError(f"checksum index paths are not UTF-8 sorted: {relative}")
    return entries, len(entries)


def _fingerprinted_references(
    value: Any,
    *,
    base: Path,
) -> Iterator[tuple[Mapping[str, Any], Path]]:
    if isinstance(value, Mapping):
        keys = value.keys()
        if {"path", "sha256", "size_bytes"} <= keys:
            yield value, base
            child_base = base
        else:
            child_base = base
            path_value = value.get("path")
            if isinstance(path_value, str) and path_value:
                candidate = Path(path_value).expanduser()
                if not candidate.is_absolute():
                    candidate = base / candidate
                try:
                    resolved = candidate.resolve(strict=True)
                except OSError:
                    resolved = None
                if resolved is not None and resolved.is_dir():
                    child_base = resolved
        for child in value.values():
            yield from _fingerprinted_references(child, base=child_base)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _fingerprinted_references(child, base=base)


def _verify_external_references(
    root: Path,
    manifest: Mapping[str, Any],
) -> int:
    references = manifest.get("references")
    if not isinstance(references, Mapping):
        raise ArtifactIntegrityError("manifest references must be an object")
    count = 0
    seen: set[tuple[str, str, int]] = set()
    for raw, reference_base in _fingerprinted_references(references, base=root):
        path_value = raw.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ArtifactIntegrityError("fingerprinted external reference path is invalid")
        expected_digest, expected_size = _validate_fingerprint_fields(
            raw,
            context=f"external reference {path_value!r}",
        )
        reference_path = Path(path_value).expanduser()
        if not reference_path.is_absolute():
            reference_path = reference_base / reference_path
        try:
            resolved = reference_path.resolve(strict=True)
        except OSError as error:
            raise ArtifactIntegrityError(f"external reference is missing: {path_value}") from error
        if not resolved.is_file():
            raise ArtifactIntegrityError(f"external reference is not a regular file: {path_value}")
        key = (str(resolved), expected_digest, expected_size)
        if key in seen:
            continue
        actual_digest, actual_size = _fingerprint(resolved, context=path_value)
        if (actual_digest, actual_size) != (expected_digest, expected_size):
            raise ArtifactIntegrityError(f"external reference fingerprint mismatch: {path_value}")
        seen.add(key)
        count += 1
    return count


def _verify_deep(
    root: Path,
    roles: Mapping[str, int],
) -> Mapping[str, Any]:
    role_names = set(roles)
    if "dataset_manifest" in role_names:
        from .dataset.verified import load_verified_dataset

        dataset = load_verified_dataset(root)
        return {
            "kind": "dataset",
            "dataset_id": dataset.dataset_id,
            "sample_count": dataset.sample_count,
            "annotation_count": int(dataset.dataset["annotation_count"]),
            "partitions": dict(dataset.partitions),
        }
    if _SCENARIO_ROLES <= role_names:
        from .scenarios.verified_plan import load_verified_scenario_plan

        plan = load_verified_scenario_plan(root)
        return {
            "kind": "scenario_plan",
            "suite_id": plan.suite.suite_id,
            "episode_count": len(plan.episodes),
            "planned_capture_count": int(plan.summary["planned_capture_count"]),
        }
    if _NATIVE_PREFLIGHT_ROLES <= role_names:
        from .native.verified_preflight import load_verified_native_preflight

        preflight = load_verified_native_preflight(root)
        return {
            "kind": "native_collection_preflight",
            "run_id": preflight.run_id,
            "ready_for_native_execution": bool(preflight.summary["ready_for_native_execution"]),
            "selected_episode_count": int(preflight.summary["selection"]["selected_episode_count"]),
            "planned_capture_count": int(preflight.summary["selection"]["planned_capture_count"]),
        }
    if _NATIVE_HOST_KIT_ROLES <= role_names:
        from .native.verified_host_kit import load_verified_native_host_kit

        kit = load_verified_native_host_kit(root)
        return {
            "kind": "native_host_kit",
            "kit_id": kit.kit_id,
            "dataset_id": str(kit.descriptor["dataset"]["dataset_id"]),
            "planned_capture_count": int(kit.descriptor["selection"]["planned_capture_count"]),
            "payload_file_count": int(kit.descriptor["payload_file_count"]),
            "payload_source_file_count": int(kit.descriptor["payload_source_file_count"]),
            "simulator_mutated": bool(kit.descriptor["generation"]["simulator_mutated"]),
        }
    if "model_release_manifest" in role_names:
        from .model_release.verified import load_verified_model

        model = load_verified_model(root)
        return {
            "kind": "model",
            "model_id": model.model_id,
            "architecture": model.architecture,
            "backend": model.backend,
            "image_size": model.image_size,
        }
    if _EVALUATION_ROLES <= role_names:
        from .evaluation.verified import load_verified_evaluation

        evaluation = load_verified_evaluation(root)
        return {
            "kind": "detector_evaluation",
            "evaluation_id": evaluation.config.evaluation_id,
            "purpose": evaluation.config.purpose,
            "sample_count": len(evaluation.frame_rows),
            "prediction_count": len(evaluation.predictions),
            "dataset_id": evaluation.dataset.dataset_id,
        }
    if "threshold_selection_release_manifest" in role_names:
        from .thresholds.verified import load_verified_threshold_selection

        selection = load_verified_threshold_selection(root)
        return {
            "kind": "threshold_selection",
            "selection_id": selection.selection_id,
            "purpose": selection.config.purpose,
            "objective": selection.config.objective,
            "operating_confidence": selection.operating_confidence,
            "source_evaluation_id": str(selection.descriptor["source_evaluation"]["run_id"]),
        }
    if "failure_mining_release_manifest" in role_names:
        from .failure_mining.verified import load_verified_failure_mining

        mining = load_verified_failure_mining(root)
        return {
            "kind": "failure_mining",
            "mining_id": mining.mining_id,
            "purpose": mining.config.purpose,
            "all_failure_count": len(mining.all_failures),
            "review_queue_count": len(mining.review_queue),
            "source_evaluation_id": str(mining.descriptor["source_evaluation"]["run_id"]),
        }
    if "failure_review_release_manifest" in role_names:
        from .failure_mining.review_verified import (
            load_verified_failure_review,
        )

        review = load_verified_failure_review(root)
        return {
            "kind": "failure_review",
            "review_id": review.review_id,
            "purpose": review.config.purpose,
            "reviewed_count": len(review.reviews),
            "confirmed_count": len(review.confirmed),
            "label_issue_count": len(review.label_issues),
            "source_mining_id": str(review.descriptor["source_mining"]["mining_id"]),
        }
    if "reproduction_bundle_release_manifest" in role_names:
        from .reproduction.verified import load_verified_reproduction_bundle

        bundle = load_verified_reproduction_bundle(root)
        return {
            "kind": "reproduction_bundle",
            "bundle_id": bundle.bundle_id,
            "purpose": bundle.config.purpose,
            "source_count": int(bundle.descriptor["source_count"]),
            "source_file_count": int(bundle.descriptor["source_file_count"]),
            "dependency_lock_count": int(bundle.descriptor["dependency_lock_count"]),
        }
    if _VISION_SHADOW_ROLES <= role_names:
        from .policy.verified import load_verified_vision_shadow_run

        shadow = load_verified_vision_shadow_run(root)
        return {
            "kind": "vision_shadow_run",
            "run_id": shadow.run_id,
            "policy": str(shadow.input_audit["policy_name"]),
            "proposal_count": len(shadow.proposal_records),
            "actuation_applied": False,
        }
    if _SHADOW_MATRIX_ROLES <= role_names:
        from .shadow.verified import load_verified_shadow_matrix

        matrix = load_verified_shadow_matrix(root)
        return {
            "kind": "vision_shadow_matrix",
            "matrix_id": matrix.config.matrix_id,
            "mode": str(matrix.descriptor["mode"]),
            "planned_run_count": len(matrix.plan_rows),
            "successful_run_count": int(matrix.descriptor["successful_run_count"]),
            "actuation_applied": False,
        }
    if "report_release_manifest" in role_names:
        from .reporting.verified import load_verified_report

        report = load_verified_report(root)
        return {
            "kind": "report",
            "report_id": report.report_id,
            "purpose": report.config.purpose,
            "source_count": int(report.descriptor["source_count"]),
            "metric_row_count": int(report.descriptor["metric_row_count"]),
        }
    if "replay_release_manifest" in role_names:
        from .replay.verified import load_verified_replay

        replay = load_verified_replay(root)
        return {
            "kind": "paired_replay",
            "replay_id": replay.replay_id,
            "purpose": replay.config.purpose,
            "model_count": int(replay.descriptor["model_count"]),
            "sample_count": int(replay.descriptor["sample_count"]),
            "reference_model_id": replay.config.reference_model_id,
        }
    return {"kind": "tracked_run"}


def _unregistered_files(root: Path, expected: set[str]) -> list[str]:
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ArtifactIntegrityError(
                f"object tree contains a symlink: {path.relative_to(root).as_posix()}"
            )
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    return sorted(actual - expected)


def verify_research_object(
    path: str | Path,
    *,
    verify_references: bool = True,
    deep: bool = True,
    reject_unregistered: bool = False,
    require_clean_git: bool = False,
    allow_non_success: bool = False,
) -> VerificationResult:
    """Verify one tracked research object without modifying it."""

    root, manifest_path = _resolve_root_and_manifest(path)
    manifest = _load_json_object(manifest_path, "tracker manifest")
    run_id, status, git = _verify_manifest_envelope(
        manifest,
        allow_non_success=allow_non_success,
        require_clean_git=require_clean_git,
    )
    _verify_reproducibility_envelope(manifest, run_id=run_id)
    artifacts, registered_paths, role_counts, artifact_bytes = _verify_tracker_artifacts(
        root,
        manifest,
    )
    expected_paths = {"manifest.json", *registered_paths}
    checksum_index_count = 0
    checksum_index_entries = 0
    for artifact in artifacts:
        role = str(artifact["role"])
        relative = str(artifact["path"])
        if relative == "checksums.sha256" or "checksum_index" in role:
            entries, count = _verify_checksum_index(root, relative)
            expected_paths.update(entries)
            checksum_index_count += 1
            checksum_index_entries += count
    external_reference_count = (
        _verify_external_references(root, manifest) if verify_references else 0
    )
    deep_result = _verify_deep(root, role_counts) if deep else {"kind": "skipped"}
    extras = _unregistered_files(root, expected_paths)
    if reject_unregistered and extras:
        preview = ", ".join(extras[:5])
        suffix = "" if len(extras) <= 5 else f" (+{len(extras) - 5} more)"
        raise ArtifactIntegrityError(f"unregistered files found: {preview}{suffix}")
    manifest_reference = fingerprint_file(manifest_path)
    return VerificationResult(
        root=str(root),
        manifest=manifest_reference,
        run_id=run_id,
        status=status,
        schema_version=str(manifest["schema_version"]),
        artifact_count=len(artifacts),
        artifact_bytes=artifact_bytes,
        role_counts=dict(sorted(role_counts.items())),
        checksum_index_count=checksum_index_count,
        checksum_index_entries=checksum_index_entries,
        external_reference_count=external_reference_count,
        deep_verification=dict(deep_result),
        git=dict(git),
        unregistered_file_count=len(extras),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only verification of tracked CARLA research runs, datasets, "
            "scenario plans, model packages, paired replays, and reports"
        )
    )
    parser.add_argument("paths", nargs="+", help="object directory or manifest.json")
    parser.add_argument(
        "--no-references",
        action="store_true",
        help="do not verify fingerprinted files referenced outside the object",
    )
    parser.add_argument(
        "--no-deep",
        action="store_true",
        help="skip dataset/scenario semantic verification",
    )
    parser.add_argument(
        "--reject-unregistered",
        action="store_true",
        help="fail if any file is absent from artifacts or a tracked checksum index",
    )
    parser.add_argument(
        "--require-clean-git",
        action="store_true",
        help="require a recorded commit and dirty=false",
    )
    parser.add_argument(
        "--allow-non-success",
        action="store_true",
        help="verify retained created/running/failed objects as non-publishable evidence",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        results = [
            verify_research_object(
                path,
                verify_references=not args.no_references,
                deep=not args.no_deep,
                reject_unregistered=args.reject_unregistered,
                require_clean_git=args.require_clean_git,
                allow_non_success=args.allow_non_success,
            ).as_dict()
            for path in args.paths
        ]
    except ArtifactIntegrityError as error:
        parser.error(str(error))
    payload: Any = results[0] if len(results) == 1 else results
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ArtifactIntegrityError",
    "VerificationResult",
    "build_argument_parser",
    "main",
    "verify_research_object",
]


if __name__ == "__main__":
    raise SystemExit(main())
