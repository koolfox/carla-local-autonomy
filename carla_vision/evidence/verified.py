"""Independent semantic verification for sealed evidence registries."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..verification import ArtifactIntegrityError, verify_research_object
from .builder import (
    EVIDENCE_REGISTRY_RELEASE_TYPE,
    Discovery,
    SourceCandidate,
    _build_snapshot,
    _expected_payloads,
    _read_regular_file_nofollow,
    _role_for,
    _strict_json_loads,
    _validate_canonical_roots,
)
from .contracts import (
    EVIDENCE_REGISTRY_OBJECT_TYPE,
    EVIDENCE_REGISTRY_SCHEMA_VERSION,
    EvidenceRegistryConfig,
)


class EvidenceRegistryIntegrityError(RuntimeError):
    """Raised when a registry is corrupt, stale, or semantically inconsistent."""


@dataclass(frozen=True)
class VerifiedEvidenceRegistry:
    root: Path
    registry_id: str
    config: EvidenceRegistryConfig
    descriptor: Mapping[str, Any]
    summary: Mapping[str, Any]
    source_count: int
    verified_count: int
    verification_failed_count: int
    registered_artifact_count: int


def _resolve_root(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_symlink():
        raise EvidenceRegistryIntegrityError("evidence registry path must not be a symbolic link")
    if expanded.name == "manifest.json":
        expanded = expanded.parent
        if expanded.is_symlink():
            raise EvidenceRegistryIntegrityError(
                "evidence registry root must not be a symbolic link"
            )
    if expanded.parent.is_symlink():
        raise EvidenceRegistryIntegrityError(
            "evidence registry canonical root must not be a symbolic link"
        )
    try:
        resolved = expanded.resolve(strict=True)
    except OSError as error:
        raise EvidenceRegistryIntegrityError(f"evidence registry does not exist: {path}") from error
    if not resolved.is_dir():
        raise EvidenceRegistryIntegrityError("evidence registry path must be a directory")
    return resolved


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = _strict_json_loads(
            _read_regular_file_nofollow(path),
            name=name,
        )
    except (OSError, RuntimeError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise EvidenceRegistryIntegrityError(f"could not read {name}: {error}") from error
    if not isinstance(value, Mapping):
        raise EvidenceRegistryIntegrityError(f"{name} must contain an object")
    return value


def _validate_roles(root: Path, manifest: Mapping[str, Any]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise EvidenceRegistryIntegrityError("registry tracker artifacts must be an array")
    actual: list[tuple[str, str]] = []
    role_counts: Counter[str] = Counter()
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise EvidenceRegistryIntegrityError("registry tracker artifact is invalid")
        relative = artifact.get("path")
        role = artifact.get("role")
        if not isinstance(relative, str) or not isinstance(role, str):
            raise EvidenceRegistryIntegrityError("registry artifact path or role is invalid")
        actual.append((relative, role))
        role_counts[role] += 1
    expected_paths = (
        "checksums.sha256",
        "plots/root_inventory.png",
        "plots/root_inventory.svg",
        "plots/verification_status.png",
        "plots/verification_status.svg",
        "registry.json",
        "registry_config.json",
        "report.md",
        "summary.json",
        "tables/artifacts.csv",
        "tables/failed_source_tree.csv",
        "tables/objects.csv",
    )
    expected = sorted((path, _role_for(path)) for path in expected_paths)
    if sorted(actual) != expected:
        raise EvidenceRegistryIntegrityError("registry artifact paths or roles are incomplete")
    expected_role_counts = Counter(role for _, role in expected)
    if role_counts != expected_role_counts:
        raise EvidenceRegistryIntegrityError("registry artifact role counts are inconsistent")
    for relative, _ in actual:
        try:
            metadata = os.lstat(root / relative)
        except OSError as error:
            raise EvidenceRegistryIntegrityError(
                f"registry artifact is missing: {relative}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise EvidenceRegistryIntegrityError(
                f"registry artifact must be a non-symlink regular file: {relative}"
            )


def _compare_payloads(root: Path, expected: Mapping[str, bytes]) -> None:
    for relative in sorted(expected, key=lambda value: value.encode("utf-8")):
        path = root / relative
        try:
            actual = _read_regular_file_nofollow(path)
        except (OSError, RuntimeError) as error:
            raise EvidenceRegistryIntegrityError(
                f"registry payload is missing: {relative}"
            ) from error
        if actual != expected[relative]:
            raise EvidenceRegistryIntegrityError(
                f"registry payload does not reproduce from current workspace state: {relative}"
            )


def _recorded_discovery(
    workspace: Path,
    config: EvidenceRegistryConfig,
) -> Discovery:
    candidates = []
    for relative in config.source_paths:
        root_kind, _ = relative.split("/", 1)
        source_root = workspace / relative
        candidates.append(
            SourceCandidate(
                root_kind=root_kind,
                relative_path=relative,
                root=source_root,
                manifest_path=source_root / "manifest.json",
            )
        )
    return Discovery(
        candidates=tuple(candidates),
        excluded_registry_paths=config.excluded_registry_paths,
    )


def _manifest_inputs(manifest: Mapping[str, Any]) -> list[Any]:
    references = manifest.get("references")
    if not isinstance(references, Mapping):
        raise EvidenceRegistryIntegrityError("registry tracker references must be an object")
    inputs = references.get("inputs")
    if not isinstance(inputs, list):
        raise EvidenceRegistryIntegrityError("registry tracker input references must be an array")
    return inputs


def _preflight_source_references(
    references: list[Mapping[str, Any]],
    *,
    workspace: Path,
) -> None:
    for reference in references:
        path_value = reference.get("path")
        source_relative = reference.get("source_relative_path")
        expected_digest = reference.get("sha256")
        expected_size = reference.get("size_bytes")
        if (
            not isinstance(path_value, str)
            or not isinstance(source_relative, str)
            or not isinstance(expected_digest, str)
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
        ):
            raise EvidenceRegistryIntegrityError("recorded source-manifest reference is invalid")
        expected_path = workspace / source_relative / "manifest.json"
        if Path(path_value) != expected_path:
            raise EvidenceRegistryIntegrityError(
                "recorded source-manifest path left its canonical source"
            )
        try:
            source_metadata = os.lstat(expected_path.parent)
            if stat.S_ISLNK(source_metadata.st_mode) or not stat.S_ISDIR(source_metadata.st_mode):
                raise EvidenceRegistryIntegrityError(
                    "recorded source is no longer a non-symlink directory"
                )
            payload = _read_regular_file_nofollow(expected_path)
        except (OSError, RuntimeError) as error:
            raise EvidenceRegistryIntegrityError(
                "recorded source manifest is no longer a non-symlink regular file"
            ) from error
        if (
            hashlib.sha256(payload).hexdigest(),
            len(payload),
        ) != (expected_digest, expected_size):
            raise EvidenceRegistryIntegrityError(
                "recorded source-manifest fingerprint changed before verification"
            )


def verify_evidence_registry(path: str | Path) -> VerifiedEvidenceRegistry:
    """Verify hashes, source drift, discovery scope, tables, report, and plots."""

    root = _resolve_root(path)
    manifest = _load_json(root / "manifest.json", "registry tracker manifest")
    _validate_roles(root, manifest)
    try:
        config = EvidenceRegistryConfig.from_mapping(
            _load_json(root / "registry_config.json", "registry configuration")
        )
    except (TypeError, ValueError) as error:
        raise EvidenceRegistryIntegrityError(
            f"registry configuration is invalid: {error}"
        ) from error
    if root.name != config.registry_id:
        raise EvidenceRegistryIntegrityError("registry identity disagrees across package files")
    invocation = manifest.get("invocation")
    invocation_config = invocation.get("config") if isinstance(invocation, Mapping) else None
    if not isinstance(invocation_config, Mapping):
        raise EvidenceRegistryIntegrityError("registry tracker invocation is missing")
    if (
        invocation_config.get("schema_version") != EVIDENCE_REGISTRY_SCHEMA_VERSION
        or invocation_config.get("object_type") != EVIDENCE_REGISTRY_OBJECT_TYPE
        or invocation_config.get("registry") != config.as_dict()
    ):
        raise EvidenceRegistryIntegrityError("registry tracker configuration changed")

    workspace = Path(config.workspace_root)
    try:
        workspace = workspace.resolve(strict=True)
    except OSError as error:
        raise EvidenceRegistryIntegrityError("registry workspace is unavailable") from error
    if workspace != Path(config.workspace_root):
        raise EvidenceRegistryIntegrityError("registry workspace path is not canonical")
    try:
        _validate_canonical_roots(workspace)
    except RuntimeError as error:
        raise EvidenceRegistryIntegrityError(str(error)) from error
    expected_relative = f"runs/{config.registry_id}"
    try:
        actual_relative = root.relative_to(workspace).as_posix()
    except ValueError as error:
        raise EvidenceRegistryIntegrityError(
            "registry is outside its recorded workspace"
        ) from error
    if actual_relative != expected_relative:
        raise EvidenceRegistryIntegrityError("registry is not at runs/<registry_id>")

    recorded_discovery = _recorded_discovery(workspace, config)
    try:
        snapshot = _build_snapshot(
            workspace=workspace,
            registry_id=config.registry_id,
            discovery=recorded_discovery,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise EvidenceRegistryIntegrityError(
            f"could not reproduce recorded source snapshot: {error}"
        ) from error
    recorded_inputs = _manifest_inputs(manifest)
    expected_inputs = snapshot["input_references"]
    if recorded_inputs != expected_inputs:
        raise EvidenceRegistryIntegrityError("recorded source-manifest reference set changed")
    _preflight_source_references(expected_inputs, workspace=workspace)
    try:
        generic = verify_research_object(
            root,
            verify_references=True,
            deep=False,
            reject_unregistered=True,
        )
    except ArtifactIntegrityError as error:
        raise EvidenceRegistryIntegrityError(str(error)) from error
    if generic.run_id != config.registry_id:
        raise EvidenceRegistryIntegrityError("registry identity disagrees across package files")
    if generic.external_reference_count != len(expected_inputs):
        raise EvidenceRegistryIntegrityError(
            "recorded source-manifest fingerprint coverage changed"
        )
    expected_payloads, expected_descriptor = _expected_payloads(config, snapshot)
    _compare_payloads(root, expected_payloads)
    descriptor = _load_json(root / "registry.json", "registry release descriptor")
    if descriptor != expected_descriptor:
        raise EvidenceRegistryIntegrityError("registry release descriptor is inconsistent")
    if (
        descriptor.get("schema_version") != EVIDENCE_REGISTRY_SCHEMA_VERSION
        or descriptor.get("object_type") != EVIDENCE_REGISTRY_RELEASE_TYPE
        or descriptor.get("status") != "complete"
    ):
        raise EvidenceRegistryIntegrityError("registry release descriptor envelope is invalid")
    summary_payload = _load_json(root / "summary.json", "registry summary")
    summary = summary_payload.get("summary")
    if not isinstance(summary, Mapping):
        raise EvidenceRegistryIntegrityError("registry summary body is invalid")
    return VerifiedEvidenceRegistry(
        root=root,
        registry_id=config.registry_id,
        config=config,
        descriptor=descriptor,
        summary=dict(summary),
        source_count=int(summary["source_count"]),
        verified_count=int(summary["verified_count"]),
        verification_failed_count=int(summary["verification_failed_count"]),
        registered_artifact_count=int(summary["registered_artifact_count"]),
    )


__all__ = [
    "EvidenceRegistryIntegrityError",
    "VerifiedEvidenceRegistry",
    "verify_evidence_registry",
]
