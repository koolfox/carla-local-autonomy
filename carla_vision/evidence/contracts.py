"""Strict configuration contract for read-only workspace evidence registries."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Self

EVIDENCE_REGISTRY_SCHEMA_VERSION = "1.0"
EVIDENCE_REGISTRY_OBJECT_TYPE = "evidence_registry"
CANONICAL_EVIDENCE_ROOTS = (
    "datasets",
    "runs",
    "models",
    "reports",
    "bundles",
    "native_kits",
    "operator_sessions",
)

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]{0,127}$")


def _strict_keys(raw: Mapping[str, Any], required: set[str], name: str) -> None:
    keys = {str(key) for key in raw}
    missing = sorted(required - keys)
    unknown = sorted(keys - required)
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(unknown)}")


def _string_array(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _validate_relative_paths(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    for value in values:
        path = PurePosixPath(value)
        if path.is_absolute() or value in {"", ".", ".."} or ".." in path.parts:
            raise ValueError(f"{name} contains an unsafe relative path: {value!r}")
        if "\\" in value or path.as_posix() != value:
            raise ValueError(f"{name} paths must use normalized forward slashes")
    if values != tuple(sorted(values, key=lambda item: item.encode("utf-8"))):
        raise ValueError(f"{name} must be UTF-8 sorted")
    return values


def _validate_canonical_children(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    _validate_relative_paths(values, name)
    for value in values:
        parts = PurePosixPath(value).parts
        if len(parts) != 2 or parts[0] not in CANONICAL_EVIDENCE_ROOTS:
            raise ValueError(
                f"{name} must contain immediate children of canonical roots: {value!r}"
            )
    return values


@dataclass(frozen=True)
class EvidenceRegistryConfig:
    """Resolved, immutable scope for one point-in-time evidence scan."""

    schema_version: str
    object_type: str
    registry_id: str
    workspace_root: str
    canonical_roots: tuple[str, ...]
    exclude_registry_objects: bool
    source_paths: tuple[str, ...]
    excluded_registry_paths: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        required = {
            "schema_version",
            "object_type",
            "registry_id",
            "workspace_root",
            "canonical_roots",
            "exclude_registry_objects",
            "source_paths",
            "excluded_registry_paths",
        }
        _strict_keys(raw, required, "evidence registry config")
        schema_version = str(raw["schema_version"])
        if schema_version != EVIDENCE_REGISTRY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported evidence registry schema {schema_version!r}; "
                f"expected {EVIDENCE_REGISTRY_SCHEMA_VERSION!r}"
            )
        object_type = str(raw["object_type"])
        if object_type != EVIDENCE_REGISTRY_OBJECT_TYPE:
            raise ValueError("evidence registry object_type is invalid")
        registry_id = str(raw["registry_id"])
        if not _ID_PATTERN.fullmatch(registry_id):
            raise ValueError(
                "registry_id must be lowercase and use only letters, digits, '.' or '-'"
            )
        workspace_root = str(raw["workspace_root"]).strip()
        if not workspace_root or not Path(workspace_root).is_absolute():
            raise ValueError("workspace_root must be an absolute path")
        canonical_roots = _string_array(raw["canonical_roots"], "canonical_roots")
        if canonical_roots != CANONICAL_EVIDENCE_ROOTS:
            raise ValueError("canonical_roots must match the frozen evidence-root order")
        if raw["exclude_registry_objects"] is not True:
            raise ValueError("exclude_registry_objects must be true")
        source_paths = _validate_canonical_children(
            _string_array(raw["source_paths"], "source_paths"),
            "source_paths",
        )
        excluded = _validate_canonical_children(
            _string_array(raw["excluded_registry_paths"], "excluded_registry_paths"),
            "excluded_registry_paths",
        )
        overlap = sorted(set(source_paths) & set(excluded))
        if overlap:
            raise ValueError(
                "source_paths and excluded_registry_paths overlap: " + ", ".join(overlap)
            )
        return cls(
            schema_version=schema_version,
            object_type=object_type,
            registry_id=registry_id,
            workspace_root=workspace_root,
            canonical_roots=canonical_roots,
            exclude_registry_objects=True,
            source_paths=source_paths,
            excluded_registry_paths=excluded,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "object_type": self.object_type,
            "registry_id": self.registry_id,
            "workspace_root": self.workspace_root,
            "canonical_roots": list(self.canonical_roots),
            "exclude_registry_objects": self.exclude_registry_objects,
            "source_paths": list(self.source_paths),
            "excluded_registry_paths": list(self.excluded_registry_paths),
        }


__all__ = [
    "CANONICAL_EVIDENCE_ROOTS",
    "EVIDENCE_REGISTRY_OBJECT_TYPE",
    "EVIDENCE_REGISTRY_SCHEMA_VERSION",
    "EvidenceRegistryConfig",
]
