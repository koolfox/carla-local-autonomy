"""Manifest-backed model discovery for external perception and driving models.

A checkpoint file is intentionally not treated as a runnable model by itself.
The registry only advertises workspace-contained model packages with an explicit
``model.json`` contract describing role, runtime, inputs, outputs and artifact.
No model code is imported and no PyTorch checkpoint is deserialized during
registry discovery.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

MODEL_REGISTRY_SCHEMA_VERSION = "1.0"
MODEL_MANIFEST_SCHEMA_VERSION = "1.0"
MODEL_ROLES = frozenset({"detector", "driving_policy", "scene_perception", "perception_guard"})
MODEL_RUNTIMES = frozenset({"ultralytics", "python_factory", "torchscript_control_v1"})
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FACTORY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")
_ALLOWED_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "id",
        "name",
        "version",
        "role",
        "runtime",
        "artifact",
        "sha256",
        "factory",
        "devices",
        "inputs",
        "outputs",
        "labels",
        "source",
    }
)
_REQUIRED_MANIFEST_KEYS = frozenset(
    {"schema_version", "id", "name", "version", "role", "runtime", "artifact"}
)


def _safe_relative_path(raw: Any, name: str) -> str:
    if not isinstance(raw, str):
        raise TypeError(f"{name} must be a string")
    value = raw.strip()
    if not value or "\\" in value or "\x00" in value or "\r" in value or "\n" in value:
        raise ValueError(f"{name} must be a safe forward-slash relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be a safe relative path")
    if any(":" in part for part in path.parts):
        raise ValueError(f"{name} must not contain drive or stream separators")
    return path.as_posix()


def _text(raw: Any, name: str, *, maximum: int = 256) -> str:
    if not isinstance(raw, str):
        raise TypeError(f"{name} must be a string")
    value = raw.strip()
    if not value or len(value) > maximum:
        raise ValueError(f"{name} must contain 1-{maximum} characters")
    return value


def _mapping(raw: Any, name: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"{name} must be an object")
    return {str(key): value for key, value in raw.items()}


def _string_list(raw: Any, name: str) -> tuple[str, ...]:
    if raw is None:
        return ("cpu",)
    if not isinstance(raw, list) or not raw:
        raise TypeError(f"{name} must be a non-empty array of strings")
    values: list[str] = []
    for index, item in enumerate(raw):
        value = _text(item, f"{name}[{index}]", maximum=64)
        if value not in values:
            values.append(value)
    return tuple(values)


@dataclass(frozen=True)
class ModelPackage:
    """Validated model package metadata without loading executable model code."""

    package_id: str
    name: str
    version: str
    role: str
    runtime: str
    artifact: str
    expected_sha256: str | None
    factory: str | None
    devices: tuple[str, ...]
    inputs: Mapping[str, Any]
    outputs: Mapping[str, Any]
    labels: Any
    source: str | None
    manifest_path: str

    @property
    def requires_trusted_code(self) -> bool:
        return self.runtime == "python_factory"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.package_id,
            "name": self.name,
            "version": self.version,
            "role": self.role,
            "runtime": self.runtime,
            "artifact": self.artifact,
            "expectedSha256": self.expected_sha256,
            "factory": self.factory,
            "devices": list(self.devices),
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
            "labels": self.labels,
            "source": self.source,
            "manifestPath": self.manifest_path,
            "requiresTrustedCode": self.requires_trusted_code,
        }


@dataclass(frozen=True)
class ResolvedModelPackage:
    package: ModelPackage
    directory: Path
    artifact_path: Path

    def verify_artifact(self) -> str:
        digest = hashlib.sha256()
        with self.artifact_path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        actual = digest.hexdigest()
        expected = self.package.expected_sha256
        if expected is not None and actual != expected:
            raise ValueError(
                f"model package {self.package.package_id!r} artifact SHA-256 does not match manifest"
            )
        return actual


def _load_manifest(workspace: Path, directory: Path) -> ModelPackage:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("model package directory must be a regular non-symlink directory")
    manifest_path = directory / "model.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("model package requires a regular non-symlink model.json")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("model.json must be valid UTF-8 JSON") from error
    if not isinstance(payload, Mapping):
        raise TypeError("model.json must contain an object")
    payload = {str(key): value for key, value in payload.items()}
    unknown = sorted(set(payload) - _ALLOWED_MANIFEST_KEYS)
    missing = sorted(_REQUIRED_MANIFEST_KEYS - set(payload))
    if unknown:
        raise ValueError("model.json has unknown fields: " + ", ".join(unknown))
    if missing:
        raise ValueError("model.json is missing fields: " + ", ".join(missing))
    if payload["schema_version"] != MODEL_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"model.json schema_version must be {MODEL_MANIFEST_SCHEMA_VERSION!r}"
        )

    package_id = _text(payload["id"], "id", maximum=128)
    if not _MODEL_ID.fullmatch(package_id):
        raise ValueError("id must use 1-128 letters, digits, '.', '_' or '-'")
    name = _text(payload["name"], "name", maximum=160)
    version = _text(payload["version"], "version", maximum=80)
    role = _text(payload["role"], "role", maximum=64).lower()
    if role not in MODEL_ROLES:
        raise ValueError("role must be one of: " + ", ".join(sorted(MODEL_ROLES)))
    runtime = _text(payload["runtime"], "runtime", maximum=64).lower()
    if runtime not in MODEL_RUNTIMES:
        raise ValueError("runtime must be one of: " + ", ".join(sorted(MODEL_RUNTIMES)))

    artifact = _safe_relative_path(payload["artifact"], "artifact")
    artifact_path = directory.joinpath(*PurePosixPath(artifact).parts)
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise ValueError("artifact must resolve to a regular non-symlink file inside the package")
    artifact_path.resolve(strict=True).relative_to(directory.resolve(strict=True))

    expected_sha256 = payload.get("sha256")
    if expected_sha256 is not None:
        expected_sha256 = _text(expected_sha256, "sha256", maximum=64).lower()
        if not _SHA256.fullmatch(expected_sha256):
            raise ValueError("sha256 must contain exactly 64 lowercase hexadecimal characters")

    factory = payload.get("factory")
    if factory is not None:
        factory = _text(factory, "factory", maximum=256)
        if not _FACTORY.fullmatch(factory):
            raise ValueError("factory must use module:callable syntax")
    if runtime == "python_factory" and factory is None:
        raise ValueError("python_factory runtime requires factory")
    if runtime != "python_factory" and factory is not None:
        raise ValueError("factory is only valid for python_factory runtime")
    if runtime == "ultralytics" and role != "detector":
        raise ValueError("ultralytics runtime currently requires role=detector")
    if runtime == "torchscript_control_v1" and role != "driving_policy":
        raise ValueError("torchscript_control_v1 runtime requires role=driving_policy")
    if runtime in {"ultralytics", "torchscript_control_v1"} and artifact_path.suffix.lower() != ".pt":
        raise ValueError(f"{runtime} artifact must end in .pt")

    source_raw = payload.get("source")
    source = None if source_raw is None else _text(source_raw, "source", maximum=512)
    labels = payload.get("labels", {})
    if not isinstance(labels, (Mapping, list)):
        raise TypeError("labels must be an object or array when supplied")

    return ModelPackage(
        package_id=package_id,
        name=name,
        version=version,
        role=role,
        runtime=runtime,
        artifact=(artifact_path.relative_to(workspace)).as_posix(),
        expected_sha256=expected_sha256,
        factory=factory,
        devices=_string_list(payload.get("devices"), "devices"),
        inputs=_mapping(payload.get("inputs"), "inputs"),
        outputs=_mapping(payload.get("outputs"), "outputs"),
        labels=dict(labels) if isinstance(labels, Mapping) else list(labels),
        source=source,
        manifest_path=(manifest_path.relative_to(workspace)).as_posix(),
    )


def discover_model_packages(workspace: str | Path) -> dict[str, Any]:
    """Return valid model packages and explicit validation failures.

    Legacy checkpoint files without a manifest are intentionally ignored. They
    remain usable by legacy flows but are not advertised as first-class model
    packages.
    """

    root = Path(workspace).expanduser().resolve(strict=True)
    models_root = root / "models"
    if models_root.is_symlink() or not models_root.is_dir():
        return {"schema_version": MODEL_REGISTRY_SCHEMA_VERSION, "packages": [], "invalid": []}

    valid: list[ModelPackage] = []
    invalid: list[dict[str, str]] = []
    for directory in sorted(models_root.iterdir(), key=lambda path: path.name.casefold()):
        if directory.is_symlink() or not directory.is_dir():
            continue
        manifest = directory / "model.json"
        if not manifest.exists():
            continue
        try:
            valid.append(_load_manifest(root, directory))
        except Exception as error:
            invalid.append(
                {
                    "path": directory.relative_to(root).as_posix(),
                    "errorType": type(error).__qualname__,
                    "message": str(error),
                }
            )

    by_id: dict[str, list[ModelPackage]] = defaultdict(list)
    for package in valid:
        by_id[package.package_id].append(package)
    duplicates = {package_id for package_id, rows in by_id.items() if len(rows) > 1}
    if duplicates:
        kept: list[ModelPackage] = []
        for package in valid:
            if package.package_id not in duplicates:
                kept.append(package)
                continue
            invalid.append(
                {
                    "path": package.manifest_path.rsplit("/", maxsplit=1)[0],
                    "errorType": "ValueError",
                    "message": f"duplicate model package id: {package.package_id!r}",
                }
            )
        valid = kept

    return {
        "schema_version": MODEL_REGISTRY_SCHEMA_VERSION,
        "packages": [package.as_dict() for package in sorted(valid, key=lambda row: row.package_id)],
        "invalid": sorted(invalid, key=lambda row: row["path"].encode("utf-8")),
    }


def resolve_model_package(
    workspace: str | Path,
    package_id: str,
    *,
    required_role: str | None = None,
) -> ResolvedModelPackage:
    root = Path(workspace).expanduser().resolve(strict=True)
    registry = discover_model_packages(root)
    rows = [row for row in registry["packages"] if row["id"] == package_id]
    if not rows:
        raise KeyError(f"model package {package_id!r} is not registered")
    row = rows[0]
    if required_role is not None and row["role"] != required_role:
        raise ValueError(
            f"model package {package_id!r} has role {row['role']!r}, expected {required_role!r}"
        )
    manifest_path = root / str(row["manifestPath"])
    package = _load_manifest(root, manifest_path.parent)
    artifact = root / package.artifact
    return ResolvedModelPackage(package=package, directory=manifest_path.parent, artifact_path=artifact)


__all__ = [
    "MODEL_MANIFEST_SCHEMA_VERSION",
    "MODEL_REGISTRY_SCHEMA_VERSION",
    "MODEL_ROLES",
    "MODEL_RUNTIMES",
    "ModelPackage",
    "ResolvedModelPackage",
    "discover_model_packages",
    "resolve_model_package",
]
