"""Strict configuration contract for standalone reproduction bundles."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

REPRODUCTION_BUNDLE_CONFIG_SCHEMA_VERSION = "1.0"
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]{0,127}$")
_SAFE_SOURCE_PATH_PATTERN = re.compile(r"^[A-Za-z0-9._/-]+$")
REPRODUCTION_SOURCE_ROLES = frozenset(
    {
        "dataset",
        "dataset-qa",
        "evaluation",
        "failure-mining",
        "failure-review",
        "model",
        "native-host-kit",
        "native-preflight",
        "operator-session",
        "replay",
        "report",
        "runtime-analysis",
        "scenario-plan",
        "shadow-matrix",
        "threshold-selection",
        "training",
        "vision-shadow",
        "other",
    }
)


def _strict_keys(raw: Mapping[str, Any], required: set[str], name: str) -> None:
    keys = {str(key) for key in raw}
    missing = sorted(required - keys)
    unknown = sorted(keys - required)
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(unknown)}")


def _nonempty(value: Any, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must not be empty")
    if "\x00" in result:
        raise ValueError(f"{name} cannot contain NUL")
    return result


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    result = tuple(str(item).strip() for item in value)
    if not result or any(not item or "\x00" in item for item in result):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} values must be unique")
    return result


@dataclass(frozen=True)
class ReproductionSource:
    source_id: str
    role: str
    path: str
    label: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        _strict_keys(raw, {"source_id", "role", "path", "label"}, "reproduction source")
        source_id = _nonempty(raw["source_id"], "source.source_id")
        if not _ID_PATTERN.fullmatch(source_id):
            raise ValueError(
                "source.source_id must be lowercase and use letters, digits, '.' or '-'"
            )
        role = str(raw["role"]).strip().lower()
        if role not in REPRODUCTION_SOURCE_ROLES:
            raise ValueError(f"unsupported reproduction source role {role!r}")
        return cls(
            source_id=source_id,
            role=role,
            path=_nonempty(raw["path"], "source.path"),
            label=_nonempty(raw["label"], "source.label"),
        )


def _commands(value: Any) -> tuple[tuple[str, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("commands must be an array of token arrays")
    commands: list[tuple[str, ...]] = []
    for command_index, raw_command in enumerate(value):
        if isinstance(raw_command, (str, bytes)) or not isinstance(raw_command, Sequence):
            raise TypeError(f"commands[{command_index}] must be a token array")
        command = tuple(str(token) for token in raw_command)
        if not command:
            raise ValueError(f"commands[{command_index}] cannot be empty")
        if any(not token or "\n" in token or "\r" in token or "\x00" in token for token in command):
            raise ValueError(
                f"commands[{command_index}] tokens must be non-empty single-line strings"
            )
        commands.append(command)
    if not commands:
        raise ValueError("commands requires at least one command")
    return tuple(commands)


def _source_paths(value: Any) -> tuple[str, ...]:
    paths = _strings(value, "source_paths")
    for path in paths:
        candidate = Path(path)
        if (
            candidate.is_absolute()
            or "\\" in path
            or not _SAFE_SOURCE_PATH_PATTERN.fullmatch(path)
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            raise ValueError(f"source_paths contains an unsafe relative path: {path!r}")
    return paths


@dataclass(frozen=True)
class ReproductionBundleConfig:
    schema_version: str
    bundle_id: str
    title: str
    authors: tuple[str, ...]
    purpose: str
    repository_root: str
    sources: tuple[ReproductionSource, ...]
    source_paths: tuple[str, ...]
    commands: tuple[tuple[str, ...], ...]
    limitations: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        required = {
            "schema_version",
            "bundle_id",
            "title",
            "authors",
            "purpose",
            "repository_root",
            "sources",
            "source_paths",
            "commands",
            "limitations",
        }
        _strict_keys(raw, required, "reproduction bundle config")
        schema_version = str(raw["schema_version"])
        if schema_version != REPRODUCTION_BUNDLE_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported reproduction config schema {schema_version!r}; "
                f"expected {REPRODUCTION_BUNDLE_CONFIG_SCHEMA_VERSION!r}"
            )
        bundle_id = str(raw["bundle_id"]).strip()
        if not _ID_PATTERN.fullmatch(bundle_id):
            raise ValueError("bundle_id must be lowercase and use only letters, digits, '.' or '-'")
        purpose = str(raw["purpose"]).strip().lower()
        if purpose not in {"development", "confirmatory"}:
            raise ValueError("purpose must be development or confirmatory")
        raw_sources = raw["sources"]
        if isinstance(raw_sources, (str, bytes)) or not isinstance(raw_sources, Sequence):
            raise TypeError("sources must be an array")
        sources = tuple(
            ReproductionSource.from_mapping(item)
            if isinstance(item, Mapping)
            else (_ for _ in ()).throw(TypeError("reproduction source must be an object"))
            for item in raw_sources
        )
        if not sources:
            raise ValueError("reproduction bundle requires at least one source")
        source_ids = [source.source_id for source in sources]
        labels = [source.label for source in sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("reproduction source IDs must be unique")
        if len(labels) != len(set(labels)):
            raise ValueError("reproduction source labels must be unique")
        return cls(
            schema_version=schema_version,
            bundle_id=bundle_id,
            title=_nonempty(raw["title"], "title"),
            authors=_strings(raw["authors"], "authors"),
            purpose=purpose,
            repository_root=_nonempty(raw["repository_root"], "repository_root"),
            sources=sources,
            source_paths=_source_paths(raw["source_paths"]),
            commands=_commands(raw["commands"]),
            limitations=_strings(raw["limitations"], "limitations"),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_reproduction_bundle_config(path: str | Path) -> ReproductionBundleConfig:
    resolved = Path(path).expanduser().resolve(strict=True)
    with resolved.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("reproduction bundle config must contain a JSON object")
    return ReproductionBundleConfig.from_mapping(payload)


__all__ = [
    "REPRODUCTION_BUNDLE_CONFIG_SCHEMA_VERSION",
    "REPRODUCTION_SOURCE_ROLES",
    "ReproductionBundleConfig",
    "ReproductionSource",
    "load_reproduction_bundle_config",
]
