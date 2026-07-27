"""Strict pre-registration contract for manifest-driven research reports."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

REPORT_CONFIG_SCHEMA_VERSION = "1.0"
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]{0,127}$")
REPORT_SOURCE_KINDS = frozenset(
    {
        "dataset",
        "dataset_qa",
        "evaluation",
        "failure_mining",
        "failure_review",
        "model",
        "native_host_kit",
        "native_preflight",
        "operator_session",
        "paired_replay",
        "reproduction_bundle",
        "report",
        "runtime_analysis",
        "scenario_plan",
        "shadow_matrix",
        "threshold_selection",
        "training",
        "vision_shadow",
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
    return result


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    result = tuple(dict.fromkeys(str(item).strip() for item in value))
    if not result or any(not item for item in result):
        raise ValueError(f"{name} must contain non-empty unique strings")
    return result


@dataclass(frozen=True)
class ReportSource:
    path: str
    kind: str
    label: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        _strict_keys(raw, {"path", "kind", "label"}, "report source")
        kind = str(raw["kind"]).strip().lower()
        if kind not in REPORT_SOURCE_KINDS:
            raise ValueError(f"unsupported report source kind {kind!r}")
        return cls(
            path=_nonempty(raw["path"], "source.path"),
            kind=kind,
            label=_nonempty(raw["label"], "source.label"),
        )


@dataclass(frozen=True)
class ReportConfig:
    schema_version: str
    report_id: str
    title: str
    authors: tuple[str, ...]
    purpose: str
    thesis_context: str
    sources: tuple[ReportSource, ...]
    claims: tuple[str, ...]
    limitations: tuple[str, ...]
    include_plots: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        required = {
            "schema_version",
            "report_id",
            "title",
            "authors",
            "purpose",
            "thesis_context",
            "sources",
            "claims",
            "limitations",
            "include_plots",
        }
        _strict_keys(raw, required, "report config")
        schema_version = str(raw["schema_version"])
        if schema_version != REPORT_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported report config schema {schema_version!r}; "
                f"expected {REPORT_CONFIG_SCHEMA_VERSION!r}"
            )
        report_id = str(raw["report_id"]).strip()
        if not _ID_PATTERN.fullmatch(report_id):
            raise ValueError("report_id must be lowercase and use only letters, digits, '.' or '-'")
        purpose = str(raw["purpose"]).strip().lower()
        if purpose not in {"development", "confirmatory"}:
            raise ValueError("purpose must be development or confirmatory")
        sources_raw = raw["sources"]
        if isinstance(sources_raw, (str, bytes)) or not isinstance(sources_raw, Sequence):
            raise TypeError("sources must be an array")
        sources = tuple(
            ReportSource.from_mapping(source)
            if isinstance(source, Mapping)
            else (_ for _ in ()).throw(TypeError("report source must be an object"))
            for source in sources_raw
        )
        if not sources:
            raise ValueError("report requires at least one source")
        labels = [source.label for source in sources]
        if len(labels) != len(set(labels)):
            raise ValueError("report source labels must be unique")
        if not isinstance(raw["include_plots"], bool):
            raise TypeError("include_plots must be boolean")
        return cls(
            schema_version=schema_version,
            report_id=report_id,
            title=_nonempty(raw["title"], "title"),
            authors=_strings(raw["authors"], "authors"),
            purpose=purpose,
            thesis_context=_nonempty(raw["thesis_context"], "thesis_context"),
            sources=sources,
            claims=_strings(raw["claims"], "claims"),
            limitations=_strings(raw["limitations"], "limitations"),
            include_plots=raw["include_plots"],
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_report_config(path: str | Path) -> ReportConfig:
    resolved = Path(path).expanduser().resolve(strict=True)
    with resolved.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("report config must contain a JSON object")
    return ReportConfig.from_mapping(payload)


__all__ = [
    "REPORT_CONFIG_SCHEMA_VERSION",
    "REPORT_SOURCE_KINDS",
    "ReportConfig",
    "ReportSource",
    "load_report_config",
]
