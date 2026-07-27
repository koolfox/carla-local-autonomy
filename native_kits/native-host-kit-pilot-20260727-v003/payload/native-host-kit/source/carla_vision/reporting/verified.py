"""Consumer-side verification for manifest-driven report releases."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import fingerprint_file
from ..verification import ArtifactIntegrityError, verify_research_object
from .builder import REPORT_RELEASE_SCHEMA_VERSION
from .contracts import ReportConfig


class ReportIntegrityError(RuntimeError):
    """Raised when a report release or one of its sources is not trustworthy."""


@dataclass(frozen=True)
class VerifiedReport:
    root: Path
    report_id: str
    descriptor: Mapping[str, Any]
    config: ReportConfig
    reference: Mapping[str, Any]


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReportIntegrityError(f"could not read {name}: {error}") from error
    if not isinstance(value, Mapping):
        raise ReportIntegrityError(f"{name} must contain a JSON object")
    return value


def _artifact_path(root: Path, manifest: Mapping[str, Any], role: str) -> Path:
    matches = [
        entry
        for entry in manifest["artifacts"]
        if isinstance(entry, Mapping) and entry.get("role") == role
    ]
    if len(matches) != 1:
        raise ReportIntegrityError(f"report package must contain exactly one {role!r} artifact")
    return (root / str(matches[0]["path"])).resolve(strict=True)


def _csv_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ReportIntegrityError(f"report CSV has no header: {path.name}")
            return sum(1 for _ in reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ReportIntegrityError(f"could not read report CSV {path.name}: {error}") from error


def load_verified_report(path: str | Path) -> VerifiedReport:
    try:
        verification = verify_research_object(
            path,
            verify_references=True,
            deep=False,
            reject_unregistered=True,
        )
    except ArtifactIntegrityError as error:
        raise ReportIntegrityError(str(error)) from error
    root = Path(verification.root)
    manifest = _load_json(root / "manifest.json", "report tracker manifest")
    descriptor_path = _artifact_path(root, manifest, "report_release_manifest")
    config_path = _artifact_path(root, manifest, "report_configuration")
    metrics_path = _artifact_path(root, manifest, "report_metric_table")
    artifacts_path = _artifact_path(root, manifest, "report_source_artifact_table")
    descriptor = _load_json(descriptor_path, "report release manifest")
    config = ReportConfig.from_mapping(_load_json(config_path, "report configuration"))
    if (
        descriptor.get("schema_version") != REPORT_RELEASE_SCHEMA_VERSION
        or descriptor.get("object_type") != "research_report_release"
        or descriptor.get("status") != "complete"
    ):
        raise ReportIntegrityError("report release descriptor envelope is invalid")
    report_id = str(descriptor.get("report_id", ""))
    if report_id != verification.run_id or report_id != config.report_id:
        raise ReportIntegrityError("report identity disagrees across package files")
    sources = descriptor.get("sources")
    if not isinstance(sources, list) or descriptor.get("source_count") != len(sources):
        raise ReportIntegrityError("report source count is inconsistent")
    if descriptor.get("metric_row_count") != _csv_rows(metrics_path):
        raise ReportIntegrityError("report metric row count is inconsistent")
    if descriptor.get("artifact_inventory_row_count") != _csv_rows(artifacts_path):
        raise ReportIntegrityError("report source-artifact row count is inconsistent")
    for source in sources:
        if not isinstance(source, Mapping):
            raise ReportIntegrityError("report source descriptor must be an object")
        source_root = source.get("root")
        if not isinstance(source_root, str) or not source_root:
            raise ReportIntegrityError("report source root is missing")
        try:
            source_verification = verify_research_object(
                source_root,
                verify_references=True,
                deep=True,
                reject_unregistered=True,
                require_clean_git=config.purpose == "confirmatory",
            )
        except ArtifactIntegrityError as error:
            raise ReportIntegrityError(
                f"report source {source.get('run_id')!r} failed verification: {error}"
            ) from error
        if (
            source_verification.run_id != source.get("run_id")
            or source_verification.manifest["sha256"] != source.get("sha256")
            or source_verification.manifest["size_bytes"] != source.get("size_bytes")
        ):
            raise ReportIntegrityError("report source identity or manifest changed")
    return VerifiedReport(
        root=root,
        report_id=report_id,
        descriptor=descriptor,
        config=config,
        reference={
            "kind": "verified_report_release",
            "report_id": report_id,
            "manifest": fingerprint_file(root / "manifest.json"),
            "report_manifest": fingerprint_file(descriptor_path),
            "report_document": fingerprint_file(_artifact_path(root, manifest, "report_document")),
        },
    )


__all__ = [
    "ReportIntegrityError",
    "VerifiedReport",
    "load_verified_report",
]
