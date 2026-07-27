"""Semantic verification for planned or executed live shadow matrices."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import fingerprint_file
from ..policy.verified import load_verified_vision_shadow_run
from ..verification import ArtifactIntegrityError, verify_research_object
from .contracts import ShadowMatrixConfig
from .matrix import SHADOW_MATRIX_RUN_SCHEMA_VERSION, build_planned_rows

_ROLES = frozenset(
    {
        "resolved_shadow_matrix_configuration",
        "shadow_matrix_plan_jsonl",
        "shadow_matrix_plan_csv",
        "shadow_matrix_execution_results",
        "shadow_matrix_release_manifest",
        "shadow_matrix_report",
        "shadow_matrix_checksum_index",
    }
)


@dataclass(frozen=True)
class VerifiedShadowMatrix:
    root: Path
    run_id: str
    config: ShadowMatrixConfig
    descriptor: Mapping[str, Any]
    plan_rows: tuple[Mapping[str, Any], ...]
    execution_results: tuple[Mapping[str, Any], ...]


def _object(path: Path, name: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(f"could not read {name}: {error}") from error
    if not isinstance(value, Mapping):
        raise ArtifactIntegrityError(f"{name} must contain an object")
    return value


def _jsonl(path: Path, name: str) -> tuple[Mapping[str, Any], ...]:
    try:
        payload = path.read_bytes()
        text = payload.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise ArtifactIntegrityError(f"could not read {name}: {error}") from error
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise ArtifactIntegrityError(f"{name} must be LF-terminated UTF-8")
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ArtifactIntegrityError(f"{name} line {line_number} is invalid JSON") from error
        if not isinstance(value, Mapping):
            raise ArtifactIntegrityError(f"{name} line {line_number} is not an object")
        rows.append(value)
    return tuple(rows)


def _paths(root: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ArtifactIntegrityError("shadow matrix artifacts must be an array")
    paths: dict[str, Path] = {}
    for role in _ROLES:
        matches = [raw for raw in artifacts if isinstance(raw, Mapping) and raw.get("role") == role]
        if len(matches) != 1:
            raise ArtifactIntegrityError(
                f"shadow matrix must contain exactly one {role!r} artifact"
            )
        relative = matches[0].get("path")
        if not isinstance(relative, str):
            raise ArtifactIntegrityError(f"shadow matrix {role!r} path is missing")
        try:
            path = (root / relative).resolve(strict=True)
            path.relative_to(root)
        except (OSError, ValueError) as error:
            raise ArtifactIntegrityError(f"shadow matrix {role!r} path is unsafe") from error
        paths[role] = path
    return paths


def _csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            expected = [
                "ordinal",
                "cell_id",
                "label",
                "repetition",
                "run_id",
                "control",
                "actuation_source",
                "vision_policy_actuation_authorized",
                "command_sha256",
                "command",
            ]
            if reader.fieldnames != expected:
                raise ArtifactIntegrityError("shadow matrix plan CSV header is invalid")
            return list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ArtifactIntegrityError(f"could not read shadow matrix CSV: {error}") from error


def load_verified_shadow_matrix(path: str | Path) -> VerifiedShadowMatrix:
    generic = verify_research_object(
        path,
        verify_references=True,
        deep=False,
        reject_unregistered=True,
    )
    root = Path(generic.root)
    manifest = _object(root / "manifest.json", "shadow matrix tracker manifest")
    paths = _paths(root, manifest)
    resolved = _object(
        paths["resolved_shadow_matrix_configuration"],
        "resolved shadow matrix",
    )
    descriptor = _object(
        paths["shadow_matrix_release_manifest"],
        "shadow matrix release",
    )
    plan_rows = _jsonl(paths["shadow_matrix_plan_jsonl"], "shadow matrix plan")
    results = _jsonl(
        paths["shadow_matrix_execution_results"],
        "shadow matrix execution results",
    )
    if set(resolved) != {
        "schema_version",
        "object_type",
        "mode",
        "matrix",
        "source_config",
        "runtime_executable",
        "runs_root",
        "vision_policy_actuation_authorized",
    }:
        raise ArtifactIntegrityError("resolved shadow matrix fields differ from schema")
    if (
        resolved.get("schema_version") != SHADOW_MATRIX_RUN_SCHEMA_VERSION
        or resolved.get("object_type") != "vision_shadow_matrix"
        or resolved.get("mode") not in {"plan", "execute"}
        or resolved.get("vision_policy_actuation_authorized") is not False
    ):
        raise ArtifactIntegrityError("resolved shadow matrix envelope is invalid")
    matrix_raw = resolved.get("matrix")
    if not isinstance(matrix_raw, Mapping):
        raise ArtifactIntegrityError("resolved shadow matrix contract is missing")
    try:
        config = ShadowMatrixConfig.from_mapping(matrix_raw, base=root)
    except (TypeError, ValueError) as error:
        raise ArtifactIntegrityError(f"resolved shadow matrix is invalid: {error}") from error
    executable = resolved.get("runtime_executable")
    runs_root = resolved.get("runs_root")
    if not isinstance(executable, str) or not isinstance(runs_root, str):
        raise ArtifactIntegrityError("shadow matrix runtime executable/root are invalid")
    expected_rows = build_planned_rows(
        config,
        runs_root=runs_root,
        executable=executable,
    )
    if list(plan_rows) != expected_rows:
        raise ArtifactIntegrityError(
            "shadow matrix planned commands do not reproduce from configuration"
        )
    expected_csv = [
        {
            "ordinal": str(row["ordinal"]),
            "cell_id": str(row["cell_id"]),
            "label": str(row["label"]),
            "repetition": str(row["repetition"]),
            "run_id": str(row["run_id"]),
            "control": str(row["control"]),
            "actuation_source": str(row["actuation_source"]),
            "vision_policy_actuation_authorized": str(row["vision_policy_actuation_authorized"]),
            "command_sha256": str(row["command_sha256"]),
            "command": json.dumps(
                row["command"],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }
        for row in expected_rows
    ]
    if _csv_rows(paths["shadow_matrix_plan_csv"]) != expected_csv:
        raise ArtifactIntegrityError("shadow matrix CSV differs from canonical JSONL")
    if any(
        "--shadow-policy" not in row["command"]
        or "vision"
        in (
            row["command"][row["command"].index("--control") + 1]
            if "--control" in row["command"]
            else "vision"
        )
        for row in expected_rows
    ):
        raise ArtifactIntegrityError("shadow matrix contains an unsafe runtime command")

    mode = str(resolved["mode"])
    expected_descriptor_status: str
    successful = 0
    failed = 0
    if mode == "plan":
        expected_results = (
            {
                "status": "not_executed",
                "planned_run_count": len(expected_rows),
                "vision_policy_actuation_authorized": False,
            },
        )
        if results != expected_results:
            raise ArtifactIntegrityError("planned shadow matrix has execution evidence")
        expected_descriptor_status = "planned"
    else:
        if len(results) > len(expected_rows):
            raise ArtifactIntegrityError("shadow matrix has too many execution results")
        for expected, result in zip(expected_rows, results, strict=False):
            if (
                result.get("ordinal") != expected["ordinal"]
                or result.get("cell_id") != expected["cell_id"]
                or result.get("run_id") != expected["run_id"]
                or result.get("status") not in {"success", "failed"}
            ):
                raise ArtifactIntegrityError("shadow matrix execution result order is invalid")
            if result["status"] == "success":
                child = load_verified_vision_shadow_run(Path(runs_root) / str(result["run_id"]))
                child_reference = fingerprint_file(child.root / "manifest.json")
                if result.get("child_manifest") != child_reference or result.get(
                    "proposal_count"
                ) != len(child.proposal_records):
                    raise ArtifactIntegrityError(
                        "shadow matrix child verification evidence changed"
                    )
                successful += 1
            else:
                if result.get("child_manifest") is not None:
                    raise ArtifactIntegrityError("failed shadow child has a manifest")
                failed += 1
        expected_descriptor_status = (
            "complete" if failed == 0 and successful == len(expected_rows) else "partial"
        )

    if (
        descriptor.get("schema_version") != SHADOW_MATRIX_RUN_SCHEMA_VERSION
        or descriptor.get("object_type") != "vision_shadow_matrix_release"
        or descriptor.get("status") != expected_descriptor_status
        or descriptor.get("matrix_id") != config.matrix_id
        or descriptor.get("run_id") != generic.run_id
        or descriptor.get("purpose") != config.purpose
        or descriptor.get("mode") != mode
        or descriptor.get("planned_run_count") != len(expected_rows)
        or descriptor.get("successful_run_count") != successful
        or descriptor.get("failed_run_count") != failed
        or descriptor.get("control") != config.control
        or descriptor.get("vision_policy_actuation_authorized") is not False
        or descriptor.get("runtime_sensor_contract") != "front_monocular_rgb_only"
    ):
        raise ArtifactIntegrityError("shadow matrix release descriptor is inconsistent")
    return VerifiedShadowMatrix(
        root=root,
        run_id=generic.run_id,
        config=config,
        descriptor=descriptor,
        plan_rows=plan_rows,
        execution_results=results,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only semantic verification of vision-shadow matrix releases"
    )
    parser.add_argument("paths", nargs="+")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = []
    for value in args.paths:
        matrix = load_verified_shadow_matrix(value)
        output.append(
            {
                "run_id": matrix.run_id,
                "matrix_id": matrix.config.matrix_id,
                "mode": matrix.descriptor["mode"],
                "planned_run_count": len(matrix.plan_rows),
                "successful_run_count": matrix.descriptor["successful_run_count"],
                "vision_policy_actuation_authorized": False,
                "status": "verified",
            }
        )
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    return 0


__all__ = [
    "VerifiedShadowMatrix",
    "load_verified_shadow_matrix",
    "main",
    "parse_args",
]
