"""Semantic verification for successful non-actuating vision-shadow runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..verification import ArtifactIntegrityError, verify_research_object
from .audit import ALLOWED_POLICY_INPUT_FIELDS, PROHIBITED_PRIVILEGED_FIELDS
from .contracts import POLICY_CONTRACT_VERSION, VisionControlProposal

_REQUIRED_ROLES = (
    "detections_jsonl",
    "run_summary",
    "vision_policy_input_audit",
    "vision_shadow_proposals",
)
_SHA256 = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class VerifiedVisionShadowRun:
    root: Path
    run_id: str
    manifest: Mapping[str, Any]
    summary: Mapping[str, Any]
    input_audit: Mapping[str, Any]
    proposal_records: tuple[Mapping[str, Any], ...]
    detection_records: tuple[Mapping[str, Any], ...]


def _object(path: Path, name: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(f"could not read {name}: {error}") from error
    if not isinstance(value, Mapping):
        raise ArtifactIntegrityError(f"{name} must contain a JSON object")
    return value


def _jsonl(path: Path, name: str) -> tuple[Mapping[str, Any], ...]:
    try:
        payload = path.read_bytes()
        text = payload.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise ArtifactIntegrityError(f"could not read {name}: {error}") from error
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise ArtifactIntegrityError(f"{name} must be non-empty LF-terminated UTF-8")
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ArtifactIntegrityError(f"{name} line {line_number} is invalid JSON") from error
        if not isinstance(value, Mapping):
            raise ArtifactIntegrityError(f"{name} line {line_number} must be an object")
        rows.append(value)
    return tuple(rows)


def _artifact_paths(
    root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Path]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ArtifactIntegrityError("shadow tracker artifacts must be an array")
    result: dict[str, Path] = {}
    for role in _REQUIRED_ROLES:
        matches = [
            entry for entry in artifacts if isinstance(entry, Mapping) and entry.get("role") == role
        ]
        if len(matches) != 1:
            raise ArtifactIntegrityError(
                f"vision shadow run must contain exactly one {role!r} artifact"
            )
        relative = matches[0].get("path")
        if not isinstance(relative, str):
            raise ArtifactIntegrityError(f"vision shadow role {role!r} has no path")
        try:
            path = (root / relative).resolve(strict=True)
            path.relative_to(root)
        except (OSError, ValueError) as error:
            raise ArtifactIntegrityError(f"vision shadow role {role!r} path is unsafe") from error
        result[role] = path
    return result


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * 0.95
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _same_float(actual: Any, expected: float) -> bool:
    return (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-9)
    )


def load_verified_vision_shadow_run(
    path: str | Path,
) -> VerifiedVisionShadowRun:
    generic = verify_research_object(
        path,
        verify_references=True,
        deep=False,
        reject_unregistered=True,
    )
    root = Path(generic.root)
    manifest = _object(root / "manifest.json", "shadow tracker manifest")
    paths = _artifact_paths(root, manifest)
    summary = _object(paths["run_summary"], "shadow run summary")
    audit = _object(paths["vision_policy_input_audit"], "policy input audit")
    proposals = _jsonl(paths["vision_shadow_proposals"], "vision shadow proposals")
    detections = _jsonl(paths["detections_jsonl"], "vision detection log")

    if (
        summary.get("status") != "success"
        or summary.get("vision_shadow_enabled") is not True
        or summary.get("vision_shadow_actuation_authorized") is not False
        or summary.get("run_id") != generic.run_id
        or summary.get("runtime_sensor_contract") != "front_monocular_rgb_only"
    ):
        raise ArtifactIntegrityError("vision shadow summary envelope is invalid")
    expected_audit_keys = {
        "schema_version",
        "status",
        "policy_name",
        "policy_backend",
        "observation_type",
        "observation_fields",
        "declared_input_fields",
        "allowed_input_fields",
        "prohibited_privileged_fields",
        "direct_carla_objects_exposed",
        "privileged_fields_exposed",
        "limitations",
    }
    if set(audit) != expected_audit_keys:
        raise ArtifactIntegrityError("vision policy input audit fields differ from schema")
    if (
        audit.get("schema_version") != POLICY_CONTRACT_VERSION
        or audit.get("status") != "pass"
        or audit.get("observation_type") != "VisionObservation"
        or audit.get("observation_fields") != list(ALLOWED_POLICY_INPUT_FIELDS)
        or audit.get("allowed_input_fields") != list(ALLOWED_POLICY_INPUT_FIELDS)
        or audit.get("prohibited_privileged_fields") != list(PROHIBITED_PRIVILEGED_FIELDS)
        or audit.get("direct_carla_objects_exposed") is not False
        or audit.get("privileged_fields_exposed") != []
    ):
        raise ArtifactIntegrityError("vision policy input audit did not pass its contract")
    declared = audit.get("declared_input_fields")
    if (
        not isinstance(declared, list)
        or not declared
        or len(declared) != len(set(declared))
        or not set(declared) <= set(ALLOWED_POLICY_INPUT_FIELDS)
    ):
        raise ArtifactIntegrityError("vision policy declared inputs are invalid")

    proposal_by_sequence: dict[int, Mapping[str, Any]] = {}
    policy_metadata: Mapping[str, Any] | None = None
    latencies: list[float] = []
    braking = 0
    throttle = 0
    prior_sequence = -1
    expected_record_keys = {
        "schema_version",
        "sequence",
        "source_timestamp",
        "rgb_sha256",
        "image_width",
        "image_height",
        "detection_count",
        "camera_fov_degrees",
        "policy",
        "input_contract",
        "proposal",
        "policy_latency_ms",
        "actuation_applied",
    }
    for index, record in enumerate(proposals):
        if set(record) != expected_record_keys:
            raise ArtifactIntegrityError(
                f"vision shadow proposal {index} fields differ from schema"
            )
        sequence = record.get("sequence")
        rgb_digest = record.get("rgb_sha256")
        latency = record.get("policy_latency_ms")
        raw_proposal = record.get("proposal")
        raw_policy = record.get("policy")
        if (
            record.get("schema_version") != POLICY_CONTRACT_VERSION
            or isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence <= prior_sequence
            or not isinstance(rgb_digest, str)
            or len(rgb_digest) != 64
            or not set(rgb_digest) <= _SHA256
            or isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(float(latency))
            or float(latency) < 0.0
            or record.get("actuation_applied") is not False
            or record.get("input_contract") != declared
            or not isinstance(raw_proposal, Mapping)
            or not isinstance(raw_policy, Mapping)
        ):
            raise ArtifactIntegrityError(f"vision shadow proposal {index} is malformed")
        if policy_metadata is None:
            policy_metadata = raw_policy
        elif raw_policy != policy_metadata:
            raise ArtifactIntegrityError("vision shadow policy metadata changed during run")
        if (
            raw_policy.get("name") != audit.get("policy_name")
            or raw_policy.get("backend") != audit.get("policy_backend")
            or raw_policy.get("contract_version") != POLICY_CONTRACT_VERSION
            or raw_policy.get("input_fields") != declared
        ):
            raise ArtifactIntegrityError("vision shadow policy metadata differs from audit")
        try:
            proposal = VisionControlProposal(
                sequence=int(raw_proposal["sequence"]),
                throttle=float(raw_proposal["throttle"]),
                steer=float(raw_proposal["steer"]),
                brake=float(raw_proposal["brake"]),
                confidence=float(raw_proposal["confidence"]),
                reason=str(raw_proposal["reason"]),
                state=raw_proposal["state"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactIntegrityError(
                f"vision shadow proposal {index} violates the output contract"
            ) from error
        if proposal.sequence != sequence:
            raise ArtifactIntegrityError("vision shadow proposal sequence changed")
        latency_value = float(latency)
        latencies.append(latency_value)
        braking += int(proposal.brake > 0.05)
        throttle += int(proposal.throttle > 0.05)
        proposal_by_sequence[sequence] = raw_proposal
        prior_sequence = sequence

    detection_by_sequence: dict[int, Mapping[str, Any]] = {}
    for record in detections:
        sequence = record.get("sequence")
        context = record.get("decision_context")
        if not isinstance(sequence, int) or not isinstance(context, Mapping):
            raise ArtifactIntegrityError("vision detection log record is malformed")
        shadow = context.get("vision_shadow")
        if not isinstance(shadow, Mapping) or shadow.get("actuation_applied") is not False:
            raise ArtifactIntegrityError(
                "vision detection log is missing a non-actuating shadow proposal"
            )
        expected = proposal_by_sequence.get(sequence)
        if (
            expected is None
            or {key: value for key, value in shadow.items() if key != "actuation_applied"}
            != expected
        ):
            raise ArtifactIntegrityError(
                "vision detection log proposal differs from isolated shadow log"
            )
        detection_by_sequence[sequence] = record
    if set(detection_by_sequence) != set(proposal_by_sequence):
        raise ArtifactIntegrityError(
            "vision shadow proposal and detection-log sequence sets differ"
        )

    stats = summary.get("vision_shadow_stats")
    if not isinstance(stats, Mapping):
        raise ArtifactIntegrityError("vision shadow summary statistics are missing")
    if (
        stats.get("proposal_count") != len(proposals)
        or stats.get("braking_proposal_count") != braking
        or stats.get("throttle_proposal_count") != throttle
        or not _same_float(stats.get("latency_mean_ms"), statistics.fmean(latencies))
        or not _same_float(stats.get("latency_median_ms"), statistics.median(latencies))
        or not _same_float(stats.get("latency_p95_ms"), _p95(latencies))
    ):
        raise ArtifactIntegrityError("vision shadow summary statistics do not reproduce")

    manifest_config = manifest.get("invocation")
    if not isinstance(manifest_config, Mapping) or not isinstance(
        manifest_config.get("config"),
        Mapping,
    ):
        raise ArtifactIntegrityError("vision shadow invocation config is missing")
    invocation_config = manifest_config["config"]
    if (
        invocation_config.get("runtime_sensor_contract") != "front_monocular_rgb_only"
        or invocation_config.get("shadow_policy") is None
        or invocation_config.get("control_mode") == "vision"
    ):
        raise ArtifactIntegrityError("vision shadow invocation violates the control boundary")

    return VerifiedVisionShadowRun(
        root=root,
        run_id=generic.run_id,
        manifest=manifest,
        summary=summary,
        input_audit=audit,
        proposal_records=proposals,
        detection_records=detections,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only semantic verification of vision-only shadow policy runs"
    )
    parser.add_argument("paths", nargs="+")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = []
    for value in args.paths:
        verified = load_verified_vision_shadow_run(value)
        results.append(
            {
                "run_id": verified.run_id,
                "root": str(verified.root),
                "policy": verified.input_audit["policy_name"],
                "proposal_count": len(verified.proposal_records),
                "actuation_applied": False,
                "status": "verified",
            }
        )
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    return 0


__all__ = [
    "VerifiedVisionShadowRun",
    "load_verified_vision_shadow_run",
    "main",
    "parse_args",
]
