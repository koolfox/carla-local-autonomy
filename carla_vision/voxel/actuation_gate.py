"""Evidence gate for any future voxel-planner vehicle actuation.

This module only validates offline/shadow evidence. It does not connect to
CARLA and exposes no vehicle-control path. A passing report is necessary but
not sufficient for actuation; a later integration still requires explicit
operator acknowledgement and code review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

VOXEL_ACTUATION_READINESS_SCHEMA_VERSION = "1.0"


def _finite_number(payload: Mapping[str, Any], name: str) -> float | None:
    value = payload.get(name)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def _sha256_json(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


@dataclass(frozen=True, slots=True)
class VoxelActuationReadinessPolicy:
    minimum_training_dataset_count: int = 3
    minimum_route_group_count: int = 20
    minimum_current_occupied_iou: float = 0.35
    minimum_future_occupied_iou_1s: float = 0.20
    maximum_brier_score: float = 0.20
    maximum_model_p95_latency_ms: float = 100.0
    minimum_shadow_run_count: int = 3
    minimum_shadow_record_count: int = 300
    maximum_shadow_error_rate: float = 0.01
    maximum_shadow_p95_latency_ms: float = 120.0
    maximum_shadow_uncertain_fraction: float = 0.45

    def __post_init__(self) -> None:
        for name in (
            "minimum_training_dataset_count",
            "minimum_route_group_count",
            "minimum_shadow_run_count",
            "minimum_shadow_record_count",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "minimum_current_occupied_iou",
            "minimum_future_occupied_iou_1s",
            "maximum_brier_score",
            "maximum_shadow_error_rate",
            "maximum_shadow_uncertain_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        for name in ("maximum_model_p95_latency_ms", "maximum_shadow_p95_latency_ms"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "VoxelActuationReadinessPolicy":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError("unknown readiness policy fields: " + ", ".join(unknown))
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    name: str
    passed: bool
    observed: Any
    required: Any
    explanation: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check(
    checks: list[ReadinessCheck],
    *,
    name: str,
    passed: bool,
    observed: Any,
    required: Any,
    explanation: str,
) -> None:
    checks.append(
        ReadinessCheck(
            name=name,
            passed=bool(passed),
            observed=observed,
            required=required,
            explanation=explanation,
        )
    )


def evaluate_voxel_actuation_readiness(
    evidence: Mapping[str, Any],
    policy: VoxelActuationReadinessPolicy | None = None,
) -> dict[str, Any]:
    """Evaluate evidence and return a deterministic fail-closed report."""

    policy = policy or VoxelActuationReadinessPolicy()
    checks: list[ReadinessCheck] = []
    model = evidence.get("model")
    training = evidence.get("training")
    validation = evidence.get("validation")
    shadow = evidence.get("shadow")
    model = model if isinstance(model, Mapping) else {}
    training = training if isinstance(training, Mapping) else {}
    validation = validation if isinstance(validation, Mapping) else {}
    shadow = shadow if isinstance(shadow, Mapping) else {}

    _check(
        checks,
        name="issue_7_acceptance_complete",
        passed=evidence.get("issue_7_acceptance_complete") is True,
        observed=evidence.get("issue_7_acceptance_complete"),
        required=True,
        explanation="Issue #7 must be explicitly completed with real-data evidence.",
    )
    _check(
        checks,
        name="rgb_only_runtime_contract",
        passed=model.get("rgb_only_runtime_contract") is True,
        observed=model.get("rgb_only_runtime_contract"),
        required=True,
        explanation="The deployable voxel model must receive RGB history only.",
    )
    checkpoint_sha = model.get("checkpoint_sha256")
    _check(
        checks,
        name="checkpoint_sha256",
        passed=isinstance(checkpoint_sha, str)
        and len(checkpoint_sha) == 64
        and all(character in "0123456789abcdef" for character in checkpoint_sha.casefold()),
        observed=checkpoint_sha,
        required="64-character SHA-256",
        explanation="Evidence must bind to one immutable checkpoint.",
    )
    _check(
        checks,
        name="dataset_verification",
        passed=training.get("dataset_verification_status") == "passed",
        observed=training.get("dataset_verification_status"),
        required="passed",
        explanation="All source datasets must pass their artifact verifier.",
    )
    _check(
        checks,
        name="route_group_leakage",
        passed=training.get("route_group_leakage") is False,
        observed=training.get("route_group_leakage"),
        required=False,
        explanation="Route/seed groups must not overlap across splits.",
    )

    integer_requirements = (
        (
            "training_dataset_count",
            training.get("dataset_count"),
            policy.minimum_training_dataset_count,
        ),
        (
            "route_group_count",
            training.get("route_group_count"),
            policy.minimum_route_group_count,
        ),
        (
            "shadow_run_count",
            shadow.get("run_count"),
            policy.minimum_shadow_run_count,
        ),
        (
            "shadow_record_count",
            shadow.get("record_count"),
            policy.minimum_shadow_record_count,
        ),
    )
    for name, observed, required in integer_requirements:
        passed = isinstance(observed, int) and not isinstance(observed, bool) and observed >= required
        _check(
            checks,
            name=name,
            passed=passed,
            observed=observed,
            required=f">= {required}",
            explanation="Minimum evidence volume requirement.",
        )

    lower_bound_requirements = (
        (
            "current_occupied_iou",
            _finite_number(validation, "current_occupied_iou"),
            policy.minimum_current_occupied_iou,
        ),
        (
            "future_occupied_iou_1s",
            _finite_number(validation, "future_occupied_iou_1s"),
            policy.minimum_future_occupied_iou_1s,
        ),
    )
    for name, observed, required in lower_bound_requirements:
        _check(
            checks,
            name=name,
            passed=observed is not None and observed >= required,
            observed=observed,
            required=f">= {required}",
            explanation="Minimum validation accuracy requirement.",
        )

    upper_bound_requirements = (
        (
            "brier_score",
            _finite_number(validation, "brier_score"),
            policy.maximum_brier_score,
        ),
        (
            "model_p95_latency_ms",
            _finite_number(validation, "p95_latency_ms"),
            policy.maximum_model_p95_latency_ms,
        ),
        (
            "shadow_error_rate",
            _finite_number(shadow, "error_rate"),
            policy.maximum_shadow_error_rate,
        ),
        (
            "shadow_p95_latency_ms",
            _finite_number(shadow, "p95_latency_ms"),
            policy.maximum_shadow_p95_latency_ms,
        ),
        (
            "shadow_uncertain_fraction",
            _finite_number(shadow, "maximum_uncertain_voxel_fraction"),
            policy.maximum_shadow_uncertain_fraction,
        ),
    )
    for name, observed, required in upper_bound_requirements:
        _check(
            checks,
            name=name,
            passed=observed is not None and observed <= required,
            observed=observed,
            required=f"<= {required}",
            explanation="Maximum risk, uncertainty, or latency requirement.",
        )

    _check(
        checks,
        name="shadow_no_actuation",
        passed=shadow.get("actuation_enabled") is False and shadow.get("control_calls") == 0,
        observed={
            "actuation_enabled": shadow.get("actuation_enabled"),
            "control_calls": shadow.get("control_calls"),
        },
        required={"actuation_enabled": False, "control_calls": 0},
        explanation="Qualification evidence must come from the read-only shadow path.",
    )
    _check(
        checks,
        name="operator_review_complete",
        passed=evidence.get("operator_review_complete") is True,
        observed=evidence.get("operator_review_complete"),
        required=True,
        explanation="A human review of failure cases and telemetry is mandatory.",
    )

    passed = all(check.passed for check in checks)
    failed_names = [check.name for check in checks if not check.passed]
    report_core = {
        "schema_version": VOXEL_ACTUATION_READINESS_SCHEMA_VERSION,
        "status": "passed" if passed else "blocked",
        "actuation_readiness": passed,
        "actuation_enabled_by_this_report": False,
        "evidence_sha256": _sha256_json(evidence),
        "policy": asdict(policy),
        "checks": [check.as_dict() for check in checks],
        "failed_checks": failed_names,
        "required_next_action": (
            "A later explicit CARLA integration and operator acknowledgement are still required."
            if passed
            else "Resolve every failed check; vehicle actuation must remain unavailable."
        ),
    }
    return report_core


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify evidence before any future voxel-planner vehicle actuation."
    )
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _dry_run_payload() -> dict[str, Any]:
    policy = VoxelActuationReadinessPolicy()
    return {
        "schema_version": VOXEL_ACTUATION_READINESS_SCHEMA_VERSION,
        "dry_run": True,
        "actuation_readiness": False,
        "actuation_enabled": False,
        "default_policy": asdict(policy),
        "required_evidence_sections": [
            "issue_7_acceptance_complete",
            "operator_review_complete",
            "model",
            "training",
            "validation",
            "shadow",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        payload = _dry_run_payload()
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.evidence is None:
        raise ValueError("--evidence is required unless --dry-run is used")
    evidence = _read_json(args.evidence.expanduser().resolve(strict=True))
    policy = (
        VoxelActuationReadinessPolicy.from_mapping(
            _read_json(args.policy.expanduser().resolve(strict=True))
        )
        if args.policy
        else VoxelActuationReadinessPolicy()
    )
    report = evaluate_voxel_actuation_readiness(evidence, policy)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["status"] == "passed" else 2


__all__ = [
    "ReadinessCheck",
    "VOXEL_ACTUATION_READINESS_SCHEMA_VERSION",
    "VoxelActuationReadinessPolicy",
    "build_parser",
    "evaluate_voxel_actuation_readiness",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
