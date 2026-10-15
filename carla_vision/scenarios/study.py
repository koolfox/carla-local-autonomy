"""Preregister and compare offline acquisition plans. No simulator/training side effects.

Candidates and their measured evidence are inputs, never invented by this CLI.
A plan comparison is not a learning-utility or Q1-publication result. Existing
collector/trainer/evaluator commands execute the plans in separate cycle stages.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..artifacts import RunArtifactTracker, fingerprint_file
from ..dataset.lineage import canonical_digest, require_train_parent
from .acquisition import METHODS, AcquisitionCandidate, AcquisitionConfig, select_acquisition


def validate_protocol(raw: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "study_id",
        "purpose",
        "hypotheses",
        "primary_metric",
        "minimum_effect",
        "methods",
        "ablations",
        "training_seeds",
        "generation_seeds",
        "selection",
        "split_manifest",
        "regimes",
        "stopping_rule",
        "power_analysis",
        "learner_configs",
        "cost_scope",
    }
    if set(raw) != required or raw["schema_version"] != "1.0":
        raise ValueError("study protocol fields/schema do not match the preregistration contract")
    if raw["purpose"] not in {"development", "confirmatory"} or not raw["study_id"]:
        raise ValueError("study identity/purpose invalid")
    if (
        not isinstance(raw["hypotheses"], list)
        or not raw["hypotheses"]
        or any(not h for h in raw["hypotheses"])
    ):
        raise ValueError("falsifiable hypotheses are required")
    if (
        not raw["primary_metric"]
        or not raw["stopping_rule"]
        or raw["cost_scope"] != "all_attempts_including_failures_reviews_and_training"
    ):
        raise ValueError("metric, stopping rule and all-attempt cost scope required")
    effect = raw["minimum_effect"]
    if (
        isinstance(effect, bool)
        or not isinstance(effect, (int, float))
        or not math.isfinite(effect)
        or effect <= 0
    ):
        raise ValueError("minimum effect must be positive, finite and preregistered")
    methods = raw["methods"]
    if not isinstance(methods, list) or len(methods) != len(set(methods)) or set(methods) - METHODS:
        raise ValueError("invalid or repeated baselines")
    if not {"random", "capability_matched", "error_source_aware"} <= set(methods):
        raise ValueError("random, capability-matched and full methods are mandatory")
    if not {"clean", "capture_fault", "supervision_fault", "mixed"} <= set(raw["regimes"]):
        raise ValueError("clean and distinct fault regimes are required")
    for key in ("training_seeds", "generation_seeds"):
        values = raw[key]
        if (
            not isinstance(values, list)
            or not values
            or len(values) != len(set(values))
            or any(
                isinstance(s, bool) or not isinstance(s, int) or s < 0 or s >= 2**63 for s in values
            )
        ):
            raise ValueError("seeds must be non-empty unique non-negative integers")
    for reference in [raw["split_manifest"], raw["power_analysis"], *raw["learner_configs"]]:
        if not isinstance(reference, dict) or fingerprint_file(reference["path"]) != reference:
            raise ValueError("preregistered source artifact changed")
    split = json.loads(Path(raw["split_manifest"]["path"]).read_text(encoding="utf-8"))
    if split.get("schema_version") != "1.0" or not isinstance(split.get("groups"), dict):
        raise ValueError("study split manifest must provide authoritative group partitions")
    from ..dataset.lineage import partition_role

    for group, partition in split["groups"].items():
        if not group:
            raise ValueError("empty split group")
        partition_role(partition)
    options = dict(raw["selection"])
    if "method" in options or "seed" in options or "ablations" in options:
        raise ValueError(
            "shared selection budget/options cannot override method, seed or ablations"
        )
    AcquisitionConfig(method="error_source_aware", **options)
    if not set(raw["ablations"]) <= {"quality", "attribution", "coverage", "exploration"}:
        raise ValueError("unknown ablation")
    if raw["purpose"] == "confirmatory":
        power = json.loads(Path(raw["power_analysis"]["path"]).read_text(encoding="utf-8"))
        if (
            power.get("status") != "registered"
            or power.get("basis") not in {"pilot_group_variance", "registered_compute_limit"}
            or not power.get("rationale")
            or not raw["learner_configs"]
        ):
            raise ValueError(
                "confirmatory protocol needs an explicit power/budget justification and learner configs"
            )
    return {**raw, "protocol_sha256": canonical_digest(raw)}


def plan_study(
    protocol_path: str | Path, candidates_path: str | Path, *, runs_root="runs", run_id=None
):
    protocol_ref, pool_ref = fingerprint_file(protocol_path), fingerprint_file(candidates_path)
    protocol = validate_protocol(json.loads(Path(protocol_path).read_text(encoding="utf-8")))
    pool = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
    if set(pool) != {"schema_version", "candidates", "evidence"} or pool["schema_version"] != "1.0":
        raise ValueError(
            "candidate pool must have schema, candidates and evidence (no injected-fault oracle)"
        )
    evidence = pool["evidence"]
    for reference in evidence.values():
        if fingerprint_file(reference["path"]) != reference:
            raise ValueError("candidate measured-evidence artifact changed")
    synthetic_input = any(
        json.loads(Path(reference["path"]).read_text(encoding="utf-8")).get("synthetic") is True
        for reference in evidence.values()
        if Path(reference["path"]).suffix.lower() == ".json"
    )
    candidates = tuple(AcquisitionCandidate(**row) for row in pool["candidates"])
    groups = json.loads(Path(protocol["split_manifest"]["path"]).read_text(encoding="utf-8"))[
        "groups"
    ]
    for candidate in candidates:
        require_train_parent(candidate.partition)
        if (
            groups.get(candidate.group_id) != candidate.partition
            or candidate.evidence_ref not in evidence
        ):
            raise ValueError(
                "candidate group partition/evidence does not match preregistered sources"
            )
    result = {
        "schema_version": "1.0",
        "study_id": protocol["study_id"],
        "empirical_results_available": False,
        "plans": [],
        "protocol": protocol,
    }
    with RunArtifactTracker(
        runs_root,
        run_id=run_id,
        config={
            "object_type": "offline_acquisition_study_plan",
            "synthetic": True if synthetic_input else None,
            "training_executed": False,
        },
        input_refs=[
            protocol_ref,
            pool_ref,
            protocol["split_manifest"],
            protocol["power_analysis"],
            *protocol["learner_configs"],
            *evidence.values(),
        ],
    ) as tracker:
        # Identical pool, grouping, costs and budgets in every arm. This is a common-pool comparison.
        for seed in protocol["generation_seeds"]:
            for method in protocol["methods"]:
                config = AcquisitionConfig(method=method, seed=seed, **protocol["selection"])
                plan = select_acquisition(candidates, config)
                result["plans"].append(
                    {"method": method, "seed": seed, "ablation": None, "plan": plan}
                )
            for ablation in protocol["ablations"]:
                config = AcquisitionConfig(
                    method="error_source_aware", seed=seed, **protocol["selection"]
                )
                plan = select_acquisition(candidates, replace(config, ablations=(ablation,)))
                result["plans"].append(
                    {
                        "method": "error_source_aware",
                        "seed": seed,
                        "ablation": ablation,
                        "plan": plan,
                    }
                )
        path = tracker.artifact_path("study_plan.json")
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        tracker.register_artifact(path, role="acquisition_study_plan")
    return {
        "run_dir": str(tracker.run_dir),
        "plan_count": len(result["plans"]),
        "empirical_results_available": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            plan_study(
                args.protocol, args.candidates, runs_root=args.runs_root, run_id=args.run_id
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
