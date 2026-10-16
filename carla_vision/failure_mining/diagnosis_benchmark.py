"""Blinded error-attribution scoring: observations and reference outcomes stay separate.

All decisions are computed before reference outcomes are opened. Fault identities
are never accepted inside AttributionEvidence. Synthetic fixtures measure software
and controlled-fault behavior, not naturally occurring CARLA error rates.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

from ..artifacts import RunArtifactTracker, fingerprint_file
from ..reproduction.review_contracts import fields, read_json, text
from .attribution import AttributionEvidence, attribute_error

SOURCES = {"capture_error", "supervision_error", "model_error", "no_detected_error", "unknown"}


def score_diagnosis(observation_path, reference_path, *, runs_root="runs", run_id=None):
    observation_path, reference_path = Path(observation_path), Path(reference_path)
    observed_ref = fingerprint_file(observation_path)
    observations = read_json(observation_path)
    fields(observations, {"schema_version", "synthetic", "observations"})
    if observations["schema_version"] != "1.0" or not isinstance(observations["synthetic"], bool):
        raise ValueError("diagnosis observation schema/synthetic flag invalid")
    if not isinstance(observations["observations"], list) or not observations["observations"]:
        raise ValueError("empty observation set")
    predictions, groups = {}, {}
    evidence_fields = {f.name for f in dataclass_fields(AttributionEvidence)}
    for row in observations["observations"]:
        fields(row, {"case_id", "group_id", "evidence", "source_artifacts"})
        identity = text(row["case_id"], "case ID")
        if identity in predictions:
            raise ValueError("duplicate observation case")
        text(row["group_id"], "group ID")
        fields(row["evidence"], evidence_fields)
        if not row["source_artifacts"]:
            raise ValueError("observations need measured source records")
        known = set()
        for source in row["source_artifacts"]:
            if fingerprint_file(source["path"]) != source:
                raise ValueError("observation source fingerprint changed")
            known.add(source["path"])
        for key in ("capture_evidence_ref", "quality_evidence_ref", "review_evidence_ref"):
            if row["evidence"][key] and row["evidence"][key] not in known:
                raise ValueError("attribution input refers to unregistered evidence")
        predictions[identity] = attribute_error(AttributionEvidence(**row["evidence"])).as_dict()
        groups[identity] = row["group_id"]
    # No outcome file content was available to the decision function above.
    truth = read_json(reference_path)
    fields(truth, {"schema_version", "synthetic", "observations_sha256", "outcomes"})
    if (
        truth["schema_version"] != "1.0"
        or truth["synthetic"] is not observations["synthetic"]
        or truth["observations_sha256"] != observed_ref["sha256"]
    ):
        raise ValueError("reference outcomes are not bound to these observations")
    if set(truth["outcomes"]) != set(predictions):
        raise ValueError("reference outcomes must cover exactly the observed cases")
    confusion, regimes, clean_actions = Counter(), {}, Counter()
    for identity, row in truth["outcomes"].items():
        fields(row, {"source_kind", "regime", "reference_evidence"})
        if row["source_kind"] not in SOURCES:
            raise ValueError("unknown attribution reference")
        text(row["regime"], "regime")
        ref = row["reference_evidence"]
        if fingerprint_file(ref["path"]) != ref:
            raise ValueError("reference audit changed")
        predicted = predictions[identity]["source_kind"]
        confusion[row["source_kind"], predicted] += 1
        regimes.setdefault(row["regime"], Counter())["cases"] += 1
        regimes[row["regime"]]["correct"] += row["source_kind"] == predicted
        if row["regime"] in {"clean", "clean_hard"}:
            clean_actions[predictions[identity]["recommended_action"]] += 1
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "synthetic": observations["synthetic"],
        "empirical_results_available": False,
        "cases": len(predictions),
        "independent_groups": len(set(groups.values())),
        "predictions": predictions,
        "confusion": [
            {"reference": a, "predicted": b, "count": n} for (a, b), n in sorted(confusion.items())
        ],
        "regimes": {k: dict(v) for k, v in sorted(regimes.items())},
        "clean_case_actions": dict(clean_actions),
        "unknown_count": sum(p["source_kind"] == "unknown" for p in predictions.values()),
        "scope": "diagnosis_agreement_with_supplied_reference_not_model_training_utility",
    }
    with RunArtifactTracker(
        runs_root,
        run_id=run_id,
        config={"object_type": "diagnosis_benchmark", "synthetic": observations["synthetic"]},
        input_refs=[observed_ref, fingerprint_file(reference_path)],
    ) as tracker:
        target = tracker.artifact_path("diagnosis.json")
        target.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        tracker.register_artifact(target, role="diagnosis_benchmark_result")
    return {"run_dir": str(tracker.run_dir), **result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            score_diagnosis(
                args.observations, args.reference, runs_root=args.runs_root, run_id=args.run_id
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
