"""Build training-side acquisition candidates from verified evaluations and blinded audits.

This is separate from validation-only failure mining: validation/test diagnostics
are never eligible inputs. The result is an action proposal, not new training data.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..artifacts import RunArtifactTracker, fingerprint_file
from ..dataset.lineage import require_train_parent, scene_group_key
from ..dataset.quality import ReviewRecord, estimate_quality
from ..evaluation.metrics import operating_metrics
from ..evaluation.verified import load_verified_evaluation
from ..scenarios.acquisition import AcquisitionCandidate
from .attribution import CHECKS, AttributionEvidence, attribute_error


def prepare_pool(
    evaluation_run: str | Path,
    review_path: str | Path,
    *,
    estimated_action_cost: float,
    runs_root: str | Path = "runs",
    run_id: str | None = None,
    minimum_quality: float = 0.9,
    seed: int = 17,
) -> dict[str, Any]:
    if not math.isfinite(estimated_action_cost) or estimated_action_cost <= 0:
        raise ValueError("a positive preregistered action cost is required")
    source = load_verified_evaluation(evaluation_run)
    for partition in source.config.partitions:
        require_train_parent(partition)
    review_ref = fingerprint_file(review_path)
    raw = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (
        set(raw) != {"schema_version", "dataset_manifest", "records", "evidence"}
        or raw["schema_version"] != "1.0"
        or raw["dataset_manifest"] != source.dataset.reference["manifest"]
    ):
        raise ValueError("independent review must identify the exact source dataset")
    for ref in raw["evidence"].values():
        if fingerprint_file(ref["path"]) != ref:
            raise ValueError("independent audit evidence changed")
    reviews = [ReviewRecord(**row) for row in raw["records"]]
    samples = [
        s for s in source.dataset.dataset["samples"] if s["split"] in source.config.partitions
    ]
    known = {s["sample_id"]: s for s in samples}
    metadata = {
        s["sample_id"]: json.loads(
            (source.dataset.root / s["metadata"]["path"]).read_text(encoding="utf-8")
        )
        for s in samples
    }
    group_by_id = {
        s["sample_id"]: scene_group_key(
            metadata[s["sample_id"]].get("context", {}), episode_id=s["episode_id"]
        )
        for s in samples
    }
    for review in reviews:
        if (
            review.sample_id not in known
            or review.group_id != group_by_id[review.sample_id]
            or review.evidence_ref not in raw["evidence"]
        ):
            raise ValueError("review sample/group/evidence does not match the source selection")
    quality = estimate_quality(reviews, task="label", seed=seed)
    ids = [int(s["image_id"]) for s in samples]
    metrics = operating_metrics(
        [a for a in source.dataset.coco["annotations"] if a["image_id"] in ids],
        source.predictions,
        image_ids=ids,
        categories=source.dataset.categories,
        confidence_threshold=source.config.operating_confidence,
        iou_threshold=source.config.matching_iou,
    )
    result = []
    triage = []
    with RunArtifactTracker(
        runs_root,
        run_id=run_id,
        config={
            "object_type": "training_acquisition_pool",
            "minimum_quality": minimum_quality,
            "estimated_action_cost": estimated_action_cost,
            "seed": seed,
            "runtime_sensor_contract": "front_monocular_rgb_only",
            "training_executed": False,
        },
        input_refs=[source.reference, review_ref, *raw["evidence"].values()],
    ) as tracker:
        for sample in samples:
            sample_id, image_id = sample["sample_id"], int(sample["image_id"])
            counts = metrics.image_counts[image_id]
            error = (counts["fp"] + counts["fn"]) / max(
                1, counts["tp"] + counts["fp"] + counts["fn"]
            )
            decisions = {
                review.decision
                for review in reviews
                if review.sample_id == sample_id and review.task == "label"
            }
            independent = (
                "valid"
                if decisions == {"valid"}
                else "invalid"
                if decisions == {"invalid"}
                else "unknown"
            )
            observed = AttributionEvidence(
                capture_checks=dict.fromkeys(CHECKS, True),
                model_error=error,
                label_quality_lower=quality["quality_lower"],
                independent_review=independent,
                capture_evidence_ref="verified_evaluation",
                quality_evidence_ref="audit",
                review_evidence_ref="audit" if independent != "unknown" else "",
            )
            attribution = attribute_error(observed, minimum_quality=minimum_quality)
            scores = [float(p["score"]) for p in source.predictions if p["image_id"] == image_id]
            context = metadata[sample_id].get("context", {})
            condition = (
                str(context.get("weather_recipe_id", "unknown"))
                + "|"
                + str(context.get("light", "unknown"))
            )
            result.append(
                asdict(
                    AcquisitionCandidate(
                        candidate_id=sample_id,
                        group_id=group_by_id[sample_id],
                        condition=condition,
                        partition=sample["split"],
                        estimated_cost=estimated_action_cost,
                        model_error=error,
                        uncertainty=1 - max(scores, default=0.0),
                        quality_lower=quality["quality_lower"],
                        action=attribution.recommended_action,
                        evidence_ref="probe",
                        repair_value=1.0 if independent == "invalid" else 0.0,
                    )
                )
            )
            triage.append({"sample_id": sample_id, "attribution": attribution.as_dict()})
        probe = tracker.artifact_path("probe.json")
        probe.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "source": source.reference,
                    "quality": quality,
                    "triage": triage,
                    "assumptions": {
                        "teacher_timing": "not_applicable_to_2d_detection",
                        "quality_scope": "pooled_independent_audit_not_per_instance_probability",
                        "uncertainty": "one_minus_max_retained_score_not_calibrated_entropy",
                        "cost_scope": "explicit_constant_estimate_actual_costs_belong_in_ledger",
                    },
                    "empirical_superiority_established": False,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        path = tracker.artifact_path("candidates.json")
        path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "candidates": result,
                    "evidence": {
                        "probe": fingerprint_file(probe),
                        "audit": review_ref,
                        "verified_evaluation": source.reference["manifest"],
                    },
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        tracker.register_artifact(probe, role="acquisition_error_source_probe")
        tracker.register_artifact(path, role="training_acquisition_candidates")
    return {
        "run_dir": str(tracker.run_dir),
        "candidate_count": len(result),
        "training_data_created": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--independent-review", required=True)
    parser.add_argument("--estimated-action-cost", type=float, required=True)
    parser.add_argument("--minimum-quality", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            prepare_pool(
                args.evaluation,
                args.independent_review,
                estimated_action_cost=args.estimated_action_cost,
                minimum_quality=args.minimum_quality,
                seed=args.seed,
                runs_root=args.runs_root,
                run_id=args.run_id,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
