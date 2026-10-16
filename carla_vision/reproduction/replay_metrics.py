"""Recompute COCO metrics from a verified relocated evidence closure, without CARLA.

Uses the original saved predictions, not rerun model inference. The distinction is
recorded so metric reproduction cannot be reported as independent training or
live-system reproduction. Original evidence and paths are never rewritten.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ..artifacts import RunArtifactTracker, fingerprint_file
from ..evaluation.coco import evaluate_coco
from .closure import verify_closure
from .review_contracts import digest, integer, parse_json, read_json, relative_path


def replay_coco_metrics(
    bundle,
    *,
    evaluation_manifest_sha256,
    runs_root="runs",
    run_id=None,
    expected_closure_sha256=None,
    tolerance=1e-10,
):
    from .review_contracts import number

    number(tolerance, "tolerance", minimum=0)
    root = Path(bundle).resolve(strict=True)
    verification = verify_closure(root, expected_sha256=expected_closure_sha256)
    nodes = {r["sha256"]: r for r in read_json(root / "closure.json")["files"]}

    def obj(key):
        key = digest(key)
        if key not in nodes:
            raise ValueError("requested evidence object is absent from verified closure")
        return relative_path(root, nodes[key]["stored_path"])

    manifest_path = obj(evaluation_manifest_sha256)
    manifest = read_json(manifest_path)
    if manifest.get("status") != "success":
        raise ValueError("metrics require a successful original evaluation")

    def role(source, name):
        items = [r for r in source["artifacts"] if r["role"] == name]
        if len(items) != 1:
            raise ValueError("missing/ambiguous source artifact role: " + name)
        return obj(items[0]["sha256"])

    resolved = read_json(role(manifest, "resolved_evaluation_configuration"))
    if resolved.get("object_type") == "offline_detector_evaluation":
        source_manifest = read_json(obj(resolved["dataset"]["manifest"]["sha256"]))
        truth_path = role(source_manifest, "canonical_coco_annotations")
    elif resolved.get("object_type") == "pooled_detector_evaluation":
        truth_path = role(manifest, "pooled_coco_annotations")
    else:
        raise ValueError("unsupported evaluation producer")
    prediction_path = role(manifest, "canonical_coco_predictions")
    frames = role(manifest, "per_image_prediction_log")
    rows = [parse_json(line) for line in frames.read_text().splitlines() if line.strip()]
    image_ids = [integer(r["image_id"], "image_id", minimum=1) for r in rows]
    if not image_ids or len(image_ids) != len(set(image_ids)):
        raise ValueError("image selection is empty/duplicated")
    predictions = read_json(prediction_path)
    selected = set(image_ids)
    if any(p["image_id"] not in selected for p in predictions):
        raise ValueError("prediction does not belong to selected holdout images")
    measured = evaluate_coco(truth_path, predictions, image_ids=image_ids)
    original = read_json(role(manifest, "evaluation_summary"))["coco"]
    agreement = {}
    for key, value in measured.aggregate.items():
        expected = original.get(key)
        agreement[key] = (
            expected is None
            if value is None
            else not isinstance(expected, bool)
            and isinstance(expected, (float, int))
            and math.isfinite(expected)
            and math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance)
        )
    result = {
        "schema_version": "1.0",
        "status": "passed" if all(agreement.values()) else "failed",
        "evaluation_manifest_sha256": evaluation_manifest_sha256,
        "closure_sha256": fingerprint_file(root / "closure.json")["sha256"],
        "image_count": len(image_ids),
        "recomputed": dict(measured.aggregate),
        "matches": agreement,
        "tolerance": tolerance,
        "model_inference_reexecuted": False,
        "training_reexecuted": False,
        "scope": "saved_prediction_metric_reproduction_not_independent_model_validation",
        "original_absolute_paths_required": False,
    }
    with RunArtifactTracker(
        runs_root,
        run_id=run_id,
        input_refs=[fingerprint_file(root / "closure.json"), fingerprint_file(manifest_path)],
        config={"object_type": "portable_coco_metric_reproduction"},
    ) as tracker:
        path = tracker.artifact_path("metric_reproduction.json")
        path.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
        tracker.register_artifact(path, role="portable_metric_reproduction")
        log = tracker.artifact_path("coco.log")
        log.write_text(measured.log)
        tracker.register_artifact(log, role="recomputed_coco_log")
    return {"run_dir": str(tracker.run_dir), "closure_files": verification["file_count"], **result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("--evaluation-manifest-sha256", required=True)
    parser.add_argument("--closure-sha256")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    result = replay_coco_metrics(
        args.bundle,
        evaluation_manifest_sha256=args.evaluation_manifest_sha256,
        runs_root=args.runs_root,
        run_id=args.run_id,
        expected_closure_sha256=args.closure_sha256,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
