"""Verify pooled receipts by re-reading instrumented child producers.

Pooling is a derivation of saved inference, never a second execution claim.
"""

from __future__ import annotations

from pathlib import Path

from ..artifacts import fingerprint_file
from ..dataset.lineage import canonical_digest, partition_role
from ..verification import verify_research_object
from .consumption import _artifact, implementation_files, sample_key
from .review_contracts import digest, fields, parse_json, read_json
from .study_links import _reference, verify_split


def verify_pooled_consumption(root, *, inventory_path=None, require_holdout=False, ancestry=()):
    from ..evaluation.coco import evaluate_coco
    from ..evaluation.verified import load_verified_evaluation
    from .consumption_verify import verify_producer_consumption

    root = Path(root).resolve(strict=True)
    verify_research_object(root, verify_references=True, deep=False, reject_unregistered=True)
    manifest = read_json(root / "manifest.json")
    report_path = _artifact(root, manifest, "pooled_consumed_sample_report")
    report = read_json(report_path)
    fields(
        report,
        {
            "schema_version",
            "inventory",
            "children",
            "keys",
            "model",
            "model_inference_reexecuted",
            "training_origin",
            "implementation",
        },
    )
    if report["schema_version"] != "1.0" or report["model_inference_reexecuted"] is not False:
        raise ValueError("invalid pooled-consumption derivation")
    if not isinstance(report["implementation"], dict) or set(report["implementation"]) != set(
        implementation_files("pooled_detector_evaluation")
    ):
        raise ValueError("pooled implementation inventory is incomplete")
    for hashed in report["implementation"].values():
        digest(hashed)
    inv_path = _reference(root, report["inventory"])
    if (
        inventory_path
        and fingerprint_file(inventory_path)["sha256"] != report["inventory"]["sha256"]
    ):
        raise ValueError("pooled inventory differs")
    inv = verify_split(inv_path)
    children = report["children"]
    if not isinstance(children, list) or len(children) < 2:
        raise ValueError("pooled receipts require canonical children")
    truth = {"images": [], "annotations": [], "categories": []}
    keys, predictions, frame_ids, sources = [], [], [], []
    policy = None
    for ref in children:
        child = _reference(root, ref).parent
        checked = verify_producer_consumption(child, inventory_path=inv_path, _ancestry=ancestry)
        if checked["training_origin"] != report["training_origin"]:
            raise ValueError("pooled trained-model origin differs from child execution")
        if checked["task"] != "detector_evaluation":
            raise ValueError("pool must reference actual canonical inference producers")
        evaluation = load_verified_evaluation(child, verify_receipts=False)
        child_policy = evaluation.config.as_dict()
        child_policy.pop("evaluation_id")
        if policy is None:
            policy = child_policy
            truth["categories"] = list(evaluation.dataset.coco["categories"])
        if (
            child_policy != policy
            or evaluation.resolved["model"] != report["model"]
            or list(evaluation.dataset.coco["categories"]) != truth["categories"]
        ):
            raise ValueError("child policy/model/ontology differs from pool")
        sources.extend(checked["source_manifests"])
        remap, images = {}, {r["id"]: r for r in evaluation.dataset.coco["images"]}
        for key, row in zip(checked["keys"], evaluation.frame_rows, strict=True):
            new_id = len(keys) + 1
            remap[row["image_id"]] = new_id
            keys.append(key)
            frame_ids.append(
                {
                    "image_id": new_id,
                    "source_image_id": row["image_id"],
                    "source_dataset_manifest_sha256": key.split(":", 1)[0],
                    "sample_id": row["sample_id"],
                    "partition": row["partition"],
                }
            )
            truth["images"].append({**images[row["image_id"]], "id": new_id})
        for row in evaluation.dataset.coco["annotations"]:
            if row["image_id"] in remap:
                truth["annotations"].append(
                    {**row, "id": len(truth["annotations"]) + 1, "image_id": remap[row["image_id"]]}
                )
        predictions.extend({**r, "image_id": remap[r["image_id"]]} for r in evaluation.predictions)
    if len(set(keys)) != len(keys) or keys != report["keys"]:
        raise ValueError("pooled coverage duplicates/omits/substitutes child consumption")
    saved_frames = [
        parse_json(r)
        for r in _artifact(root, manifest, "per_image_prediction_log").read_text().splitlines()
    ]
    if len(saved_frames) != len(frame_ids) or any(
        any(actual.get(k) != v for k, v in expected.items())
        for expected, actual in zip(frame_ids, saved_frames, strict=True)
    ):
        raise ValueError("pooled frame identities differ from child inference")
    for role, expected in (
        ("pooled_coco_annotations", truth),
        ("canonical_coco_predictions", predictions),
    ):
        if canonical_digest(read_json(_artifact(root, manifest, role))) != canonical_digest(
            expected
        ):
            raise ValueError(f"{role} does not reproduce from attested children")
    resolved = read_json(_artifact(root, manifest, "resolved_evaluation_configuration"))
    if (
        resolved["source_evaluations"] != children
        or resolved["model"] != report["model"]
        or canonical_digest(resolved["evaluation"]) != canonical_digest(policy)
    ):
        raise ValueError("pooled resolved configuration differs from child consumption")
    summary = read_json(_artifact(root, manifest, "evaluation_summary"))
    recomputed = evaluate_coco(truth, predictions, image_ids=[r["id"] for r in truth["images"]])
    if (
        summary["image_count"] != len(keys)
        or summary["coco"] != dict(recomputed.aggregate)
        or summary["model_inference_reexecuted"] is not False
    ):
        raise ValueError("pooled AP/count is not recomputed child evidence")
    if require_holdout:
        wanted = {
            sample_key(s["manifest_sha256"], s["sample_id"])
            for s in inv["samples"]
            if partition_role(s["partition"]) == "test"
        }
        if not wanted or set(keys) != wanted:
            raise ValueError("pooled child execution does not cover the exact frozen holdout")
    return {
        "status": "passed",
        "task": "pooled_detector_evaluation",
        "keys": keys,
        "phases": {"evaluation": sorted({r["partition"] for r in frame_ids})},
        "checkpoints": {
            "evaluated": {k: report["model"][k] for k in ("path", "sha256", "size_bytes")}
        },
        "source_manifests": sources,
        "training_origin": report["training_origin"],
        "report": fingerprint_file(report_path),
        "scope": "derived_saved_child_consumption_not_new_inference",
    }
