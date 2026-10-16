"""Pool canonical evaluation shards before computing AP; never average shard AP.

This producer uses retained predictions from the same model and evaluation policy.
It does not rerun inference and it rejects duplicated source samples. Annotation
and prediction IDs are remapped together. COCO evaluation is memory-resident and
has an explicit image cap; this is not an unbounded streaming COCO implementation.
"""

from __future__ import annotations

import argparse
import json

from ..artifacts import RunArtifactTracker, fingerprint_file
from ..dataset.lineage import scene_group_key
from ..reproduction.review_contracts import integer, read_json
from .coco import evaluate_coco
from .metrics import operating_metrics
from .uncertainty import bootstrap_coco, supported_ratios
from .verified import load_verified_evaluation


def pool_evaluations(
    roots, *, runs_root="runs", run_id=None, maximum_images=1_000_000, bootstrap_replicates=0
):
    integer(maximum_images, "maximum_images", minimum=1)
    integer(bootstrap_replicates, "bootstrap_replicates")
    if len(roots) < 2:
        raise ValueError("pool at least two explicit canonical evaluations")
    truth = {"images": [], "annotations": [], "categories": []}
    predictions, frames, references = [], [], []
    seen, groups, group_partitions = set(), {}, {}
    policy, model, config = None, None, None
    for raw_root in roots:
        evaluation = load_verified_evaluation(raw_root)
        current_policy = evaluation.config.as_dict()
        # Execution identity differs per shard; scientific policy may not.
        current_policy.pop("evaluation_id")
        if policy is None:
            policy, model, config = current_policy, evaluation.resolved["model"], evaluation.config
            truth["categories"] = list(evaluation.dataset.coco["categories"])
        elif (
            current_policy != policy
            or evaluation.resolved["model"] != model
            or list(evaluation.dataset.coco["categories"]) != truth["categories"]
        ):
            raise ValueError("pooled evaluations differ in model, ontology or evaluation policy")
        references.append(fingerprint_file(evaluation.root / "manifest.json"))
        source_hash = fingerprint_file(evaluation.dataset.root / "manifest.json")["sha256"]
        samples = {s["image_id"]: s for s in evaluation.dataset.dataset["samples"]}
        source_images = {i["id"]: i for i in evaluation.dataset.coco["images"]}
        remap = {}
        for row in evaluation.frame_rows:
            old_id = row["image_id"]
            sample = samples[old_id]
            identity = (source_hash, sample["sample_id"])
            if identity in seen:
                raise ValueError("same source sample counted more than once")
            if len(truth["images"]) >= maximum_images:
                raise ValueError("pooled COCO image budget exceeded")
            seen.add(identity)
            new_id = len(truth["images"]) + 1
            remap[old_id] = new_id
            metadata = read_json(evaluation.dataset.root / sample["metadata"]["path"])
            group = scene_group_key(
                metadata["context"], episode_id=f"{sample['scenario_id']}/{sample['episode_id']}"
            )
            if group_partitions.setdefault(group, sample["split"]) != sample["split"]:
                raise ValueError("pooled source scene crosses partitions")
            groups[new_id] = group
            truth["images"].append({**source_images[old_id], "id": new_id})
            frames.append(
                {
                    "image_id": new_id,
                    "source_image_id": old_id,
                    "source_dataset_manifest_sha256": source_hash,
                    "sample_id": sample["sample_id"],
                    "group_id": group,
                    "partition": sample["split"],
                }
            )
        for annotation in evaluation.dataset.coco["annotations"]:
            if annotation["image_id"] in remap:
                truth["annotations"].append(
                    {
                        **annotation,
                        "id": len(truth["annotations"]) + 1,
                        "image_id": remap[annotation["image_id"]],
                    }
                )
        predictions.extend({**p, "image_id": remap[p["image_id"]]} for p in evaluation.predictions)
    image_ids = [i["id"] for i in truth["images"]]
    result = evaluate_coco(truth, predictions, image_ids=image_ids)
    operating = operating_metrics(
        truth["annotations"],
        predictions,
        image_ids=image_ids,
        categories={c["id"]: c["name"] for c in truth["categories"]},
        confidence_threshold=config.operating_confidence,
        iou_threshold=config.matching_iou,
    )
    summary = {
        "schema_version": "1.0",
        "status": "success",
        "coco": dict(result.aggregate),
        "purpose": config.purpose,
        "model": model,
        "image_count": len(image_ids),
        "source_evaluation_count": len(roots),
        "supported_operating_point": supported_ratios(
            int(operating.overall["tp"]), int(operating.overall["fp"]), int(operating.overall["fn"])
        ),
        "ap_group_bootstrap": bootstrap_coco(
            truth,
            predictions,
            group_by_image=groups,
            replicates=bootstrap_replicates,
            seed=config.master_seed,
            confidence=config.bootstrap_confidence,
        )
        if bootstrap_replicates
        else None,
        "shard_aps_averaged": False,
        "model_inference_reexecuted": False,
        "scope": "pooled_saved_predictions_same_model_and_evaluation_policy",
    }
    resolved = {
        "object_type": "pooled_detector_evaluation",
        "evaluation": policy,
        "model": model,
        "source_evaluations": references,
    }
    with RunArtifactTracker(
        runs_root, run_id=run_id, input_refs=references, config=resolved
    ) as tracker:
        for name, value, role in (
            ("resolved_evaluation_config.json", resolved, "resolved_evaluation_configuration"),
            ("instances.coco.json", truth, "pooled_coco_annotations"),
            ("predictions.coco.json", predictions, "canonical_coco_predictions"),
            ("summary_metrics.json", summary, "evaluation_summary"),
        ):
            path = tracker.artifact_path(name)
            path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
            tracker.register_artifact(path, role=role)
        frame_path = tracker.artifact_path("frames.jsonl")
        frame_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in frames))
        tracker.register_artifact(frame_path, role="per_image_prediction_log")
        log = tracker.artifact_path("coco.log")
        log.write_text(result.log)
        tracker.register_artifact(log, role="coco_evaluation_log")
    return {"run_dir": str(tracker.run_dir), **summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", action="append", required=True)
    parser.add_argument("--maximum-images", type=int, default=1_000_000)
    parser.add_argument("--bootstrap-replicates", type=int, default=0)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            pool_evaluations(
                args.evaluation,
                runs_root=args.runs_root,
                run_id=args.run_id,
                maximum_images=args.maximum_images,
                bootstrap_replicates=args.bootstrap_replicates,
            ),
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
