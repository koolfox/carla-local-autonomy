"""Bind canonical and pooled saved predictions to producer-observed sample use."""

from __future__ import annotations

from pathlib import Path

from ..artifacts import fingerprint_file
from ..dataset.lineage import canonical_digest, partition_role
from ..evaluation.verified import load_verified_evaluation
from ..verification import verify_research_object
from .consumption import sample_key, verify_consumption
from .review_contracts import fields, parse_json, read_json, relative_path
from .study_links import _reference, verify_split


def verify_evaluation_coverage(root, *, split_path, require_full_holdout=False):
    root = Path(root).resolve(strict=True)
    verify_research_object(root, deep=False, verify_references=True, reject_unregistered=True)
    manifest = read_json(root / "manifest.json")

    def artifact(role):
        entries = [a for a in manifest["artifacts"] if a["role"] == role]
        if len(entries) != 1:
            raise ValueError(f"missing/ambiguous consumption coverage role {role}")
        return relative_path(root, entries[0]["path"])

    resolved = read_json(artifact("resolved_evaluation_configuration"))
    if resolved["object_type"] == "offline_detector_evaluation":
        evaluation = load_verified_evaluation(root)
        receipt = verify_consumption(root, split_path=split_path)
        if receipt["producer"] != "detector_evaluation" or len(receipt["passes"]) != 1:
            raise ValueError("invalid detector consumption")
        observed = receipt["passes"][0]
        keys, outputs = observed["keys"], observed["outputs"]
        weight = receipt["selected_weights"]
        references = [evaluation.dataset.reference]
    elif resolved["object_type"] == "pooled_detector_evaluation":
        receipt = read_json(artifact("pooled_sample_consumption"))
        fields(
            receipt,
            {"schema_version", "split_inventory", "source_evaluations", "inference_reexecuted"},
        )
        if receipt["schema_version"] != "1.0" or receipt["inference_reexecuted"] is not False:
            raise ValueError("pool must not claim repeated inference")
        if _reference(root, receipt["split_inventory"]) != Path(split_path).resolve():
            # Relocation of equivalent inventory files is allowed, not changed membership.
            if receipt["split_inventory"]["sha256"] != fingerprint_file(split_path)["sha256"]:
                raise ValueError("pool split differs from study")
        refs = receipt["source_evaluations"]
        if refs != resolved["source_evaluations"] or not isinstance(refs, list) or len(refs) < 2:
            raise ValueError("pool source evaluations differ")
        expected_frames, expected_predictions, keys, outputs, references = [], [], [], [], []
        weight = None
        for ref in refs:
            child_root = _reference(root, ref).parent
            # Canonical children only: no recursive reference cycles/duplicate pooling.
            child = load_verified_evaluation(child_root)
            child_policy = child.config.as_dict()
            child_policy.pop("evaluation_id")
            if (
                canonical_digest(child_policy) != canonical_digest(resolved["evaluation"])
                or child.resolved["model"] != resolved["model"]
            ):
                raise ValueError("pool model or policy differs from its consumed children")
            observed = verify_consumption(child_root, split_path=split_path)
            if observed["producer"] != "detector_evaluation" or len(observed["passes"]) != 1:
                raise ValueError("pool child lacks canonical observed consumption")
            if weight is None:
                weight = observed["selected_weights"]
            elif weight != observed["selected_weights"]:
                raise ValueError("pool mixes consumed model weights")
            p = observed["passes"][0]
            references.append(child.dataset.reference)
            for frame, key, prediction in zip(
                child.frame_rows, p["keys"], p["outputs"], strict=True
            ):
                new_id = len(keys) + 1
                keys.append(key)
                remapped = [{**r, "image_id": new_id} for r in prediction]
                outputs.append(remapped)
                expected_predictions.extend(remapped)
                expected_frames.append(
                    {
                        "image_id": new_id,
                        "source_image_id": frame["image_id"],
                        "source_dataset_manifest_sha256": key[:64],
                        "sample_id": frame["sample_id"],
                        "partition": frame["partition"],
                    }
                )
        frames = [
            parse_json(line)
            for line in artifact("per_image_prediction_log").read_text().splitlines()
        ]
        if len(frames) != len(expected_frames):
            raise ValueError("pool omitted or duplicated consumed frames")
        for actual, expected in zip(frames, expected_frames, strict=True):
            if canonical_digest({k: actual.get(k) for k in expected}) != canonical_digest(expected):
                raise ValueError("pool remapping differs from observed source samples")
        if canonical_digest(read_json(artifact("canonical_coco_predictions"))) != canonical_digest(
            expected_predictions
        ):
            raise ValueError("pool predictions differ from consumed child outputs")
    else:
        raise ValueError("unsupported producer for empirical sample coverage")
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("duplicate or empty observed evaluation coverage")
    inventory = verify_split(split_path)
    membership = {sample_key(s): s for s in inventory["samples"]}
    if not set(keys) <= membership.keys():
        raise ValueError("processed samples outside frozen inventory")
    if require_full_holdout:
        expected = {k for k, s in membership.items() if partition_role(s["partition"]) == "test"}
        if not expected or set(keys) != expected:
            raise ValueError("actual evaluation does not cover the exact frozen holdout")
    return {
        "status": "passed",
        "keys": keys,
        "outputs": outputs,
        "selected_weights": weight,
        "dataset_references": references,
        "sample_count": len(keys),
        "complete_frozen_holdout": require_full_holdout,
        "scope": "producer_observed_inputs_and_saved_outputs_not_independent_execution_proof",
    }


def check_consumed_study(paths):
    """Mandatory empirical gate; legacy declarations cannot satisfy observed use."""
    protocol = read_json(paths["protocol"])
    split = _reference(paths["protocol"].parent, protocol["split_manifest"])
    results = read_json(paths["results"])
    card = read_json(paths["model_card"])
    trained, pretrained = {}, set()
    training_refs, baseline_refs = (
        card.get("training_consumption"),
        card.get("pretrained_baselines"),
    )
    if not isinstance(training_refs, list) or not isinstance(baseline_refs, list):
        raise ValueError(
            "model card must distinguish observed training from declared pretrained baselines"
        )
    for ref in training_refs:
        run = _reference(paths["model_card"].parent, ref).parent
        receipt = verify_consumption(run, split_path=split)
        if not receipt["producer"].endswith("training"):
            raise ValueError("model training reference is not a training consumption producer")
        key = receipt["selected_weights"]["sha256"]
        if key in trained:
            raise ValueError("ambiguous selected training checkpoint")
        trained[key] = ref
    for ref in baseline_refs:
        fields(ref, {"weights", "reason"})
        from .review_contracts import text

        text(ref["reason"], "pretrained baseline reason")
        weight = _reference(paths["model_card"].parent, ref["weights"])
        key = fingerprint_file(weight)["sha256"]
        if key in pretrained or key in trained:
            raise ValueError("duplicate/ambiguous pretrained model identity")
        pretrained.add(key)
    counts, evaluated_training = {}, set()
    for run_id, binding in results["producers"].items():
        root = _reference(paths["results"].parent, binding["manifest"]).parent
        result = verify_evaluation_coverage(root, split_path=split, require_full_holdout=True)
        weight = result["selected_weights"]["sha256"]
        if weight not in trained and weight not in pretrained:
            raise ValueError(
                "evaluated weights have no training receipt or declared pretrained baseline"
            )
        if weight in trained:
            evaluated_training.add(weight)
        counts[run_id] = result["sample_count"]
    claims = read_json(paths["claim_registry"])
    if (
        any(c.get("scope") == "training_utility" for c in claims["claims"])
        and not evaluated_training
    ):
        raise ValueError("training-utility claim has no evaluated observed-training checkpoint")
    if not counts:
        raise ValueError("study lacks evaluated consumption producers")
    return {"status": "passed", "processed_by_run": counts, "evidence": "actual_recorded_coverage"}
