"""Cross-check in-loop receipts with the producer's saved model outputs.

The threat model is accidental omission, substitution, or inconsistent retained
artifacts. This is not hardware attestation against a malicious producer. No
untrusted model checkpoint is deserialized by these verification routines.
"""

from __future__ import annotations

import math
from pathlib import Path

from ..artifacts import fingerprint_file
from ..dataset.lineage import canonical_digest, partition_role
from ..dataset.verified import load_verified_dataset
from .consumption import _artifact, sample_key, verify_consumption
from .review_contracts import fields, integer, number, read_json
from .study_links import _reference, verify_split


def _plain(ref):
    return {k: ref[k] for k in ("path", "sha256", "size_bytes")}


def _control_metrics(pass_record, metrics, *, targets=None):
    """Replay pooled (nonlinear) control metrics, not averages of batch RMSE."""
    outputs = [o for b in pass_record["batches"] for o in b["outputs"]]
    if metrics.get("samples") != len(outputs) or metrics.get("batches") != len(
        pass_record["batches"]
    ):
        raise ValueError("saved control counts differ from actual consumption")
    squares = [0.0, 0.0]
    absolute = [0.0, 0.0]
    direction = tp = fp = fn = 0
    for key, output in zip(pass_record["keys"], outputs, strict=True):
        fields(output, {"steer", "longitudinal", "target"})
        values = [number(output[k], k) for k in ("steer", "longitudinal")]
        target = output["target"]
        if not isinstance(target, list) or len(target) != 2:
            raise ValueError("consumed target shape is invalid")
        target = [number(x, "control target") for x in target]
        if targets is not None and any(
            not math.isclose(a, b, abs_tol=1e-6) for a, b in zip(target, targets[key], strict=True)
        ):
            raise ValueError("consumed evaluation target differs from verified supervision")
        for i in range(2):
            squares[i] += (values[i] - target[i]) ** 2
            absolute[i] += abs(values[i] - target[i])
        direction += (values[0] > 0) - (values[0] < 0) == (target[0] > 0) - (target[0] < 0)
        true_brake, pred_brake = target[1] < -0.05, values[1] < -0.05
        tp += true_brake and pred_brake
        fp += not true_brake and pred_brake
        fn += true_brake and not pred_brake
    n = len(outputs)
    computed = {
        "steer_rmse": math.sqrt(squares[0] / n),
        "longitudinal_rmse": math.sqrt(squares[1] / n),
        "steer_mae": absolute[0] / n,
        "longitudinal_mae": absolute[1] / n,
        "steer_direction_accuracy": direction / n,
        "brake_precision": tp / (tp + fp) if tp + fp else None,
        "brake_recall": tp / (tp + fn) if tp + fn else None,
        "brake_prediction_support": tp + fp,
        "brake_ground_truth_support": tp + fn,
    }
    for name, value in computed.items():
        actual = metrics.get(name)
        if value is None:
            if actual is not None:
                raise ValueError(f"unsupported control metric {name} must be null")
        elif actual is None or not math.isclose(
            number(actual, name), value, abs_tol=2e-6, rel_tol=2e-6
        ):
            raise ValueError(f"saved control metric {name} differs from consumption")


def _targets(trace):
    targets = {}
    for reference in trace["plan"]["source_manifests"]:
        root = Path(reference["path"]).parent
        data = load_verified_dataset(root)
        for sample in data.dataset["samples"]:
            meta = read_json(root / sample["metadata"]["path"])
            control = meta["context"]["privileged_teacher_control"]
            targets[sample_key(reference["sha256"], sample["sample_id"])] = [
                control["steer"],
                control["throttle"] - control["brake"],
            ]
    return targets


def _single_pass(trace, phase):
    if (
        set(trace["expected"]) != {phase}
        or len(trace["passes"]) != 1
        or trace["passes"][0]["epoch"] != 0
    ):
        raise ValueError("evaluation must have one complete inference pass")
    return trace["passes"][0]


def _detector(root, trace, *, ancestry):
    from ..evaluation.verified import load_verified_evaluation

    evaluation = load_verified_evaluation(root, verify_receipts=False)
    record = _single_pass(trace, "evaluation")
    source = evaluation.dataset.reference["manifest"]["sha256"]
    keys = [sample_key(source, row["sample_id"]) for row in evaluation.frame_rows]
    outputs = [row["mapped_predictions"] for row in evaluation.frame_rows]
    actual = [o for batch in record["batches"] for o in batch["outputs"]]
    if keys != record["keys"] or canonical_digest(outputs) != canonical_digest(actual):
        raise ValueError("saved detector predictions differ from consumed sample outputs")
    if trace["plan"]["model"] != evaluation.resolved["model"] or trace["checkpoints"] != {
        "evaluated": _plain(evaluation.resolved["model"])
    }:
        raise ValueError("evaluated weights differ from the consumption model binding")
    _verify_origin(root, trace, evaluation.resolved, ancestry=ancestry)
    return keys


def _imitation(root, manifest, trace, *, ancestry):
    config = manifest["invocation"]["config"]
    if trace["plan"]["task"] == "imitation_evaluation":
        record = _single_pass(trace, "test")
        model = trace["plan"]["model"]
        if (
            trace["checkpoints"] != {"evaluated": model}
            or model not in manifest["references"]["models"]
        ):
            raise ValueError("imitation evaluated checkpoint is not bound to the producer")
        _verify_origin(root, trace, config, ancestry=ancestry, required=True)
        summary = read_json(_artifact(root, manifest, "imitation_test_summary"))
        _control_metrics(record, summary["metrics"], targets=_targets(trace))
        return record["keys"]
    if set(trace["expected"]) != {"train", "val"}:
        raise ValueError("attested imitation training requires train/validation phases")
    first = integer(config["start_epoch"], "first epoch", minimum=1)
    last = integer(config["total_epochs"], "total epochs", minimum=first)
    expected = [(phase, epoch) for epoch in range(first, last + 1) for phase in ("train", "val")]
    if [(p["phase"], p["epoch"]) for p in trace["passes"]] != expected:
        raise ValueError("training epochs/phases differ from actual consumed passes")
    model = trace["plan"]["model"]
    if model["initial_epoch"] != first - 1:
        raise ValueError("training initial epoch differs from receipt")
    history = read_json(_artifact(root, manifest, "training_history"))
    if [r["epoch"] for r in history] != list(range(1, last + 1)):
        raise ValueError("training history is incomplete")
    targets = _targets(trace)
    for record in trace["passes"]:
        _control_metrics(
            record,
            history[record["epoch"] - 1][record["phase"]],
            targets=targets if record["phase"] == "val" else None,
        )
    refs = {
        name: fingerprint_file(_artifact(root, manifest, name + "_checkpoint"))
        for name in ("best", "last")
    }
    if trace["checkpoints"] != refs:
        raise ValueError("selected best/last checkpoints differ from consumption")
    summary = read_json(_artifact(root, manifest, "training_summary"))
    best = min(history, key=lambda r: number(r["val"]["loss"], "validation loss"))
    if (
        summary["best_epoch"] != best["epoch"]
        or summary["best_validation_loss"] != best["val"]["loss"]
        or summary["test_evaluated"] is not False
    ):
        raise ValueError("chosen weights/validation decision inconsistent with consumed history")
    if first > 1:
        parent_path = _reference(root, model["resume_manifest"])
        parent = verify_producer_consumption(
            parent_path.parent,
            inventory_path=trace["plan"]["inventory"]["path"],
            _ancestry=ancestry,
        )
        if (
            parent["task"] != "imitation_training"
            or parent["checkpoints"]["last"] != model["resume_checkpoint"]
        ):
            raise ValueError("resume lineage does not bind prior consumed batches")
        parent_manifest = read_json(parent_path)
        if parent_manifest["invocation"]["config"]["total_epochs"] != first - 1:
            raise ValueError("resume skipped unverified epochs")
        parent_history = read_json(
            _artifact(parent_path.parent, parent_manifest, "training_history")
        )
        if history[: first - 1] != parent_history:
            raise ValueError("resume changed already published training history")
    elif "resume_manifest" in model or "resume_checkpoint" in model:
        raise ValueError("unexpected resume source for first epoch")
    return []


def verify_producer_consumption(root, *, inventory_path=None, require_holdout=False, _ancestry=()):
    """Verify one instrumented producer; opaque/legacy producers fail closed."""
    root = Path(root).resolve(strict=True)
    manifest_ref = fingerprint_file(root / "manifest.json")
    if manifest_ref["sha256"] in _ancestry or len(_ancestry) >= 64:
        raise ValueError("cyclic or excessive consumed-sample lineage")
    ancestry = (*_ancestry, manifest_ref["sha256"])
    manifest = read_json(root / "manifest.json")
    kind = manifest["invocation"]["config"].get("object_type")
    if kind == "pooled_detector_evaluation":
        from .pooled_consumption import verify_pooled_consumption

        return verify_pooled_consumption(
            root, inventory_path=inventory_path, require_holdout=require_holdout, ancestry=ancestry
        )
    trace = verify_consumption(root, inventory_path=inventory_path)
    task = trace["plan"]["task"]
    kinds = {
        "detector_evaluation": "offline_detector_evaluation",
        "imitation_evaluation": "imitation_locked_evaluation",
        "imitation_training": "imitation_training",
        "detector_training": "detector_training",
    }
    if kinds[task] != kind:
        raise ValueError("producer type differs from claimed consumption task")
    if task == "detector_evaluation":
        keys = _detector(root, trace, ancestry=ancestry)
    elif task.startswith("imitation_"):
        keys = _imitation(root, manifest, trace, ancestry=ancestry)
    else:
        from ..training.consumption import verify_training_receipts

        keys = verify_training_receipts(root, manifest, trace)
    if require_holdout:
        inv = verify_split(trace["plan"]["inventory"]["path"])
        wanted = {
            sample_key(s["manifest_sha256"], s["sample_id"])
            for s in inv["samples"]
            if partition_role(s["partition"]) == "test"
        }
        if not wanted or len(keys) != len(wanted) or set(keys) != wanted:
            raise ValueError("actual consumed evaluation does not cover the exact frozen holdout")
    return {
        "status": "passed",
        "task": task,
        "keys": keys,
        "phases": trace["expected"],
        "checkpoints": trace["checkpoints"],
        "report": trace["report"],
        "source_manifests": trace["plan"]["source_manifests"],
        "training_origin": manifest["invocation"]["config"].get("training_origin"),
        "scope": "recorded_post_success_software_execution_not_hardware_attestation",
    }


def training_origin_reference(root, *, checkpoint, inventory_path):
    """Called BEFORE inference: verify the producing trainer and its chosen best file."""
    root = Path(root).resolve(strict=True)
    result = verify_producer_consumption(root, inventory_path=inventory_path)
    if (
        result["task"] not in {"detector_training", "imitation_training"}
        or _plain(checkpoint) != result["checkpoints"]["best"]
    ):
        raise ValueError("evaluation weights are not the selected best checkpoint of this trainer")
    return fingerprint_file(root / "manifest.json")


def _verify_origin(root, trace, config, *, ancestry, required=False):
    origin = config.get("training_origin")
    if origin is None:
        if required:
            raise ValueError("attested model evaluation lacks its producing training run")
        return
    parent = _reference(root, origin)
    result = verify_producer_consumption(
        parent.parent, inventory_path=trace["plan"]["inventory"]["path"], _ancestry=ancestry
    )
    if (
        result["task"] not in {"detector_training", "imitation_training"}
        or result["checkpoints"]["best"] != trace["checkpoints"]["evaluated"]
    ):
        raise ValueError("evaluated model is not the trainer's selected best checkpoint")
    manifest = read_json(root / "manifest.json")
    if origin not in manifest["references"]["inputs"]:
        raise ValueError("training origin is not registered by the evaluator")


def validate_execution_policy(policy, methods):
    """Freeze whether each arm trains or evaluates a specified external baseline."""
    if not isinstance(policy, dict) or set(policy) != set(methods):
        raise ValueError("sample execution policy must cover every registered method")
    for method, row in policy.items():
        fields(row, {"model_origin"}, optional={"checkpoint"})
        if row["model_origin"] == "trained":
            if "checkpoint" in row:
                raise ValueError("trained arm must bind weights at production, not guess them")
        elif row["model_origin"] == "external_frozen":
            if "checkpoint" not in row:
                raise ValueError("external baseline must pin its exact checkpoint")
            _reference(Path.cwd(), row["checkpoint"])
        else:
            raise ValueError(f"unsupported model origin for {method}")


def check_study_consumption(paths, *, source_commit):
    """Mandatory empirical gate: declared held-out membership is not enough."""
    from .test_execution import verify_test_execution

    if not {"protocol", "results", "tests", "source_snapshot", "claim_registry"} <= paths.keys():
        raise ValueError("actual sample execution evidence is incomplete")
    protocol = read_json(paths["protocol"])
    inventory = _reference(paths["protocol"].parent, protocol["split_manifest"])
    policy = protocol.get("sample_execution")
    validate_execution_policy(policy, protocol["methods"])
    claims = read_json(paths["claim_registry"])
    if any(c["scope"] == "training_utility" for c in claims["claims"]) and not any(
        v["model_origin"] == "trained" for v in policy.values()
    ):
        raise ValueError("training utility cannot be justified only by externally frozen models")
    tests = verify_test_execution(
        paths["tests"], source_archive=paths["source_snapshot"], source_commit=source_commit
    )
    results = read_json(paths["results"])
    if not results["producers"] or not results["rows"]:
        raise ValueError("no actual result producers")
    checked = {}
    for row in results["rows"]:
        binding = results["producers"][row["run_id"]]
        root = _reference(paths["results"].parent, binding["manifest"]).parent
        result = verify_producer_consumption(root, inventory_path=inventory, require_holdout=True)
        verify_execution_source_tree(root, paths["source_snapshot"], source_commit=source_commit)
        origin = result["training_origin"]
        arm = policy[row["method"]]
        if arm["model_origin"] == "trained":
            if origin is None:
                raise ValueError("trained result lacks producer-bound training sample evidence")
            training = read_json(_reference(root, origin))
            if (
                training.get("git", {}).get("commit") != source_commit
                or training.get("git", {}).get("dirty") is not False
            ):
                raise ValueError("training execution is not bound to selected clean source")
        else:
            if origin is not None or result["checkpoints"]["evaluated"] != arm["checkpoint"]:
                raise ValueError("external baseline differs from preregistered weights/origin")
        checked[row["run_id"]] = {
            "receipt": result["report"],
            "consumed_holdout_samples": len(result["keys"]),
            "model_origin": arm["model_origin"],
        }
    return {
        "status": "passed",
        "tests": tests,
        "producers": checked,
        "scope": "producer_recorded_execution_not_hardware_attestation_or_empirical_truth",
    }


def verify_execution_source_tree(root, archive_path, *, source_commit):
    """Compare executed producer byte digests with the selected, Git-proven source ZIP.

    Package file digests are recorded in the producer, not guessed from its working
    directory. This still trusts the Python process, not hostile monkeypatches.
    """
    import hashlib
    import zipfile

    from .reviewer import _source_archive

    _source_archive(Path(archive_path), source_commit)
    seen = set()
    with zipfile.ZipFile(archive_path) as archive:

        def visit(directory):
            directory = Path(directory).resolve(strict=True)
            identity = fingerprint_file(directory / "manifest.json")["sha256"]
            if identity in seen:
                return
            if len(seen) >= 64:
                raise ValueError("execution source lineage exceeds review budget")
            seen.add(identity)
            manifest = read_json(directory / "manifest.json")
            kind = manifest["invocation"]["config"].get("object_type")
            if kind == "pooled_detector_evaluation":
                pooled = read_json(_artifact(directory, manifest, "pooled_consumed_sample_report"))
                for relative, expected in pooled["implementation"].items():
                    if hashlib.sha256(archive.read(relative)).hexdigest() != expected:
                        raise ValueError(
                            "executed pooling bytes differ from selected source archive"
                        )
                for reference in pooled["children"]:
                    visit(_reference(directory, reference).parent)
                return
            plan = read_json(_artifact(directory, manifest, "consumed_sample_plan"))
            for relative, expected in plan["implementation"].items():
                if hashlib.sha256(archive.read(relative)).hexdigest() != expected:
                    raise ValueError(
                        "executed producer bytes differ from the selected source archive"
                    )
            for reference in (
                manifest["invocation"]["config"].get("training_origin"),
                plan["model"].get("resume_manifest"),
            ):
                if reference is not None:
                    visit(_reference(directory, reference).parent)

        visit(root)
    return {"status": "passed", "producer_runs": len(seen), "source_commit": source_commit}


def main(argv=None):
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--sample-inventory")
    parser.add_argument("--require-holdout", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            verify_producer_consumption(
                args.run, inventory_path=args.sample_inventory, require_holdout=args.require_holdout
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
