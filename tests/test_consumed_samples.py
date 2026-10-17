"""Producer-bound consumption fixtures, not empirical model-performance evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_evaluation import build_dataset

from carla_vision.artifacts import RunArtifactTracker, fingerprint_file
from carla_vision.reproduction.consumption import (
    consumption_session,
    sample_key,
    verify_consumption,
)
from carla_vision.reproduction.study_links import freeze_split


def case(tmp_path, partition="test_seen"):
    root = build_dataset(tmp_path / "data", split=partition)
    split = tmp_path / "split.json"
    inventory = freeze_split([root], split)
    rows = inventory["samples"]
    weights = tmp_path / "model.bin"
    weights.write_bytes(b"test-only fake model weights")
    spec = {
        "pass_id": "evaluation",
        "role": "evaluation",
        "partitions": [partition],
        "source_manifests": [fingerprint_file(root / "manifest.json")["sha256"]],
        "expected_keys": [sample_key(r) for r in rows],
        "ordered": True,
    }
    return root, split, weights, rows, spec


def run_case(tmp_path, *, omit=False, outputs=None):
    root, split, weights, rows, spec = case(tmp_path)
    tracker = RunArtifactTracker(
        tmp_path / "runs", run_id="run", config={"object_type": "offline_detector_evaluation"}
    )
    with (
        tracker,
        consumption_session(
            tracker, split, [spec], producer="detector_evaluation", initial_weights=weights
        ) as audit,
    ):
        if not omit:
            with audit.batch("evaluation", spec["expected_keys"]) as observed:
                observed["outputs"] = [[]] if outputs is None else outputs
        audit.seal(weights)
    return tracker.run_dir, split, rows


def test_empty_predictions_still_attest_processed_sample(tmp_path):
    root, split, rows = run_case(tmp_path)
    report = verify_consumption(root, split_path=split, require_complete=True)
    assert report["processed_samples"] == 1
    assert report["passes"][0]["outputs"] == [[]]
    assert report["passes"][0]["keys"] == [sample_key(rows[0])]


def test_skipped_sample_never_becomes_completed_evaluation(tmp_path):
    with pytest.raises(ValueError, match="coverage|missing"):
        run_case(tmp_path, omit=True)
    manifest = json.loads((tmp_path / "runs/run/manifest.json").read_text())
    assert manifest["status"] != "success"


@pytest.mark.parametrize("mutation", ["duplicate", "substitute", "wrong_pass"])
def test_invalid_batch_rejected_before_callback(tmp_path, mutation):
    _, split, weights, rows, spec = case(tmp_path)
    tracker = RunArtifactTracker(tmp_path / "runs", run_id="run")
    with pytest.raises(ValueError):
        with (
            tracker,
            consumption_session(
                tracker, split, [spec], producer="detector_evaluation", initial_weights=weights
            ) as audit,
        ):
            keys = spec["expected_keys"]
            if mutation == "duplicate":
                keys = keys * 2
            if mutation == "substitute":
                keys = ["a" * 64 + ":other"]
            with audit.batch("bad" if mutation == "wrong_pass" else "evaluation", keys):
                pytest.fail("invalid batch reached model call")


def test_failed_inference_records_failure_not_success(tmp_path):
    _, split, weights, _, spec = case(tmp_path)
    tracker = RunArtifactTracker(tmp_path / "runs", run_id="run")
    with pytest.raises(RuntimeError, match="model failed"):
        with (
            tracker,
            consumption_session(
                tracker, split, [spec], producer="detector_evaluation", initial_weights=weights
            ) as audit,
        ):
            with audit.batch("evaluation", spec["expected_keys"]):
                raise RuntimeError("model failed")
    report = json.loads((tracker.run_dir / "consumption/summary.json").read_text())
    assert report["status"] == "failed"
    assert report["processed_samples"] == 0
    assert report["omissions"][0]["reason"] == "run_failed_before_completion"


def test_changed_weights_abort_before_sealing(tmp_path):
    _, split, weights, _, spec = case(tmp_path)
    tracker = RunArtifactTracker(tmp_path / "runs", run_id="run")
    with pytest.raises(ValueError, match="weight"):
        with (
            tracker,
            consumption_session(
                tracker, split, [spec], producer="detector_evaluation", initial_weights=weights
            ) as audit,
        ):
            with audit.batch("evaluation", spec["expected_keys"]) as observed:
                observed["outputs"] = [[]]
            weights.write_bytes(b"changed")
            audit.seal(weights)


def test_train_may_not_claim_test_partition(tmp_path):
    _, split, _, _, spec = case(tmp_path)
    spec["role"] = "train"
    tracker = RunArtifactTracker(tmp_path / "runs", run_id="run")
    with pytest.raises(ValueError, match="partition|role"):
        with tracker, consumption_session(tracker, split, [spec], producer="detector_training"):
            pytest.fail("forbidden train role")


def test_old_run_without_receipt_fails_empirical_coverage(tmp_path):
    with RunArtifactTracker(tmp_path, run_id="old") as tracker:
        pass
    with pytest.raises((ValueError, RuntimeError), match="consumption"):
        verify_consumption(tracker.run_dir, require_complete=True)


def test_outputs_require_one_record_per_consumed_input(tmp_path):
    with pytest.raises(ValueError, match="output"):
        run_case(tmp_path, outputs=[])


def evaluate(tmp_path, root=None, split=None, weights=None, detector=None, name="eval"):
    from test_evaluation import FakeDetector, evaluation_config

    from carla_vision.contracts import DetectorConfig
    from carla_vision.evaluation.runner import run_evaluation

    if root is None:
        root, split, weights, _, _ = case(tmp_path)
    config = tmp_path / f"{name}.json"
    config.write_text(json.dumps(evaluation_config("test_seen")))
    result = run_evaluation(
        config_path=config,
        dataset_path=root,
        split_inventory_path=split,
        detector_config=DetectorConfig(backend="custom", weights=weights, factory="test:fake"),
        detector_override=detector or FakeDetector(),
        acknowledge_locked_test=True,
        runs_root=tmp_path / "runs",
        run_id=name,
    )
    return Path(result["run_dir"]), split


def rehash_run(root):
    """Test-only corruption: remove superficial hash detection, not semantic detection."""
    p = root / "manifest.json"
    raw = json.loads(p.read_text())
    for entry in raw["artifacts"]:
        ref = fingerprint_file(root / entry["path"])
        entry.update({k: ref[k] for k in ("sha256", "size_bytes")})
    p.write_text(json.dumps(raw))


def test_real_evaluation_loop_binds_predictions_to_consumed_inputs(tmp_path):
    from carla_vision.reproduction.coverage import verify_evaluation_coverage

    root, split = evaluate(tmp_path)
    report = verify_evaluation_coverage(root, split_path=split, require_full_holdout=True)
    assert report["sample_count"] == 1
    assert report["outputs"][0][0]["category_id"] == 5


def test_zero_detection_model_produces_complete_coverage(tmp_path):
    from test_evaluation import FakeDetector

    from carla_vision.reproduction.coverage import verify_evaluation_coverage

    class Empty(FakeDetector):
        def infer(self, image):
            return ()

    root, split = evaluate(tmp_path, detector=Empty())
    assert verify_evaluation_coverage(root, split_path=split, require_full_holdout=True)[
        "outputs"
    ] == [[]]


def test_posthoc_frames_cannot_replace_actual_loop_outputs(tmp_path):
    from carla_vision.evaluation.verified import load_verified_evaluation

    root, _ = evaluate(tmp_path)
    path = root / "consumption/batches.jsonl"
    trace = json.loads(path.read_text())
    trace["outputs"] = [[]]
    from carla_vision.dataset.lineage import canonical_digest

    trace["outputs_sha256"] = canonical_digest(trace["outputs"])
    path.write_text(json.dumps(trace) + "\n")
    summary = root / "consumption/summary.json"
    value = json.loads(summary.read_text())
    value["batches"] = fingerprint_file(path)
    summary.write_text(json.dumps(value))
    rehash_run(root)
    with pytest.raises(RuntimeError, match="consumption|outputs|predictions"):
        load_verified_evaluation(root)


def test_trace_removed_is_not_legacy_downgrade(tmp_path):
    from carla_vision.evaluation.verified import load_verified_evaluation

    root, _ = evaluate(tmp_path)
    path = root / "manifest.json"
    doc = json.loads(path.read_text())
    doc["artifacts"] = [a for a in doc["artifacts"] if a["role"] != "sample_consumption"]
    (root / "consumption/summary.json").unlink()
    path.write_text(json.dumps(doc))
    with pytest.raises(RuntimeError, match="consumption"):
        load_verified_evaluation(root)


def test_changed_rgb_is_not_decoded_as_frozen_input(tmp_path):
    from carla_vision.reproduction.consumption import decode_rgb

    root, _, _, _, _ = case(tmp_path)
    sample = json.loads((root / "dataset.json").read_text())["samples"][0]
    p = root / sample["rgb"]["path"]
    p.write_bytes(p.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="RGB bytes"):
        decode_rgb(p, sample["rgb"]["sha256"])


def test_pool_attests_union_and_does_not_run_model_again(tmp_path):
    from test_evaluation import make_pair

    from carla_vision.dataset.shards import ShardedDatasetWriter, expand_dataset_roots
    from carla_vision.evaluation.pool import pool_evaluations
    from carla_vision.reproduction.coverage import verify_evaluation_coverage

    with ShardedDatasetWriter(
        tmp_path / "data",
        dataset_id="shards",
        shard_samples=1,
        carla_endpoint={},
        carla_version="0.9.16",
        carla_map="fixture",
        minimum_pixels=4,
    ) as writer:
        for i in (1, 2):
            writer.add_pair(
                make_pair(i, i, 20 + i), split="test_seen", scenario_id="s", episode_id="e"
            )
    roots = expand_dataset_roots([tmp_path / "data/shards"])
    split = tmp_path / "split.json"
    freeze_split([tmp_path / "data/shards"], split)
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"synthetic fixture weights")
    runs = [
        evaluate(tmp_path, root, split, weights, name=f"eval{i}")[0] for i, root in enumerate(roots)
    ]
    # A single shard has internally complete inference but not the full study holdout.
    with pytest.raises(ValueError, match="exact frozen holdout"):
        verify_evaluation_coverage(runs[0], split_path=split, require_full_holdout=True)
    result = pool_evaluations(
        runs, runs_root=tmp_path / "runs", run_id="pooled", split_inventory_path=split
    )
    pooled = Path(result["run_dir"])
    coverage = verify_evaluation_coverage(pooled, split_path=split, require_full_holdout=True)
    assert coverage["sample_count"] == 2
    assert coverage["keys"][0] != coverage["keys"][1]  # repeated shard-local sample ID
    path = pooled / "predictions.coco.json"
    path.write_text("[]")
    rehash_run(pooled)
    with pytest.raises(ValueError, match="pool predictions"):
        verify_evaluation_coverage(pooled, split_path=split, require_full_holdout=True)


def test_opaque_training_backend_cannot_turn_data_yaml_into_receipt(tmp_path):
    from test_training import FakeBackend, build_trainable_dataset, training_config

    from carla_vision.training.runner import run_training

    data = build_trainable_dataset(tmp_path)
    split = tmp_path / "split.json"
    freeze_split([data], split)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(training_config()))
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="instrumentation"):
        run_training(
            config_path=config,
            dataset_path=data,
            pretrained_weights=weights,
            split_inventory_path=split,
            backend_override=FakeBackend(),
            runs_root=tmp_path / "runs",
        )


def test_instrumented_training_request_requires_complete_passes(tmp_path):
    from test_training import FakeBackend, build_trainable_dataset, training_config

    from carla_vision.training.runner import run_training

    class Instrumented(FakeBackend):
        consumption_contract_version = "1.0"

        def train(self, request):
            assert request.sample_consumption is not None
            for name, spec in request.sample_consumption.specs.items():
                with request.sample_consumption.batch(name, spec["expected_keys"]) as observed:
                    observed["outputs"] = [{"fixture_loss": 0.0} for _ in spec["expected_keys"]]
            return super().train(request)

    data = build_trainable_dataset(tmp_path)
    split = tmp_path / "split.json"
    freeze_split([data], split)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(training_config()))
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"fixture")
    result = run_training(
        config_path=config,
        dataset_path=data,
        pretrained_weights=weights,
        split_inventory_path=split,
        backend_override=Instrumented(),
        runs_root=tmp_path / "runs",
    )
    report = verify_consumption(result["run_dir"], split_path=split)
    assert len(report["passes"]) == 4
    assert report["processed_samples"] == 4


def teacher_release(tmp_path):
    from test_evaluation import make_pair

    from carla_vision.dataset.writer import DatasetWriter
    from carla_vision.native.teacher_timing import ObservationActionTransition

    with DatasetWriter(
        tmp_path / "data",
        dataset_id="teacher",
        carla_endpoint={},
        carla_version="0.9.16",
        carla_map="fixture",
        minimum_pixels=4,
    ) as writer:
        for i, partition in enumerate(("train", "train", "val_seen", "test_seen"), 1):
            writer.add_pair(
                make_pair(i, i, 40 + i),
                split=partition,
                scenario_id=f"s{i}",
                episode_id=f"e{i}",
                context={
                    "control_mode": "behavior_agent_teacher",
                    "route": {
                        "route_id": f"route-{i}",
                        "start_spawn_index": i * 2,
                        "destination_spawn_index": i * 2 + 1,
                        "destination_world_transform": {"x": float(i), "y": 0.0, "z": 0.0},
                    },
                    "privileged_teacher_control": {
                        "source": "BehaviorAgent.run_step",
                        "carla_frame": i,
                        "throttle": 0.2,
                        "steer": 0.1,
                        "brake": 0.0,
                    },
                    "privileged_evaluation": {"velocity_mps": {"speed": 2.0}},
                    "teacher_transition": ObservationActionTransition(
                        i, i, i + 1, "fixture"
                    ).as_dict(),
                },
            )
        writer.set_release_metadata(
            {
                "control_mode": "behavior_agent_teacher",
                "episode_ids": [f"e{i}" for i in range(1, 5)],
            }
        )
    return writer.dataset_dir


def test_actual_cpu_imitation_training_and_test_have_separate_receipts(tmp_path):
    from carla_vision.imitation import runner
    from carla_vision.imitation.evaluate import main as evaluate_imitation

    data = teacher_release(tmp_path)
    split = tmp_path / "split.json"
    freeze_split([data], split)
    output = tmp_path / "training"
    args = runner.build_parser().parse_args(
        [
            "--dataset",
            str(data),
            "--split-inventory",
            str(split),
            "--output",
            str(output),
            "--image-size",
            "16x24",
            "--batch-size",
            "1",
            "--device",
            "cpu",
            "--epochs",
            "2",
            "--encoder-channels",
            "8",
            "--hidden-dim",
            "16",
            "--dropout",
            "0",
        ]
    )
    runner.run(args)
    receipt = verify_consumption(output, split_path=split)
    assert receipt["processed_samples"] == 6  # (2 train + 1 val) x 2 epochs
    assert all("test_seen" not in p["partitions"] for p in receipt["passes"])
    assert receipt["selected_weights"] == fingerprint_file(output / "checkpoint_best.pt")
    assert (
        evaluate_imitation(
            [
                "--checkpoint",
                str(output / "checkpoint_best.pt"),
                "--dataset",
                str(data),
                "--split-inventory",
                str(split),
                "--output",
                str(tmp_path / "testing"),
                "--device",
                "cpu",
                "--acknowledge-locked-test",
            ]
        )
        == 0
    )
    evaluated = verify_consumption(tmp_path / "testing", split_path=split)
    assert evaluated["processed_samples"] == 1
    assert evaluated["passes"][0]["partitions"] == ["test_seen"]
    assert evaluated["selected_weights"] == receipt["selected_weights"]


def test_imitation_prefetch_without_successful_optimizer_is_not_consumed(tmp_path, monkeypatch):
    from carla_vision.imitation import runner

    data = teacher_release(tmp_path)
    split = tmp_path / "split.json"
    freeze_split([data], split)

    def fail(*args, **kwargs):
        raise RuntimeError("optimizer failed")

    monkeypatch.setattr(runner.torch.optim.AdamW, "step", fail)
    with pytest.raises(RuntimeError, match="optimizer failed"):
        runner.run(
            runner.build_parser().parse_args(
                [
                    "--dataset",
                    str(data),
                    "--split-inventory",
                    str(split),
                    "--output",
                    str(tmp_path / "training"),
                    "--image-size",
                    "16x24",
                    "--batch-size",
                    "1",
                    "--device",
                    "cpu",
                    "--epochs",
                    "1",
                    "--encoder-channels",
                    "8",
                    "--hidden-dim",
                    "16",
                ]
            )
        )
    raw = json.loads((tmp_path / "training/consumption/summary.json").read_text())
    assert raw["status"] == "failed" and raw["processed_samples"] == 0
    assert len(raw["omissions"]) == 3


def test_training_dry_run_does_not_manufacture_consumption(tmp_path):
    from carla_vision.imitation import runner

    data = teacher_release(tmp_path)
    split = tmp_path / "split.json"
    freeze_split([data], split)
    out = tmp_path / "plan-only"
    result = runner.run(
        runner.build_parser().parse_args(
            [
                "--dataset",
                str(data),
                "--split-inventory",
                str(split),
                "--output",
                str(out),
                "--dry-run",
            ]
        )
    )
    assert result["training_executed"] is False
    assert not out.exists()


def test_changed_pass_order_and_duplicate_after_success_are_rejected(tmp_path):
    from test_training import build_trainable_dataset

    from carla_vision.dataset.verified import load_verified_dataset
    from carla_vision.reproduction.consumption import make_pass

    data = build_trainable_dataset(tmp_path)
    split = tmp_path / "split.json"
    freeze_split([data], split)
    ds = load_verified_dataset(data)
    spec = make_pass("train", "train", ["train"], [ds], ordered=False)
    tracker = RunArtifactTracker(tmp_path / "runs", run_id="duplicate")
    with pytest.raises(ValueError, match="duplicated"):
        with (
            tracker,
            consumption_session(tracker, split, [spec], producer="detector_training") as audit,
        ):
            with audit.batch("train", spec["expected_keys"]) as observed:
                observed["outputs"] = [0.0]
            with audit.batch("train", spec["expected_keys"]):
                pytest.fail("duplicate input reached optimizer")


def test_declared_support_without_runtime_events_cannot_seal(tmp_path):
    from test_training import FakeBackend, build_trainable_dataset, training_config

    from carla_vision.training.runner import run_training

    class LyingCapability(FakeBackend):
        consumption_contract_version = "1.0"

    data = build_trainable_dataset(tmp_path)
    split = tmp_path / "split.json"
    freeze_split([data], split)
    config, weights = tmp_path / "config.json", tmp_path / "weights.pt"
    config.write_text(json.dumps(training_config()))
    weights.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="coverage"):
        run_training(
            config_path=config,
            dataset_path=data,
            pretrained_weights=weights,
            split_inventory_path=split,
            backend_override=LyingCapability(),
            runs_root=tmp_path / "runs",
            run_id="unobserved",
        )
    assert (
        json.loads((tmp_path / "runs/unobserved/manifest.json").read_text())["status"] == "failed"
    )


@pytest.mark.parametrize("mutation", ["reordered", "missing", "phase_order", "nan"])
def test_registered_order_and_finite_outputs_are_enforced(tmp_path, mutation):
    from carla_vision.dataset.verified import load_verified_dataset
    from carla_vision.reproduction.consumption import make_pass

    data = teacher_release(tmp_path)
    split = tmp_path / "split.json"
    freeze_split([data], split)
    ds = load_verified_dataset(data)
    specs = [
        make_pass("train/1", "train", ["train"], [ds]),
        make_pass("validation/1", "validation", ["val_seen"], [ds]),
    ]
    tracker = RunArtifactTracker(tmp_path / "runs", run_id="bad")
    with pytest.raises(ValueError):
        with (
            tracker,
            consumption_session(tracker, split, specs, producer="imitation_training") as audit,
        ):
            name = "validation/1" if mutation == "phase_order" else "train/1"
            keys = specs[1 if mutation == "phase_order" else 0]["expected_keys"]
            if mutation == "reordered":
                keys = keys[::-1]
            if mutation == "missing":
                keys = keys[:-1]
            with audit.batch(name, keys) as observed:
                observed["outputs"] = [float("nan") if mutation == "nan" else 0.0 for _ in keys]
            checkpoint = tmp_path / "fixture-checkpoint"
            checkpoint.write_bytes(b"fixture")
            audit.seal(checkpoint)


def test_empirical_gate_cannot_treat_bare_predictions_as_consumption(tmp_path):
    from carla_vision.reproduction.coverage import check_consumed_study

    root, split = evaluate(tmp_path)
    weights = json.loads((root / "consumption/summary.json").read_text())["selected_weights"]

    def dump(name, obj):
        p = tmp_path / name
        p.write_text(json.dumps(obj))
        return p

    paths = {
        "protocol": dump("protocol.json", {"split_manifest": fingerprint_file(split)}),
        "results": dump(
            "results.json",
            {"producers": {"run": {"manifest": fingerprint_file(root / "manifest.json")}}},
        ),
        "model_card": dump(
            "card.json",
            {
                "training_consumption": [],
                "pretrained_baselines": [
                    {"weights": weights, "reason": "synthetic pretrained control"}
                ],
            },
        ),
        "claim_registry": dump("claims.json", {"claims": [{"scope": "implementation"}]}),
    }
    assert check_consumed_study(paths)["status"] == "passed"
    paths["claim_registry"].write_text(json.dumps({"claims": [{"scope": "training_utility"}]}))
    with pytest.raises(ValueError, match="training-utility"):
        check_consumed_study(paths)


def test_uninstrumented_ultralytics_has_no_attestation_capability():
    from carla_vision.training.backends import UltralyticsTrainingBackend

    for architecture in ("yolo", "rtdetr"):
        assert (
            getattr(UltralyticsTrainingBackend(architecture), "consumption_contract_version", None)
            is None
        )


def test_audited_resume_cannot_hide_parent_epochs(tmp_path):
    from carla_vision.imitation import runner

    args = runner.build_parser().parse_args(
        [
            "--dataset",
            str(tmp_path / "source"),
            "--split-inventory",
            str(tmp_path / "split.json"),
            "--resume",
            str(tmp_path / "last.pt"),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(ValueError, match="parent-consumption chain"):
        runner.run(args)
    assert not (tmp_path / "out").exists()


def test_backend_cannot_mutate_the_frozen_schedule_by_alias(tmp_path):
    _, split, weights, _, spec = case(tmp_path)
    expected = list(spec["expected_keys"])
    tracker = RunArtifactTracker(tmp_path / "runs", run_id="frozen")
    with (
        tracker,
        consumption_session(
            tracker, split, [spec], producer="detector_evaluation", initial_weights=weights
        ) as audit,
    ):
        spec["expected_keys"].clear()
        assert audit.specs["evaluation"]["expected_keys"] == expected
        with audit.batch("evaluation", expected) as observed:
            observed["outputs"] = [[]]
        audit.seal(weights)


def test_plan_edit_during_run_cannot_be_sealed(tmp_path):
    _, split, weights, _, spec = case(tmp_path)
    tracker = RunArtifactTracker(tmp_path / "runs", run_id="changed-plan")
    with pytest.raises(ValueError, match="plan changed"):
        with (
            tracker,
            consumption_session(
                tracker, split, [spec], producer="detector_evaluation", initial_weights=weights
            ) as audit,
        ):
            with audit.batch("evaluation", spec["expected_keys"]) as observed:
                observed["outputs"] = [[]]
            path = tracker.run_dir / "consumption/plan.json"
            value = json.loads(path.read_text())
            value["passes"][0]["ordered"] = False
            path.write_text(json.dumps(value))
            audit.seal(weights)
