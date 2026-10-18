"""Successful in-loop events, not data-loader availability, establish coverage."""

import json
from pathlib import Path

import numpy as np
import pytest
import test_reviewer_study_links as link_fixtures

from carla_vision.artifacts import RunArtifactTracker, fingerprint_file
from carla_vision.reproduction.study_links import freeze_split


@pytest.fixture
def releases(tmp_path):
    return link_fixtures.releases.__wrapped__(tmp_path)


def setup_trace(tmp_path, roots, *, task="detector_evaluation"):
    from carla_vision.reproduction.consumption import ConsumptionRecorder

    inventory = tmp_path / "inventory.json"
    if not inventory.exists():
        freeze_split(roots, inventory)
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"fixture weights; no trained-model claim")
    tracker = RunArtifactTracker(tmp_path / "runs", run_id="trace", config={"synthetic": True})
    return tracker, inventory, weights, ConsumptionRecorder


def run_complete(tmp_path, roots):
    tracker, inventory, weights, cls = setup_trace(tmp_path, roots)
    with tracker:
        with cls(
            tracker,
            inventory,
            roots,
            phases={"test": ["test_seen"]},
            task="detector_evaluation",
            model=fingerprint_file(weights),
        ) as recorder:
            recorder.begin("test", 0)
            keys = recorder.expected["test"]
            for key in keys:
                token = recorder.prepare(
                    [key], [recorder.rows[key]["rgb_sha256"]], input_sha256="a" * 64
                )
                recorder.commit(token, [[]])  # legitimate zero detections still consumed
            recorder.end()
            recorder.finish({"evaluated": fingerprint_file(weights)})
    return tracker.run_dir, inventory, weights


def test_empty_predictions_are_consumed_and_bound(releases, tmp_path):
    from carla_vision.reproduction.consumption import verify_consumption

    root, inv, weights = run_complete(tmp_path, releases)
    result = verify_consumption(root, inventory_path=inv)
    assert result["passes"][0]["sample_count"] == 2
    assert result["passes"][0]["keys"] == result["expected"]["test"]
    assert result["checkpoints"]["evaluated"] == fingerprint_file(weights)


@pytest.mark.parametrize("fault", ["drop", "duplicate", "substitute", "wrong_hash", "prefetch"])
def test_bad_actual_coverage_cannot_finish(releases, tmp_path, fault):
    tracker, inventory, weights, cls = setup_trace(tmp_path, releases)
    with pytest.raises(ValueError):
        with tracker:
            with cls(
                tracker,
                inventory,
                releases,
                phases={"test": ["test_seen"]},
                task="detector_evaluation",
                model=fingerprint_file(weights),
            ) as recorder:
                recorder.begin("test", 0)
                keys = list(recorder.expected["test"])
                if fault == "substitute":
                    keys[0] = next(k for k, r in recorder.rows.items() if r["partition"] == "train")
                if fault in {"drop", "prefetch"}:
                    keys = keys[:1]
                if fault == "duplicate":
                    keys[1] = keys[0]
                for key in keys:
                    token = recorder.prepare(
                        [key],
                        [("f" * 64) if fault == "wrong_hash" else recorder.rows[key]["rgb_sha256"]],
                        input_sha256="a" * 64,
                    )
                    if fault != "prefetch":
                        recorder.commit(token, [[]])
                recorder.end()
                recorder.finish({"evaluated": fingerprint_file(weights)})
    assert not (tracker.run_dir / "consumption/report.json").exists()
    assert json.loads((tracker.run_dir / "manifest.json").read_text())["status"] == "failed"
    assert (tracker.run_dir / "consumption/events.jsonl").is_file()


def test_inference_exception_is_not_acknowledged(releases, tmp_path):
    tracker, inventory, weights, cls = setup_trace(tmp_path, releases)
    with pytest.raises(RuntimeError, match="model failed"):
        with tracker:
            with cls(
                tracker,
                inventory,
                releases,
                phases={"test": ["test_seen"]},
                task="detector_evaluation",
                model=fingerprint_file(weights),
            ) as recorder:
                recorder.begin("test", 0)
                k = recorder.expected["test"][0]
                recorder.prepare([k], [recorder.rows[k]["rgb_sha256"]], input_sha256="a" * 64)
                raise RuntimeError("model failed")
    rows = [
        json.loads(x)
        for x in (tracker.run_dir / "consumption/events.jsonl").read_text().splitlines()
    ]
    assert not any(x["kind"] == "batch" for x in rows)


def test_changed_checkpoint_is_rejected(releases, tmp_path):
    from carla_vision.reproduction.consumption import verify_consumption

    root, inventory, weights = run_complete(tmp_path, releases)
    weights.write_bytes(b"changed")
    with pytest.raises((ValueError, RuntimeError)):
        verify_consumption(root, inventory_path=inventory)


def test_hash_checked_decode_reads_the_same_bytes(releases):
    from carla_vision.dataset.verified import load_verified_dataset
    from carla_vision.reproduction.consumption import checked_image

    ds = load_verified_dataset(releases[0])
    sample = ds.dataset["samples"][0]
    image, digest = checked_image(ds.root / sample["rgb"]["path"], sample["rgb"]["sha256"])
    assert image.dtype == np.uint8 and digest == sample["rgb"]["sha256"]
    with pytest.raises(ValueError):
        checked_image(ds.root / sample["rgb"]["path"], "0" * 64)


def test_train_forbids_test_samples_before_work(releases, tmp_path):
    tracker, inventory, weights, cls = setup_trace(tmp_path, releases)
    with pytest.raises(ValueError, match="partition|role"):
        with tracker:
            cls(
                tracker,
                inventory,
                releases,
                phases={"train": ["test_seen"]},
                task="detector_training",
                model=fingerprint_file(weights),
            )


def test_detector_receipts_are_written_after_inference(releases, tmp_path):
    from test_thresholds import _evaluation_config, _ScoredDetector

    from carla_vision.contracts import DetectorConfig
    from carla_vision.evaluation.runner import run_evaluation
    from carla_vision.reproduction.consumption_verify import verify_producer_consumption

    class EmptyDetector(_ScoredDetector):
        calls = 0

        def infer(self, image_bgr):
            self.calls += 1
            return ()

    inventory = tmp_path / "inventory.json"
    freeze_split(releases, inventory)
    config = tmp_path / "evaluation.json"
    config.write_text(json.dumps(_evaluation_config() | {"partitions": ["test_seen"]}))
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"fixture, not a model")
    detector = EmptyDetector()
    result = run_evaluation(
        config_path=config,
        dataset_path=releases[2],
        detector_config=DetectorConfig(backend="custom", weights=weights, factory="fixture:fake"),
        runs_root=tmp_path / "runs",
        run_id="evaluation",
        detector_override=detector,
        sample_inventory=inventory,
        acknowledge_locked_test=True,
    )
    verified = verify_producer_consumption(
        result["run_dir"], inventory_path=inventory, require_holdout=True
    )
    assert detector.calls == 2 == len(verified["keys"])
    # Removing a consumed record and rehashing report/manifest must still be rejected.
    from research_fixtures import ref

    root = Path(result["run_dir"])
    frames = root / "predictions/frames.jsonl"
    frames.write_text(frames.read_text().splitlines()[0] + "\n")
    manifest = json.loads((root / "manifest.json").read_text())
    for artifact in manifest["artifacts"]:
        artifact.update(ref(root / artifact["path"], artifact["path"]))
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, RuntimeError)):
        verify_producer_consumption(root, inventory_path=inventory, require_holdout=True)


def teacher_releases(tmp_path):
    from test_dataset_writer import camera_frame

    from carla_vision.dataset.sync import SynchronizedFramePair
    from carla_vision.dataset.writer import DatasetWriter
    from carla_vision.native.teacher_timing import ObservationActionTransition

    roots = []
    for i, partition in enumerate(("train", "val_seen", "test_seen")):
        with DatasetWriter(
            tmp_path / "teacher",
            dataset_id=f"teacher-{i}",
            minimum_pixels=1,
            carla_version="0.9.16",
            carla_map="fixture",
            carla_endpoint={"host": "synthetic-fixture"},
        ) as writer:
            for frame in (1, 2):
                rgb = np.full((32, 48, 4), 20 + i * 50 + frame, dtype=np.uint8)
                rgb[:, :, 3] = 255
                mask = np.zeros_like(rgb)
                pair = SynchronizedFramePair(
                    camera_frame(frame, frame, rgb), camera_frame(frame, frame, mask), 0, 0
                )
                writer.add_pair(
                    pair,
                    split=partition,
                    scenario_id=f"s{i}",
                    episode_id=f"e{i}",
                    context={
                        "map_family": "fixture",
                        "control_mode": "behavior_agent_teacher",
                        "route": {
                            "route_id": f"route{i}",
                            "start_spawn_index": i * 2,
                            "destination_spawn_index": i * 2 + 1,
                            "destination_world_transform": dict(
                                x=1.0, y=2.0, z=0.0, pitch=0.0, yaw=0.0, roll=0.0
                            ),
                        },
                        "privileged_teacher_control": dict(
                            source="BehaviorAgent.run_step",
                            carla_frame=frame,
                            throttle=0.2,
                            steer=0.1,
                            brake=0.0,
                        ),
                        "privileged_evaluation": {"velocity_mps": {"speed": 2.5}},
                        "teacher_transition": ObservationActionTransition(
                            frame, frame, frame + 1, "fixture"
                        ).as_dict(),
                    },
                )
            writer.set_release_metadata(
                {"control_mode": "behavior_agent_teacher", "episode_ids": [f"e{i}"]}
            )
        roots.append(writer.dataset_dir)
    return roots


def test_imitation_real_cpu_batches_resume_and_explicit_test(tmp_path):
    import torch

    from carla_vision.artifacts import fingerprint_file
    from carla_vision.imitation import runner
    from carla_vision.imitation.dataset import BehaviorImitationDataset
    from carla_vision.imitation.evaluate import main as evaluate
    from carla_vision.reproduction.consumption_verify import verify_producer_consumption

    roots = teacher_releases(tmp_path)
    inventory = tmp_path / "inventory.json"
    freeze_split(roots, inventory)
    argv = [
        "--dataset",
        str(roots[0]),
        "--dataset",
        str(roots[1]),
        "--sample-inventory",
        str(inventory),
        "--image-size",
        "32x48",
        "--encoder-channels",
        "4",
        "--hidden-dim",
        "8",
        "--batch-size",
        "1",
        "--device",
        "cpu",
    ]
    for name, epochs, extra in [
        ("first", "1", []),
        ("resumed", "2", ["--resume", str(tmp_path / "first/checkpoint_last.pt")]),
    ]:
        runner.run(
            runner.build_parser().parse_args(
                argv + ["--output", str(tmp_path / name), "--epochs", epochs] + extra
            )
        )
        result = verify_producer_consumption(tmp_path / name, inventory_path=inventory)
        assert result["task"] == "imitation_training"
        assert set(result["phases"]) == {"train", "val"}
    checkpoint = tmp_path / "resumed/checkpoint_best.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    test_data = BehaviorImitationDataset([roots[2]], partition="test", image_size=(32, 48))
    protocol = tmp_path / "test-protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "checkpoint": fingerprint_file(checkpoint),
                "training_manifest_sha256": [r["sha256"] for r in payload["source_references"]],
                "test_manifests": test_data.source_references,
                "group_policy": "disjoint_scene_roots_and_exact_rgb",
            }
        )
    )
    assert (
        evaluate(
            [
                "--checkpoint",
                str(checkpoint),
                "--dataset",
                str(roots[2]),
                "--output",
                str(tmp_path / "tested"),
                "--sample-inventory",
                str(inventory),
                "--evaluation-protocol",
                str(protocol),
                "--acknowledge-locked-test",
            ]
        )
        == 0
    )
    result = verify_producer_consumption(
        tmp_path / "tested", inventory_path=inventory, require_holdout=True
    )
    assert len(result["keys"]) == 2


def test_uninstrumented_backend_cannot_claim_consumption(tmp_path):
    from test_training import FakeBackend, build_trainable_dataset, training_config

    from carla_vision.training.runner import run_training

    data = build_trainable_dataset(tmp_path)
    inventory = tmp_path / "split.json"
    freeze_split([data], inventory)
    config = tmp_path / "training.json"
    config.write_text(json.dumps(training_config()))
    weights = tmp_path / "initial.bin"
    weights.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="in-loop"):
        run_training(
            config_path=config,
            dataset_path=data,
            pretrained_weights=weights,
            sample_inventory=inventory,
            runs_root=tmp_path / "runs",
            run_id="no-hook",
            backend_override=FakeBackend(),
        )
    assert not list((tmp_path / "runs/no-hook").rglob("best.pt"))


@pytest.mark.parametrize("backend", ["yolo", "rtdetr"])
def test_real_ultralytics_tiny_cpu_training_receipts(tmp_path, monkeypatch, backend):
    pytest.importorskip("ultralytics")
    monkeypatch.setenv("YOLO_AUTOINSTALL", "false")
    monkeypatch.setenv("YOLO_OFFLINE", "true")
    from test_training import image_frame, training_config
    from ultralytics import RTDETR, YOLO
    from ultralytics.utils import checks

    from carla_vision.reproduction.consumption_verify import verify_producer_consumption
    from carla_vision.training.runner import run_training

    # Only font discovery is stubbed: no upstream weights/assets or network needed.
    monkeypatch.setattr(checks, "check_font", lambda *a, **k: None)
    import torch
    from ultralytics.data import utils as data_utils

    monkeypatch.setattr(data_utils, "check_font", lambda *a, **k: None)
    torch.manual_seed(0)
    from carla_vision.dataset.sync import SynchronizedFramePair
    from carla_vision.dataset.writer import DatasetWriter

    with DatasetWriter(
        tmp_path / "datasets",
        dataset_id="tiny-train",
        minimum_pixels=4,
        carla_endpoint={"host": "fixture"},
        carla_version="0.9.16",
        carla_map="fixture",
    ) as writer:
        for i, partition in enumerate(("train", "val_seen", "test_seen")):
            rgb = np.full((32, 48, 4), 45 + i * 60, dtype=np.uint8)
            rgb[:, :, 3] = 255
            mask = np.zeros_like(rgb)
            mask[5:25, 8:38, 2] = 14
            mask[5:25, 8:38, 0] = 4
            writer.add_pair(
                SynchronizedFramePair(
                    image_frame(i + 1, i + 1, rgb), image_frame(i + 1, i + 1, mask), 0, 0
                ),
                split=partition,
                scenario_id=partition,
                episode_id=partition,
            )
    data = writer.dataset_dir
    assert json.loads((data / "annotations/instances.coco.json").read_text())["annotations"], (
        "fixture must include labeled objects"
    )
    inventory = tmp_path / "split.json"
    freeze_split([data], inventory)
    model_config = tmp_path / "tiny.yaml"
    model_config.write_text(
        "nc: 10\nbackbone:\n  - [-1, 1, Conv, [8, 3, 2]]\n  - [-1, 1, Conv, [16, 3, 2]]\n  - [-1, 1, Conv, [16, 3, 2]]\nhead:\n  - [[0, 1, 2], 1, Detect, [nc]]\n"
    )
    weights = tmp_path / "initial.pt"
    if backend == "rtdetr":
        model_config.write_text(
            "nc: 10\nbackbone:\n  - [-1, 1, Conv, [8, 3, 2]]\n  - [-1, 1, Conv, [16, 3, 2]]\n  - [-1, 1, Conv, [32, 3, 2]]\n  - [-1, 1, Conv, [32, 3, 2]]\n  - [-1, 1, Conv, [32, 3, 2]]\nhead:\n  - [[2, 3, 4], 1, RTDETRDecoder, [nc, 32, 10, 4, 4, 1]]\n"
        )
    (YOLO if backend == "yolo" else RTDETR)(str(model_config)).save(weights)
    import shutil
    import subprocess

    from test_execution_receipts import source

    from carla_vision.reproduction.source_snapshot import build_source_snapshot
    from carla_vision.reproduction.test_execution import run_tests

    source_root = source(tmp_path)
    actual_package = Path(__file__).resolve().parents[1] / "carla_vision"
    for module in actual_package.rglob("*.py"):
        target = source_root / "carla_vision" / module.relative_to(actual_package)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(module, target)
    subprocess.run(["git", "-C", str(source_root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "exact producer bytes fixture",
        ],
        check=True,
    )
    source_zip = tmp_path / "executed-source.zip"
    snapshot = build_source_snapshot(source_root, source_zip)
    execution = tmp_path / "test-execution"
    assert run_tests(source_root, execution, timeout_seconds=30)["status"] == "passed"
    config = tmp_path / "training.json"
    config.write_text(
        json.dumps(
            training_config(
                backend=backend,
                custom_factory=None,
                epochs=1,
                image_size=64,
                patience=0,
                extra_options={"mosaic": 0.0, "nbs": 1, "plots": False, "close_mosaic": 0},
            )
        )
    )
    result = run_training(
        config_path=config,
        dataset_path=data,
        pretrained_weights=weights,
        sample_inventory=inventory,
        runs_root=tmp_path / "runs",
        run_id="tiny-yolo",
        repository_root=source_root,
    )
    verified = verify_producer_consumption(result["run_dir"], inventory_path=inventory)
    assert verified["task"] == "detector_training"
    assert set(verified["phases"]) == {"train", "val", "val_final"}
    assert not list(data.rglob("*.cache"))
    assert not list(data.rglob("*.npy"))

    from test_thresholds import _evaluation_config

    from carla_vision.contracts import DetectorConfig
    from carla_vision.evaluation.runner import run_evaluation
    from carla_vision.reproduction.consumption_verify import training_origin_reference

    best = Path(verified["checkpoints"]["best"]["path"])
    evaluation_config = tmp_path / "evaluation.json"
    evaluation_config.write_text(json.dumps(_evaluation_config() | {"partitions": ["test_seen"]}))
    evaluated = run_evaluation(
        config_path=evaluation_config,
        dataset_path=data,
        detector_config=DetectorConfig(backend=backend, weights=best, device="cpu", image_size=64),
        runs_root=tmp_path / "runs",
        run_id="test-trained-model",
        sample_inventory=inventory,
        training_run=result["run_dir"],
        acknowledge_locked_test=True,
        repository_root=source_root,
    )
    actual = verify_producer_consumption(
        evaluated["run_dir"], inventory_path=inventory, require_holdout=True
    )
    assert len(actual["keys"]) == 1 and actual["training_origin"] == fingerprint_file(
        Path(result["run_dir"]) / "manifest.json"
    )
    with pytest.raises(ValueError, match="best"):
        training_origin_reference(
            result["run_dir"], checkpoint=fingerprint_file(weights), inventory_path=inventory
        )

    from carla_vision.reproduction.consumption_verify import (
        check_study_consumption,
        verify_execution_source_tree,
    )

    protocol = tmp_path / "execution-policy.json"
    protocol.write_text(
        json.dumps(
            {
                "methods": ["ours"],
                "split_manifest": fingerprint_file(inventory),
                "sample_execution": {"ours": {"model_origin": "trained"}},
            }
        )
    )
    results = tmp_path / "cell.json"
    results.write_text(
        json.dumps(
            {
                "rows": [{"method": "ours", "run_id": "cell"}],
                "producers": {
                    "cell": {
                        "manifest": fingerprint_file(Path(evaluated["run_dir"]) / "manifest.json")
                    }
                },
            }
        )
    )
    claims = tmp_path / "claims.json"
    claims.write_text(json.dumps({"claims": [{"scope": "training_utility"}]}))
    paths = {
        "protocol": protocol,
        "results": results,
        "claim_registry": claims,
        "tests": execution / "execution.json",
        "source_snapshot": source_zip,
    }
    # Necessary execution links only. These synthetic fixtures do NOT pass a full empirical submission.
    bound = check_study_consumption(paths, source_commit=snapshot["source_commit"])
    assert bound["producers"]["cell"]["consumed_holdout_samples"] == 1
    assert bound["producers"]["cell"]["model_origin"] == "trained"
    original_module = source_root / "carla_vision/evaluation/runner.py"
    original_module.write_text(original_module.read_text() + "\n# source mutation fixture\n")
    subprocess.run(["git", "-C", str(source_root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "different source fixture",
        ],
        check=True,
    )
    other = tmp_path / "different-source.zip"
    changed = build_source_snapshot(source_root, other)
    with pytest.raises(ValueError, match="producer bytes"):
        verify_execution_source_tree(
            evaluated["run_dir"], other, source_commit=changed["source_commit"]
        )


@pytest.mark.parametrize("ordered", [False, True])
def test_recorded_order_is_preserved_and_enforced_when_declared(releases, tmp_path, ordered):
    from contextlib import nullcontext

    from carla_vision.reproduction.consumption import verify_consumption

    tracker, inventory, weights, cls = setup_trace(tmp_path, releases)
    with pytest.raises(ValueError, match="order") if ordered else nullcontext():
        with tracker:
            with cls(
                tracker,
                inventory,
                releases,
                phases={"test": ["test_seen"]},
                task="detector_evaluation",
                model=fingerprint_file(weights),
            ) as recorder:
                recorder.begin("test", 0, ordered=ordered)
                keys = list(reversed(recorder.expected["test"]))
                token = recorder.prepare(
                    keys, [recorder.rows[k]["rgb_sha256"] for k in keys], input_sha256="a" * 64
                )
                recorder.commit(token, [[], []])
                recorder.end()
                recorder.finish({"evaluated": fingerprint_file(weights)})
    if not ordered:
        assert verify_consumption(tracker.run_dir)["passes"][0]["keys"] == keys


def test_forged_success_report_cannot_hide_aborted_execution(releases, tmp_path):
    from carla_vision.reproduction.consumption import verify_consumption

    # A missing report is rejected even when a producer status is changed by hand.
    tracker, inventory, weights, cls = setup_trace(tmp_path, releases)
    with pytest.raises(ValueError):
        with tracker:
            with cls(
                tracker,
                inventory,
                releases,
                phases={"test": ["test_seen"]},
                task="detector_evaluation",
                model=fingerprint_file(weights),
            ) as recorder:
                recorder.begin("test", 0)
                recorder.end()
    with pytest.raises((ValueError, RuntimeError)):
        verify_consumption(tracker.run_dir)


def test_recorder_rejects_reused_token_and_overlapping_pass(releases, tmp_path):
    tracker, inventory, weights, cls = setup_trace(tmp_path, releases)
    with pytest.raises(ValueError, match="pass"):
        with tracker:
            with cls(
                tracker,
                inventory,
                releases,
                phases={"test": ["test_seen"]},
                task="detector_evaluation",
                model=fingerprint_file(weights),
            ) as recorder:
                recorder.begin("test", 0)
                recorder.begin("test", 1)


@pytest.mark.parametrize(
    "option,value",
    [
        ("workers", 1),
        ("amp", True),
        ("cache", True),
        ("patience", 3),
        ("extra_options", {"mosaic": 1, "nbs": 1}),
        ("extra_options", {"mosaic": 0, "nbs": 64}),
        ("extra_options", {"mosaic": 0, "nbs": 1, "fraction": 0.5}),
    ],
)
def test_unsupported_training_sampling_fails_closed(option, value):
    from test_training import training_config

    from carla_vision.training.contracts import TrainingConfig
    from carla_vision.training.ultralytics_consumption import validate_settings

    base = training_config(
        backend="yolo",
        custom_factory=None,
        epochs=1,
        patience=0,
        extra_options={"mosaic": 0, "nbs": 1},
    )
    base[option] = value
    with pytest.raises(ValueError):
        validate_settings(TrainingConfig.from_mapping(base))


def test_installed_hook_source_changes_are_not_silently_accepted(monkeypatch):
    pytest.importorskip("ultralytics")
    from carla_vision.training import ultralytics_consumption as adapter

    monkeypatch.setattr(adapter, "HOOK_SOURCE_HASHES", {"engine/trainer.py": "f" * 64})
    with pytest.raises(ValueError, match="hook source"):
        adapter.check_hook_sources()


def test_execution_policy_requires_explicit_model_origins(tmp_path):
    from carla_vision.reproduction.consumption_verify import validate_execution_policy

    weight = tmp_path / "weight.bin"
    weight.write_bytes(b"fixture")
    good = {
        "ours": {"model_origin": "trained"},
        "base": {"model_origin": "external_frozen", "checkpoint": fingerprint_file(weight)},
    }
    validate_execution_policy(good, ["ours", "base"])
    with pytest.raises(ValueError, match="every"):
        validate_execution_policy({"ours": good["ours"]}, ["ours", "base"])
    with pytest.raises(ValueError, match="pin"):
        validate_execution_policy({"base": {"model_origin": "external_frozen"}}, ["base"])


def test_empirical_reviewer_rejects_bare_junit(tmp_path):
    from carla_vision.reproduction.reviewer import _check_role

    path = tmp_path / "tests.xml"
    path.write_text('<testsuite><testcase name="pass"/></testsuite>')
    assert _check_role("tests", path, {"source_commit": "a" * 40}, empirical=False)["cases"] == 1
    with pytest.raises(ValueError):
        _check_role("tests", path, {"source_commit": "a" * 40}, empirical=True)


def test_pool_receipts_cover_every_shard_and_recompute_outputs(tmp_path):
    from research_fixtures import ref
    from test_thresholds import _evaluation_config, _ScoredDetector
    from test_training import sample_pair

    from carla_vision.contracts import DetectorConfig
    from carla_vision.dataset.writer import DatasetWriter
    from carla_vision.evaluation.pool import pool_evaluations
    from carla_vision.evaluation.runner import run_evaluation
    from carla_vision.reproduction.consumption_verify import verify_producer_consumption

    datasets = []
    for i in range(3):
        with DatasetWriter(
            tmp_path / "datasets",
            dataset_id=f"shard-{i}",
            minimum_pixels=4,
            carla_endpoint={"host": "fixture"},
            carla_version="0.9.16",
            carla_map="fixture",
        ) as writer:
            writer.add_pair(
                sample_pair(1, 1, 20 + i * 20),
                split="test_seen",
                scenario_id=f"s{i}",
                episode_id=f"e{i}",
            )
        datasets.append(writer.dataset_dir)
    inv = tmp_path / "split.json"
    freeze_split(datasets, inv)
    config = tmp_path / "eval.json"
    config.write_text(json.dumps(_evaluation_config() | {"partitions": ["test_seen"]}))
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"fixture")
    children = []
    for i, root in enumerate(datasets):
        result = run_evaluation(
            config_path=config,
            dataset_path=root,
            detector_config=DetectorConfig(
                backend="custom", weights=weights, factory="fixture:fake"
            ),
            runs_root=tmp_path / "runs",
            run_id=f"child-{i}",
            detector_override=_ScoredDetector(),
            sample_inventory=inv,
            acknowledge_locked_test=True,
        )
        children.append(result["run_dir"])
    partial = pool_evaluations(
        children[:2], runs_root=tmp_path / "runs", run_id="partial", sample_inventory=inv
    )
    with pytest.raises(ValueError, match="holdout"):
        verify_producer_consumption(partial["run_dir"], inventory_path=inv, require_holdout=True)
    complete = pool_evaluations(
        children, runs_root=tmp_path / "runs", run_id="complete", sample_inventory=inv
    )
    check = verify_producer_consumption(
        complete["run_dir"], inventory_path=inv, require_holdout=True
    )
    assert len(check["keys"]) == 3 and check["scope"].startswith("derived_")
    root = Path(complete["run_dir"])
    pred = root / "predictions.coco.json"
    rows = json.loads(pred.read_text())
    rows[0]["score"] = 0.01
    pred.write_text(json.dumps(rows))
    manifest = json.loads((root / "manifest.json").read_text())
    for artifact in manifest["artifacts"]:
        artifact.update(ref(root / artifact["path"], artifact["path"]))
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reproduce"):
        verify_producer_consumption(root, inventory_path=inv)
