"""Synthetic software acceptance only; no human audit, simulator, or model benchmark."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from carla_vision.artifacts import RunArtifactTracker, fingerprint_file


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    return path


def test_crossed_seed_intervals_do_not_flatten_cartesian_pairs():
    from carla_vision.evaluation.study_statistics import paired_crossed_analysis

    left = {(g, s): g / 100 + s / 50 for g in (1, 2, 3) for s in (11, 12)}
    right = {key: value + 0.1 for key, value in left.items()}
    result = paired_crossed_analysis(left, right, replicates=30)
    assert result["mean_difference"] == pytest.approx(0.1)
    assert result["lower"] == pytest.approx(0.1)
    assert result["independent_pair_count_claimed"] is False
    right.pop((1, 11))
    with pytest.raises(ValueError):
        paired_crossed_analysis(left, right)


def test_published_coreset_cli_executes_and_rejects_test_pool(tmp_path):
    from carla_vision.scenarios.baselines import select_feature_pool

    features = tmp_path / "features.npy"
    np.save(features, np.array([[0.0], [2.0], [10.0], [6.0]]))
    encoder = tmp_path / "encoder.fixture"
    encoder.write_bytes(b"synthetic encoder identity, not actual trained weights")
    raw = {
        "schema_version": "1.0",
        "features": fingerprint_file(features),
        "feature_model": fingerprint_file(encoder),
        "count": 2,
        "unit": "equal_cost_feature_row",
        "synthetic": True,
        "items": [
            {"sample_id": str(i), "group_id": str(i), "partition": "train", "labeled": i == 0}
            for i in range(4)
        ],
    }
    path = dump(tmp_path / "pool.json", raw)
    result = select_feature_pool(path, runs_root=tmp_path / "runs", run_id="core")
    assert result["selected_indices"] == [2, 3]
    assert result["empirical_results_available"] is False
    raw["items"][1]["partition"] = "test_map_ood"
    dump(path, raw)
    with pytest.raises(ValueError, match="train"):
        select_feature_pool(path, runs_root=tmp_path / "runs", run_id="bad")


def test_diagnosis_benchmark_locks_predictions_before_truth_and_rejects_oracle_fields(tmp_path):
    from carla_vision.failure_mining.attribution import CHECKS, AttributionEvidence
    from carla_vision.failure_mining.diagnosis_benchmark import score_diagnosis

    evidence = dump(tmp_path / "actual.fixture.json", {"synthetic": True, "signal": "fixture"})
    values = AttributionEvidence(
        {k: True for k in CHECKS}, 1.0, None, "valid", str(evidence), "", str(evidence)
    )
    observation = {
        "schema_version": "1.0",
        "synthetic": True,
        "observations": [
            {
                "case_id": "hard-night",
                "group_id": "one",
                "evidence": asdict(values),
                "source_artifacts": [fingerprint_file(evidence)],
            }
        ],
    }
    observed = dump(tmp_path / "observed.json", observation)
    truth = {
        "schema_version": "1.0",
        "synthetic": True,
        "observations_sha256": fingerprint_file(observed)["sha256"],
        "outcomes": {
            "hard-night": {
                "source_kind": "model_error",
                "regime": "clean_hard",
                "reference_evidence": fingerprint_file(evidence),
            }
        },
    }
    reference = dump(tmp_path / "truth.json", truth)
    a = score_diagnosis(observed, reference, runs_root=tmp_path / "runs", run_id="d1")
    truth["outcomes"]["hard-night"]["source_kind"] = "capture_error"
    dump(reference, truth)
    b = score_diagnosis(observed, reference, runs_root=tmp_path / "runs", run_id="d2")
    assert a["predictions"] == b["predictions"]
    assert a["regimes"] != b["regimes"]
    assert a["clean_case_actions"] == {"NEW_SCENE": 1}
    assert a["empirical_results_available"] is False
    observation["observations"][0]["evidence"]["oracle_fault"] = "capture_error"
    dump(observed, observation)
    with pytest.raises(ValueError, match="fields"):
        score_diagnosis(observed, reference, runs_root=tmp_path / "runs", run_id="d3")


def captured_release(tmp_path, *, sharded=False):
    from test_training import sample_pair

    from carla_vision.dataset.shards import ShardedDatasetWriter
    from carla_vision.dataset.writer import DatasetWriter
    from carla_vision.native.map_evidence import record_map_evidence

    writer_cls = ShardedDatasetWriter if sharded else DatasetWriter
    options = {"shard_samples": 1} if sharded else {}
    with writer_cls(
        tmp_path,
        dataset_id="captured",
        carla_endpoint={"host": "fixture"},
        carla_version="0.9.16",
        carla_map="fixture_map",
        minimum_pixels=4,
        **options,
    ) as writer:
        world = SimpleNamespace(
            get_map=lambda: SimpleNamespace(
                name="fixture_map", to_opendrive=lambda: "<OpenDRIVE>synthetic fixture</OpenDRIVE>"
            )
        )
        identity = record_map_evidence(world, writer, "one")
        for i in range(3):
            writer.add_pair(
                sample_pair(i + 1, 10 + i, 30 + i),
                split="train",
                scenario_id="s",
                episode_id="e",
                context={
                    **identity,
                    "map": "fixture_map",
                    "privileged_evaluation": {"world_transform": {"x": i, "y": 0, "z": 0}},
                },
            )
    return writer.dataset_dir, writer.dataset_dir / "maps/one.opendrive.json"


@pytest.mark.parametrize("sharded", [False, True])
def test_geometry_bound_to_captured_trajectories_and_recomputed_by_reviewer(tmp_path, sharded):
    from carla_vision.dataset.trajectory_audit import captured_routes, run_trajectory_audit
    from carla_vision.reproduction.reviewer import _check_role

    root, map_path = captured_release(tmp_path, sharded=sharded)
    routes, refs = captured_routes([root], [map_path], maximum_step_m=1.5)
    assert len(routes) == 1 and len(routes[0]["points_xyz"]) == 3
    result = run_trajectory_audit(
        [root],
        [map_path],
        clearance_m=1.0,
        maximum_step_m=1.5,
        runs_root=tmp_path / "runs",
        run_id="trajectory",
    )
    path = Path(result["run_dir"]) / "trajectory_audit.json"
    assert _check_role("geometry", path, {}, empirical=True)["status"] == "passed"
    raw = json.loads(path.read_text())
    raw["routes"][0]["points_xyz"][0][0] = 25
    dump(path, raw)
    with pytest.raises(ValueError, match="differs"):
        _check_role("geometry", path, {}, empirical=True)
    with pytest.raises(ValueError, match="gap"):
        captured_routes([root], [map_path], maximum_step_m=0.5)
    assert all("sha256" in r for r in refs)


def test_source_less_geometry_cannot_certify_empirical_routes(tmp_path):
    from carla_vision.reproduction.reviewer import _check_role

    path = dump(tmp_path / "geometry.json", {"routes": [], "source_artifacts": []})
    with pytest.raises(ValueError, match="recorded"):
        _check_role("geometry", path, {}, empirical=True)


def test_detector_training_consumes_shards_with_source_partitions(tmp_path):
    import yaml
    from test_training import sample_pair, training_config

    from carla_vision.dataset.shards import ShardedDatasetWriter
    from carla_vision.training.runner import run_training

    with ShardedDatasetWriter(
        tmp_path / "datasets",
        dataset_id="sharded",
        shard_samples=1,
        carla_endpoint={"host": "fixture"},
        carla_version="0.9.16",
        carla_map="test",
        minimum_pixels=4,
    ) as writer:
        for i, part in enumerate(("train", "train", "val_seen", "test_seen")):
            writer.add_pair(
                sample_pair(i, 10 + i, 20 + i),
                split=part,
                scenario_id=part,
                episode_id=part,
                context={"map": "test"},
            )
    config = dump(tmp_path / "training.json", training_config())
    weights = tmp_path / "initial.fixture"
    weights.write_bytes(b"not-a-model")
    result = run_training(
        config_path=config,
        dataset_path=writer.dataset_dir,
        pretrained_weights=weights,
        runs_root=tmp_path / "runs",
        run_id="dry",
        dry_run=True,
    )
    data = yaml.safe_load((Path(result["run_dir"]) / "data/training_data.yaml").read_text())
    assert len(data["train"]) == 2 and len(data["val"]) == 1 and data["test"] == []
    assert all("/images/train" in p for p in data["train"])
    assert result["summary"]["training_executed"] is False


def bound_result_matrix(tmp_path):
    from carla_vision.scenarios.costs import CostLedger

    matrix = dump(
        tmp_path / "matrix.json",
        {
            "methods": ["one"],
            "seeds": [1],
            "holdout_sha256": "a" * 64,
            "budget": 5.0,
            "cost_weights": {"wall_seconds": 1.0},
        },
    )
    ledger_path = tmp_path / "costs.jsonl"
    with CostLedger(ledger_path) as ledger:
        ledger.record(
            "stage-one",
            kind="train",
            outcome="failed",
            resources={"wall_seconds": 1.0},
            evidence_ref="failed-fixture",
        )
        ledger.record(
            "stage-two",
            kind="train",
            outcome="success",
            resources={"wall_seconds": 2.0},
            evidence_ref="success-fixture",
        )
    contract = {
        "matrix_sha256": fingerprint_file(matrix)["sha256"],
        "method": "one",
        "seed": 1,
        "holdout_sha256": "a" * 64,
        "cost_ledger_sha256": fingerprint_file(ledger_path)["sha256"],
    }
    with RunArtifactTracker(
        tmp_path / "runs", run_id="result", config={"synthetic": True, "review_binding": contract}
    ) as tracker:
        summary = dump(tracker.artifact_path("summary.json"), {"metric": 0.4, "synthetic": True})
        tracker.register_artifact(summary, role="registered_experiment_result")
    raw = {
        "protocol": fingerprint_file(matrix),
        "synthetic": True,
        "rows": [
            {
                "method": "one",
                "seed": 1,
                "run_id": "result",
                "status": "success",
                "cost": 3.0,
                "holdout_sha256": "a" * 64,
                "metric": 0.4,
            }
        ],
        "producers": {
            "result": {
                "manifest": fingerprint_file(tracker.manifest_path),
                "summary": fingerprint_file(summary),
                "metric_field": "metric",
                "cost_ledger": fingerprint_file(ledger_path),
            }
        },
    }
    return dump(tmp_path / "results.json", raw), raw, matrix


def test_reviewer_results_bind_source_method_seed_holdout_and_all_attempt_costs(tmp_path):
    from carla_vision.reproduction.reviewer import _check_role

    path, raw, matrix = bound_result_matrix(tmp_path)
    assert _check_role("results", path, {"source_commit": "b" * 40}, empirical=False)["complete"]
    raw["rows"][0]["cost"] = 2.0
    dump(path, raw)
    with pytest.raises(ValueError, match="cost differs"):
        _check_role("results", path, {}, empirical=False)
    raw["rows"][0]["cost"] = 3.0
    raw["rows"][0]["method"] = "renamed"
    m = json.loads(matrix.read_text())
    m["methods"] = ["renamed"]
    dump(matrix, m)
    raw["protocol"] = fingerprint_file(matrix)
    dump(path, raw)
    with pytest.raises(ValueError, match="registered run"):
        _check_role("results", path, {}, empirical=False)


def test_source_snapshot_is_commit_bound_and_excludes_dirty_and_private_files(tmp_path):
    import os
    import subprocess
    import zipfile

    from carla_vision.reproduction.reviewer import _source_archive
    from carla_vision.reproduction.source_snapshot import build_source_snapshot

    source = tmp_path / "repo"
    source.mkdir()
    (source / "carla_vision").mkdir()
    (source / "tests").mkdir()
    (source / "carla_vision/a.py").write_text("VALUE = 1\n")
    (source / "tests/test_a.py").write_text("def test_a(): assert True\n")
    (source / ".env.local").write_text("SECRET=not-real-test-secret")
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "add", "carla_vision", "tests"], check=True)
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME="Fixture",
        GIT_AUTHOR_EMAIL="fixture@example.invalid",
        GIT_COMMITTER_NAME="Fixture",
        GIT_COMMITTER_EMAIL="fixture@example.invalid",
    )
    subprocess.run(
        ["git", "-C", str(source), "commit", "-m", "Synthetic test only"],
        env=env,
        check=True,
        capture_output=True,
    )
    (source / "carla_vision/a.py").write_text("VALUE = 2\n")
    a = build_source_snapshot(source, tmp_path / "a.zip")
    b = build_source_snapshot(source, tmp_path / "b.zip")
    assert a["archive"]["sha256"] == b["archive"]["sha256"]
    _source_archive(tmp_path / "a.zip", a["source_commit"])
    with zipfile.ZipFile(tmp_path / "a.zip") as archive:
        assert ".env.local" not in archive.namelist()
        assert archive.read("carla_vision/a.py") == b"VALUE = 1\n"
    with pytest.raises(FileExistsError):
        build_source_snapshot(source, tmp_path / "a.zip")
    with pytest.raises(ValueError):
        build_source_snapshot(source, tmp_path / "c.zip", commit="--help")


def test_portable_coco_reproduction_runs_after_original_workspace_is_removed(tmp_path):
    import shutil

    from test_evaluation import FakeDetector, build_dataset, evaluation_config

    from carla_vision.contracts import DetectorConfig
    from carla_vision.evaluation.runner import run_evaluation
    from carla_vision.reproduction.closure import build_closure
    from carla_vision.reproduction.replay_metrics import replay_coco_metrics

    original = tmp_path / "original"
    original.mkdir()
    dataset = build_dataset(original)
    config = dump(original / "eval.json", evaluation_config())
    result = run_evaluation(
        config_path=config,
        dataset_path=dataset,
        detector_config=DetectorConfig(
            backend="custom", factory="fixture:factory", device="cpu", image_size=320
        ),
        runs_root=original / "runs",
        run_id="original-eval",
        detector_override=FakeDetector(),
    )
    path = Path(result["run_dir"]) / "manifest.json"
    identity = fingerprint_file(path)["sha256"]
    packed = build_closure([path], tmp_path / "pack")
    shutil.rmtree(original)
    reproduced = replay_coco_metrics(
        tmp_path / "pack",
        evaluation_manifest_sha256=identity,
        expected_closure_sha256=packed["index"]["sha256"],
        runs_root=tmp_path / "reproduced",
        run_id="replay",
    )
    assert reproduced["status"] == "passed"
    assert reproduced["original_absolute_paths_required"] is False
    assert reproduced["training_reexecuted"] is False


def test_source_archive_race_does_not_delete_another_publishers_output(tmp_path, monkeypatch):
    import subprocess

    from carla_vision.reproduction import source_snapshot

    target = tmp_path / "source.zip"
    real = subprocess.check_output

    def interpose(args, **kwargs):
        value = real(args, **kwargs)
        if "ls-tree" in args:
            target.write_bytes(b"another publisher owns these bytes")
        return value

    monkeypatch.setattr(source_snapshot.subprocess, "check_output", interpose)
    with pytest.raises(FileExistsError):
        source_snapshot.build_source_snapshot(Path(__file__).parents[1], target)
    assert target.read_bytes() == b"another publisher owns these bytes"


def test_captured_trajectory_rejects_duplicate_release_identity(tmp_path):
    from carla_vision.dataset.trajectory_audit import captured_routes

    first, map_path = captured_release(tmp_path / "first")
    second, _ = captured_release(tmp_path / "second")
    with pytest.raises(ValueError, match="duplicate.*identity"):
        captured_routes([first, second], [map_path], maximum_step_m=2)


def test_seed_analysis_emits_recomputed_table_from_bound_producers(tmp_path):
    from carla_vision.evaluation.study_statistics import run_seed_analysis
    from carla_vision.scenarios.costs import CostLedger

    matrix = dump(
        tmp_path / "matrix.json",
        {
            "methods": ["random", "ours"],
            "seeds": [1, 2, 3],
            "holdout_sha256": "a" * 64,
            "budget": 5.0,
            "cost_weights": {"wall_seconds": 1.0},
        },
    )
    rows, producers = [], {}
    for method in ("random", "ours"):
        for seed in (1, 2, 3):
            name = f"{method}-{seed}"
            costs = tmp_path / f"{name}.jsonl"
            with CostLedger(costs) as ledger:
                ledger.record(
                    name,
                    kind="train",
                    outcome="success",
                    resources={"wall_seconds": 2.0},
                    evidence_ref="synthetic-fixture",
                )
            review_binding = {
                "matrix_sha256": fingerprint_file(matrix)["sha256"],
                "method": method,
                "seed": seed,
                "holdout_sha256": "a" * 64,
                "cost_ledger_sha256": fingerprint_file(costs)["sha256"],
            }
            value = seed / 10 + (0.1 if method == "ours" else 0)
            with RunArtifactTracker(
                tmp_path / "runs",
                run_id=name,
                config={"synthetic": True, "review_binding": review_binding},
            ) as tracker:
                summary = dump(
                    tracker.artifact_path("summary.json"), {"metric": value, "synthetic": True}
                )
                tracker.register_artifact(summary, role="registered_experiment_result")
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "run_id": name,
                    "status": "success",
                    "cost": 2.0,
                    "holdout_sha256": "a" * 64,
                    "metric": value,
                }
            )
            producers[name] = {
                "manifest": fingerprint_file(tracker.manifest_path),
                "summary": fingerprint_file(summary),
                "metric_field": "metric",
                "cost_ledger": fingerprint_file(costs),
            }
    source = dump(
        tmp_path / "results.json",
        {
            "protocol": fingerprint_file(matrix),
            "rows": rows,
            "synthetic": True,
            "producers": producers,
        },
    )
    result = run_seed_analysis(
        source,
        baseline="random",
        source_commit="b" * 40,
        runs_root=tmp_path / "analysis",
        replicates=20,
        run_id="paired",
    )
    assert result["comparisons"]["ours"]["mean_difference"] == pytest.approx(0.1)
    assert result["total_cost_all_attempts"] == 12.0
    assert result["synthetic"] is True and result["statistical_significance_claimed"] is False
    assert (Path(result["run_dir"]) / "paired_effects.csv").is_file()


def test_pooled_evaluations_remap_ids_and_reproduce_without_original_sources(tmp_path):
    import shutil

    from test_evaluation import FakeDetector, build_dataset, evaluation_config

    from carla_vision.contracts import DetectorConfig
    from carla_vision.evaluation.pool import pool_evaluations
    from carla_vision.evaluation.runner import run_evaluation
    from carla_vision.reproduction.closure import build_closure
    from carla_vision.reproduction.replay_metrics import replay_coco_metrics

    original = tmp_path / "original"
    outputs = []
    for i in range(2):
        folder = original / str(i)
        folder.mkdir(parents=True)
        dataset = build_dataset(folder)
        config = dump(folder / "eval.json", evaluation_config())
        run = run_evaluation(
            config_path=config,
            dataset_path=dataset,
            detector_config=DetectorConfig(
                backend="custom", factory="fixture:factory", device="cpu", image_size=320
            ),
            runs_root=folder / "runs",
            run_id="source-eval",
            detector_override=FakeDetector(),
        )
        outputs.append(Path(run["run_dir"]))
    result = pool_evaluations(outputs, runs_root=original / "pooled", run_id="pooled")
    assert result["image_count"] == 2
    assert result["shard_aps_averaged"] is False
    assert result["model_inference_reexecuted"] is False
    truth = json.loads((Path(result["run_dir"]) / "instances.coco.json").read_text())
    assert [i["id"] for i in truth["images"]] == [1, 2]
    with pytest.raises(ValueError, match="more than once"):
        pool_evaluations([outputs[0], outputs[0]], runs_root=tmp_path / "bad")
    with pytest.raises(ValueError, match="budget"):
        pool_evaluations(outputs, runs_root=tmp_path / "bad2", maximum_images=1)
    manifest = Path(result["run_dir"]) / "manifest.json"
    sha = fingerprint_file(manifest)["sha256"]
    closure = build_closure([manifest], tmp_path / "bundle")
    shutil.rmtree(original)
    replay = replay_coco_metrics(
        tmp_path / "bundle",
        evaluation_manifest_sha256=sha,
        expected_closure_sha256=closure["index"]["sha256"],
        runs_root=tmp_path / "replayed",
    )
    assert replay["status"] == "passed" and replay["image_count"] == 2


def test_source_archive_cannot_forge_git_identity_with_a_marker(tmp_path):
    import zipfile

    from carla_vision.reproduction.reviewer import _source_archive
    from carla_vision.reproduction.source_snapshot import build_source_snapshot

    source = tmp_path / "source.zip"
    built = build_source_snapshot(Path(__file__).parents[1], source)
    changed = tmp_path / "changed.zip"
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(changed, "x") as dst:
        target = next(
            n for n in src.namelist() if n.startswith("carla_vision/") and n.endswith(".py")
        )
        for name in src.namelist():
            dst.writestr(
                name, b"# unrelated replacement source\n" if name == target else src.read(name)
            )
    with pytest.raises(ValueError, match="source blob"):
        _source_archive(changed, built["source_commit"])
