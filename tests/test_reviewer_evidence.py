"""Reviewer-facing negative/golden tests; these are not empirical ADAS results."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from carla_vision.artifacts import RunArtifactTracker, fingerprint_file


def dump(path, data):
    path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    return path


def route(name, points, split="train", *, group=None, map_id="a" * 64):
    return dict(
        route_id=name,
        scene_root_id=group or name,
        partition=split,
        map_sha256=map_id,
        points_xyz=points,
    )


def test_corridor_checks_segment_interiors_and_transitive_components():
    from carla_vision.dataset.leakage import audit_corridors

    routes = [
        route("a", [[0, 0, 0], [10, 0, 0]]),
        route("b", [[5, -5, 0], [5, 5, 0]], group="root-b"),
        route("c", [[5, 5.1, 0], [6, 6, 0]], "test", group="root-c"),
    ]
    result = audit_corridors(routes, clearance_m=0.2)
    assert result["status"] == "failed"
    assert len(result["components"]) == 1
    assert set(result["components"][0]["route_ids"]) == {"a", "b", "c"}
    assert audit_corridors(list(reversed(routes)), clearance_m=0.2) == result


def test_corridors_separate_overpass_and_map_but_bind_descendants():
    from carla_vision.dataset.leakage import audit_corridors

    routes = [route("a", [[0, 0, 0], [10, 0, 0]]), route("b", [[5, -5, 5], [5, 5, 5]], "test")]
    assert audit_corridors(routes, clearance_m=1)["status"] == "passed"
    routes[1]["scene_root_id"] = "a"
    assert audit_corridors(routes, clearance_m=1)["status"] == "failed"
    routes[1]["scene_root_id"] = "b"
    routes[1]["points_xyz"][0][2] = routes[1]["points_xyz"][1][2] = 0
    routes[1]["map_sha256"] = "b" * 64
    assert audit_corridors(routes, clearance_m=1)["status"] == "passed"


@pytest.mark.parametrize("bad", [float("nan"), -1, True, float("inf")])
def test_corridor_rejects_invalid_clearance(bad):
    from carla_vision.dataset.leakage import audit_corridors

    with pytest.raises((TypeError, ValueError)):
        audit_corridors([route("r", [[0, 0, 0], [1, 0, 0]])], clearance_m=bad)


def test_corridor_budget_exhaustion_never_means_disjoint():
    from carla_vision.dataset.leakage import audit_corridors

    with pytest.raises(ValueError, match="budget"):
        audit_corridors(
            [route("a", [[0, 0, 0], [1, 0, 0]]), route("b", [[0, 0, 0], [1, 1, 0]], "test")],
            clearance_m=0,
            maximum_segment_checks=0,
        )


def test_voxel_honors_locked_source_partitions(tmp_path):
    from test_voxel_training_dataset import _write_episode

    from carla_vision.voxel.training.dataset import TemporalVoxelDataset

    root = tmp_path / "voxel"
    _write_episode(root)
    path = root / "manifest.json"
    raw = json.loads(path.read_text())
    raw["split"] = "test_map_ood"
    dump(path, raw)
    for seed in range(6):
        data = TemporalVoxelDataset(
            [root], partition="train", split_seed=seed, history_frames=2, horizons_s=(0.0, 0.5)
        )
        assert not len(data)
    assert (
        len(TemporalVoxelDataset([root], partition="test", history_frames=2, horizons_s=(0.0, 0.5)))
        == 4
    )


def test_voxel_legacy_split_requires_explicit_compatibility(tmp_path):
    from test_voxel_training_dataset import _write_episode

    from carla_vision.voxel.training.dataset import TemporalVoxelDataset

    root = tmp_path / "voxel"
    _write_episode(root)
    with pytest.raises(ValueError, match="source partition"):
        TemporalVoxelDataset([root], partition="train")
    with pytest.warns(UserWarning, match="development"):
        TemporalVoxelDataset([root], partition="train", allow_legacy_splits=True)


def test_voxel_rejects_artifact_path_escape(tmp_path):
    from test_voxel_training_dataset import _write_episode

    from carla_vision.voxel.training.dataset import load_voxel_episode

    root = tmp_path / "voxel"
    _write_episode(root)
    path = root / "sequence.json"
    raw = json.loads(path.read_text())
    raw["records"][0]["rgb"] = "../outside.png"
    dump(path, raw)
    with pytest.raises(ValueError, match="path"):
        load_voxel_episode(root)


def test_empirical_claim_cannot_borrow_an_unrelated_manifest(tmp_path):
    from carla_vision.scenarios.claims import check_claims

    with RunArtifactTracker(tmp_path, run_id="fake", config={"synthetic": True}) as tracker:
        p = dump(tracker.artifact_path("test.json"), {"synthetic": True})
        tracker.register_artifact(p, role="offline_contract")
    fake = dump(
        tmp_path / "empirical.json",
        {
            "synthetic": False,
            "empirical_results_available": True,
            "scope": "training_utility",
            "producer_manifest": fingerprint_file(tracker.manifest_path),
        },
    )
    registry = dump(
        tmp_path / "claims.json",
        {
            "schema_version": "1.0",
            "claims": [
                {
                    "claim_id": "c1",
                    "scope": "training_utility",
                    "statement": "improved AP",
                    "evidence": [fingerprint_file(fake)],
                    "required_fields": {},
                }
            ],
            "redistribution": {},
        },
    )
    result = check_claims(registry)
    assert result["claims"][0]["ready"] is False
    assert result["acceptance_guaranteed"] is False


def test_crossed_analysis_keeps_method_seed_pairing():
    from carla_vision.evaluation.study_statistics import paired_seed_analysis

    left = {1: 0.3, 2: 0.4, 3: 0.5, 4: 0.6}
    right = {k: v + 0.1 for k, v in left.items()}
    r = paired_seed_analysis(left, right, replicates=100, seed=3)
    assert r["mean_difference"] == pytest.approx(0.1)
    assert r["seed_count"] == 4
    assert r["interval_scope"] == "training_seed_variation_fixed_evaluation_set"
    with pytest.raises(ValueError, match="paired"):
        paired_seed_analysis(left, {2: 0.5})


def test_power_requires_independent_pilot_not_final_test():
    from carla_vision.evaluation.study_statistics import plan_paired_sample_size

    with pytest.raises(ValueError, match="pilot"):
        plan_paired_sample_size([0.1, 0.2, 0.3], effect=0.1, source_role="test")
    r = plan_paired_sample_size([0.1, 0.2, 0.15, 0.35, 0.05], effect=0.05, source_role="pilot")
    assert r["required_independent_pairs"] > 5
    assert r["empirical_power_established"] is False


def test_holm_and_kappa_hand_answers():
    from carla_vision.evaluation.study_statistics import agreement, holm_adjust

    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert agreement(["valid", "invalid"], ["valid", "invalid"])["kappa"] == 1
    assert agreement(["valid", "valid"], ["valid", "valid"])["kappa"] is None


def test_result_aggregation_rejects_missing_seeds_and_changed_holdout():
    from carla_vision.evaluation.study_statistics import validate_run_matrix

    protocol = dict(methods=["random", "ours"], seeds=[1, 2], holdout_sha256="a" * 64, budget=10.0)
    rows = [
        dict(
            method=m,
            seed=s,
            status="success",
            cost=9.0,
            holdout_sha256="a" * 64,
            metric=0.3,
            run_id=f"{m}-{s}",
        )
        for m in protocol["methods"]
        for s in protocol["seeds"]
    ]
    assert validate_run_matrix(protocol, rows)["complete"] is True
    assert validate_run_matrix(protocol, rows[:-1])["complete"] is False
    rows[0]["holdout_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="holdout"):
        validate_run_matrix(protocol, rows)


def test_portable_evidence_closure_survives_relocation_and_detects_change(tmp_path):
    from carla_vision.reproduction.closure import build_closure, verify_closure

    root = tmp_path / "original"
    root.mkdir()
    child = dump(root / "child.json", {"synthetic": True})
    parent = dump(root / "parent.json", {"input": fingerprint_file(child)})
    bundle = tmp_path / "bundle"
    build_closure([parent], bundle)
    moved = tmp_path / "moved"
    shutil.move(bundle, moved)
    shutil.rmtree(root)
    result = verify_closure(moved)
    assert result["file_count"] == 2
    assert result["status"] == "passed"
    copied = next((moved / "objects").glob("*"))
    copied.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest|size"):
        verify_closure(moved)


def test_closure_collects_registered_artifacts_not_just_manifest(tmp_path):
    from carla_vision.reproduction.closure import build_closure, verify_closure

    with RunArtifactTracker(tmp_path / "runs", run_id="one") as tracker:
        p = tracker.artifact_path("table.csv")
        p.write_text("name,value\na,1\n")
        tracker.register_artifact(p, role="result_table")
    bundle = tmp_path / "bundle"
    build_closure([tracker.manifest_path], bundle)
    assert verify_closure(bundle)["file_count"] >= 2
    manifest = json.loads((bundle / "closure.json").read_text())
    assert not any(Path(row["stored_path"]).is_absolute() for row in manifest["files"])


def test_closure_rejects_empty_unknown_unregistered_and_traversal(tmp_path):
    from carla_vision.reproduction.closure import build_closure, verify_closure

    with pytest.raises(ValueError):
        build_closure([], tmp_path / "empty")
    p = dump(tmp_path / "a.json", {})
    build_closure([p], tmp_path / "b")
    index = tmp_path / "b/closure.json"
    raw = json.loads(index.read_text())
    raw["files"][0]["stored_path"] = "../a.json"
    dump(index, raw)
    with pytest.raises(ValueError, match="path|inventory"):
        verify_closure(tmp_path / "b")


def test_published_greedy_coreset_algorithm_has_exact_small_answer():
    from carla_vision.scenarios.baselines import kcenter_greedy

    x = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0], [10.0, 0.0]])
    assert kcenter_greedy(x, labeled=[0], count=2) == [3, 2]
    assert kcenter_greedy(x, labeled=[0], count=0) == []
    with pytest.raises(ValueError):
        kcenter_greedy(x, labeled=[0, 0], count=1)
    with pytest.raises(ValueError):
        kcenter_greedy(x, labeled=[], count=1)


def test_cost_ledger_empty_or_rehashed_bad_records_do_not_pass(tmp_path):
    from carla_vision.dataset.lineage import canonical_digest
    from carla_vision.scenarios.costs import verify_cost_ledger

    p = tmp_path / "empty.jsonl"
    p.write_text("")
    with pytest.raises(ValueError, match="empty"):
        verify_cost_ledger(p)
    row = {
        "schema_version": "1.0",
        "event_id": "x",
        "kind": "train",
        "outcome": "success",
        "resources": {},
        "evidence_ref": "",
        "previous_sha256": "0" * 64,
    }
    row["sha256"] = canonical_digest(row)
    p.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        verify_cost_ledger(p)


def test_reviewer_template_reports_missing_not_ready(tmp_path):
    from carla_vision.reproduction.reviewer import review_submission, template

    p = dump(tmp_path / "submission.json", template())
    result = review_submission(p)
    assert result["ready"] is False
    assert result["acceptance_guaranteed"] is False
    assert len(result["missing"]) >= 8
    bad = template()
    bad["requirements"] = []
    dump(p, bad)
    with pytest.raises(ValueError):
        review_submission(p)


def test_corridors_near_parallel_segment_intersection():
    from carla_vision.dataset.leakage import segment_distance

    assert segment_distance([0, 0, 0], [1, 0, 0], [0, 1e-12, 0], [1, -1e-12, 0]) == pytest.approx(
        0, abs=1e-16
    )


def test_portable_closure_supports_cost_jsonl_and_dataset_root_references(tmp_path):
    from carla_vision.reproduction.closure import build_closure, verify_closure
    from carla_vision.scenarios.costs import CostLedger

    p = tmp_path / "costs.jsonl"
    with CostLedger(p) as ledger:
        ledger.record(
            "one", kind="train", outcome="success", resources={"wall_seconds": 1}, evidence_ref="x"
        )
        ledger.record(
            "two", kind="train", outcome="failed", resources={"wall_seconds": 2}, evidence_ref="y"
        )
    (tmp_path / "metadata").mkdir()
    (tmp_path / "images").mkdir()
    image = tmp_path / "images/a.png"
    image.write_bytes(b"not-a-real-image-for-path-resolution-test")
    reference = fingerprint_file(image)
    reference["path"] = "images/a.png"
    nested = dump(tmp_path / "metadata/a.json", {"image": reference})
    top = dump(
        tmp_path / "root.json",
        {"artifacts": [reference, fingerprint_file(nested), fingerprint_file(p)]},
    )
    build_closure([top], tmp_path / "bundle")
    assert verify_closure(tmp_path / "bundle")["file_count"] == 4


def test_voxel_future_target_cannot_reference_past_or_skipped_horizon(tmp_path):
    from test_voxel_training_dataset import _write_episode

    from carla_vision.voxel.training.dataset import load_voxel_episode

    root = tmp_path / "voxel"
    _write_episode(root)
    p = root / "sequence.json"
    raw = json.loads(p.read_text())
    raw["future_targets"][0]["future_targets"]["0.500"] = 4
    dump(p, raw)
    with pytest.raises(ValueError, match="horizon"):
        load_voxel_episode(root)


def test_tracker_costs_measured_in_success_and_failure_without_double_count_claim(tmp_path):
    for fail in (False, True):
        try:
            with RunArtifactTracker(tmp_path, run_id=f"cost-{fail}") as tracker:
                if fail:
                    raise RuntimeError("injected")
        except RuntimeError:
            pass
        raw = json.loads(tracker.manifest_path.read_text())
        resources = raw["resource_usage"]
        assert resources["wall_seconds"] >= 0 and resources["cpu_seconds"] >= 0
        assert resources["gpu_seconds"] is None
        assert resources["child_processes_included"] is False
        assert resources["overlapping_scopes_additive"] is False


def test_claim_registry_requires_bound_registered_result_not_self_declared_wrapper(tmp_path):
    from carla_vision.scenarios.claims import check_claims

    with RunArtifactTracker(
        tmp_path, run_id="measured", config={"synthetic": False, "evidence_kind": "empirical"}
    ) as tracker:
        summary = dump(
            tracker.artifact_path("result.json"),
            {
                "scope": "training_utility",
                "synthetic": False,
                "empirical_results_available": True,
                "improvement": False,
            },
        )
        tracker.register_artifact(summary, role="empirical_study_result")
    wrapper = dump(
        tmp_path / "wrapper.json",
        {
            "producer_manifest": fingerprint_file(tracker.manifest_path),
            "result_artifact": fingerprint_file(summary),
            "improvement": True,
        },
    )
    registry = dump(
        tmp_path / "registry.json",
        {
            "schema_version": "1.0",
            "claims": [
                {
                    "claim_id": "h",
                    "scope": "training_utility",
                    "statement": "test-only simulated claim binding",
                    "evidence": [fingerprint_file(wrapper)],
                    "required_fields": {"improvement": True},
                }
            ],
            "redistribution": {},
        },
    )
    assert check_claims(registry)["claims"][0]["ready"] is False


def test_complete_offline_gate_is_executable_and_cannot_be_promoted_to_empirical(tmp_path):
    from carla_vision.reproduction.reviewer import DOCUMENTS, review_submission, template
    from carla_vision.reproduction.source_snapshot import build_source_snapshot

    snapshot = build_source_snapshot(Path(__file__).parents[1], tmp_path / "source.zip")
    split = dump(tmp_path / "split.json", {"schema_version": "1.0", "groups": {"one": "train"}})
    pilot = dump(tmp_path / "pilot.json", {"synthetic": True})
    learner = dump(tmp_path / "learner.json", {"synthetic": True})
    protocol = dump(
        tmp_path / "protocol.json",
        {
            "schema_version": "1.0",
            "study_id": "software-fixture",
            "purpose": "development",
            "hypotheses": ["software invariants, not model improvement"],
            "primary_metric": "fixture_only",
            "minimum_effect": 0.01,
            "methods": ["random", "capability_matched", "error_source_aware"],
            "ablations": ["quality", "attribution", "coverage", "exploration"],
            "training_seeds": [1, 2],
            "generation_seeds": [3, 4],
            "selection": {"budget": 10.0, "max_actions": 2},
            "split_manifest": fingerprint_file(split),
            "regimes": ["clean", "capture_fault", "supervision_fault", "mixed"],
            "stopping_rule": "synthetic fixed count",
            "power_analysis": fingerprint_file(pilot),
            "learner_configs": [fingerprint_file(learner)],
            "cost_scope": "all_attempts_including_failures_reviews_and_training",
        },
    )
    geometry = dump(
        tmp_path / "geometry.json",
        {
            "routes": [route("one", [[0, 0, 0], [1, 0, 0]])],
            "clearance_m": 1.0,
            "maximum_segment_checks": 100,
            "source_artifacts": [fingerprint_file(split)],
        },
    )
    # Handwritten XML here tests the parser only; it is NOT an execution record.
    junit = tmp_path / "parser-fixture.xml"
    junit.write_text(
        '<testsuites><testsuite><testcase name="synthetic-parser-fixture"/></testsuite></testsuites>'
    )
    raw = template()
    raw.update(
        study_id="software-fixture", mode="offline_fixture", source_commit=snapshot["source_commit"]
    )
    for role, p in {
        "source_snapshot": tmp_path / "source.zip",
        "protocol": protocol,
        "tests": junit,
        "geometry": geometry,
        "dependency_lock": Path(__file__).parents[1] / "uv.lock",
    }.items():
        raw["artifacts"][role] = fingerprint_file(p)
    for role in DOCUMENTS:
        if role == "independent_reproduction":
            continue
        content = {
            name: "Synthetic schema test, not human approval or measurements"
            for name in DOCUMENTS[role]
        }
        if role == "manuscript":
            content["author_approval"] = True  # parser fixture only, offline mode
        raw["artifacts"][role] = fingerprint_file(dump(tmp_path / f"{role}.json", content))
    path = dump(tmp_path / "submission.json", raw)
    result = review_submission(path)
    assert result["ready"] is True, result
    assert result["empirical_submission_ready"] is False
    lock = tmp_path / "unrelated.lock"
    lock.write_text("Unrelated dependency lock bytes, despite a valid fingerprint." * 2)
    raw["artifacts"]["dependency_lock"] = fingerprint_file(lock)
    dump(path, raw)
    assert "source_lock_binding" in review_submission(path)["errors"]
    raw["mode"] = "empirical_simulation"
    dump(path, raw)
    promoted = review_submission(path)
    assert promoted["ready"] is False and promoted["empirical_submission_ready"] is False
