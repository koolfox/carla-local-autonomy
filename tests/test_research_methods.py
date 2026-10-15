"""Falsification/golden tests for the offline research method, not empirical ADAS results."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from carla_vision.artifacts import fingerprint_file
from carla_vision.dataset.derivation import derive_dataset
from carla_vision.dataset.lineage import scene_group_key, validate_derivation
from carla_vision.dataset.quality import ReviewRecord, estimate_quality
from carla_vision.evaluation.uncertainty import bootstrap_coco, resample_coco, supported_ratios
from carla_vision.failure_mining.attribution import CHECKS, AttributionEvidence, attribute_error
from carla_vision.native.session_lease import NativeSessionLease
from carla_vision.native.synchronization import (
    NativeSensorQueue,
    carla_transform_matrix,
    compose_relative_transform,
)
from carla_vision.native.worker import _response_ids
from carla_vision.scenarios.acquisition import (
    METHODS,
    AcquisitionCandidate,
    AcquisitionConfig,
    select_acquisition,
)
from carla_vision.scenarios.contracts import TransformRecipe
from carla_vision.scenarios.costs import CostLedger, verify_cost_ledger
from carla_vision.scenarios.cycle import CycleJournal, verify_cycle
from carla_vision.scenarios.study import plan_study, validate_protocol


def candidate(i=0, **kwargs):
    values = dict(
        candidate_id=str(i),
        group_id=f"g{i}",
        condition="night" if i % 2 else "day",
        partition="acquisition_pool",
        estimated_cost=2.0 + i % 3,
        model_error=float(i % 4),
        uncertainty=0.5,
        quality_lower=0.95,
        action="NEW_SCENE",
        evidence_ref="probe",
        repair_value=0.3,
    )
    return AcquisitionCandidate(**(values | kwargs))


def evidence(**kwargs):
    values = dict(
        capture_checks=dict.fromkeys(CHECKS, True),
        model_error=1.0,
        label_quality_lower=None,
        capture_evidence_ref="probe.json",
    )
    return AttributionEvidence(**(values | kwargs))


def test_error_is_not_teacher_invalidity():
    assert attribute_error(evidence(model_error=999.0)).source_kind == "unknown"
    good = evidence(independent_review="valid", review_evidence_ref="independent.json")
    assert attribute_error(good).source_kind == "model_error"
    assert attribute_error(replace(good, model_error=0.0)).recommended_action == "KEEP"
    bad = replace(good, capture_checks=dict.fromkeys(CHECKS, True) | {"frame": False})
    assert attribute_error(bad).recommended_action == "RECAPTURE"
    assert (
        attribute_error(replace(good, independent_review="invalid")).recommended_action == "RELABEL"
    )


def test_unknown_capture_and_action_quality_are_explicit():
    partial = evidence(capture_checks=dict.fromkeys(CHECKS, True) | {"teacher_timing": None})
    assert attribute_error(partial).unknown_reason == "capture_check_unresolved"
    label_only = evidence(task="imitation", label_quality_lower=1.0, quality_evidence_ref="review")
    assert attribute_error(label_only).source_kind == "unknown"
    with pytest.raises(ValueError, match="audit evidence"):
        evidence(action_quality_lower=1.0)
    with pytest.raises(TypeError):
        AttributionEvidence(**(asdict(partial) | {"fault_oracle": "bad_label"}))


def test_quality_keeps_ambiguity_and_is_not_certain_with_few_all_valid_groups():
    records = [ReviewRecord(str(i), f"g{i}", "r1", "valid", "review") for i in range(6)]
    result = estimate_quality(records, task="label", replicates=10)
    assert 0 < result["quality_lower"] < 1
    assert estimate_quality(records, task="action", replicates=10)["quality_lower"] is None
    records.append(ReviewRecord("0", "g0", "r2", "invalid", "review"))
    mixed = estimate_quality(records, task="label", replicates=10)
    assert mixed["unresolved_count"] == 1 and mixed["sample_count"] == 6
    with pytest.raises(ValueError, match="duplicate review"):
        estimate_quality(records + [records[0]], task="label")


@pytest.mark.parametrize("method", sorted(METHODS))
def test_all_baselines_share_budget_group_exclusion_and_determinism(method):
    pool = [candidate(i) for i in range(12)]
    config = AcquisitionConfig(method, 12.0, 4, minimum_per_condition={"night": 1, "day": 1})
    a = select_acquisition(pool, config)
    b = select_acquisition(list(reversed(pool)), config)
    assert a == b
    assert a["estimated_cost"] <= config.budget
    assert len(a["selected"]) <= 4
    assert len({row["group_id"] for row in a["selected"]}) == len(a["selected"])
    assert a["planned_condition_counts"]["night"] >= 1
    assert not a["empirical_superiority_established"]


@pytest.mark.parametrize("partition", ["val_seen", "test_seen", "test_map_ood", "unassigned"])
def test_candidate_test_and_validation_data_cannot_drive_acquisition(partition):
    with pytest.raises(ValueError):
        candidate(partition=partition)


def test_cost_ledger_counts_failures_and_detects_changes(tmp_path):
    path = tmp_path / "costs.jsonl"
    with CostLedger(path) as ledger:
        ledger.record(
            "success",
            kind="render",
            outcome="success",
            resources={"gpu_seconds": 3.0},
            evidence_ref="capture",
        )
        ledger.record(
            "reject",
            kind="probe",
            outcome="rejected",
            resources={"gpu_seconds": 2.0},
            evidence_ref="probe",
        )
        with pytest.raises(RuntimeError):
            with ledger.measure("failure", kind="audit", evidence_ref="audit"):
                raise RuntimeError("deliberate injected failure")
    report = verify_cost_ledger(path)
    assert report["totals"]["gpu_seconds"] == 5.0
    assert report["outcomes"] == {"success": 1, "rejected": 1, "failed": 1}
    text = path.read_text().replace('"gpu_seconds": 3.0', '"gpu_seconds": 0.0')
    path.write_text(text)
    with pytest.raises(ValueError, match="hash chain"):
        verify_cost_ledger(path)


def test_cycle_order_resume_and_test_exclusion(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    path = tmp_path / "cycle.jsonl"
    with CycleJournal(
        path, cycle_id="c", partition="acquisition_pool", protocol_ref=protocol
    ) as cycle:
        with pytest.raises(ValueError, match="skipped"):
            cycle.advance("frozen", {"training_dataset": protocol})
        cycle.advance("planned", {"plan": protocol})
    with CycleJournal(
        path, cycle_id="c", partition="acquisition_pool", protocol_ref=protocol, resume=True
    ) as cycle:
        cycle.advance("probed", {"probe": protocol})
    assert verify_cycle(path)["stage"] == "probed"
    with pytest.raises(ValueError):
        CycleJournal(tmp_path / "bad.jsonl", cycle_id="c", partition="test", protocol_ref=protocol)
    protocol.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="changed"):
        verify_cycle(path)


def test_grouping_ignores_route_seed_and_recapture_episode():
    base = {
        "map_family": "Town10HD_Opt",
        "route": {"start_spawn_index": 2, "destination_spawn_index": 8, "route_seed": 1},
    }
    other = {**base, "route": {**base["route"], "route_seed": 999}}
    assert scene_group_key(base, episode_id="a") == scene_group_key(other, episode_id="b")
    with pytest.raises(ValueError):
        validate_derivation("test_map_ood", "train", "RELABEL")


def test_se3_composition_non_commuting_and_positive_carla_pitch():
    a = TransformRecipe(1.0, 2.0, 3.0, pitch=25.0, yaw=40.0, roll=15.0)
    b = TransformRecipe(3.0, -2.0, 1.0, pitch=-35.0, yaw=20.0, roll=50.0)
    combined = compose_relative_transform(a, b)
    actual = np.asarray(carla_transform_matrix(combined))
    expected = np.asarray(carla_transform_matrix(a)) @ np.asarray(carla_transform_matrix(b))
    assert np.allclose(actual, expected, atol=1e-8)
    assert not np.allclose(
        expected, np.asarray(carla_transform_matrix(b)) @ np.asarray(carla_transform_matrix(a))
    )
    pitched = compose_relative_transform(
        TransformRecipe(0, 0, 0, 90, 0, 0), TransformRecipe(1, 0, 0, 0, 0, 0)
    )
    assert pitched.z == pytest.approx(1.0)


def test_partial_batch_registration_precedes_error():
    registered = []
    responses = [
        SimpleNamespace(error="", actor_id=12),
        SimpleNamespace(error="failed", actor_id=0),
    ]
    with pytest.raises(RuntimeError):
        _response_ids(responses, operation="spawn", strict=True, register=registered.extend)
    assert registered == [12]


def test_cooperative_lease_released_on_error(tmp_path):
    # Defaults can be shared across processes; inject an isolated lock root for this test.
    with pytest.raises(RuntimeError):
        with NativeSessionLease("localhost", 28766, directory=tmp_path):
            with NativeSessionLease("localhost", 28766, directory=tmp_path):
                pytest.fail("duplicate local tick owner acquired lease")
    with NativeSessionLease("localhost", 28766, directory=tmp_path):
        pass


def test_sensor_queue_default_is_bounded_and_counts_discard():
    queue = NativeSensorQueue("front")
    for i in range(20):
        queue.callback(SimpleNamespace(frame=i))
    assert queue.received == 20 and queue.discarded == 16
    assert queue.get_exact(19, 0.1).image.frame == 19


def test_bootstrap_duplicate_groups_keep_multiplicity_and_perfect_ap():
    truth = {
        "images": [{"id": 1, "width": 20, "height": 20}, {"id": 2, "width": 20, "height": 20}],
        "categories": [{"id": 1, "name": "object"}],
        "annotations": [
            {
                "id": i,
                "image_id": i,
                "category_id": 1,
                "bbox": [1, 1, 5, 5],
                "area": 25,
                "iscrowd": 0,
            }
            for i in (1, 2)
        ],
    }
    pred = [{"image_id": i, "category_id": 1, "bbox": [1, 1, 5, 5], "score": 0.9} for i in (1, 2)]
    mapping = {1: "g1", 2: "g2"}
    doubled, doubled_pred = resample_coco(truth, pred, mapping, ["g1", "g1"])
    assert len(doubled["images"]) == len(doubled["annotations"]) == len(doubled_pred) == 2
    assert {row["id"] for row in doubled["images"]} == {1, 2}
    result = bootstrap_coco(
        truth, pred, group_by_image=mapping, replicates=3, reference_predictions=pred
    )
    assert result["ap_50_95"]["lower"] == pytest.approx(1.0)
    assert result["paired_ap_delta"]["upper"] == pytest.approx(0.0)
    assert supported_ratios(0, 0, 0)["recall"] is None
    assert supported_ratios(0, 1, 0)["ground_truth_support"] == 0


def test_immutable_freeze_and_test_rejection(tmp_path):
    from test_dataset_writer import synchronized_pair

    from carla_vision.dataset.verified import load_verified_dataset
    from carla_vision.dataset.writer import DatasetWriter

    for split in ("acquisition_pool", "test_seen"):
        with DatasetWriter(
            tmp_path,
            dataset_id=split,
            carla_endpoint={},
            carla_version="fixture",
            carla_map="fixture",
        ) as writer:
            pair, _, _ = synchronized_pair()
            writer.add_pair(pair, split=split, scenario_id="s", episode_id="e")
    before = fingerprint_file(tmp_path / "acquisition_pool" / "dataset.json")
    result = derive_dataset(
        tmp_path / "acquisition_pool", datasets_root=tmp_path, dataset_id="frozen", action="FREEZE"
    )
    assert result["sample_count"] == 1
    assert load_verified_dataset(tmp_path / "frozen").partitions == {"train": 1}
    assert fingerprint_file(tmp_path / "acquisition_pool" / "dataset.json") == before
    with pytest.raises(ValueError):
        derive_dataset(
            tmp_path / "test_seen", datasets_root=tmp_path, dataset_id="bad", action="FREEZE"
        )


def test_observer_cleanup_after_partial_sensor_creation(tmp_path, monkeypatch):
    from carla_vision import closed_loop_evaluation as observer

    location = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    vehicle = SimpleNamespace(get_location=lambda: location)
    client = SimpleNamespace(set_timeout=lambda _: None, get_world=lambda: object())
    sensor = Mock()
    sensor.destroy.return_value = True
    monkeypatch.setattr(observer, "_load_carla", lambda: SimpleNamespace(Client=lambda *_: client))
    monkeypatch.setattr(observer, "_find_vehicle", lambda *_, **__: vehicle)
    monkeypatch.setattr(
        observer, "_choose_destination", lambda *_, **__: (SimpleNamespace(location=location), 0)
    )
    monkeypatch.setattr(observer, "_route_points", lambda *_: np.array([[0.0, 0.0], [10.0, 0.0]]))
    monkeypatch.setattr(
        observer,
        "_spawn_event_sensor",
        Mock(side_effect=[sensor, RuntimeError("second spawn failed")]),
    )
    args = observer.build_parser().parse_args(["--output", str(tmp_path / "observer")])
    with pytest.raises(RuntimeError, match="second spawn"):
        observer.run(args)
    sensor.destroy.assert_called_once()
    assert json.loads((tmp_path / "observer" / "manifest.json").read_text())["status"] == "failed"


def test_route_loop_cannot_jump_to_far_future_segment():
    from carla_vision.closed_loop_evaluation import RouteProgressTracker

    points = np.array(
        [[0.0, 0.0], [10.0, 0.0], [30.0, 0.0], [60.0, 0.0], [60.0, 40.0], [0.0, 40.0], [0.0, 0.1]]
    )
    tracker = RouteProgressTracker(points)
    tracker.update(0.0, 0.09)
    assert tracker.completion == 0.0


def test_study_plan_is_executable_but_not_empirical_result(tmp_path):
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return path

    pool = [candidate(i) for i in range(8)]
    split = save(
        "split.json", {"schema_version": "1.0", "groups": {c.group_id: c.partition for c in pool}}
    )
    power = save(
        "power.json",
        {"status": "registered", "basis": "registered_compute_limit", "rationale": "fixture_only"},
    )
    learner = save("learner.json", {"fixture": True})
    probe = save("probe.json", {"synthetic": True, "scope": "software_contract_fixture"})
    protocol = {
        "schema_version": "1.0",
        "study_id": "fixture-study",
        "purpose": "development",
        "hypotheses": ["triage actions are reproducible"],
        "primary_metric": "heldout_AP",
        "minimum_effect": 0.01,
        "methods": ["random", "capability_matched", "error_source_aware"],
        "ablations": ["quality", "attribution", "coverage", "exploration"],
        "training_seeds": [11, 12],
        "generation_seeds": [20, 21],
        "selection": {"budget": 10.0, "max_actions": 3},
        "split_manifest": fingerprint_file(split),
        "regimes": ["clean", "capture_fault", "supervision_fault", "mixed"],
        "stopping_rule": "fixed_budget",
        "power_analysis": fingerprint_file(power),
        "learner_configs": [fingerprint_file(learner)],
        "cost_scope": "all_attempts_including_failures_reviews_and_training",
    }
    protocol_path = save("protocol.json", protocol)
    pool_path = save(
        "pool.json",
        {
            "schema_version": "1.0",
            "candidates": [asdict(c) for c in pool],
            "evidence": {"probe": fingerprint_file(probe)},
        },
    )
    result = plan_study(protocol_path, pool_path, runs_root=tmp_path / "runs", run_id="study")
    assert result["plan_count"] == 14 and not result["empirical_results_available"]
    with pytest.raises(ValueError, match="mandatory"):
        validate_protocol(protocol | {"methods": ["random", "error_source_aware"]})


def test_current_voxel_tags_do_not_treat_terrain_and_walls_as_road_users():
    from carla_vision.dataset.semantic_version import DYNAMIC_ROAD_USER_TAGS
    from carla_vision.voxel.training.dataset import compact_semantic_labels

    assert 4 not in DYNAMIC_ROAD_USER_TAGS and 10 not in DYNAMIC_ROAD_USER_TAGS
    assert {12, 13, 14, 19} <= set(DYNAMIC_ROAD_USER_TAGS)
    tags = np.asarray([1, 2, 3, 12, 14, 24, 10, 4], dtype=np.uint8)
    mapped = compact_semantic_labels(tags, np.ones(tags.shape, dtype=np.int8))
    assert mapped.tolist() == [1, 2, 0, 4, 3, 1, 6, 6]


def test_publication_gate_does_not_promote_fixture_to_empirical_claim(tmp_path):
    from carla_vision.scenarios.claims import check_claims

    proof = tmp_path / "fixture.json"
    proof.write_text(
        json.dumps({"synthetic": True, "empirical_results_available": False, "passed": True})
    )
    claim = {
        "claim_id": "h1",
        "scope": "training_utility",
        "statement": "test-only claim",
        "evidence": [fingerprint_file(proof)],
        "required_fields": {"passed": True},
    }
    registry = tmp_path / "claims.json"
    registry.write_text(
        json.dumps({"schema_version": "1.0", "claims": [claim], "redistribution": {}})
    )
    result = check_claims(registry)
    assert not result["claims"][0]["ready"]
    assert not result["submission_evidence_ready"]
    claim["scope"] = "offline_contract"
    registry.write_text(
        json.dumps({"schema_version": "1.0", "claims": [claim], "redistribution": {}})
    )
    assert check_claims(registry)["claims"][0]["ready"]
    assert not check_claims(registry)["redistribution_ready"]


def test_derivation_root_is_stable_across_generations():
    context = {
        "map_family": "Town10HD_Opt",
        "route": {"start_spawn_index": 1, "destination_spawn_index": 2},
    }
    parent = scene_group_key(context, episode_id="parent")
    child = {**context, "lineage": {"scene_root_id": parent}}
    assert scene_group_key(child, episode_id="child") == parent
    grandchild = {
        **context,
        "lineage": {"scene_root_id": scene_group_key(child, episode_id="child")},
    }
    assert scene_group_key(grandchild, episode_id="grandchild") == parent


def test_failed_cost_write_does_not_hide_primary_error(tmp_path, monkeypatch):
    with CostLedger(tmp_path / "costs.jsonl") as ledger:
        monkeypatch.setattr(ledger, "record", Mock(side_effect=OSError("disk full")))
        with pytest.raises(RuntimeError, match="original") as caught:
            with ledger.measure("job", kind="render", evidence_ref="job-log"):
                raise RuntimeError("original")
        assert any("finalization" in note for note in caught.value.__notes__)
