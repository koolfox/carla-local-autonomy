"""Cross-role contract tests on synthetic files, not empirical paper evidence."""

from __future__ import annotations

import copy
import json

import pytest

from carla_vision.artifacts import fingerprint_file


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    return path


def rebind_table_fixture(paths):
    """Test-only deliberate mutation: keep the synthetic table pointer in sync."""
    raw = json.loads(paths["manuscript"].read_text())
    raw["tables"][0]["results"] = fingerprint_file(paths["results"])
    dump(paths["manuscript"], raw)


@pytest.fixture
def releases(tmp_path):
    from test_training import sample_pair

    from carla_vision.dataset.writer import DatasetWriter

    roots = []
    for i, partition in enumerate(("train", "val_seen", "test_seen")):
        with DatasetWriter(
            tmp_path / "data",
            dataset_id=f"ds-{i}",
            minimum_pixels=4,
            carla_endpoint={"host": "synthetic-fixture"},
            carla_version="0.9.16",
            carla_map="fixture",
        ) as writer:
            for frame in (1, 2):
                writer.add_pair(
                    sample_pair(frame, frame, i * 30 + frame),
                    split=partition,
                    scenario_id=f"s-{i}",
                    episode_id=f"e-{i}",
                    context={
                        "map_family": "fixture",
                        "route": {
                            "start_spawn_index": i * 2,
                            "destination_spawn_index": i * 2 + 1,
                        },
                    },
                )
        roots.append(writer.dataset_dir)
    return roots


def test_freeze_recomputes_all_samples_and_never_repartitions(releases, tmp_path):
    from carla_vision.reproduction.study_links import freeze_split, verify_split

    path = tmp_path / "split.json"
    raw = freeze_split(releases, path)
    assert raw["schema_version"] == "1.1"
    assert len(raw["samples"]) == 6
    assert raw["holdout_count"] == 2
    assert set(raw["groups"].values()) == {"train", "val_seen", "test_seen"}
    assert verify_split(path) == raw
    with pytest.raises(FileExistsError):
        freeze_split(releases, path)
    assert json.loads(path.read_text()) == raw
    changed = copy.deepcopy(raw)
    next(s for s in changed["samples"] if s["partition"] == "train")["partition"] = "test_seen"
    dump(path, changed)
    with pytest.raises(ValueError, match="source|inventory"):
        verify_split(path)


def test_freeze_rejects_duplicate_roots_budget_and_unknown_split(releases, tmp_path):
    from carla_vision.reproduction.study_links import freeze_split

    with pytest.raises(ValueError, match="duplicate"):
        freeze_split([releases[0], releases[0]], tmp_path / "duplicate.json")
    with pytest.raises(ValueError, match="budget"):
        freeze_split(releases, tmp_path / "short.json", maximum_samples=5)
    assert not (tmp_path / "short.json").exists()
    with pytest.raises(ValueError):
        freeze_split([], tmp_path / "empty.json")


def test_freeze_detects_cross_release_exact_rgb_and_group_leakage(releases, tmp_path):
    from test_training import sample_pair

    from carla_vision.dataset.writer import DatasetWriter
    from carla_vision.reproduction.study_links import freeze_split

    for name, color, source in (("rgb", 1, 9), ("group", 199, 0)):
        with DatasetWriter(
            tmp_path / "other",
            dataset_id=name,
            minimum_pixels=4,
            carla_endpoint={"host": "synthetic-fixture"},
            carla_version="0.9.16",
            carla_map="fixture",
        ) as writer:
            writer.add_pair(
                sample_pair(1, 1, color),
                split="test_seen",
                scenario_id=name,
                episode_id=name,
                context={
                    "map_family": "fixture",
                    "route": {
                        "start_spawn_index": source * 2,
                        "destination_spawn_index": source * 2 + 1,
                    },
                },
            )
        with pytest.raises(ValueError, match="crosses|overlap"):
            freeze_split([releases[0], writer.dataset_dir], tmp_path / f"{name}.json")


@pytest.fixture
def linked_files(releases, tmp_path):
    """A link-validator fixture; it deliberately does not claim a valid empirical run."""
    from carla_vision.reproduction.study_links import freeze_split
    from carla_vision.scenarios.costs import CostLedger

    split_path = tmp_path / "split.json"
    split = freeze_split(releases, split_path)
    protocol = dump(
        tmp_path / "protocol.json",
        {
            "study_id": "fixture-study",
            "split_manifest": fingerprint_file(split_path),
            "methods": ["one", "two"],
            "training_seeds": [1, 2],
            "primary_metric": "coco.ap",
        },
    )
    matrix = dump(
        tmp_path / "matrix.json",
        {
            "methods": ["one", "two"],
            "seeds": [1, 2],
            "holdout_sha256": split["holdout_sha256"],
        },
    )
    ledger_path = tmp_path / "costs.jsonl"
    with CostLedger(ledger_path) as ledger:
        ledger.record(
            "event-1",
            kind="train",
            outcome="success",
            resources={"wall_seconds": 2.0},
            evidence_ref="synthetic",
        )
    # Result role verification independently checks actual run/summary bindings.
    producer = dump(
        tmp_path / "producer.json",
        {
            "references": {"datasets": [{"manifest": r} for r in split["dataset_manifests"]]},
        },
    )
    results = dump(
        tmp_path / "results.json",
        {
            "protocol": fingerprint_file(matrix),
            "rows": [],
            "synthetic": True,
            "producers": {
                "run-1": {
                    "manifest": fingerprint_file(producer),
                    "metric_field": "coco.ap",
                    "cost_ledger": fingerprint_file(ledger_path),
                }
            },
        },
    )
    geometry = dump(
        tmp_path / "geometry.json",
        {
            "source_artifacts": split["dataset_manifests"],
        },
    )
    claims = dump(
        tmp_path / "claims.json",
        {
            "claims": [
                {"claim_id": "h1", "scope": "offline_contract", "evidence": []},
            ]
        },
    )
    manuscript = dump(
        tmp_path / "manuscript.json",
        {
            "claims": ["h1"],
            "tables": [{"table_id": "t1", "results": fingerprint_file(results)}],
        },
    )
    selected = split["samples"][0]
    audit = dump(
        tmp_path / "audit.json",
        {
            "records": [
                {
                    "sample_id": selected["sample_id"],
                    "group_id": selected["group_id"],
                    "dataset_manifest_sha256": selected["manifest_sha256"],
                }
            ]
        },
    )
    baseline = dump(tmp_path / "baselines.json", {"methods": ["one", "two"]})
    paths = dict(
        protocol=protocol,
        results=results,
        geometry=geometry,
        claim_registry=claims,
        manuscript=manuscript,
        label_audit=audit,
        baseline_registry=baseline,
        costs=ledger_path,
    )
    return paths, matrix, split_path


def test_matching_links_are_consistency_only_not_empirical_success(linked_files):
    from carla_vision.reproduction.study_links import check_study_links

    paths, _, _ = linked_files
    result = check_study_links(paths, study_id="fixture-study")
    assert result["status"] == "passed"
    assert result["sample_count"] == 6
    assert result["holdout_count"] == 2
    assert result["scope"] == "cross_artifact_consistency_not_empirical_validation"


@pytest.mark.parametrize(
    "field,value",
    [
        ("methods", ["one"]),
        ("seeds", [1]),
        ("holdout_sha256", "f" * 64),
    ],
)
def test_rejects_individually_valid_but_different_matrix(linked_files, field, value):
    from carla_vision.reproduction.study_links import check_study_links

    paths, matrix, _ = linked_files
    raw = json.loads(matrix.read_text())
    raw[field] = value
    dump(matrix, raw)
    r = json.loads(paths["results"].read_text())
    r["protocol"] = fingerprint_file(matrix)
    dump(paths["results"], r)
    with pytest.raises(ValueError, match="method|seed|holdout"):
        check_study_links(paths, study_id="fixture-study")


def test_unrelated_geometry_audit_cannot_cover_the_study(linked_files):
    from carla_vision.reproduction.study_links import check_study_links

    paths, _, _ = linked_files
    data = json.loads(paths["geometry"].read_text())
    data["source_artifacts"].pop()
    dump(paths["geometry"], data)
    with pytest.raises(ValueError, match="geometry"):
        check_study_links(paths, study_id="fixture-study")


@pytest.mark.parametrize(
    "role,field,value",
    [
        ("manuscript", "claims", ["unregistered"]),
        ("manuscript", "tables", []),
        ("baseline_registry", "methods", ["one"]),
        ("label_audit", "records", [{"sample_id": "other", "group_id": "other"}]),
    ],
)
def test_manuscript_baselines_and_audit_must_match_the_selected_study(
    linked_files,
    role,
    field,
    value,
):
    from carla_vision.reproduction.study_links import check_study_links

    paths, _, _ = linked_files
    raw = json.loads(paths[role].read_text())
    raw[field] = value
    dump(paths[role], raw)
    with pytest.raises(ValueError):
        check_study_links(paths, study_id="fixture-study")


def test_table_inventory_cannot_reference_another_results_file(linked_files, tmp_path):
    from carla_vision.reproduction.study_links import check_study_links

    paths, _, _ = linked_files
    other = dump(tmp_path / "other.json", {"synthetic": True})
    raw = json.loads(paths["manuscript"].read_text())
    raw["tables"][0]["results"] = fingerprint_file(other)
    dump(paths["manuscript"], raw)
    with pytest.raises(ValueError, match="table"):
        check_study_links(paths, study_id="fixture-study")


def test_global_cost_ledger_cannot_omit_or_change_per_run_events(linked_files, tmp_path):
    from carla_vision.reproduction.study_links import check_study_links
    from carla_vision.scenarios.costs import CostLedger

    paths, _, _ = linked_files
    other = tmp_path / "different-costs.jsonl"
    with CostLedger(other) as ledger:
        ledger.record(
            "event-1",
            kind="train",
            outcome="success",
            resources={"wall_seconds": 1.0},
            evidence_ref="synthetic",
        )
    paths["costs"] = other
    with pytest.raises(ValueError, match="cost event"):
        check_study_links(paths, study_id="fixture-study")


def test_empirical_reviewer_requires_cross_role_check_even_when_role_checks_pass(
    tmp_path,
    monkeypatch,
):
    from carla_vision.reproduction import reviewer

    # Isolate the orchestration boundary: real role validators are tested elsewhere.
    raw = reviewer.template()
    raw.update(source_commit="a" * 40, study_id="unrelated")
    for role in raw["artifacts"]:
        raw["artifacts"][role] = fingerprint_file(dump(tmp_path / f"{role}.json", {}))
    monkeypatch.setattr(reviewer, "_check_role", lambda *a, **kw: {"status": "passed"})
    monkeypatch.setattr(reviewer, "verify_cost_ledger", lambda *a, **kw: {"status": "passed"})
    result = reviewer.review_submission(dump(tmp_path / "submission.json", raw))
    assert "study_linkage" in result["errors"]
    assert result["empirical_submission_ready"] is False


def test_sharded_inventory_uses_manifest_not_repeated_shard_name(tmp_path):
    from test_training import sample_pair

    from carla_vision.dataset.shards import ShardedDatasetWriter
    from carla_vision.reproduction.study_links import freeze_split, verify_split

    roots = []
    for i, split in enumerate(("train", "test_seen")):
        with ShardedDatasetWriter(
            tmp_path,
            dataset_id=f"release-{i}",
            shard_samples=1,
            carla_endpoint={},
            carla_version="fixture",
            carla_map="fixture",
        ) as writer:
            for frame in (1, 2):
                writer.add_pair(
                    sample_pair(frame, frame, 20 * i + frame),
                    split=split,
                    scenario_id=f"s-{i}",
                    episode_id=f"e-{i}",
                )
        roots.append(writer.dataset_dir)
    path = tmp_path / "split.json"
    raw = freeze_split(roots, path)
    assert len(raw["samples"]) == 4
    assert len(raw["dataset_manifests"]) == 4
    assert len({s["dataset_id"] for s in raw["samples"]}) == 2
    assert len({(s["manifest_sha256"], s["sample_id"]) for s in raw["samples"]}) == 4
    assert raw["holdout_count"] == 2
    assert verify_split(path) == raw
    with pytest.raises(ValueError, match="duplicate"):
        freeze_split([roots[0], roots[0] / "shards/shard-000000"], tmp_path / "dup.json")


def test_unassigned_source_and_changed_source_never_get_a_frozen_inventory(tmp_path, releases):
    from test_training import sample_pair

    from carla_vision.dataset.writer import DatasetWriter
    from carla_vision.reproduction.study_links import freeze_split, verify_split

    with DatasetWriter(
        tmp_path,
        dataset_id="unassigned",
        carla_endpoint={},
        carla_version="fixture",
        carla_map="fixture",
    ) as writer:
        writer.add_pair(sample_pair(1, 1, 22), split="unassigned", scenario_id="s", episode_id="e")
    with pytest.raises(ValueError, match="unassigned"):
        freeze_split([writer.dataset_dir], tmp_path / "invalid.json")
    assert not (tmp_path / "invalid.json").exists()
    path = tmp_path / "valid.json"
    freeze_split(releases, path)
    with (releases[0] / "manifest.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="changed"):
        verify_split(path)


def test_publication_race_preserves_winning_file(releases, tmp_path, monkeypatch):
    import os

    from carla_vision.reproduction import study_links

    link = os.link
    output = tmp_path / "winner.json"

    def race(source, target):
        target.write_text("another writer's complete result")
        return link(source, target)

    monkeypatch.setattr(study_links.os, "link", race)
    with pytest.raises(FileExistsError):
        study_links.freeze_split(releases, output)
    assert output.read_text() == "another writer's complete result"
    assert not list(tmp_path.glob(".split-*"))


def test_frozen_inventory_rejects_bool_count_and_unknown_fields(releases, tmp_path):
    from carla_vision.reproduction.study_links import freeze_split, verify_split

    path = tmp_path / "split.json"
    raw = freeze_split(releases, path)
    for changed in (raw | {"holdout_count": True}, raw | {"acceptance_guaranteed": True}):
        dump(path, changed)
        with pytest.raises(ValueError):
            verify_split(path)


def test_freeze_cli_returns_fingerprint_not_readiness(releases, tmp_path, capsys):
    from carla_vision.reproduction.reviewer import main

    output = tmp_path / "split.json"
    args = ["freeze-split", "--output", str(output)]
    for root in releases:
        args.extend(["--dataset", str(root)])
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["sample_count"] == 6
    assert result["split"] == fingerprint_file(output)
    assert "empirical_submission_ready" not in result


@pytest.mark.parametrize(
    "field,value",
    [
        ("methods", ["one", "one"]),
        ("seeds", [True, 2]),
        ("seeds", []),
    ],
)
def test_linked_matrix_cannot_weaken_types_or_duplicate_cells(linked_files, field, value):
    from carla_vision.reproduction.study_links import check_study_links

    paths, matrix, _ = linked_files
    raw = json.loads(matrix.read_text())
    raw[field] = value
    dump(matrix, raw)
    raw = json.loads(paths["results"].read_text())
    raw["protocol"] = fingerprint_file(matrix)
    dump(paths["results"], raw)
    with pytest.raises(ValueError):
        check_study_links(paths, study_id="fixture-study")


def test_producer_metric_and_dataset_must_match_frozen_protocol(linked_files, tmp_path):
    from carla_vision.reproduction.study_links import check_study_links

    paths, _, _ = linked_files
    raw = json.loads(paths["results"].read_text())
    raw["producers"]["run-1"]["metric_field"] = "ap_best_condition_only"
    dump(paths["results"], raw)
    rebind_table_fixture(paths)
    with pytest.raises(ValueError, match="metric"):
        check_study_links(paths, study_id="fixture-study")
    raw["producers"]["run-1"]["metric_field"] = "coco.ap"
    other = dump(tmp_path / "other-manifest.json", {"synthetic": True})
    producer = dump(
        tmp_path / "other-producer.json",
        {
            "references": {"datasets": [{"manifest": fingerprint_file(other)}]},
        },
    )
    raw["producers"]["run-1"]["manifest"] = fingerprint_file(producer)
    dump(paths["results"], raw)
    rebind_table_fixture(paths)
    with pytest.raises(ValueError, match="outside"):
        check_study_links(paths, study_id="fixture-study")


def test_same_cost_event_cannot_be_charged_to_two_producers(linked_files):
    from carla_vision.reproduction.study_links import check_study_links

    paths, _, _ = linked_files
    raw = json.loads(paths["results"].read_text())
    raw["producers"]["run-2"] = raw["producers"]["run-1"]
    dump(paths["results"], raw)
    rebind_table_fixture(paths)
    with pytest.raises(ValueError, match="cost event"):
        check_study_links(paths, study_id="fixture-study")


def test_wrong_study_duplicate_claims_and_missing_roles_are_rejected(linked_files):
    from carla_vision.reproduction.study_links import check_study_links

    paths, _, _ = linked_files
    with pytest.raises(ValueError, match="different study"):
        check_study_links(paths, study_id="other-study")
    with pytest.raises(ValueError, match="incomplete"):
        check_study_links(
            {k: v for k, v in paths.items() if k != "costs"}, study_id="fixture-study"
        )
    raw = json.loads(paths["claim_registry"].read_text())
    raw["claims"] *= 2
    dump(paths["claim_registry"], raw)
    with pytest.raises(ValueError, match="duplicate"):
        check_study_links(paths, study_id="fixture-study")


def test_split_reference_through_parent_symlink_is_rejected(releases, tmp_path):
    from carla_vision.reproduction.study_links import freeze_split

    alias = tmp_path / "alias"
    alias.symlink_to(releases[0].parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        freeze_split([alias / releases[0].name], tmp_path / "bad.json")


def test_study_planner_consumes_frozen_inventory_without_promoting_test(releases, tmp_path):
    from dataclasses import asdict

    from test_research_methods import candidate

    from carla_vision.reproduction.study_links import freeze_split
    from carla_vision.scenarios.study import plan_study, validate_protocol

    split_path = tmp_path / "split.json"
    split = freeze_split(releases, split_path)
    power = dump(
        tmp_path / "power.json",
        {
            "status": "registered",
            "basis": "registered_compute_limit",
            "rationale": "fixture only",
        },
    )
    probe = dump(tmp_path / "probe.json", {"synthetic": True})
    learner = dump(tmp_path / "learner.json", {"synthetic": True})
    raw = {
        "schema_version": "1.0",
        "study_id": "fixture",
        "purpose": "development",
        "hypotheses": ["fixture software contract"],
        "primary_metric": "coco.ap",
        "minimum_effect": 0.01,
        "methods": ["random", "capability_matched", "error_source_aware"],
        "ablations": [],
        "training_seeds": [1],
        "generation_seeds": [2],
        "selection": {"budget": 10.0, "max_actions": 1},
        "split_manifest": fingerprint_file(split_path),
        "regimes": ["clean", "capture_fault", "supervision_fault", "mixed"],
        "stopping_rule": "fixture budget",
        "power_analysis": fingerprint_file(power),
        "learner_configs": [fingerprint_file(learner)],
        "cost_scope": "all_attempts_including_failures_reviews_and_training",
    }
    assert validate_protocol(raw)["split_manifest"] == fingerprint_file(split_path)
    protocol = dump(tmp_path / "protocol.json", raw)
    train_group = next(s["group_id"] for s in split["samples"] if s["partition"] == "train")
    item = asdict(candidate(0, group_id=train_group, partition="train"))
    pool = {
        "schema_version": "1.0",
        "candidates": [item],
        "evidence": {
            "probe": fingerprint_file(probe),
        },
    }
    pool_path = dump(tmp_path / "pool.json", pool)
    planned = plan_study(protocol, pool_path, runs_root=tmp_path / "runs", run_id="fixture")
    assert planned["empirical_results_available"] is False
    item["group_id"] = next(
        s["group_id"] for s in split["samples"] if s["partition"] == "test_seen"
    )
    dump(pool_path, pool)
    with pytest.raises(ValueError, match="partition"):
        plan_study(protocol, pool_path, runs_root=tmp_path / "runs", run_id="bad")
    assert not (tmp_path / "runs/bad").exists()


def test_frozen_inventory_closure_survives_original_workspace_removal(releases, tmp_path):
    import shutil

    from carla_vision.reproduction.closure import build_closure, verify_closure
    from carla_vision.reproduction.study_links import freeze_split

    path = tmp_path / "split.json"
    freeze_split(releases, path)
    bundle = tmp_path / "bundle"
    result = build_closure([path], bundle)
    path.unlink()
    for root in releases:
        shutil.rmtree(root)
    checked = verify_closure(bundle, expected_sha256=result["index"]["sha256"])
    assert checked["status"] == "passed"
    assert checked["scope"] == "portable_byte_and_reference_integrity_not_empirical_truth"


def test_additional_study_costs_cannot_disappear_from_the_method_budgets(linked_files, tmp_path):
    from carla_vision.reproduction.study_links import check_study_links
    from carla_vision.scenarios.costs import CostLedger

    paths, _, _ = linked_files
    global_path = tmp_path / "with-probe.jsonl"
    with CostLedger(global_path) as ledger:
        ledger.record(
            "event-1",
            kind="train",
            outcome="success",
            resources={"wall_seconds": 2.0},
            evidence_ref="synthetic",
        )
        ledger.record(
            "unallocated-probe",
            kind="probe",
            outcome="failed",
            resources={"wall_seconds": 100.0},
            evidence_ref="synthetic",
        )
    paths["costs"] = global_path
    with pytest.raises(ValueError, match="unallocated"):
        check_study_links(paths, study_id="fixture-study")
