"""Research actions use the existing allow-listed Operator job owner."""

from __future__ import annotations

import sys
import time

import pytest

from carla_vision.operator.commands import build_command_plan
from carla_vision.operator.contracts import OperatorJobRequest
from carla_vision.operator.jobs import JobManager


@pytest.fixture(autouse=True)
def importable_child_source(monkeypatch):
    # The source-only review environment is not an editable install. Real CI
    # installs the project; explicitly expose the same source to child Python.
    import os
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[1])
    monkeypatch.setenv("PYTHONPATH", root + os.pathsep + os.environ.get("PYTHONPATH", ""))


def request(kind, parameters):
    return OperatorJobRequest.from_mapping(
        {"schema_version": "1.0", "kind": kind, "parameters": parameters}
    )


@pytest.mark.parametrize(
    "kind,params",
    [
        ("acquisition_plan", {"protocol": "input.json", "candidates": "input.json", "run_id": "r"}),
        (
            "diagnosis_benchmark",
            {"observations": "input.json", "reference": "input.json", "run_id": "r"},
        ),
        ("coreset_select", {"protocol": "input.json", "run_id": "r"}),
        ("reviewer_check", {"submission": "input.json"}),
    ],
)
def test_research_command_plans_are_allowlisted_and_scoped(tmp_path, kind, params):
    (tmp_path / "input.json").write_text("{}")
    plan = build_command_plan(request(kind, params), workspace=tmp_path, executable=sys.executable)
    assert not plan.motion_authorized and not plan.destructive
    assert plan.command[1] == "-c" and "import main" in plan.command[2]
    bad = {**params, "shell": "touch somefile"}
    with pytest.raises(ValueError):
        build_command_plan(request(kind, bad), workspace=tmp_path, executable=sys.executable)
    field = next(k for k in params if k != "run_id")
    with pytest.raises((ValueError, RuntimeError)):
        build_command_plan(
            request(kind, {**params, field: "../outside.json"}),
            workspace=tmp_path,
            executable=sys.executable,
        )


def test_derivation_action_cannot_disguise_live_recapture(tmp_path):
    (tmp_path / "parent").mkdir()
    p = {
        "parent": "parent",
        "dataset_id": "new",
        "action": "FREEZE",
        "sample_ids": ["a"],
        "review": None,
    }
    plan = build_command_plan(
        request("dataset_derive", p), workspace=tmp_path, executable=sys.executable
    )
    assert "--sample-id" in plan.command
    p["action"] = "RECAPTURE"
    with pytest.raises(ValueError):
        build_command_plan(
            request("dataset_derive", p), workspace=tmp_path, executable=sys.executable
        )


def test_existing_job_manager_reports_incomplete_review_not_success(tmp_path):
    import json

    from carla_vision.reproduction.reviewer import template

    p = tmp_path / "submission.json"
    p.write_text(json.dumps(template()))
    req = request("reviewer_check", {"submission": "submission.json"})
    plan = build_command_plan(req, workspace=tmp_path, executable=sys.executable)
    manager = JobManager(workspace=tmp_path)
    try:
        job = manager.submit(req, plan)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            result = manager.get(job["job_id"])
            if result["status"] in {"failed", "success", "stopped"}:
                break
            time.sleep(0.03)
        assert result["status"] == "failed" and result["returncode"] == 1
        assert '"acceptance_guaranteed": false' in manager.log_tail(job["job_id"])
        assert "ModuleNotFoundError" not in manager.log_tail(job["job_id"], stream="stderr")
    finally:
        manager.shutdown()


def test_real_job_manager_cycle_records_actual_plan_and_no_duplicate_submission(tmp_path):
    import json

    from test_scenarios import split_plan, suite

    from carla_vision.operator.research_cycle import execute_cycle_job, inspect_cycle_receipt
    from carla_vision.scenarios.cycle import CycleJournal, verify_cycle

    for name, value in (
        ("suite.json", suite()),
        ("split.json", split_plan()),
        ("protocol.json", {"synthetic": True}),
    ):
        (tmp_path / name).write_text(json.dumps(value))
    manager = JobManager(workspace=tmp_path)
    req = request(
        "scenario_plan", {"suite": "suite.json", "split_plan": "split.json", "run_id": "cycle-plan"}
    )
    receipt = tmp_path / "receipt.json"
    try:
        with CycleJournal(
            tmp_path / "cycle.jsonl",
            cycle_id="one",
            partition="acquisition_pool",
            protocol_ref=tmp_path / "protocol.json",
        ) as cycle:
            result = execute_cycle_job(
                cycle, manager, req, stage="planned", receipt_path=receipt, timeout_seconds=15
            )
            assert result["status"] == "recorded" and cycle.stage == "planned"
            with pytest.raises(ValueError, match="next"):
                execute_cycle_job(
                    cycle, manager, req, stage="planned", receipt_path=tmp_path / "second.json"
                )
        assert verify_cycle(tmp_path / "cycle.jsonl")["stage"] == "planned"
        inspection = inspect_cycle_receipt(receipt, manager)
        assert inspection["resubmitted"] is False and inspection["job"]["status"] == "success"
    finally:
        manager.shutdown()


def test_cycle_receipt_ambiguity_never_retries(tmp_path):
    from carla_vision.operator.research_cycle import inspect_cycle_receipt

    path = tmp_path / "receipt.json"
    path.write_text('{"status":"submitting","job_id":null}')
    manager = JobManager(workspace=tmp_path)
    try:
        result = inspect_cycle_receipt(path, manager)
        assert result["resubmitted"] is False and "ambiguity" in result
    finally:
        manager.shutdown()
