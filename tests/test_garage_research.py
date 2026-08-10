from __future__ import annotations

from pathlib import Path

import pytest

from carla_vision.operator.garage_research import (
    GarageResearchRequest,
    build_garage_research_plan,
)
from carla_vision.operator.garage_server import _validate_research_runtime

HOST = "127.0.0.1"
PORT = 2000


def request(kind: str, **parameters: object) -> GarageResearchRequest:
    return GarageResearchRequest.from_mapping(
        {
            "schema_version": "1.0",
            "kind": kind,
            "parameters": parameters,
        }
    )


def plan(workspace: Path, kind: str, **parameters: object):
    return build_garage_research_plan(
        request(kind, **parameters),
        workspace=workspace,
        carla_host=HOST,
        carla_port=PORT,
    )


def test_request_rejects_unknown_job_kind_and_fields() -> None:
    with pytest.raises(ValueError, match="unsupported Garage research kind"):
        request("shell", command="rm -rf /")
    with pytest.raises(ValueError, match="requires schema_version"):
        GarageResearchRequest.from_mapping(
            {"schema_version": "1.0", "kind": "voxel_train", "parameters": {}, "command": "x"}
        )


def test_imitation_training_is_fixed_module_and_workspace_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "datasets" / "teacher-a"
    dataset.mkdir(parents=True)
    result = plan(
        tmp_path,
        "imitation_train",
        dataset="datasets/teacher-a",
        run_id="imitation-a",
        device="cpu",
        epochs=2,
        dry_run=False,
    )

    assert result.kind == "imitation_train"
    assert result.motion_authorized is False
    assert result.destructive is False
    assert "carla_vision.imitation.runner" in result.command[2]
    assert "--dataset" in result.command
    assert str(dataset.resolve()) in result.command
    assert result.expected_output == (tmp_path / "models" / "imitation" / "imitation-a")
    assert all("rm -rf" not in token for token in result.command)


def test_teacher_capture_requires_ack_and_marks_destructive_motion(tmp_path: Path) -> None:
    scenario = tmp_path / "runs" / "scenario-plan" / "scenario-plan.json"
    scenario.parent.mkdir(parents=True)
    scenario.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="acknowledgement"):
        plan(
            tmp_path,
            "teacher_capture",
            scenario_plan="runs/scenario-plan/scenario-plan.json",
            dataset_id="teacher-a",
            behavior="normal",
            max_episodes=2,
            acknowledge=False,
        )

    result = plan(
        tmp_path,
        "teacher_capture",
        scenario_plan="runs/scenario-plan/scenario-plan.json",
        dataset_id="teacher-a",
        behavior="normal",
        max_episodes=2,
        acknowledge=True,
    )
    assert result.motion_authorized is True
    assert result.destructive is True
    assert "carla_vision.native.behavior_teacher" in result.command[2]
    assert "--acknowledge-exclusive-tick-owner" in result.command


def test_voxel_live_jobs_use_fixed_factories_and_current_garage_role(tmp_path: Path) -> None:
    checkpoint = tmp_path / "models" / "voxel" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    shadow = plan(
        tmp_path,
        "voxel_shadow",
        run_id="shadow-a",
        checkpoint="models/voxel/best.pt",
        device="cpu",
        frames=25,
        dry_run=False,
    )
    assert shadow.motion_authorized is False
    assert "carla_vision.voxel.shadow" in shadow.command[2]
    assert "research_drive_ego" in shadow.command
    assert "carla_vision.voxel.model_examples.temporal_flow:create_predictor" in shadow.command

    capture = plan(
        tmp_path,
        "voxel_capture",
        run_id="rgb-a",
        frames=20,
        dry_run=False,
        mode="rgb-only",
        checkpoint="models/voxel/best.pt",
        device="cpu",
    )
    assert "carla_vision.voxel.capture" in capture.command[2]
    assert "--mode" in capture.command
    assert "rgb-only" in capture.command
    assert "--predictor-checkpoint" in capture.command


def test_closed_loop_observer_is_read_only_plan(tmp_path: Path) -> None:
    result = plan(
        tmp_path,
        "closed_loop_evaluate",
        run_id="eval-a",
        driver_label="voxel",
        duration=15,
        dry_run=False,
    )

    assert result.motion_authorized is False
    assert result.destructive is False
    assert "carla_vision.closed_loop_cli" in result.command[2]
    assert "research_drive_ego" in result.command
    assert result.expected_output == (tmp_path / "runs" / "eval-a")


def test_benchmark_cannot_escape_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-run"
    outside.mkdir(exist_ok=True)
    with pytest.raises(ValueError):
        plan(tmp_path, "voxel_benchmark", run=str(outside), run_id="bench")


def test_output_names_are_non_overwriting(tmp_path: Path) -> None:
    dataset = tmp_path / "datasets" / "teacher-a"
    dataset.mkdir(parents=True)
    existing = tmp_path / "models" / "voxel" / "occupied"
    existing.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="output already exists"):
        plan(
            tmp_path,
            "voxel_train",
            dataset="datasets/teacher-a",
            run_id="occupied",
            device="cpu",
            epochs=1,
            dry_run=False,
        )


def test_runtime_guards_keep_destructive_and_live_workflows_in_their_safe_contexts() -> None:
    teacher = request(
        "teacher_capture",
        scenario_plan="plan.json",
        dataset_id="teacher-a",
        behavior="normal",
        max_episodes=1,
        acknowledge=True,
    )
    with pytest.raises(RuntimeError, match="end the active Garage drive"):
        _validate_research_runtime(teacher, {"status": "running"})
    _validate_research_runtime(teacher, {"status": "idle"})

    live = request(
        "closed_loop_evaluate",
        run_id="eval",
        driver_label="manual",
        duration=10,
        dry_run=False,
    )
    with pytest.raises(RuntimeError, match="requires a running Garage ego"):
        _validate_research_runtime(live, {"status": "idle"})
    _validate_research_runtime(live, {"status": "running"})

    dry = request(
        "closed_loop_evaluate",
        run_id="eval",
        driver_label="manual",
        duration=10,
        dry_run=True,
    )
    _validate_research_runtime(dry, {"status": "idle"})
