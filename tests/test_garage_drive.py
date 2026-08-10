from __future__ import annotations

import time
from pathlib import Path

import pytest

from carla_vision.operator.drive_contracts import DriveInput
from carla_vision.operator.garage_drive import GarageDriveSession, GarageDriveStartConfig
from carla_vision.operator.garage_server import GarageOperatorDriveManager

CARLA_HOST = "127.0.0.1"
CARLA_PORT = 65534


def base_start(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": "garage-drive-test",
        "host": CARLA_HOST,
        "port": CARLA_PORT,
        "vehicle_blueprint": "vehicle.tesla.model3",
        "color": "",
        "seed": 20260810,
        "weather_preset": "clear-day",
        "prop_preset": "none",
        "detector_enabled": False,
        "detector": "rtdetr",
        "weights": "",
        "device": "cpu",
        "image_size": 640,
        "confidence": 0.35,
        "resolution": "640x384",
        "camera_fps": 10.0,
        "camera_fov": 90.0,
        "record_video": False,
        "spectator_follow": False,
        "control_mode": "manual",
        "behavior": "normal",
        "acknowledge_autonomy": False,
        "traffic_vehicles": 0,
        "walkers": 0,
        "tm_port": 8000,
        "target_speed_kmh": 35.0,
        "policy_checkpoint": "",
        "policy_device": "cpu",
        "voxel_readiness_report": "",
    }
    payload.update(overrides)
    return payload


def config(workspace: Path, **overrides: object) -> GarageDriveStartConfig:
    return GarageDriveStartConfig.from_mapping(
        base_start(**overrides),
        workspace=workspace,
        expected_host=CARLA_HOST,
        expected_port=CARLA_PORT,
    )


def control(**overrides: object) -> DriveInput:
    payload: dict[str, object] = {
        "session_id": "garage-drive-test",
        "sequence": 1,
        "throttle": 0.4,
        "steer": -0.2,
        "brake": 0.0,
        "hand_brake": False,
        "reverse": False,
    }
    payload.update(overrides)
    return DriveInput.from_mapping(payload)


def test_manual_mode_keeps_existing_browser_contract_without_pythonapi(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    session = GarageDriveSession(cfg, workspace=tmp_path)
    drive_input = control()
    with session._lock:
        session._last_input = (drive_input, 100.0)

    command, source, age = session._command(100.1, camera_stale=False)

    assert source == "browser_manual"
    assert age == pytest.approx(0.1)
    assert command.throttle == pytest.approx(0.4)
    assert command.steer == pytest.approx(-0.2)
    assert session._garage_context is None


def test_autonomous_modes_require_explicit_acknowledgement(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit operator acknowledgement"):
        config(tmp_path, control_mode="behavior", acknowledge_autonomy=False)

    cfg = config(tmp_path, control_mode="behavior", acknowledge_autonomy=True)
    assert cfg.autonomous is True
    assert cfg.model_output_actuated is False


def test_model_modes_require_workspace_checkpoint_and_disable_detector_overlay(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "detector.pt").write_bytes(b"detector")
    (models / "imitation.pt").write_bytes(b"policy")

    cfg = config(
        tmp_path,
        control_mode="imitation",
        acknowledge_autonomy=True,
        policy_checkpoint="models/imitation.pt",
        detector_enabled=True,
        weights="models/detector.pt",
    )

    assert cfg.policy_checkpoint == (models / "imitation.pt").resolve()
    assert cfg.base.detector_enabled is False
    assert cfg.base.weights is None
    assert cfg.model_output_actuated is True
    assert cfg.manifest_config()["control_owner"] == "imitation"


def test_checkpoint_and_readiness_paths_cannot_escape_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.pt"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError):
        config(
            tmp_path,
            control_mode="imitation",
            acknowledge_autonomy=True,
            policy_checkpoint=str(outside),
        )

    checkpoint = tmp_path / "voxel.pt"
    checkpoint.write_bytes(b"voxel")
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="requires control_mode=voxel"):
        config(
            tmp_path,
            control_mode="imitation",
            acknowledge_autonomy=True,
            policy_checkpoint="voxel.pt",
            voxel_readiness_report="report.json",
        )


def test_emergency_and_camera_staleness_fail_closed_before_policy_initialization(tmp_path: Path) -> None:
    cfg = config(tmp_path, control_mode="behavior", acknowledge_autonomy=True)
    session = GarageDriveSession(cfg, workspace=tmp_path)

    with session._lock:
        session._emergency = True
    command, source, _ = session._command(time.monotonic(), camera_stale=False)
    assert source == "emergency_stop"
    assert command.throttle == 0.0
    assert command.brake == 1.0
    assert session._policy is None

    with session._lock:
        session._emergency = False
    command, source, _ = session._command(time.monotonic(), camera_stale=True)
    assert source == "camera_deadman"
    assert command.throttle == 0.0
    assert command.brake == 1.0
    assert session._policy is None


def test_browser_control_is_rejected_when_autonomy_owns_control(tmp_path: Path) -> None:
    cfg = config(tmp_path, control_mode="behavior", acknowledge_autonomy=True)
    session = GarageDriveSession(cfg, workspace=tmp_path)

    with pytest.raises(RuntimeError, match="browser manual control is disabled"):
        session.submit_control(control())


def test_catalog_finds_nested_policy_checkpoints_without_exposing_environment_dirs(
    tmp_path: Path,
) -> None:
    (tmp_path / "models" / "imitation").mkdir(parents=True)
    (tmp_path / "models" / "imitation" / "best.pt").write_bytes(b"checkpoint")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "ignored.pt").write_bytes(b"ignored")

    manager = GarageOperatorDriveManager(
        workspace=tmp_path,
        carla_host=CARLA_HOST,
        carla_port=CARLA_PORT,
    )
    catalog = manager.catalog()

    assert "models/imitation/best.pt" in catalog["policy_checkpoints"]
    assert ".venv/ignored.pt" not in catalog["policy_checkpoints"]
    modes = {row["id"]: row for row in catalog["control_modes"]}
    assert modes["manual"]["available"] is True
