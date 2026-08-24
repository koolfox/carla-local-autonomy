from __future__ import annotations

import time
from pathlib import Path

import pytest

from carla_vision.operator import garage_drive
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


def config(
    workspace: Path,
    *,
    world_worker_configured: bool = False,
    experimental_enabled: bool = True,
    **overrides: object,
) -> GarageDriveStartConfig:
    return GarageDriveStartConfig.from_mapping(
        base_start(**overrides),
        workspace=workspace,
        expected_host=CARLA_HOST,
        expected_port=CARLA_PORT,
        world_worker_configured=world_worker_configured,
        experimental_enabled=experimental_enabled,
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


def test_manual_output_registration_forwards_latest_worker_jpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = GarageDriveSession(config(tmp_path), workspace=tmp_path)
    captured: dict[str, object] = {}

    def capture_registration(_session: object, _tracker: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(garage_drive.DriveSession, "_register_outputs", capture_registration)
    paths = {
        name: tmp_path / name
        for name in (
            "controls.jsonl",
            "detections.jsonl",
            "events.jsonl",
            "raw.mp4",
            "overlay.mp4",
        )
    }
    raw_frame = b"already-encoded-worker-jpeg"

    session._register_outputs(
        object(),
        controls_path=paths["controls.jsonl"],
        detections_path=paths["detections.jsonl"],
        events_path=paths["events.jsonl"],
        raw_video_path=paths["raw.mp4"],
        overlay_video_path=paths["overlay.mp4"],
        latest_raw=None,
        latest_raw_jpeg=raw_frame,
        latest_overlay=None,
    )

    assert captured["latest_raw_jpeg"] is raw_frame


def test_experimental_control_modes_are_disabled_by_default(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="--enable-experimental"):
        GarageDriveStartConfig.from_mapping(
            base_start(control_mode="behavior", acknowledge_autonomy=True),
            workspace=tmp_path,
            expected_host=CARLA_HOST,
            expected_port=CARLA_PORT,
        )


def test_autonomous_modes_require_explicit_acknowledgement(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit operator acknowledgement"):
        config(tmp_path, control_mode="behavior", acknowledge_autonomy=False)

    cfg = config(tmp_path, control_mode="behavior", acknowledge_autonomy=True)
    assert cfg.autonomous is True
    assert cfg.model_output_actuated is False


def test_worker_runtime_mode_and_garage_mode_remain_distinct(tmp_path: Path) -> None:
    cfg = config(
        tmp_path,
        world_worker_configured=True,
        initial_control_mode="autopilot",
        traffic_count=7,
        walker_count=8,
        traffic_vehicles=3,
        walkers=4,
    )
    session = GarageDriveSession(cfg, workspace=tmp_path, world_worker=object())  # type: ignore[arg-type]

    snapshot = session.snapshot()
    manifest = cfg.manifest_config()

    assert snapshot["control_mode"] == "autopilot"
    assert snapshot["garage_mode"] == "manual"
    assert manifest["control_mode"] == "world_worker_autopilot"
    assert manifest["garage_mode"] == "manual"
    assert manifest["traffic_count"] == 7
    assert manifest["walker_count"] == 8
    assert manifest["garage_traffic_vehicles"] == 3
    assert manifest["garage_walkers"] == 4


def test_garage_autonomy_rejects_worker_autopilot_ownership(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="World Worker autopilot owns control"):
        config(
            tmp_path,
            world_worker_configured=True,
            control_mode="behavior",
            acknowledge_autonomy=True,
            initial_control_mode="autopilot",
        )

    cfg = config(
        tmp_path,
        world_worker_configured=True,
        control_mode="behavior",
        acknowledge_autonomy=True,
    )
    session = GarageDriveSession(cfg, workspace=tmp_path, world_worker=object())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="cannot take over"):
        session.request_mode("autopilot")


def test_model_modes_require_workspace_checkpoint_and_disable_detector_overlay(
    tmp_path: Path,
) -> None:
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


def test_emergency_and_camera_staleness_fail_closed_before_policy_initialization(
    tmp_path: Path,
) -> None:
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
        experimental_enabled=True,
    )
    catalog = manager.catalog()

    assert "models/imitation/best.pt" in catalog["policy_checkpoints"]
    assert ".venv/ignored.pt" not in catalog["policy_checkpoints"]
    modes = {row["id"]: row for row in catalog["control_modes"]}
    assert modes["manual"]["available"] is True


def test_catalog_namespaces_garage_capabilities_without_overwriting_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(garage_drive, "_module_available", lambda _name: True)
    manager = GarageOperatorDriveManager(
        workspace=tmp_path,
        carla_host=CARLA_HOST,
        carla_port=CARLA_PORT,
        experimental_enabled=True,
    )

    capabilities = manager.catalog()["capabilities"]

    assert capabilities["autopilot"] is False
    assert capabilities["traffic_manager"] is False
    assert capabilities["walkers"] is False
    assert "pythonapi" not in capabilities
    assert "behavior_agent" not in capabilities
    assert capabilities["garage_behavior_drive"] is True
    assert capabilities["garage_traffic_population"] is True
    assert capabilities["garage_walker_population"] is True
    assert capabilities["garage_imitation_drive"] is True
    assert capabilities["garage_voxel_drive"] is True


def test_production_catalog_and_start_hide_experimental_autonomy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(garage_drive, "_module_available", lambda _name: True)
    manager = GarageOperatorDriveManager(
        workspace=tmp_path,
        carla_host=CARLA_HOST,
        carla_port=CARLA_PORT,
    )

    catalog = manager.catalog()

    assert catalog["control_modes"] == [{"id": "manual", "label": "Manual", "available": True}]
    assert catalog["capabilities"]["garage_experimental"] is False
    assert "garage_behavior_drive" not in catalog["capabilities"]
    assert "garage_imitation_drive" not in catalog["capabilities"]
    assert "garage_voxel_drive" not in catalog["capabilities"]
    assert "policy_checkpoints" not in catalog
    with pytest.raises(PermissionError, match="--enable-experimental"):
        manager.start(base_start(control_mode="behavior", acknowledge_autonomy=True))


def test_manager_uses_ready_worker_and_falls_back_only_for_exact_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[object] = []

    class FakeSession:
        def __init__(self, cfg: object, *, workspace: Path, world_worker: object = None) -> None:
            self.config = cfg
            self.workspace = workspace
            self.world_worker = world_worker
            self.session_id = "garage-drive-test"
            created.append(self)

        def start(self) -> None:
            pass

        def snapshot(self) -> dict[str, object]:
            return {"status": "running", "session_id": self.session_id}

    class FakeWorker:
        def __init__(self, *, ready: bool) -> None:
            self.ready = ready

        def health(self) -> dict[str, object]:
            if not self.ready:
                raise RuntimeError("offline")
            return {"ready": True}

        def catalog(self) -> dict[str, object]:
            return {"catalog": {"capabilities": {}}}

    monkeypatch.setattr(garage_drive, "GarageDriveSession", FakeSession)
    monkeypatch.setattr(garage_drive, "_module_available", lambda _name: False)

    ready_worker = FakeWorker(ready=True)
    ready_manager = GarageOperatorDriveManager(
        workspace=tmp_path,
        carla_host=CARLA_HOST,
        carla_port=CARLA_PORT,
        world_worker=ready_worker,  # type: ignore[arg-type]
    )
    ready_manager.start(
        base_start(
            map_name="Town10HD_Opt",
            traffic_count=5,
            initial_control_mode="autopilot",
        )
    )
    assert created[-1].world_worker is ready_worker  # type: ignore[attr-defined]
    assert created[-1].config.base.traffic_count == 5  # type: ignore[attr-defined]
    assert created[-1].config.base.initial_control_mode == "autopilot"  # type: ignore[attr-defined]

    unavailable_worker = FakeWorker(ready=False)
    unavailable_manager = GarageOperatorDriveManager(
        workspace=tmp_path,
        carla_host=CARLA_HOST,
        carla_port=CARLA_PORT,
        world_worker=unavailable_worker,  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="configured World Worker is required"):
        unavailable_manager.start(base_start(traffic_count=1))
    unavailable_manager.start(base_start())
    assert created[-1].world_worker is None  # type: ignore[attr-defined]
