from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_garage_drive import CARLA_HOST, CARLA_PORT, base_start
from test_garage_preview import _scene, preview_payload

from carla_vision.operator import garage_drive
from carla_vision.operator.configuration import build_garage_preview_request, session_defaults
from carla_vision.operator.drive import DriveSession
from carla_vision.operator.drive_contracts import EXPERIMENT_PRESETS
from carla_vision.operator.garage_drive import GarageDriveSession, GarageDriveStartConfig
from carla_vision.operator.garage_preview import (
    GaragePreviewConfig,
    GaragePreviewManager,
    GaragePreviewSession,
)
from carla_vision.operator.garage_server import GarageOperatorDriveManager
from carla_vision.operator.world_worker_client import WorldWorkerScene


class HandoffWorker:
    def __init__(self) -> None:
        self.prepared: list[dict[str, Any]] = []
        self.started: list[WorldWorkerScene] = []
        self.stopped: list[WorldWorkerScene] = []
        self.modes: list[tuple[WorldWorkerScene, str]] = []
        self.heartbeats: list[WorldWorkerScene] = []

    def prepare_scene(self, payload: dict[str, Any]) -> WorldWorkerScene:
        self.prepared.append(dict(payload))
        return replace(
            ready_scene(),
            scene_id="fresh-scene",
            ego_actor_id=902,
            control_mode=payload["initial_control_mode"],
            route_mode=payload["route_mode"],
            traffic_count=payload["traffic_count"],
            walker_count=payload["walker_count"],
        )

    def heartbeat(self, scene: WorldWorkerScene) -> WorldWorkerScene:
        self.heartbeats.append(scene)
        return scene

    def mode(self, scene: WorldWorkerScene, control_mode: str) -> WorldWorkerScene:
        assert scene.status == "prepared", "handoff must not activate the ego before Drive is ready"
        self.modes.append((scene, control_mode))
        return replace(scene, control_mode=control_mode)

    def start_scene(self, scene: WorldWorkerScene) -> WorldWorkerScene:
        self.started.append(scene)
        return replace(scene, status="running")

    def stop_scene(self, scene: WorldWorkerScene) -> WorldWorkerScene:
        self.stopped.append(scene)
        return replace(scene, status="stopped", cleanup_guard_passed=True)


class ClosingStream:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def ready_scene(**overrides: Any) -> WorldWorkerScene:
    values = {
        "traffic_count": 15,
        "walker_count": 10,
        "traffic_count_requested": 15,
        "walker_count_requested": 10,
        "prop_actor_ids": (31, 32),
        "pedestrian_crossing_factor": 0.2,
        "speed_difference_percent": 12.0,
        "following_distance_metres": 2.0,
        "capabilities": {
            "prepared_scene_handoff": True,
            "compressed_camera_relay": True,
        },
    }
    values.update(overrides)
    return replace(_scene(), **values)


def drive_payload(**overrides: Any) -> dict[str, Any]:
    return base_start(**{**preview_payload(), **overrides})


def drive_config(workspace: Path, **overrides: Any) -> GarageDriveStartConfig:
    return GarageDriveStartConfig.from_mapping(
        drive_payload(**overrides),
        workspace=workspace,
        expected_host=CARLA_HOST,
        expected_port=CARLA_PORT,
        world_worker_configured=True,
    )


def ready_preview(
    worker: HandoffWorker,
    *,
    scene: WorldWorkerScene | None = None,
    **config_overrides: Any,
) -> tuple[GaragePreviewManager, GaragePreviewSession, ClosingStream]:
    preview = GaragePreviewSession(
        GaragePreviewConfig.from_mapping(preview_payload(**config_overrides)),
        carla_host=CARLA_HOST,
        carla_port=CARLA_PORT,
        world_worker=worker,  # type: ignore[arg-type]
    )
    prepared = ready_scene() if scene is None else scene
    stream = ClosingStream()
    preview._scene = prepared
    preview._status = "running"
    preview._episode_id = prepared.episode_id
    preview._vehicle_id = prepared.ego_actor_id
    preview._camera_id = 55
    preview._worker_camera = True
    preview._stream = stream  # type: ignore[assignment]
    preview._frame_sequence = 1
    preview._jpeg = b"ready-frame"
    manager = GaragePreviewManager(
        carla_host=CARLA_HOST,
        carla_port=CARLA_PORT,
        world_worker=worker,  # type: ignore[arg-type]
        drive_state=lambda: {"status": "idle"},
        world_mode_lock=threading.RLock(),
    )
    manager._session = preview
    return manager, preview, stream


@pytest.fixture
def no_drive_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DriveSession, "_start_worker_heartbeat", lambda self: None)
    monkeypatch.setattr(garage_drive, "_module_available", lambda _name: False)


def test_matching_preview_transfers_lease_and_closes_only_local_transports(tmp_path: Path) -> None:
    worker = HandoffWorker()
    manager, preview, stream = ready_preview(worker)
    original = preview._scene

    transferred = manager.take_for_drive(drive_config(tmp_path))

    assert transferred is not None
    assert transferred.scene_id == original.scene_id
    assert transferred.lease_token == original.lease_token
    assert transferred.ego_actor_id == original.ego_actor_id
    assert transferred.episode_id == original.episode_id
    assert stream.closed
    assert preview._heartbeat_stop.is_set()
    assert preview._frame_stop.is_set()
    assert manager.state()["active"] is False
    assert worker.prepared == worker.stopped == []
    preview.close(reason="stale_preview_close")
    manager.shutdown()
    assert worker.stopped == []  # The old owner cannot later destroy the transferred lease.


@pytest.mark.parametrize(
    "overrides",
    [
        {"map_name": "Town05"},
        {"vehicle_blueprint": "vehicle.audi.tt"},
        {"color": "0,0,255"},
        {"seed": 43},
        {"weather_preset": "heavy-rain"},
        {"traffic_count": 16},
        {"walker_count": 11},
        {"prop_preset": "none"},
        {"route_mode": "random_destination"},
        {"pedestrian_crossing_factor": 0.8},
        {"speed_difference_percent": -10.0},
        {"following_distance_metres": 5.0},
    ],
)
def test_scene_affecting_config_mismatch_is_not_reused(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    worker = HandoffWorker()
    manager, _preview, _stream = ready_preview(worker)
    try:
        assert manager.take_for_drive(drive_config(tmp_path, **overrides)) is None
    finally:
        manager.shutdown()


@pytest.mark.parametrize("preset", sorted(EXPERIMENT_PRESETS - {"free_drive"}))
def test_experiment_presets_require_fresh_scene(tmp_path: Path, preset: str) -> None:
    worker = HandoffWorker()
    manager, _preview, _stream = ready_preview(worker)
    try:
        assert manager.take_for_drive(drive_config(tmp_path, experiment_preset=preset)) is None
    finally:
        manager.shutdown()


@pytest.mark.parametrize(
    "scene",
    [
        ready_scene(capabilities={}),
        ready_scene(capabilities={"prepared_scene_handoff": False}),
        ready_scene(status="running"),
        ready_scene(ego_actor_id=None),
        ready_scene(episode_id=None),
    ],
)
def test_unsupported_or_nonprepared_preview_is_not_reused(
    tmp_path: Path, scene: WorldWorkerScene
) -> None:
    worker = HandoffWorker()
    manager, _preview, _stream = ready_preview(worker, scene=scene)
    try:
        assert manager.take_for_drive(drive_config(tmp_path)) is None
    finally:
        manager.shutdown()


def test_drive_camera_settings_and_initial_autopilot_do_not_require_new_population(
    tmp_path: Path,
) -> None:
    worker = HandoffWorker()
    manager, _preview, _stream = ready_preview(worker, profile="detail", fov=65.0)
    transferred = manager.take_for_drive(
        drive_config(
            tmp_path,
            resolution="1280x720",
            camera_fps=30.0,
            camera_fov=100.0,
            initial_control_mode="autopilot",
        )
    )
    assert transferred is not None
    assert transferred.ego_actor_id == 20
    assert worker.prepared == worker.stopped == []


@pytest.mark.parametrize("session_type", [DriveSession, GarageDriveSession])
def test_drive_adopts_prepared_scene_and_sets_mode_before_start_without_prepare_or_stop(
    tmp_path: Path, no_drive_threads: None, session_type: type[DriveSession]
) -> None:
    del no_drive_threads
    worker = HandoffWorker()
    config = drive_config(tmp_path, initial_control_mode="autopilot")
    original = ready_scene()
    session = session_type(
        config.base if session_type is DriveSession else config,
        workspace=tmp_path,
        world_worker=worker,  # type: ignore[arg-type]
        prepared_scene=original,
    )

    prepared = session._begin_worker_scene()
    assert prepared.scene_id == original.scene_id
    assert prepared.ego_actor_id == original.ego_actor_id
    assert prepared.episode_id == original.episode_id
    assert prepared.status == "prepared"
    assert prepared.control_mode == "autopilot"
    assert worker.prepared == worker.stopped == worker.started == []
    assert worker.modes == [(original, "autopilot")]
    assert session.snapshot()["traffic_count_actual"] == 15
    assert session.snapshot()["walker_count_actual"] == 10

    activated = session._activate_worker_scene()
    assert activated.status == "running"
    assert activated.scene_id == original.scene_id
    assert len(worker.started) == 1
    assert worker.prepared == worker.stopped == []
    session._stop_worker_scene()
    session._stop_worker_scene()
    assert len(worker.stopped) == 1
    assert worker.stopped[0].scene_id == original.scene_id


def test_random_route_is_prepared_in_garage_and_survives_handoff(tmp_path: Path) -> None:
    canonical = session_defaults(detector_enabled=False, worker_configured=True)
    canonical["identity"]["runId"] = "handoff-random-route"
    canonical["route"]["mode"] = "random_destination"
    assert build_garage_preview_request(canonical)["route_mode"] == "random_destination"
    route = {"mode": "random_destination", "planned": True, "waypoint_count": 30}
    scene = ready_scene(route_mode="random_destination", route=route, destination={"x": 150.0})
    worker = HandoffWorker()
    manager, _preview, _stream = ready_preview(worker, scene=scene, route_mode="random_destination")
    transferred = manager.take_for_drive(drive_config(tmp_path, route_mode="random_destination"))
    assert transferred is not None
    assert transferred.route == route
    assert transferred.destination == {"x": 150.0}
    assert worker.prepared == worker.stopped == []


def test_operator_start_reuses_preview_scene_without_second_prepare_or_intermediate_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_drive_threads: None
) -> None:
    del no_drive_threads
    worker = HandoffWorker()
    preview_manager, preview, stream = ready_preview(worker)
    manager = GarageOperatorDriveManager(
        workspace=tmp_path,
        carla_host=CARLA_HOST,
        carla_port=CARLA_PORT,
        world_worker=worker,  # type: ignore[arg-type]
    )
    manager.attach_preview(preview_manager, preview_manager._world_mode_lock)

    def start_without_camera(session: GarageDriveSession) -> None:
        session._begin_worker_scene()
        session._activate_worker_scene()
        session._status = "running"

    monkeypatch.setattr(GarageDriveSession, "start", start_without_camera)
    result = manager.start(drive_payload())
    assert result["status"] == "running"
    assert stream.closed
    assert worker.prepared == worker.stopped == []
    assert [scene.scene_id for scene in worker.started] == ["preview-scene"]
    assert preview_manager.state()["active"] is False
    preview.close(reason="stale_preview_shutdown")
    assert worker.stopped == []
    assert manager._session is not None
    manager._session._stop_worker_scene()
    assert [scene.scene_id for scene in worker.stopped] == ["preview-scene"]


def test_invalid_drive_request_does_not_consume_ready_preview(
    tmp_path: Path, no_drive_threads: None
) -> None:
    del no_drive_threads
    worker = HandoffWorker()
    preview_manager, _preview, stream = ready_preview(worker)
    manager = GarageOperatorDriveManager(
        workspace=tmp_path,
        carla_host=CARLA_HOST,
        carla_port=CARLA_PORT,
        world_worker=worker,  # type: ignore[arg-type]
    )
    manager.attach_preview(preview_manager, preview_manager._world_mode_lock)
    try:
        with pytest.raises(ValueError, match="run_id"):
            manager.start(drive_payload(run_id="invalid run id"))
        assert preview_manager.state()["active"] is True
        assert not stream.closed
        assert worker.prepared == worker.stopped == []
    finally:
        preview_manager.shutdown()


@pytest.mark.parametrize("reason", ["old_worker", "changed_scene", "fresh_experiment"])
def test_operator_start_uses_fresh_prepare_when_handoff_is_ineligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_drive_threads: None,
    reason: str,
) -> None:
    del no_drive_threads
    worker = HandoffWorker()
    scene = ready_scene(capabilities={}) if reason == "old_worker" else ready_scene()
    preview_manager, _preview, stream = ready_preview(worker, scene=scene)
    manager = GarageOperatorDriveManager(
        workspace=tmp_path,
        carla_host=CARLA_HOST,
        carla_port=CARLA_PORT,
        world_worker=worker,  # type: ignore[arg-type]
    )
    manager.attach_preview(preview_manager, preview_manager._world_mode_lock)

    def start_without_camera(session: GarageDriveSession) -> None:
        assert [stopped.scene_id for stopped in worker.stopped] == ["preview-scene"]
        session._begin_worker_scene()
        session._activate_worker_scene()
        session._status = "running"

    monkeypatch.setattr(GarageDriveSession, "start", start_without_camera)
    overrides = (
        {"seed": 43}
        if reason == "changed_scene"
        else {"experiment_preset": "traffic_stress"}
        if reason == "fresh_experiment"
        else {}
    )
    result = manager.start(drive_payload(**overrides))
    assert result["status"] == "running"
    assert stream.closed
    assert len(worker.prepared) == 1
    assert [started.scene_id for started in worker.started] == ["fresh-scene"]
    assert [stopped.scene_id for stopped in worker.stopped] == ["preview-scene"]
    assert manager._session is not None
    manager._session._stop_worker_scene()


@pytest.mark.parametrize("phase", ["constructor", "start"])
def test_failed_drive_startup_releases_transferred_scene_once_and_leaves_manager_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_drive_threads: None, phase: str
) -> None:
    del no_drive_threads
    worker = HandoffWorker()
    preview_manager, preview, _stream = ready_preview(worker)
    manager = GarageOperatorDriveManager(
        workspace=tmp_path,
        carla_host=CARLA_HOST,
        carla_port=CARLA_PORT,
        world_worker=worker,  # type: ignore[arg-type]
    )
    manager.attach_preview(preview_manager, preview_manager._world_mode_lock)

    def fail_startup(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("injected Drive startup failure")

    if phase == "constructor":
        monkeypatch.setattr(garage_drive, "GarageDriveSession", fail_startup)
    else:
        monkeypatch.setattr(GarageDriveSession, "start", fail_startup)
    with pytest.raises(RuntimeError, match="startup failure"):
        manager.start(drive_payload())
    assert manager._session is None
    assert manager.state()["status"] == "idle"
    assert worker.prepared == worker.started == []
    assert [scene.scene_id for scene in worker.stopped] == ["preview-scene"]
    preview.close(reason="stale_owner_shutdown")
    preview_manager.shutdown()
    assert len(worker.stopped) == 1


def test_handoff_cancels_pending_garage_reconfiguration(tmp_path: Path) -> None:
    worker = HandoffWorker()
    manager, _preview, _stream = ready_preview(worker)
    with manager._configure_condition:
        manager._configure_requested_revision = 2
    transferred = manager.take_for_drive(drive_config(tmp_path))
    assert transferred is not None
    with manager._configure_condition:
        assert manager._configure_completed_revision == 2
        assert set(manager._configure_outcomes) == {1, 2}
        assert all(
            isinstance(outcome[2], RuntimeError) for outcome in manager._configure_outcomes.values()
        )
    assert worker.prepared == worker.stopped == []


def test_failed_start_and_failed_cleanup_retain_lease_until_shutdown_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_drive_threads: None
) -> None:
    del no_drive_threads
    worker = HandoffWorker()
    preview_manager, _preview, _stream = ready_preview(worker)
    manager = GarageOperatorDriveManager(
        workspace=tmp_path,
        carla_host=CARLA_HOST,
        carla_port=CARLA_PORT,
        world_worker=worker,  # type: ignore[arg-type]
    )
    manager.attach_preview(preview_manager, preview_manager._world_mode_lock)
    stop = worker.stop_scene

    def fail_start(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("injected start failure")

    def fail_stop(scene: WorldWorkerScene) -> WorldWorkerScene:
        raise TimeoutError("cleanup unavailable")

    monkeypatch.setattr(GarageDriveSession, "start", fail_start)
    monkeypatch.setattr(worker, "stop_scene", fail_stop)
    with pytest.raises(RuntimeError, match="start failure") as caught:
        manager.start(drive_payload())
    assert any("cleanup unavailable" in note for note in caught.value.__notes__)
    assert manager._pending_start_cleanup is not None
    with pytest.raises(TimeoutError, match="cleanup unavailable"):
        manager.start(drive_payload())
    assert worker.prepared == []
    monkeypatch.setattr(worker, "stop_scene", stop)
    manager.shutdown()
    assert manager._pending_start_cleanup is None
    assert [scene.scene_id for scene in worker.stopped] == ["preview-scene"]


def test_preview_manager_rejects_scene_owned_by_different_worker_client(tmp_path: Path) -> None:
    worker = HandoffWorker()
    manager, _preview, stream = ready_preview(worker)
    manager.world_worker = HandoffWorker()  # type: ignore[assignment]
    try:
        assert manager.take_for_drive(drive_config(tmp_path)) is None
        assert not stream.closed
        assert manager.state()["active"] is True
        assert worker.prepared == worker.stopped == []
    finally:
        manager.shutdown()


def test_terminal_cleanup_diagnostics_do_not_block_future_starts_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = HandoffWorker()
    manager = GarageOperatorDriveManager(
        workspace=tmp_path,
        carla_host=CARLA_HOST,
        carla_port=CARLA_PORT,
        world_worker=worker,  # type: ignore[arg-type]
    )
    scene = ready_scene()
    manager._pending_start_cleanup = (worker, scene)
    monkeypatch.setattr(
        worker,
        "stop_scene",
        lambda item: replace(
            item, status="stopped", cleanup_errors=("episode changed; cleanup skipped",)
        ),
    )
    with pytest.raises(RuntimeError, match="episode changed"):
        manager._retry_start_cleanup()
    assert manager._pending_start_cleanup is None
    manager._retry_start_cleanup()


def test_drive_manager_does_not_adopt_scene_from_different_worker_client(tmp_path: Path) -> None:
    preview_worker = HandoffWorker()
    drive_worker = HandoffWorker()
    preview_manager, _preview, stream = ready_preview(preview_worker)
    manager = GarageOperatorDriveManager(
        workspace=tmp_path,
        carla_host=CARLA_HOST,
        carla_port=CARLA_PORT,
        world_worker=drive_worker,  # type: ignore[arg-type]
    )
    manager.attach_preview(preview_manager, preview_manager._world_mode_lock)
    assert manager._prepare_worker_scene(drive_config(tmp_path)) is None
    assert stream.closed
    assert [scene.scene_id for scene in preview_worker.stopped] == ["preview-scene"]
    assert drive_worker.prepared == drive_worker.started == drive_worker.stopped == []


@pytest.mark.parametrize("failure", ["timeout", "changed_episode"])
def test_failed_handoff_heartbeat_releases_original_lease_and_never_starts_drive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    worker = HandoffWorker()
    manager, preview, stream = ready_preview(worker)
    original = preview._scene

    def fail_refresh(scene: WorldWorkerScene) -> WorldWorkerScene:
        if failure == "timeout":
            raise TimeoutError("injected handoff heartbeat timeout")
        return replace(scene, episode_id=999)

    monkeypatch.setattr(worker, "heartbeat", fail_refresh)
    with pytest.raises(
        (TimeoutError, RuntimeError), match="heartbeat timeout|changed before handoff"
    ):
        manager.take_for_drive(drive_config(tmp_path))
    assert stream.closed
    assert worker.prepared == worker.started == []
    assert worker.stopped == [original]
    manager.shutdown()
    assert worker.stopped == [original]
