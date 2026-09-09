from __future__ import annotations

import json
import threading
import time
import urllib.request
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest

from carla_vision.operator.configuration import session_defaults
from carla_vision.operator.garage_preview import (
    GarageOrbitRequest,
    GaragePreviewConfig,
    GaragePreviewManager,
    GaragePreviewSession,
)
from carla_vision.operator.garage_server import create_server
from carla_vision.operator.world_worker_client import WorldWorkerScene


def preview_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "map_name": "Town10HD_Opt",
        "weather_preset": "clear-day",
        "vehicle_blueprint": "vehicle.tesla.model3",
        "color": "255,0,0",
        "seed": 42,
        "traffic_count": 15,
        "walker_count": 10,
        "prop_preset": "construction",
    }
    payload.update(overrides)
    return payload


def test_preview_config_is_strict_and_uses_authoritative_camera_defaults() -> None:
    config = GaragePreviewConfig.from_mapping(preview_payload())

    assert config.yaw == 325.0
    assert config.pitch == -10.0
    assert config.distance == 6.5
    assert config.width == 1280
    assert config.height == 720
    assert config.fps == 30.0
    assert config.profile == "balanced"
    assert config.traffic_count == 15
    assert config.walker_count == 10
    assert config.prop_preset == "construction"
    assert config.pedestrian_crossing_factor == 0.2
    assert config.speed_difference_percent == 12.0
    assert config.following_distance_metres == 2.0
    assert config.spectator_mirror is False

    mirrored = GaragePreviewConfig.from_mapping(preview_payload(spectator_mirror=True))
    assert mirrored.spectator_mirror is True
    tuned = GaragePreviewConfig.from_mapping(
        preview_payload(
            pedestrian_crossing_factor=0.9,
            speed_difference_percent=-25.0,
            following_distance_metres=9.5,
            fov=105.0,
        )
    )
    assert tuned.pedestrian_crossing_factor == 0.9
    assert tuned.speed_difference_percent == -25.0
    assert tuned.following_distance_metres == 9.5
    assert tuned.fov == 105.0
    assert tuned.as_dict()["fov"] == 105.0

    with pytest.raises(ValueError, match="unknown fields"):
        GaragePreviewConfig.from_mapping(preview_payload(shell="anything"))
    with pytest.raises(ValueError, match="missing fields"):
        raw = preview_payload()
        raw.pop("traffic_count")
        GaragePreviewConfig.from_mapping(raw)


@pytest.mark.parametrize(
    ("profile", "width", "height", "fps"),
    [
        ("balanced", 1280, 720, 30.0),
        ("high-refresh", 1280, 720, 60.0),
        ("detail", 1920, 1080, 30.0),
        ("compatibility", 640, 384, 10.0),
    ],
)
def test_preview_config_maps_bounded_camera_profiles(
    profile: str,
    width: int,
    height: int,
    fps: float,
) -> None:
    config = GaragePreviewConfig.from_mapping(preview_payload(profile=profile))

    assert (config.width, config.height, config.fps) == (width, height, fps)
    assert config.profile == profile

    with pytest.raises(ValueError, match="profile must be one of"):
        GaragePreviewConfig.from_mapping(preview_payload(profile="unbounded"))
    with pytest.raises(ValueError, match="unknown fields"):
        GaragePreviewConfig.from_mapping(preview_payload(fps=60))


def test_orbit_contract_normalizes_yaw_and_clamps_bounded_camera_controls() -> None:
    request = GarageOrbitRequest.from_mapping(
        {"sequence": 9, "yaw": -10.0, "pitch": -80.0, "distance": 30.0}
    )

    assert request.sequence == 9
    assert request.yaw == 350.0
    assert request.pitch == -25.0
    assert request.distance == 10.0
    assert request.preset == "orbit"

    cockpit = GarageOrbitRequest.from_mapping(
        {
            "sequence": 10,
            "yaw": 0,
            "pitch": 0,
            "distance": 6,
            "preset": "cockpit",
        }
    )
    assert cockpit.preset == "cockpit"

    with pytest.raises(ValueError, match="preset must be"):
        GarageOrbitRequest.from_mapping(
            {
                "sequence": 10,
                "yaw": 0,
                "pitch": 0,
                "distance": 6,
                "preset": "cinematic",
            }
        )

    with pytest.raises(ValueError, match="unknown fields"):
        GarageOrbitRequest.from_mapping(
            {"sequence": 10, "yaw": 0, "pitch": 0, "distance": 6, "actor_id": 1}
        )


class _FakePreviewSession:
    events: list[str]

    def __init__(self, config: GaragePreviewConfig, **_: Any) -> None:
        self.config = config
        self.events = []
        self.active = False

    def start(self) -> dict[str, Any]:
        self.events.append("start")
        self.active = True
        return self.snapshot()

    def close(self, *, reason: str) -> dict[str, Any]:
        self.events.append(f"close:{reason}")
        self.active = False
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {
            "status": "running" if self.active else "stopped",
            "active": self.active,
            "applied_config": self.config.as_dict(),
            "frame_sequence": 1,
            "yaw": self.config.yaw,
            "pitch": self.config.pitch,
            "distance": self.config.distance,
            "error": None,
            "cleanup_errors": [],
            "camera_id": id(self),
            "weather_preset": self.config.weather_preset,
        }

    def update_weather(self, weather_preset: str) -> dict[str, Any]:
        self.events.append(f"weather:{weather_preset}")
        self.config = replace(self.config, weather_preset=weather_preset)
        return self.snapshot()

    def orbit(self, request: GarageOrbitRequest) -> dict[str, Any]:
        self.events.append(f"orbit:{request.sequence}")
        return self.snapshot()

    def frame(self) -> tuple[int, bytes]:
        return 1, b"jpeg"


def _manager(
    drive_state: Mapping[str, Any],
) -> tuple[GaragePreviewManager, list[_FakePreviewSession]]:
    sessions: list[_FakePreviewSession] = []

    def factory(config: GaragePreviewConfig, **kwargs: Any) -> _FakePreviewSession:
        del kwargs
        session = _FakePreviewSession(config)
        sessions.append(session)
        return session

    manager = GaragePreviewManager(
        carla_host="127.0.0.1",
        carla_port=2000,
        world_worker=object(),  # type: ignore[arg-type]
        drive_state=lambda: drive_state,
        world_mode_lock=threading.RLock(),
        session_factory=factory,  # type: ignore[arg-type]
    )
    return manager, sessions


def test_preview_manager_rejects_drive_overlap_and_reconfigure_cleans_old_scene() -> None:
    active_manager, active_sessions = _manager({"status": "running"})
    with pytest.raises(RuntimeError, match="end the active Drive"):
        active_manager.configure(preview_payload())
    assert active_sessions == []

    manager, sessions = _manager({"status": "idle"})
    first = manager.configure(preview_payload())
    second = manager.configure(preview_payload(vehicle_blueprint="vehicle.audi.tt"))

    assert first["active"] is True
    assert first["configure_action"] == "started"
    assert second["active"] is True
    assert second["configure_action"] == "restarted"
    assert sessions[0].events == ["start", "close:reconfigure"]
    assert sessions[1].events == ["start"]

    manager.stop_for_drive()
    assert sessions[1].events == ["start", "close:drive_start"]
    assert manager.state()["active"] is False


def test_preview_manager_identical_config_is_a_no_op() -> None:
    manager, sessions = _manager({"status": "idle"})

    first = manager.configure(preview_payload())
    second = manager.configure(preview_payload())

    assert len(sessions) == 1
    assert sessions[0].events == ["start"]
    assert second["camera_id"] == first["camera_id"]
    assert second["configure_action"] == "noop"


def test_preview_manager_updates_only_weather_in_place() -> None:
    manager, sessions = _manager({"status": "idle"})

    first = manager.configure(preview_payload())
    second = manager.configure(preview_payload(weather_preset="heavy-rain"))

    assert len(sessions) == 1
    assert sessions[0].events == ["start", "weather:heavy-rain"]
    assert second["camera_id"] == first["camera_id"]
    assert second["weather_preset"] == "heavy-rain"
    assert second["configure_action"] == "weather"


def test_preview_manager_weather_failure_preserves_existing_scene() -> None:
    manager, sessions = _manager({"status": "idle"})
    first = manager.configure(preview_payload())

    def fail_weather(weather_preset: str) -> dict[str, Any]:
        sessions[0].events.append(f"weather-failed:{weather_preset}")
        raise TimeoutError("weather RPC stalled")

    sessions[0].update_weather = fail_weather  # type: ignore[method-assign]

    with pytest.raises(TimeoutError, match="weather RPC stalled"):
        manager.configure(preview_payload(weather_preset="heavy-rain"))

    current = manager.state()
    assert len(sessions) == 1
    assert current["active"] is True
    assert current["camera_id"] == first["camera_id"]
    assert current["weather_preset"] == "clear-day"


def test_preview_manager_coalesces_overlapping_requests_to_latest_config() -> None:
    sessions: list[_FakePreviewSession] = []
    first_reconfigure_started = threading.Event()
    release_first_reconfigure = threading.Event()

    class BlockingSession(_FakePreviewSession):
        def start(self) -> dict[str, Any]:
            if self.config.traffic_count == 20:
                first_reconfigure_started.set()
                assert release_first_reconfigure.wait(2.0)
            return super().start()

    def factory(config: GaragePreviewConfig, **kwargs: Any) -> BlockingSession:
        del kwargs
        session = BlockingSession(config)
        sessions.append(session)
        return session

    manager = GaragePreviewManager(
        carla_host="127.0.0.1",
        carla_port=2000,
        world_worker=object(),  # type: ignore[arg-type]
        drive_state=lambda: {"status": "idle"},
        world_mode_lock=threading.RLock(),
        session_factory=factory,  # type: ignore[arg-type]
    )
    manager.configure(preview_payload(traffic_count=10))
    results: dict[str, dict[str, Any]] = {}
    errors: list[BaseException] = []

    def configure(name: str, traffic_count: int) -> None:
        try:
            results[name] = manager.configure(preview_payload(traffic_count=traffic_count))
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=configure, args=("first", 20))
    middle = threading.Thread(target=configure, args=("middle", 30))
    latest = threading.Thread(target=configure, args=("latest", 40))
    first.start()
    assert first_reconfigure_started.wait(1.0)
    middle.start()
    latest.start()
    with manager._configure_condition:
        assert manager._configure_condition.wait_for(
            lambda: manager._configure_requested_revision == 4,
            timeout=1.0,
        )
    release_first_reconfigure.set()
    for thread in (first, middle, latest):
        thread.join(timeout=3.0)
        assert not thread.is_alive()

    assert errors == []
    assert [session.config.traffic_count for session in sessions] == [10, 20, 40]
    assert manager.state()["applied_config"]["traffic_count"] == 40
    assert results["first"]["requested_config_superseded"] is True
    assert results["middle"]["requested_config_superseded"] is True
    assert results["latest"]["configure_action"] == "restarted"
    assert all(result["applied_config"]["traffic_count"] == 40 for result in results.values())


def test_preview_manager_recovers_when_superseded_configuration_fails() -> None:
    sessions: list[_FakePreviewSession] = []
    failed_start_entered = threading.Event()
    release_failed_start = threading.Event()

    class FailingSession(_FakePreviewSession):
        def start(self) -> dict[str, Any]:
            if self.config.traffic_count == 20:
                failed_start_entered.set()
                assert release_failed_start.wait(2.0)
                raise TimeoutError("superseded CARLA prepare stalled")
            return super().start()

    def factory(config: GaragePreviewConfig, **kwargs: Any) -> FailingSession:
        del kwargs
        session = FailingSession(config)
        sessions.append(session)
        return session

    manager = GaragePreviewManager(
        carla_host="127.0.0.1",
        carla_port=2000,
        world_worker=object(),  # type: ignore[arg-type]
        drive_state=lambda: {"status": "idle"},
        world_mode_lock=threading.RLock(),
        session_factory=factory,  # type: ignore[arg-type]
    )
    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []

    def configure(traffic_count: int) -> None:
        try:
            results.append(manager.configure(preview_payload(traffic_count=traffic_count)))
        except BaseException as error:
            errors.append(error)

    failed = threading.Thread(target=configure, args=(20,))
    latest = threading.Thread(target=configure, args=(40,))
    failed.start()
    assert failed_start_entered.wait(1.0)
    latest.start()
    with manager._configure_condition:
        assert manager._configure_condition.wait_for(
            lambda: manager._configure_requested_revision == 2,
            timeout=1.0,
        )
    release_failed_start.set()
    for thread in (failed, latest):
        thread.join(timeout=3.0)
        assert not thread.is_alive()

    assert errors == []
    assert [session.config.traffic_count for session in sessions] == [20, 40]
    assert len(results) == 2
    assert all(result["applied_config"]["traffic_count"] == 40 for result in results)
    assert manager.state()["active"] is True


def test_preview_manager_stop_cancels_config_waiting_for_world_lock() -> None:
    sessions: list[_FakePreviewSession] = []
    world_mode_lock = threading.RLock()

    def factory(config: GaragePreviewConfig, **kwargs: Any) -> _FakePreviewSession:
        del kwargs
        session = _FakePreviewSession(config)
        sessions.append(session)
        return session

    manager = GaragePreviewManager(
        carla_host="127.0.0.1",
        carla_port=2000,
        world_worker=object(),  # type: ignore[arg-type]
        drive_state=lambda: {"status": "idle"},
        world_mode_lock=world_mode_lock,
        session_factory=factory,  # type: ignore[arg-type]
    )
    manager.configure(preview_payload(traffic_count=10))
    errors: list[BaseException] = []

    def configure_waiting() -> None:
        try:
            manager.configure(preview_payload(traffic_count=20))
        except BaseException as error:
            errors.append(error)

    waiting = threading.Thread(target=configure_waiting)
    with world_mode_lock:
        waiting.start()
        with manager._configure_condition:
            assert manager._configure_condition.wait_for(
                lambda: manager._configure_requested_revision == 2,
                timeout=1.0,
            )
        stopped = manager.stop()
        assert stopped["active"] is False
    waiting.join(timeout=3.0)

    assert not waiting.is_alive()
    assert len(errors) == 1
    assert "cancelled by operator stop" in str(errors[0])
    assert len(sessions) == 1
    assert sessions[0].events == ["start", "close:operator_stop"]

    restarted = manager.configure(preview_payload(traffic_count=30))
    assert restarted["active"] is True
    assert restarted["applied_config"]["traffic_count"] == 30


def test_preview_manager_shutdown_cancels_waiting_config_and_stays_closed() -> None:
    sessions: list[_FakePreviewSession] = []
    world_mode_lock = threading.RLock()

    def factory(config: GaragePreviewConfig, **kwargs: Any) -> _FakePreviewSession:
        del kwargs
        session = _FakePreviewSession(config)
        sessions.append(session)
        return session

    manager = GaragePreviewManager(
        carla_host="127.0.0.1",
        carla_port=2000,
        world_worker=object(),  # type: ignore[arg-type]
        drive_state=lambda: {"status": "idle"},
        world_mode_lock=world_mode_lock,
        session_factory=factory,  # type: ignore[arg-type]
    )
    manager.configure(preview_payload(traffic_count=10))
    errors: list[BaseException] = []

    def configure_waiting() -> None:
        try:
            manager.configure(preview_payload(traffic_count=20))
        except BaseException as error:
            errors.append(error)

    waiting = threading.Thread(target=configure_waiting)
    with world_mode_lock:
        waiting.start()
        with manager._configure_condition:
            assert manager._configure_condition.wait_for(
                lambda: manager._configure_requested_revision == 2,
                timeout=1.0,
            )
        manager.shutdown()
    waiting.join(timeout=3.0)

    assert not waiting.is_alive()
    assert len(errors) == 1
    assert "cancelled by Operator server shutdown" in str(errors[0])
    assert len(sessions) == 1
    assert sessions[0].events == ["start", "close:operator_server_shutdown"]
    assert manager.state()["active"] is False
    with pytest.raises(RuntimeError, match="shutting down"):
        manager.configure(preview_payload(traffic_count=30))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("traffic_count", 30),
        ("walker_count", 22),
        ("prop_preset", "accident"),
    ],
)
def test_preview_manager_restarts_for_scene_population_changes(
    field: str,
    value: object,
) -> None:
    manager, sessions = _manager({"status": "idle"})

    manager.configure(preview_payload())
    changed = manager.configure(preview_payload(**{field: value}))

    assert changed["configure_action"] == "restarted"
    assert len(sessions) == 2
    assert sessions[0].events == ["start", "close:reconfigure"]
    assert sessions[1].events == ["start"]


@pytest.mark.parametrize("end_error", [EOFError, TimeoutError])
def test_garage_server_streams_multipart_preview_frames_without_polling(
    tmp_path, end_error,
) -> None:
    class StreamSession:
        def wait_for_frame(self, after_sequence: int, *, timeout: float) -> tuple[int, bytes]:
            assert timeout == 5.0
            if after_sequence < 7:
                return 7, b"jpeg-payload"
            raise end_error("done")

    class Preview:
        def subscribe(self) -> StreamSession:
            return StreamSession()

        def shutdown(self) -> None:
            pass

    server = create_server(
        workspace=tmp_path,
        bind="127.0.0.1",
        port=0,
        sessions_root=tmp_path / "sessions",
        carla_host="127.0.0.1",
        carla_port=65534,
    )
    server.application.preview = Preview()  # type: ignore[assignment]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/garage/preview/stream.mjpg",
            timeout=3.0,
        ) as response:
            body = response.read()
            assert response.headers.get_content_type() == "multipart/x-mixed-replace"
            assert b"--carla-garage-preview" in body
            assert b"Content-Type: image/jpeg" in body
            assert b"X-Garage-Preview-Frame-Sequence: 7" in body
            assert b"jpeg-payload" in body
    finally:
        server.shutdown()
        thread.join(timeout=3.0)
        server.server_close()
        server.application.jobs.shutdown()


def test_canonical_preview_api_resolves_one_session_and_reports_applied_values(
    tmp_path,
) -> None:
    captured: dict[str, Any] = {}

    class Preview:
        def configure(self, request: Mapping[str, Any]) -> dict[str, Any]:
            captured.update(request)
            return {
                "status": "running",
                "traffic_count": 12,
                "walker_count": 7,
                "pedestrian_crossing_factor": 0.95,
            }

        def shutdown(self) -> None:
            pass

    server = create_server(
        workspace=tmp_path,
        bind="127.0.0.1",
        port=0,
        sessions_root=tmp_path / "sessions",
        carla_host="127.0.0.1",
        carla_port=65534,
    )
    server.application.preview = Preview()  # type: ignore[assignment]
    session = session_defaults(detector_enabled=False)
    session["identity"]["runId"] = "preview-evidence-test"
    session["vehicle"]["blueprint"] = "vehicle.tesla.model3"
    session["scene"].update(
        {
            "mapName": "Town10HD_Opt",
            "weatherPreset": "clear-day",
            "trafficCount": 14,
            "walkerCount": 9,
            "pedestrianCrossingFactor": 0.95,
        }
    )
    session["camera"].update({"resolution": "1920x1080", "fps": 60.0, "fov": 103.0})
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        request = urllib.request.Request(
            f"http://{host}:{port}/api/garage/preview/configure",
            data=json.dumps({"schema_version": "1.0", "session": session}).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Operator-Token": server.application.token,
            },
        )
        with urllib.request.urlopen(request, timeout=3.0) as response:
            payload = json.loads(response.read().decode("utf-8"))

        assert captured["traffic_count"] == 14
        assert captured["walker_count"] == 9
        assert captured["pedestrian_crossing_factor"] == 0.95
        assert captured["fov"] == 103.0
        assert payload["configuration"]["requested"] == session
        assert payload["configuration"]["resolved"]["profile"] == "detail"
        assert payload["configuration"]["resolved"]["fps"] == 30.0
        assert payload["configuration"]["applied"]["traffic_count"] == 12
    finally:
        server.shutdown()
        thread.join(timeout=3.0)
        server.server_close()
        server.application.jobs.shutdown()


def test_preview_api_coalesces_concurrent_configure_requests_without_409(
    tmp_path,
) -> None:
    first_start_entered = threading.Event()
    release_first_start = threading.Event()

    class BlockingSession(_FakePreviewSession):
        def start(self) -> dict[str, Any]:
            if self.config.traffic_count == 20:
                first_start_entered.set()
                assert release_first_start.wait(2.0)
            return super().start()

    def factory(config: GaragePreviewConfig, **kwargs: Any) -> BlockingSession:
        del kwargs
        return BlockingSession(config)

    manager = GaragePreviewManager(
        carla_host="127.0.0.1",
        carla_port=2000,
        world_worker=object(),  # type: ignore[arg-type]
        drive_state=lambda: {"status": "idle"},
        world_mode_lock=threading.RLock(),
        session_factory=factory,  # type: ignore[arg-type]
    )
    server = create_server(
        workspace=tmp_path,
        bind="127.0.0.1",
        port=0,
        sessions_root=tmp_path / "sessions",
        carla_host="127.0.0.1",
        carla_port=65534,
    )
    server.application.preview = manager
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    responses: dict[str, tuple[int, dict[str, Any]]] = {}
    errors: list[BaseException] = []

    def post(name: str, traffic_count: int) -> None:
        session = session_defaults(detector_enabled=False)
        session["identity"]["runId"] = f"preview-{name}"
        session["vehicle"]["blueprint"] = "vehicle.tesla.model3"
        session["scene"].update(
            {
                "mapName": "Town10HD_Opt",
                "weatherPreset": "clear-day",
                "trafficCount": traffic_count,
            }
        )
        host, port = server.server_address[:2]
        request = urllib.request.Request(
            f"http://{host}:{port}/api/garage/preview/configure",
            data=json.dumps({"schema_version": "1.0", "session": session}).encode(
                "utf-8"
            ),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Operator-Token": server.application.token,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=3.0) as response:
                responses[name] = (
                    response.status,
                    json.loads(response.read().decode("utf-8")),
                )
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=post, args=("first", 20))
    latest = threading.Thread(target=post, args=("latest", 40))
    try:
        first.start()
        assert first_start_entered.wait(1.0)
        latest.start()
        with manager._configure_condition:
            assert manager._configure_condition.wait_for(
                lambda: manager._configure_requested_revision == 2,
                timeout=1.0,
            )
        release_first_start.set()
        for request_thread in (first, latest):
            request_thread.join(timeout=3.0)
            assert not request_thread.is_alive()

        assert errors == []
        assert {name: status for name, (status, _) in responses.items()} == {
            "first": 201,
            "latest": 201,
        }
        assert responses["first"][1]["requested_config_superseded"] is True
        assert all(
            payload["applied_config"]["traffic_count"] == 40
            for _, payload in responses.values()
        )
        assert responses["first"][1]["configuration"]["requested"]["scene"][
            "trafficCount"
        ] == 20
        assert responses["latest"][1]["configuration"]["requested"]["scene"][
            "trafficCount"
        ] == 40
        assert manager.state()["applied_config"]["traffic_count"] == 40
    finally:
        release_first_start.set()
        first.join(timeout=3.0)
        latest.join(timeout=3.0)
        server.shutdown()
        server_thread.join(timeout=3.0)
        server.server_close()
        server.application.jobs.shutdown()


def _scene() -> WorldWorkerScene:
    return WorldWorkerScene(
        scene_id="preview-scene",
        lease_token="preview-lease",
        status="prepared",
        episode_id=10,
        ego_actor_id=20,
        map_name="Town10HD_Opt",
        spawn_index=1,
        route_mode="free",
        route={},
        destination=None,
        control_mode="manual",
        traffic_count=0,
        walker_count=0,
        prop_actor_ids=(),
        lease_expires_in_seconds=30.0,
        cleanup_guard_passed=None,
        cleanup_errors=(),
        capabilities={},
    )


def test_preview_snapshot_separates_requested_and_worker_applied_scene_values() -> None:
    session = GaragePreviewSession(
        GaragePreviewConfig.from_mapping(
            preview_payload(
                traffic_count=15,
                walker_count=10,
                pedestrian_crossing_factor=0.9,
                speed_difference_percent=-25.0,
                following_distance_metres=9.5,
                fov=100.0,
            )
        ),
        carla_host="127.0.0.1",
        carla_port=2000,
        world_worker=object(),  # type: ignore[arg-type]
    )
    with session._lock:
        session._scene = replace(
            _scene(),
            traffic_count=13,
            walker_count=8,
            pedestrian_crossing_factor=0.9,
            speed_difference_percent=-25.0,
            following_distance_metres=9.5,
        )

    snapshot = session.snapshot()

    assert snapshot["traffic_count_requested"] == 15
    assert snapshot["walker_count_requested"] == 10
    assert snapshot["traffic_count"] == 13
    assert snapshot["walker_count"] == 8
    assert snapshot["pedestrian_crossing_factor"] == 0.9
    assert snapshot["speed_difference_percent"] == -25.0
    assert snapshot["following_distance_metres"] == 9.5
    assert snapshot["camera_fov"] == 100.0


class _CleanupWorker:
    def __init__(self) -> None:
        self.stopped: list[WorldWorkerScene] = []

    def stop_scene(self, scene: WorldWorkerScene) -> WorldWorkerScene:
        self.stopped.append(scene)
        return replace(scene, status="stopped", cleanup_guard_passed=True)


def test_preview_session_weather_update_preserves_scene_camera_and_stream() -> None:
    class Worker(_CleanupWorker):
        def __init__(self) -> None:
            super().__init__()
            self.weather_updates: list[tuple[WorldWorkerScene, str]] = []

        def weather(
            self,
            scene: WorldWorkerScene,
            weather_preset: str,
        ) -> WorldWorkerScene:
            self.weather_updates.append((scene, weather_preset))
            return scene

    worker = Worker()
    session = GaragePreviewSession(
        GaragePreviewConfig.from_mapping(preview_payload()),
        carla_host="127.0.0.1",
        carla_port=2000,
        world_worker=worker,  # type: ignore[arg-type]
    )
    scene = _scene()
    stream = object()
    with session._lock:
        session._scene = scene
        session._stream = stream  # type: ignore[assignment]
        session._vehicle_id = 20
        session._camera_id = 30
        session._status = "running"

    updated = session.update_weather("heavy-rain")

    assert worker.weather_updates == [(scene, "heavy-rain")]
    assert session._stream is stream
    assert updated["vehicle_id"] == 20
    assert updated["camera_id"] == 30
    assert updated["weather_preset"] == "heavy-rain"
    assert updated["applied_config"]["weather_preset"] == "heavy-rain"


class _CleanupRpc:
    def __init__(self, *, episode: int, role: str) -> None:
        self.episode = episode
        self.role = role
        self.destroyed: list[int] = []
        self.closed = False

    def episode_id(self) -> int:
        return self.episode

    def actor(self, actor_id: int) -> list[Any]:
        return [
            actor_id,
            None,
            [1, "sensor.camera.rgb", [["role_name", 0, self.role]]],
        ]

    def destroy_actor(self, actor_id: int) -> None:
        self.destroyed.append(actor_id)

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    ("episode", "role", "worker_camera", "destroyed"),
    [
        (10, "garage_preview", False, [30]),
        (11, "garage_preview", False, []),
        (10, "somebody_else", False, []),
        (10, "world_worker_camera", True, []),
    ],
)
def test_preview_cleanup_requires_same_episode_type_and_owned_role(
    episode: int,
    role: str,
    worker_camera: bool,
    destroyed: list[int],
) -> None:
    worker = _CleanupWorker()
    rpc = _CleanupRpc(episode=episode, role=role)
    session = GaragePreviewSession(
        GaragePreviewConfig.from_mapping(preview_payload()),
        carla_host="127.0.0.1",
        carla_port=2000,
        world_worker=worker,  # type: ignore[arg-type]
    )
    with session._lock:
        session._scene = _scene()
        session._episode_id = 10
        session._vehicle_id = 20
        session._camera_id = 30
        session._camera_type = "sensor.camera.rgb"
        session._worker_camera = worker_camera
        session._rpc = rpc  # type: ignore[assignment]
        session._status = "running"

    result = session.close()

    assert rpc.destroyed == destroyed
    assert rpc.closed is True
    assert len(worker.stopped) == 1
    assert result["status"] == "stopped"


def test_worker_orbit_rpc_remains_serialized_with_scene_lifecycle() -> None:
    class Worker(_CleanupWorker):
        session: GaragePreviewSession | None = None
        presets: list[str]

        def __init__(self) -> None:
            super().__init__()
            self.presets = []

        def orbit_camera(
            self,
            scene: WorldWorkerScene,
            *,
            yaw: float,
            pitch: float,
            distance: float,
            preset: str,
        ) -> dict[str, Any]:
            del scene, yaw, pitch, distance
            assert self.session is not None
            assert self.session._worker_request_lock.locked()
            self.presets.append(preset)
            return {"status": "running"}

    class Rpc(_CleanupRpc):
        def actor_transform(self, actor_id: int, component: str) -> list[Any]:
            assert (actor_id, component) == (20, "VehicleMesh")
            return [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    worker = Worker()
    rpc = Rpc(episode=10, role="world_worker_camera")
    session = GaragePreviewSession(
        GaragePreviewConfig.from_mapping(preview_payload()),
        carla_host="127.0.0.1",
        carla_port=2000,
        world_worker=worker,  # type: ignore[arg-type]
    )
    worker.session = session
    with session._lock:
        session._scene = _scene()
        session._episode_id = 10
        session._vehicle_id = 20
        session._camera_id = 30
        session._camera_type = "sensor.camera.rgb"
        session._worker_camera = True
        session._rpc = rpc  # type: ignore[assignment]
        session._status = "running"

    result = session.orbit(
        GarageOrbitRequest.from_mapping(
            {"sequence": 1, "yaw": 0.0, "pitch": -8.0, "distance": 6.0, "preset": "front"}
        )
    )

    assert worker.presets == ["front"]
    assert result["camera_preset"] == "front"
    session.close()


def test_preview_uses_worker_jpeg_relay_without_raw_lan_camera(monkeypatch) -> None:
    scene = replace(_scene(), capabilities={"compressed_camera_relay": True})

    class Worker(_CleanupWorker):
        camera_requests: list[dict[str, Any]]

        def __init__(self) -> None:
            super().__init__()
            self.camera_requests = []

        def prepare_scene(self, payload: dict[str, Any]) -> WorldWorkerScene:
            del payload
            return scene

        def start_camera(self, active: WorldWorkerScene, **payload: Any) -> dict[str, Any]:
            assert active is scene
            self.camera_requests.append(dict(payload))
            return {"camera": {"actor_id": 30}}

        def heartbeat(self, active: WorldWorkerScene) -> WorldWorkerScene:
            return active

    class Rpc:
        closed = False

        def value_call(self, method: str) -> str:
            assert method == "version"
            return "0.9.16"

        def episode_id(self) -> int:
            return 10

        def actor(self, actor_id: int) -> list[Any]:
            assert actor_id == 20
            return [20, None, [1, "vehicle.tesla.model3", []]]

        def actor_transform(self, actor_id: int, component: str) -> list[Any]:
            assert (actor_id, component) == (20, "VehicleMesh")
            return [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

        def destroy_actor(self, actor_id: int) -> None:
            raise AssertionError(f"Worker-owned camera {actor_id} must not be raw-destroyed")

        def close(self) -> None:
            self.closed = True

    class Frame:
        sequence = 1
        jpeg = b"\xff\xd8worker-jpeg\xff\xd9"

    class Stream:
        transport = "worker_mjpeg"

        def __init__(self, *_: Any, **__: Any) -> None:
            self.closed = threading.Event()
            self.first = True

        def wait_for_frame(self, after_sequence: int = -1, timeout: float = 5.0) -> Frame:
            del after_sequence
            if self.first:
                self.first = False
                return Frame()
            self.closed.wait(min(timeout, 0.05))
            raise TimeoutError

        def close(self) -> None:
            self.closed.set()

    worker = Worker()
    rpc = Rpc()
    monkeypatch.setattr(
        "carla_vision.operator.garage_preview.WorldWorkerCameraStream",
        Stream,
    )
    monkeypatch.setattr(
        "carla_vision.operator.garage_preview.spawn_unparented_rgb_camera",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("raw BGRA camera must not be spawned")
        ),
    )
    session = GaragePreviewSession(
        GaragePreviewConfig.from_mapping(preview_payload(profile="high-refresh")),
        carla_host="127.0.0.1",
        carla_port=2000,
        world_worker=worker,  # type: ignore[arg-type]
        rpc_factory=lambda *_args, **_kwargs: rpc,  # type: ignore[arg-type]
    )

    started = session.start()
    sequence, jpeg = session.frame()
    time.sleep(0.01)
    stopped = session.close()

    assert started["active"] is True
    assert (sequence, jpeg) == (1, Frame.jpeg)
    assert worker.camera_requests[0]["mode"] == "garage"
    assert worker.camera_requests[0]["width"] == 1280
    assert worker.camera_requests[0]["height"] == 720
    assert worker.camera_requests[0]["fps"] == 60.0
    assert started["camera_profile"] == "high-refresh"
    assert started["camera_target_fps"] == 60.0
    assert started["camera_resolution"] == "1280x720"
    assert started["camera_transport"] == "worker_mjpeg"
    assert started["stream"]["source_fps"] == 0.0
    assert started["stream"]["stale"] is False
    assert stopped["status"] == "stopped"
    assert rpc.closed is True


def test_raw_fallback_keeps_last_frame_during_long_lan_pause(monkeypatch) -> None:
    class PausedStream:
        def wait_for_frame(self, *, after_sequence: int, timeout: float) -> None:
            del after_sequence, timeout
            raise TimeoutError

    session = GaragePreviewSession(
        GaragePreviewConfig.from_mapping(preview_payload()),
        carla_host="127.0.0.1",
        carla_port=2000,
        world_worker=object(),  # type: ignore[arg-type]
    )
    with session._lock:
        session._stream = PausedStream()  # type: ignore[assignment]
        session._frame_sequence = 4
        session._jpeg = b"last-good-frame"
        session._status = "running"
        session._worker_camera = False
    monkeypatch.setattr(
        "carla_vision.operator.garage_preview.time.monotonic",
        lambda: 10_000.0,
    )

    thread = threading.Thread(target=session._frame_loop)
    thread.start()
    time.sleep(0.01)
    session._frame_stop.set()
    thread.join(timeout=1.0)

    assert session._status == "running"
    assert session._error is None
    assert session._jpeg == b"last-good-frame"


def test_transient_heartbeat_failure_recovers_without_closing_scene(monkeypatch) -> None:
    class Worker(_CleanupWorker):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            self.parked = threading.Event()
            self.release = threading.Event()

        def heartbeat(self, scene: WorldWorkerScene) -> WorldWorkerScene:
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("temporary LAN stall")
            if self.calls == 3:
                self.parked.set()
                self.release.wait(1.0)
            return scene

    monkeypatch.setattr("carla_vision.operator.garage_preview._HEARTBEAT_SECONDS", 0.001)
    worker = Worker()
    session = GaragePreviewSession(
        GaragePreviewConfig.from_mapping(preview_payload()),
        carla_host="127.0.0.1",
        carla_port=2000,
        world_worker=worker,  # type: ignore[arg-type]
    )
    scene = _scene()
    with session._lock:
        session._scene = scene
        session._vehicle_id = 20
        session._camera_id = 30
        session._status = "running"

    thread = threading.Thread(target=session._heartbeat_loop)
    thread.start()
    assert worker.parked.wait(1.0)

    assert session._status == "running"
    assert session._error is None
    assert session._closed is False
    assert session._scene is scene
    assert worker.stopped == []

    session._heartbeat_stop.set()
    worker.release.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive()


def test_persistent_heartbeat_failure_closes_after_bounded_retries(monkeypatch) -> None:
    class Worker(_CleanupWorker):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def heartbeat(self, scene: WorldWorkerScene) -> WorldWorkerScene:
            del scene
            self.calls += 1
            raise TimeoutError("persistent LAN failure")

    monkeypatch.setattr("carla_vision.operator.garage_preview._HEARTBEAT_SECONDS", 0.001)
    worker = Worker()
    session = GaragePreviewSession(
        GaragePreviewConfig.from_mapping(preview_payload()),
        carla_host="127.0.0.1",
        carla_port=2000,
        world_worker=worker,  # type: ignore[arg-type]
    )
    scene = _scene()
    with session._lock:
        session._scene = scene
        session._status = "running"

    session._heartbeat_loop()

    assert worker.calls == 3
    assert worker.stopped == [scene]
    assert session._status == "failed"
    assert session._closed is True
