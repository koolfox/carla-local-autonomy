from __future__ import annotations

import threading
import urllib.request
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest

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
    assert config.fps == 20.0
    assert config.traffic_count == 15
    assert config.walker_count == 10
    assert config.prop_preset == "construction"
    assert config.spectator_mirror is False

    mirrored = GaragePreviewConfig.from_mapping(
        preview_payload(spectator_mirror=True)
    )
    assert mirrored.spectator_mirror is True

    with pytest.raises(ValueError, match="unknown fields"):
        GaragePreviewConfig.from_mapping(preview_payload(shell="anything"))
    with pytest.raises(ValueError, match="missing fields"):
        raw = preview_payload()
        raw.pop("traffic_count")
        GaragePreviewConfig.from_mapping(raw)


def test_orbit_contract_normalizes_yaw_and_clamps_bounded_camera_controls() -> None:
    request = GarageOrbitRequest.from_mapping(
        {"sequence": 9, "yaw": -10.0, "pitch": -80.0, "distance": 30.0}
    )

    assert request.sequence == 9
    assert request.yaw == 350.0
    assert request.pitch == -25.0
    assert request.distance == 10.0

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
            "frame_sequence": 1,
            "yaw": self.config.yaw,
            "pitch": self.config.pitch,
            "distance": self.config.distance,
            "error": None,
            "cleanup_errors": [],
        }

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
    assert second["active"] is True
    assert sessions[0].events == ["start", "close:reconfigure"]
    assert sessions[1].events == ["start"]

    manager.stop_for_drive()
    assert sessions[1].events == ["start", "close:drive_start"]
    assert manager.state()["active"] is False


def test_garage_server_streams_multipart_preview_frames_without_polling(
    tmp_path,
) -> None:
    class StreamSession:
        def wait_for_frame(self, after_sequence: int, *, timeout: float) -> tuple[int, bytes]:
            assert timeout == 5.0
            if after_sequence < 7:
                return 7, b"jpeg-payload"
            raise EOFError("done")

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


class _CleanupWorker:
    def __init__(self) -> None:
        self.stopped: list[WorldWorkerScene] = []

    def stop_scene(self, scene: WorldWorkerScene) -> WorldWorkerScene:
        self.stopped.append(scene)
        return replace(scene, status="stopped", cleanup_guard_passed=True)


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
    ("episode", "role", "destroyed"),
    [
        (10, "garage_preview", [30]),
        (11, "garage_preview", []),
        (10, "somebody_else", []),
    ],
)
def test_preview_cleanup_requires_same_episode_type_and_owned_role(
    episode: int,
    role: str,
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
        session._rpc = rpc  # type: ignore[assignment]
        session._status = "running"

    result = session.close()

    assert rpc.destroyed == destroyed
    assert rpc.closed is True
    assert len(worker.stopped) == 1
    assert result["status"] == "stopped"
