from __future__ import annotations

import threading
import time
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
        )
    )
    assert tuned.pedestrian_crossing_factor == 0.9
    assert tuned.speed_difference_percent == -25.0
    assert tuned.following_distance_metres == 9.5

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
