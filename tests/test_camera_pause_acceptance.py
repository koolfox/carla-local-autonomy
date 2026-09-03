from __future__ import annotations

from typing import Any
from unittest import mock

import pytest
from test_world_worker import (
    FakeBlueprintLibrary,
    FakeCarla,
    FakeClient,
    FakeClock,
    FakePlanner,
    FakeTrafficManager,
    FakeWorld,
)

from carla_vision.native.world_worker import (
    _SCENE_PATH,
    CompressedCameraRelay,
    WorldWorker,
)
from carla_vision.operator.world_worker_client import (
    WorldWorkerClient,
    WorldWorkerError,
    WorldWorkerScene,
)


class Sensor:
    id = 901
    is_alive = True

    def __init__(self) -> None:
        self.callback: Any = None
        self.is_listening = False
        self.listen_calls = 0
        self.stop_calls = 0

    def listen(self, callback: Any) -> None:
        assert not self.is_listening
        self.callback = callback
        self.is_listening = True
        self.listen_calls += 1

    def stop(self) -> None:
        assert self.is_listening
        self.is_listening = False
        self.stop_calls += 1

    def emit(self, frame: int) -> None:
        if not self.is_listening or self.callback is None:
            return
        image = type(
            "Image",
            (),
            {
                "frame": frame,
                "timestamp": frame / 10.0,
                "width": 320,
                "height": 180,
                "fov": 90.0,
                "transform": None,
            },
        )()
        self.callback(image)


def scene_payload(*, capability: bool = True) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "scene": {
            "scene_id": "scene-acceptance-0001",
            "lease_token": "lease-acceptance-0001",
            "status": "running",
            "episode_id": 44,
            "ego_actor_id": 101,
            "map_name": "Town10HD_Opt",
            "spawn_index": 2,
            "route_mode": "free",
            "route": {},
            "destination": None,
            "control_mode": "manual",
            "traffic_count": 5,
            "walker_count": 5,
            "prop_actor_ids": [],
            "lease_expires_in_seconds": 4.0,
            "cleanup_guard_passed": None,
            "cleanup_errors": [],
            "capabilities": {"camera_pause_resume": capability},
        },
    }


def test_relay_pauses_and_resumes_same_official_sensor_subscription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(CompressedCameraRelay, "_STOP_DRAIN_SECONDS", 0.0)
    sensor = Sensor()
    relay = CompressedCameraRelay(
        sensor,
        jpeg_encoder=lambda image, quality: b"\xff\xd8" + bytes([image.frame]) + b"\xff\xd9",
    )
    relay.listen()
    sensor.emit(1)
    assert relay.wait(-1, 1.0)[0] == 0

    relay.pause()
    assert sensor.is_alive
    assert not sensor.is_listening
    assert relay.snapshot()["paused"] is True
    sensor.emit(2)
    with pytest.raises(Exception, match="timed out"):
        relay.wait(0, 0.01)

    relay.resume()
    assert sensor.is_alive
    assert sensor.is_listening
    assert relay.snapshot()["paused"] is False
    sensor.emit(3)
    sequence, payload, _ = relay.wait(0, 1.0)
    assert sequence == 1
    assert payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")
    assert sensor.listen_calls == 2
    assert sensor.stop_calls == 1

    relay.close()
    assert sensor.stop_calls == 2


def test_world_worker_camera_pause_resume_keeps_scene_and_ego() -> None:
    clock = FakeClock()
    library = FakeBlueprintLibrary()
    world = FakeWorld("Town10HD_Opt", library)
    tm = FakeTrafficManager(8000)
    client = FakeClient(world, library, tm)
    worker = WorldWorker(
        carla_loader=lambda: FakeCarla(client),
        route_planner_loader=lambda: FakePlanner,
        clock=clock,
        lease_seconds=5,
        start_monitor=False,
    )
    try:
        response = worker.prepare({})
        scene = worker._scene
        assert scene is not None
        ego_id = scene.ego.id

        class Relay:
            paused = False

            def pause(self) -> None:
                self.paused = True

            def resume(self) -> None:
                self.paused = False

            def snapshot(self) -> dict[str, Any]:
                return {"actor_id": 900, "paused": self.paused, "telemetry": {}}

            def close(self) -> None:
                return None

        relay = Relay()
        scene.camera_relay = relay  # type: ignore[assignment]
        scene_id = response["scene"]["scene_id"]
        token = response["scene"]["lease_token"]

        paused = worker.camera_pause(scene_id, {"lease_token": token})
        assert paused["scene"]["scene_id"] == scene_id
        assert relay.paused is True
        assert scene.ego.id == ego_id
        assert scene.status == "prepared"

        resumed = worker.camera_resume(scene_id, {"lease_token": token})
        assert resumed["scene"]["scene_id"] == scene_id
        assert relay.paused is False
        assert scene.ego.id == ego_id
    finally:
        worker.close()


def test_worker_route_and_client_use_authenticated_scene_actions() -> None:
    assert _SCENE_PATH.fullmatch("/v1/scenes/scene-acceptance-0001/camera_pause")
    assert _SCENE_PATH.fullmatch("/v1/scenes/scene-acceptance-0001/camera_resume")

    client = WorldWorkerClient(
        "http://127.0.0.1:8766",
        bearer_token="test-only-worker-token",
    )
    scene = WorldWorkerScene.from_response(scene_payload())
    with mock.patch.object(client, "_request", return_value=scene_payload()) as request:
        client.pause_camera(scene)
        request.assert_called_once_with(
            "POST",
            f"/v1/scenes/{scene.scene_id}/camera_pause",
            {"lease_token": scene.lease_token},
            timeout=None,
        )

    with mock.patch.object(client, "_request", return_value=scene_payload()) as request:
        client.resume_camera(scene)
        request.assert_called_once_with(
            "POST",
            f"/v1/scenes/{scene.scene_id}/camera_resume",
            {"lease_token": scene.lease_token},
            timeout=None,
        )


def test_old_worker_rejects_camera_fault_gate_without_network() -> None:
    client = WorldWorkerClient(
        "http://127.0.0.1:8766",
        bearer_token="test-only-worker-token",
    )
    scene = WorldWorkerScene.from_response(scene_payload(capability=False))
    with mock.patch.object(client, "_request", side_effect=AssertionError("network")):
        with pytest.raises(WorldWorkerError) as caught:
            client.pause_camera(scene)
    assert caught.value.code == "camera_pause_resume_unavailable"
