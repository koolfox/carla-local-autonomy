from __future__ import annotations

import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest import mock

import pytest
from test_world_worker_client import _scene_payload

from carla_vision.operator.world_worker_client import (
    WorldWorkerCameraStream,
    WorldWorkerClient,
    WorldWorkerError,
    WorldWorkerScene,
)


class BlockingCameraServer(ThreadingHTTPServer):
    def __init__(self, *, delay_headers: bool = False) -> None:
        self.delay_headers = delay_headers
        self.requested = threading.Event()
        self.release = threading.Event()
        self.paths: list[str] = []
        super().__init__(("127.0.0.1", 0), BlockingCameraHandler)


class BlockingCameraHandler(BaseHTTPRequestHandler):
    server: BlockingCameraServer

    def log_message(self, *_: Any) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802
        self.server.paths.append(self.path)
        self.server.requested.set()
        if self.server.delay_headers:
            self.server.release.wait(5.0)
        self.send_response(200)
        if self.path.endswith("stream.mjpg"):
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=camera")
        else:
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", "100")
        try:
            self.end_headers()
            self.wfile.flush()
            self.server.release.wait(5.0)
        except (BrokenPipeError, ConnectionResetError):
            pass


def camera_scene(*, persistent: bool = True) -> WorldWorkerScene:
    scene = WorldWorkerScene.from_response(_scene_payload())
    return replace(scene, capabilities={"persistent_mjpeg_camera_relay": persistent})


@pytest.mark.parametrize("persistent", [True, False])
def test_close_interrupts_blocked_response_and_client_is_reusable(persistent: bool) -> None:
    server = BlockingCameraServer()
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    host, port = server.server_address
    client = WorldWorkerClient(f"http://{host}:{port}", "local-test-token")
    streams: list[WorldWorkerCameraStream] = []
    try:
        for _ in range(2):
            stream = WorldWorkerCameraStream(
                client, camera_scene(persistent=persistent), timeout=15
            )
            streams.append(stream)
            with stream._condition:
                assert stream._condition.wait_for(
                    lambda current=stream: current._response is not None, timeout=1
                )
            if len(streams) > 1:
                streams[0].close()
                assert stream._thread.is_alive()
            started = time.monotonic()
            stream.close()
            assert time.monotonic() - started < 0.75
            assert not stream._thread.is_alive()
            assert stream._error is None
            with pytest.raises(WorldWorkerError, match="closed"):
                stream.wait_for_frame(timeout=15)
            stream.close()  # Repeated close is harmless.
        assert len(server.paths) == 2  # Cancelled streams never reconnect.
    finally:
        server.release.set()
        for stream in streams:
            stream.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def test_close_is_bounded_before_headers_and_rejects_late_response() -> None:
    server = BlockingCameraServer(delay_headers=True)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    host, port = server.server_address
    client = WorldWorkerClient(f"http://{host}:{port}", "local-test-token")
    stream = WorldWorkerCameraStream(client, camera_scene(), timeout=15)
    try:
        assert server.requested.wait(1)
        started = time.monotonic()
        stream.close()
        assert time.monotonic() - started < 1.5
        server.release.set()
        stream._thread.join(timeout=1)
        assert not stream._thread.is_alive()
        assert stream._error is None
        assert len(server.paths) == 1
    finally:
        server.release.set()
        stream.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def configured_payload() -> dict[str, Any]:
    return {
        "map_name": "current",
        "weather_preset": "clear-day",
        "vehicle_blueprint": "vehicle.tesla.model3",
        "color": "255,0,0",
        "seed": 41,
        "traffic_count": 20,
        "walker_count": 10,
        "prop_preset": "none",
        "route_mode": "free",
        "initial_control_mode": "manual",
        "pedestrian_crossing_factor": 0.8,
        "speed_difference_percent": -10.0,
        "following_distance_metres": 7.0,
    }


def test_configure_scene_uses_existing_lease_and_lifecycle_timeout() -> None:
    client = WorldWorkerClient("http://127.0.0.1:8766", "local-test-token")
    response = _scene_payload()
    response["scene"]["capabilities"]["prepared_scene_reconfigure"] = True
    scene = WorldWorkerScene.from_response(response)
    with mock.patch.object(client, "_request", return_value=response) as request:
        configured = client.configure_scene(scene, configured_payload())
    assert configured.scene_id == scene.scene_id
    request.assert_called_once_with(
        "POST",
        f"/v1/scenes/{scene.scene_id}/configure",
        {"lease_token": scene.lease_token, **configured_payload()},
        timeout=120.0,
    )


def test_configure_scene_rejects_unsupported_worker_and_unbounded_payload_before_request() -> None:
    client = WorldWorkerClient("http://127.0.0.1:8766", "local-test-token")
    scene = camera_scene()
    with mock.patch.object(client, "_request") as request:
        with pytest.raises(WorldWorkerError, match="outdated"):
            client.configure_scene(scene, configured_payload())
        with pytest.raises(ValueError, match="invalid fields"):
            client.configure_scene(scene, {**configured_payload(), "lease_token": "replacement"})
        with pytest.raises(ValueError, match="invalid fields"):
            client.configure_scene(scene, {"traffic_count": 20})
        request.assert_not_called()


def test_configure_scene_rejects_changed_scene_identity() -> None:
    client = WorldWorkerClient("http://127.0.0.1:8766", "local-test-token")
    scene = replace(camera_scene(), capabilities={"prepared_scene_reconfigure": True})
    response = _scene_payload()
    response["scene"]["scene_id"] = "unexpected-scene"
    with mock.patch.object(client, "_request", return_value=response):
        with pytest.raises(WorldWorkerError, match="changed the active scene_id"):
            client.configure_scene(scene, configured_payload())
