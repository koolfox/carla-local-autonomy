from __future__ import annotations

import json
import tempfile
import threading
import unittest
from dataclasses import replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest import mock

import carla_vision.operator.drive as drive_module
from carla_vision.operator.drive import DriveSession, DriveSessionManager
from carla_vision.operator.drive_contracts import DriveInput, DriveStartConfig
from carla_vision.operator.world_worker_client import (
    WorldWorkerCameraFrame,
    WorldWorkerClient,
    WorldWorkerError,
    WorldWorkerScene,
)

CARLA_HOST = "172.20.10.7"
CARLA_PORT = 2000
WORKER_TOKEN = "test-worker-token"


def _start_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": "worker-drive-test",
        "host": CARLA_HOST,
        "port": CARLA_PORT,
        "vehicle_blueprint": "vehicle.tesla.model3",
        "color": "255,0,0",
        "seed": 41,
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
    }
    payload.update(overrides)
    return payload


def _scene_payload(*, status: str = "prepared", control_mode: str = "manual") -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "status": status,
        "scene": {
            "scene_id": "scene-test-1",
            "lease_token": "scene-lease-secret",
            "status": status,
            "episode_id": 1234,
            "ego_actor_id": 77,
            "map_name": "Town10HD_Opt",
            "spawn_index": 5,
            "route_mode": "random_destination",
            "route": {"planned": True, "enforced": control_mode == "autopilot"},
            "destination": {"spawn_index": 9},
            "control_mode": control_mode,
            "traffic_count": 20,
            "walker_count": 10,
            "prop_actor_ids": [81, 82],
            "lease_expires_in_seconds": 2.0,
            "cleanup_guard_passed": True if status == "stopped" else None,
            "cleanup_errors": [],
            "capabilities": {"autopilot": True},
        },
    }


class _RecordingWorkerServer(ThreadingHTTPServer):
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.fail_path: str | None = None
        self.redirect_path: str | None = None
        self.redirect_location: str | None = None
        super().__init__(("127.0.0.1", 0), _RecordingWorkerHandler)


class _RecordingWorkerHandler(BaseHTTPRequestHandler):
    server: _RecordingWorkerServer

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _handle(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        body = json.loads(raw.decode("utf-8")) if raw else None
        self.server.requests.append(
            {
                "method": self.command,
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "body": body,
            }
        )
        if self.path == self.server.redirect_path:
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", str(self.server.redirect_location))
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == self.server.fail_path:
            self._json(
                HTTPStatus.CONFLICT,
                {
                    "schema_version": "1.0",
                    "error": {"code": "scene_busy", "message": "another scene is active"},
                },
            )
            return
        if self.path == "/v1/health":
            self._json(
                HTTPStatus.OK,
                {"schema_version": "1.0", "status": "ready", "carla_version": "0.9.16"},
            )
            return
        if self.path == "/v1/catalog":
            self._json(
                HTTPStatus.OK,
                {
                    "schema_version": "1.0",
                    "maps": [{"id": "Town10HD_Opt", "label": "Town 10 HD"}],
                    "vehicles": [],
                    "capabilities": {"map_reload": True, "autopilot": True},
                },
            )
            return
        if self.path.endswith("/camera/frame.jpg"):
            payload = b"\xff\xd8worker-jpeg\xff\xd9"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Camera-Sequence", "7")
            self.send_header("X-CARLA-Frame", "314")
            self.send_header("X-CARLA-Timestamp", "12.5")
            self.send_header("X-Camera-Width", "1920")
            self.send_header("X-Camera-Height", "1080")
            self.send_header("X-Camera-FOV", "65")
            self.send_header(
                "X-Camera-Transform",
                json.dumps(
                    {
                        "location": {"x": 1, "y": 2, "z": 3},
                        "rotation": {"pitch": 4, "yaw": 5, "roll": 6},
                    }
                ),
            )
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path.endswith("/camera") or self.path.endswith("/camera_orbit"):
            self._json(
                HTTPStatus.OK,
                {
                    "schema_version": "1.0",
                    "status": "running",
                    "scene_id": "scene-test-1",
                    "camera": {"actor_id": 91},
                },
            )
            return
        status = "active"
        control_mode = "manual"
        if self.path.endswith("/mode"):
            control_mode = str(body["control_mode"])
        if self.path.endswith("/stop"):
            status = "stopped"
        self._json(HTTPStatus.OK, _scene_payload(status=status, control_mode=control_mode))

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    do_GET = _handle
    do_POST = _handle


class WorldWorkerClientHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = _RecordingWorkerServer()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.client = WorldWorkerClient(f"http://{host}:{port}", WORKER_TOKEN)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)

    def test_full_scene_contract_uses_bearer_and_lease_without_leaking_into_paths(self) -> None:
        self.assertEqual(self.client.health()["status"], "ready")
        self.assertTrue(self.client.catalog()["capabilities"]["autopilot"])
        prepared = self.client.prepare_scene(
            {
                "map_name": "Town10HD_Opt",
                "weather_preset": "clear-day",
                "vehicle_blueprint": "vehicle.tesla.model3",
                "color": "255,0,0",
                "seed": 41,
                "traffic_count": 20,
                "walker_count": 10,
                "prop_preset": "none",
                "route_mode": "random_destination",
                "initial_control_mode": "manual",
            }
        )
        active = self.client.start_scene(prepared)
        active = self.client.heartbeat(active)
        active = self.client.control(
            active,
            {
                "sequence": 1,
                "throttle": 0.2,
                "steer": -0.1,
                "brake": 0.0,
                "hand_brake": False,
                "reverse": False,
            },
        )
        active = self.client.mode(active, "autopilot")
        self.assertEqual(active.control_mode, "autopilot")
        active = self.client.weather(active, "soft-rain-sunset")
        stopped = self.client.stop_scene(active)

        self.assertEqual(stopped.status, "stopped")
        self.assertEqual(stopped.ego_actor_id, 77)
        self.assertEqual(stopped.episode_id, 1234)
        self.assertTrue(stopped.cleanup_guard_passed)
        self.assertTrue(stopped.route["planned"])
        self.assertTrue(self.server.requests)
        self.assertTrue(
            all(row["authorization"] == f"Bearer {WORKER_TOKEN}" for row in self.server.requests)
        )
        self.assertTrue(
            all("scene-lease-secret" not in row["path"] for row in self.server.requests)
        )
        scene_requests = [row for row in self.server.requests if "/scene-test-1/" in row["path"]]
        self.assertTrue(scene_requests)
        self.assertTrue(
            all(row["body"]["lease_token"] == "scene-lease-secret" for row in scene_requests)
        )

    def test_worker_error_envelope_preserves_status_and_code(self) -> None:
        self.server.fail_path = "/v1/scenes/prepare"

        with self.assertRaisesRegex(WorldWorkerError, "another scene is active") as raised:
            self.client.prepare_scene(
                {
                    "map_name": "current",
                    "weather_preset": "keep",
                    "vehicle_blueprint": "vehicle.tesla.model3",
                    "color": None,
                    "seed": 0,
                    "traffic_count": 0,
                    "walker_count": 0,
                    "prop_preset": "none",
                    "route_mode": "free",
                    "initial_control_mode": "manual",
                }
            )

        self.assertEqual(raised.exception.status, 409)
        self.assertEqual(raised.exception.code, "scene_busy")

    def test_camera_relay_uses_authenticated_jpeg_transport(self) -> None:
        scene = WorldWorkerScene.from_response(_scene_payload())

        started = self.client.start_camera(
            scene,
            mode="garage",
            width=1920,
            height=1080,
            fps=10.0,
            fov=65.0,
        )
        self.client.orbit_camera(scene, yaw=90.0, pitch=-8.0, distance=6.0)
        frame = self.client.camera_frame(scene, after_sequence=-1, timeout=1.0)

        self.assertEqual(started["camera"]["actor_id"], 91)
        self.assertIsInstance(frame, WorldWorkerCameraFrame)
        self.assertEqual((frame.sequence, frame.frame), (7, 314))
        self.assertEqual((frame.width, frame.height, frame.fov), (1920, 1080, 65.0))
        self.assertEqual(frame.transform, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        self.assertEqual(frame.jpeg, b"\xff\xd8worker-jpeg\xff\xd9")
        camera_requests = [row for row in self.server.requests if "/camera" in row["path"]]
        self.assertEqual(len(camera_requests), 3)
        self.assertTrue(
            all(row["authorization"] == f"Bearer {WORKER_TOKEN}" for row in camera_requests)
        )
        self.assertEqual(camera_requests[0]["body"]["width"], 1920)
        self.assertEqual(camera_requests[1]["body"]["yaw"], 90.0)

    def test_url_and_token_validation_rejects_unsafe_configuration(self) -> None:
        invalid = (
            ("ftp://worker:8766", "token"),
            ("http://user:pass@worker:8766", "token"),
            ("http://worker:8766/v1", "token"),
            ("http://worker:8766", "line\nbreak"),
        )
        for url, token in invalid:
            with self.subTest(url=url), self.assertRaises(ValueError):
                WorldWorkerClient(url, token)

    def test_scene_requires_authoritative_integer_episode_id(self) -> None:
        payload = _scene_payload()
        payload["scene"]["episode_id"] = ["world", "1234", "Town10HD_Opt"]

        with self.assertRaisesRegex(WorldWorkerError, "scene.episode_id"):
            WorldWorkerScene.from_response(payload)

    def test_redirect_is_rejected_without_forwarding_bearer(self) -> None:
        redirect_target = _RecordingWorkerServer()
        target_thread = threading.Thread(target=redirect_target.serve_forever, daemon=True)
        target_thread.start()
        try:
            target_host, target_port = redirect_target.server_address
            self.server.redirect_path = "/v1/health"
            self.server.redirect_location = f"http://{target_host}:{target_port}/v1/health"

            with self.assertRaisesRegex(WorldWorkerError, "redirects are not allowed") as raised:
                self.client.health()

            self.assertEqual(raised.exception.status, 302)
            self.assertEqual(raised.exception.code, "world_worker_redirect")
            self.assertEqual(redirect_target.requests, [])
        finally:
            redirect_target.shutdown()
            redirect_target.server_close()
            target_thread.join(timeout=2.0)


class WorkerDriveContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name).resolve()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def config(self, *, worker: bool, **overrides: object) -> DriveStartConfig:
        return DriveStartConfig.from_mapping(
            _start_payload(**overrides),
            workspace=self.workspace,
            expected_host=CARLA_HOST,
            expected_port=CARLA_PORT,
            world_worker_configured=worker,
        )

    def test_worker_fields_are_normalized_and_manifest_contains_no_worker_secret(self) -> None:
        config = self.config(
            worker=True,
            map_name="Town10HD_Opt",
            traffic_count=24,
            walker_count=12,
            route_mode="random_destination",
            initial_control_mode="autopilot",
        )

        self.assertTrue(config.world_worker_enabled)
        self.assertEqual(config.map_name, "Town10HD_Opt")
        self.assertEqual(config.traffic_count, 24)
        self.assertEqual(config.walker_count, 12)
        self.assertEqual(config.route_mode, "random_destination")
        self.assertEqual(config.initial_control_mode, "autopilot")
        manifest = config.manifest_config()
        self.assertEqual(manifest["control_mode"], "world_worker_autopilot")
        serialized = json.dumps(manifest)
        self.assertNotIn(WORKER_TOKEN, serialized)
        self.assertNotIn("lease_token", serialized)

    def test_legacy_path_accepts_only_exact_world_defaults(self) -> None:
        defaults = self.config(worker=False)
        self.assertFalse(defaults.world_worker_enabled)
        self.assertEqual(defaults.map_name, "current")
        invalid = (
            {"map_name": "Town03"},
            {"traffic_count": 1},
            {"walker_count": 1},
            {"route_mode": "random_destination"},
            {"initial_control_mode": "autopilot"},
        )
        for overrides in invalid:
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, "configured World Worker is required"),
            ):
                self.config(worker=False, **overrides)


class _FakeWorker:
    def __init__(self, *, reachable: bool = True) -> None:
        self.reachable = reachable
        self.modes: list[str] = []
        self.controls: list[dict[str, Any]] = []
        self.cameras: list[dict[str, Any]] = []
        self.events: list[str] = []
        self.stop_after_control: DriveSession | None = None

    def health(self) -> dict[str, Any]:
        if not self.reachable:
            return {"status": "unavailable", "ready": False}
        return {
            "status": "ready",
            "ready": True,
            "carla": {"connected": True, "server_version": "0.9.16"},
        }

    def catalog(self) -> dict[str, Any]:
        return {
            "maps": [{"id": "Town03", "label": "Town 03"}],
            "vehicles": [{"id": "vehicle.tesla.model3", "label": "Tesla", "colors": []}],
            "carla": {
                "connected": True,
                "server_version": "0.9.16",
                "current_map": "Town03",
            },
            "spawn_count": 11,
            "capabilities": {
                "map_reload": True,
                "random_route": True,
                "traffic_manager": True,
                "walkers": True,
                "autopilot": True,
            },
        }

    def mode(self, scene: WorldWorkerScene, mode: str) -> WorldWorkerScene:
        self.modes.append(mode)
        return WorldWorkerScene(
            **{**scene.__dict__, "control_mode": mode},
        )

    def start_scene(self, scene: WorldWorkerScene) -> WorldWorkerScene:
        self.events.append("worker_start")
        return replace(scene, status="running")

    def control(
        self,
        scene: WorldWorkerScene,
        payload: dict[str, Any],
    ) -> WorldWorkerScene:
        self.controls.append(dict(payload))
        self.events.append("worker_control")
        if self.stop_after_control is not None:
            self.stop_after_control._stop_event.set()
        return scene

    def heartbeat(self, scene: WorldWorkerScene) -> WorldWorkerScene:
        self.events.append("worker_heartbeat")
        return scene

    def start_camera(self, scene: WorldWorkerScene, **payload: Any) -> dict[str, Any]:
        del scene
        self.cameras.append(dict(payload))
        self.events.append("camera_start")
        return {"camera": {"actor_id": 91}}

    def stop_scene(self, scene: WorldWorkerScene) -> WorldWorkerScene:
        self.events.append("worker_stop")
        return replace(scene, status="stopped", lease_expires_in_seconds=0.0)


class _CatalogRpc:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def __enter__(self) -> "_CatalogRpc":
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def value_call(self, method: str) -> Any:
        values = {
            "version": "0.9.16",
            "get_map_info": ["/Game/Carla/Maps/Town10HD_Opt", [[[0, 0, 0], [0, 0, 0]]]],
            "get_available_maps": ["/Game/Carla/Maps/Town10HD_Opt"],
            "get_actor_definitions": [],
        }
        return values[method]


class _CapturedSession:
    instances: list["_CapturedSession"] = []

    def __init__(
        self,
        config: DriveStartConfig,
        *,
        workspace: Path,
        world_worker: _FakeWorker | None = None,
    ) -> None:
        self.config = config
        self.workspace = workspace
        self.world_worker = world_worker
        self.session_id = config.run_id
        self.status = "starting"
        self.__class__.instances.append(self)

    def start(self) -> None:
        self.status = "running"

    def snapshot(self) -> dict[str, Any]:
        return {"status": self.status, "session_id": self.session_id}


class WorkerManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name).resolve()
        _CapturedSession.instances.clear()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def manager(self, worker: _FakeWorker) -> DriveSessionManager:
        return DriveSessionManager(
            workspace=self.workspace,
            carla_host=CARLA_HOST,
            carla_port=CARLA_PORT,
            session_factory=_CapturedSession,
            rpc_factory=_CatalogRpc,
            world_worker=worker,  # type: ignore[arg-type]
        )

    def test_catalog_merges_live_worker_maps_and_capabilities(self) -> None:
        catalog = self.manager(_FakeWorker()).catalog()

        self.assertTrue(catalog["world_worker"]["connected"])
        self.assertTrue(catalog["capabilities"]["native_worker"])
        self.assertTrue(catalog["capabilities"]["map_reload"])
        self.assertTrue(catalog["capabilities"]["autopilot"])
        self.assertEqual(catalog["maps"], [{"id": "Town03", "label": "Town 03"}])
        self.assertEqual(catalog["map"], "Town03")

    def test_unreachable_worker_falls_back_only_for_legacy_defaults(self) -> None:
        manager = self.manager(_FakeWorker(reachable=False))

        state = manager.start(_start_payload())
        self.assertEqual(state["status"], "running")
        self.assertIsNone(_CapturedSession.instances[-1].world_worker)
        self.assertFalse(_CapturedSession.instances[-1].config.world_worker_enabled)

        _CapturedSession.instances[-1].status = "success"
        with self.assertRaisesRegex(ValueError, "configured World Worker is required"):
            manager.start(_start_payload(traffic_count=1, run_id="worker-required"))

    def test_reachable_worker_is_injected_into_session(self) -> None:
        worker = _FakeWorker()
        manager = self.manager(worker)

        manager.start(
            _start_payload(
                map_name="Town03",
                route_mode="random_destination",
                initial_control_mode="autopilot",
            )
        )

        session = _CapturedSession.instances[-1]
        self.assertIs(session.world_worker, worker)
        self.assertTrue(session.config.world_worker_enabled)


class WorkerModeArbitrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name).resolve()
        self.worker = _FakeWorker()
        config = DriveStartConfig.from_mapping(
            _start_payload(initial_control_mode="autopilot"),
            workspace=self.workspace,
            expected_host=CARLA_HOST,
            expected_port=CARLA_PORT,
            world_worker_configured=True,
        )
        self.session = DriveSession(
            config,
            workspace=self.workspace,
            world_worker=self.worker,  # type: ignore[arg-type]
        )
        self.scene = WorldWorkerScene.from_response(
            _scene_payload(status="active", control_mode="autopilot")
        )
        with self.session._lock:
            self.session._status = "running"
            self.session._worker_scene = self.scene
            self.session._worker_scene_id = self.scene.scene_id

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_browser_input_is_ignored_until_mode_is_manual(self) -> None:
        control = DriveInput.from_mapping(
            {
                "session_id": "worker-drive-test",
                "sequence": 1,
                "throttle": 0.8,
                "steer": 0.2,
                "brake": 0.0,
                "hand_brake": False,
                "reverse": False,
            }
        )
        self.session.submit_control(control)
        with self.session._lock:
            self.assertIsNone(self.session._last_input)

        state = self.session.request_mode("manual")
        self.assertEqual(state["control_mode"], "manual")
        self.assertEqual(self.worker.modes, ["manual"])
        self.assertEqual(self.session._mode_history[-1]["reason"], "operator_request")

        self.session.submit_control(
            DriveInput.from_mapping(
                {
                    **control.__dict__,
                    "sequence": 2,
                }
            )
        )
        with self.session._lock:
            self.assertIsNotNone(self.session._last_input)

    def test_emergency_leaves_autopilot_then_sends_full_brake(self) -> None:
        state = self.session.emergency_stop()

        self.assertEqual(state["control_mode"], "manual")
        self.assertEqual(self.worker.modes, ["manual"])
        self.assertEqual(len(self.worker.controls), 1)
        self.assertEqual(self.worker.controls[0]["throttle"], 0.0)
        self.assertEqual(self.worker.controls[0]["brake"], 1.0)
        self.assertTrue(state["deadman_active"])
        with self.assertRaisesRegex(RuntimeError, "latched"):
            self.session.request_mode("autopilot")


class _ExecuteRpc:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[str] = []
        self.closed = False

    def value_call(self, method: str, *args: Any) -> Any:
        del args
        self.calls.append(method)
        if method == "version":
            return "0.9.16"
        if method == "get_map_info":
            return ["/Game/Carla/Maps/Town10HD_Opt", [[[0, 0, 0], [0, 0, 0]]]]
        raise AssertionError(f"unexpected raw RPC method {method}")

    def episode_id(self) -> int:
        return 1234

    def actor(self, actor_id: int) -> list[Any] | None:
        if actor_id == 77:
            return [77, None, [None, "vehicle.tesla.model3"]]
        if actor_id == 91:
            return None
        raise AssertionError(f"unexpected actor {actor_id}")

    def telemetry(self, actor_id: int) -> Any:
        self.assert_ego(actor_id)
        return type(
            "Telemetry",
            (),
            {"speed": 0.0, "gear": 0, "throttle": 0.0, "steer": 0.0, "brake": 1.0},
        )()

    def assert_ego(self, actor_id: int) -> None:
        if actor_id != 77:
            raise AssertionError(f"unexpected ego {actor_id}")

    def destroy_actor(self, actor_id: int) -> None:
        raise AssertionError(f"worker-owned cleanup should make camera {actor_id} absent")

    def close(self) -> None:
        self.closed = True


class _ExecuteStream:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def wait_for_frame(self, timeout: float) -> None:
        del timeout

    def latest(self) -> None:
        return None

    def close(self) -> None:
        self.events.append("camera_close")


class _ExecuteTracker:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True)

    def artifact_path(self, name: str) -> Path:
        return self.root / name

    def register_artifact(self, path: Path, **kwargs: Any) -> None:
        del path, kwargs


class WorkerExecuteOwnershipTests(unittest.TestCase):
    def test_worker_camera_path_avoids_raw_bgra_stream_and_raw_actuator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            config = DriveStartConfig.from_mapping(
                _start_payload(),
                workspace=workspace,
                expected_host=CARLA_HOST,
                expected_port=CARLA_PORT,
                world_worker_configured=True,
            )
            worker = _FakeWorker()
            session = DriveSession(
                config,
                workspace=workspace,
                world_worker=worker,  # type: ignore[arg-type]
            )
            scene = WorldWorkerScene.from_response(_scene_payload(status="prepared"))
            scene = replace(
                scene,
                capabilities={**scene.capabilities, "compressed_camera_relay": True},
            )
            with session._lock:
                session._worker_scene = scene
                session._worker_scene_id = scene.scene_id
            worker.stop_after_control = session
            rpc = _ExecuteRpc(worker.events)
            stream = _ExecuteStream(worker.events)
            tracker = _ExecuteTracker(workspace / "artifacts")

            with (
                mock.patch.object(drive_module, "CarlaRpc", return_value=rpc),
                mock.patch.object(
                    drive_module,
                    "spawn_front_camera",
                    side_effect=AssertionError(
                        "compressed Worker path must not open CARLA raw BGRA streaming"
                    ),
                ),
                mock.patch.object(
                    drive_module,
                    "WorldWorkerCameraStream",
                    return_value=stream,
                ),
                mock.patch.object(
                    drive_module,
                    "SafeActuator",
                    side_effect=AssertionError("raw actuator must not exist in worker mode"),
                ),
                mock.patch.object(
                    drive_module,
                    "_spawn_vehicle",
                    side_effect=AssertionError("raw ego spawn must not run in worker mode"),
                ),
                mock.patch.object(
                    drive_module,
                    "_spawn_props",
                    side_effect=AssertionError("raw prop spawn must not run in worker mode"),
                ),
            ):
                session._execute(tracker)  # type: ignore[arg-type]

            self.assertEqual(len(worker.controls), 1)
            self.assertEqual(worker.controls[0]["brake"], 1.0)
            self.assertEqual(worker.cameras[0]["width"], 640)
            self.assertEqual(worker.cameras[0]["mode"], "drive")
            self.assertNotIn("get_weather_parameters", rpc.calls)
            self.assertNotIn("set_weather_parameters", rpc.calls)
            self.assertLess(worker.events.index("worker_stop"), worker.events.index("camera_close"))
            self.assertTrue(session._worker_scene_stopped)
            self.assertTrue(rpc.closed)


if __name__ == "__main__":
    unittest.main()
