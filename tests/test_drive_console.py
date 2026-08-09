from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from carla_vision.operator.drive import DriveSession, DriveSessionManager
from carla_vision.operator.drive_contracts import (
    DriveInput,
    DriveStartConfig,
    weather_payload,
)
from carla_vision.operator.server import create_server

CARLA_HOST = "172.20.10.7"
CARLA_PORT = 2000


def valid_start(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": "drive-console-test",
        "host": CARLA_HOST,
        "port": CARLA_PORT,
        "vehicle_blueprint": "vehicle.tesla.model3",
        "color": "255,0,0",
        "seed": 20260809,
        "weather_preset": "clear-day",
        "prop_preset": "none",
        "detector_enabled": True,
        "detector": "rtdetr",
        "weights": "models/detector.pt",
        "device": "cpu",
        "image_size": 640,
        "confidence": 0.35,
        "resolution": "640x384",
        "camera_fps": 10.0,
        "camera_fov": 90.0,
        "record_video": True,
        "spectator_follow": False,
    }
    payload.update(overrides)
    return payload


def valid_control(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "session_id": "drive-console-test",
        "sequence": 1,
        "throttle": 0.4,
        "steer": -0.25,
        "brake": 0.0,
        "hand_brake": False,
        "reverse": False,
    }
    payload.update(overrides)
    return payload


class _WorkspaceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.workspace = self.root / "workspace"
        (self.workspace / "models").mkdir(parents=True)
        (self.workspace / "models" / "detector.pt").write_bytes(b"test weights")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def config(self, **overrides: object) -> DriveStartConfig:
        return DriveStartConfig.from_mapping(
            valid_start(**overrides),
            workspace=self.workspace,
            expected_host=CARLA_HOST,
            expected_port=CARLA_PORT,
        )


class WeatherPayloadTests(unittest.TestCase):
    def test_weather_serialization_uses_exact_carla_rpc_field_order(self) -> None:
        self.assertEqual(
            weather_payload("soft-rain-sunset"),
            [
                70.0,
                20.0,
                25.0,
                35.0,
                250.0,
                8.0,
                4.0,
                40.0,
                0.5,  # fog_falloff precedes wetness in the CARLA RPC tuple
                55.0,
                1.0,
                0.08,
                0.0331,
                0.0,
            ],
        )

    def test_unknown_weather_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown weather preset"):
            weather_payload("not-a-weather")


class DriveStartConfigTests(_WorkspaceTestCase):
    def test_valid_start_is_normalized_and_keeps_model_advisory(self) -> None:
        config = self.config()

        self.assertEqual(config.run_id, "drive-console-test")
        self.assertEqual(config.weights, self.workspace / "models" / "detector.pt")
        self.assertEqual((config.width, config.height), (640, 384))
        self.assertEqual(config.prop_preset, "none")
        self.assertTrue(config.record_video)
        self.assertFalse(config.spectator_follow)
        self.assertFalse(config.manifest_config()["model_output_actuated"])

    def test_detector_can_be_disabled_without_resolving_weights(self) -> None:
        config = self.config(detector_enabled=False, weights="")

        self.assertIsNone(config.weights)
        self.assertFalse(config.detector_enabled)

    def test_endpoint_must_equal_operator_endpoint(self) -> None:
        for field, value in (("host", "127.0.0.1"), ("port", 2001)):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(
                    ValueError,
                    "endpoint must match",
                ),
            ):
                self.config(**{field: value})

    def test_weights_cannot_escape_workspace(self) -> None:
        outside = self.root / "outside.pt"
        outside.write_bytes(b"outside")

        with self.assertRaises(ValueError):
            self.config(weights=str(outside))

        link = self.workspace / "models" / "linked.pt"
        link.symlink_to(outside)
        with self.assertRaises(ValueError):
            self.config(weights="models/linked.pt")

    def test_invalid_prop_preset_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown prop preset"):
            self.config(prop_preset="surprise-road-clutter")

    def test_invalid_start_shapes_are_rejected(self) -> None:
        invalid_cases = (
            ({"run_id": "../escape"}, "run_id"),
            ({"vehicle_blueprint": "sensor.camera.rgb"}, "vehicle_blueprint"),
            ({"resolution": "640-by-384"}, "resolution"),
            ({"camera_fps": 0.0}, "camera_fps"),
            ({"detector_enabled": 1}, "detector_enabled"),
        )
        for overrides, message in invalid_cases:
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(
                    (TypeError, ValueError),
                    message,
                ),
            ):
                self.config(**overrides)


class DriveInputTests(unittest.TestCase):
    def test_bounds_are_strict(self) -> None:
        invalid_cases = (
            ({"sequence": -1}, "sequence"),
            ({"throttle": 1.01}, "throttle"),
            ({"steer": -1.01}, "steer"),
            ({"brake": -0.01}, "brake"),
            ({"reverse": 1}, "reverse"),
        )
        for overrides, message in invalid_cases:
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(
                    (TypeError, ValueError),
                    message,
                ),
            ):
                DriveInput.from_mapping(valid_control(**overrides))

    def test_reverse_is_preserved_and_throttle_is_capped(self) -> None:
        drive_input = DriveInput.from_mapping(valid_control(throttle=0.9, steer=0.5, reverse=True))

        command = drive_input.command(max_throttle=0.55)

        self.assertEqual(command.throttle, 0.55)
        self.assertEqual(command.steer, 0.5)
        self.assertTrue(command.reverse)
        self.assertEqual(command.brake, 0.0)

    def test_brake_and_hand_brake_each_arbitrate_throttle_to_zero(self) -> None:
        for overrides in ({"brake": 0.2}, {"hand_brake": True}):
            with self.subTest(overrides=overrides):
                drive_input = DriveInput.from_mapping(valid_control(throttle=0.8, **overrides))
                command = drive_input.command(max_throttle=0.55)
                self.assertEqual(command.throttle, 0.0)


class DriveSessionControlTests(_WorkspaceTestCase):
    def session(self) -> DriveSession:
        return DriveSession(self.config(), workspace=self.workspace)

    def test_fresh_control_is_manual_and_stale_control_uses_deadman(self) -> None:
        session = self.session()
        drive_input = DriveInput.from_mapping(valid_control(throttle=0.4, steer=-0.3, reverse=True))
        with session._lock:
            session._last_input = (drive_input, 100.0)

        fresh, source, age = session._command(100.2, camera_stale=False)
        self.assertEqual(source, "browser_manual")
        self.assertAlmostEqual(age or 0.0, 0.2)
        self.assertEqual(fresh.throttle, 0.4)
        self.assertTrue(fresh.reverse)
        self.assertFalse(session.snapshot()["deadman_active"])

        stale, source, age = session._command(100.41, camera_stale=False)
        self.assertEqual(source, "browser_deadman")
        self.assertAlmostEqual(age or 0.0, 0.41)
        self.assertEqual(stale.throttle, 0.0)
        self.assertEqual(stale.brake, 1.0)
        self.assertEqual(stale.steer, -0.3)
        self.assertTrue(session.snapshot()["deadman_active"])

    def test_camera_staleness_overrides_fresh_browser_control(self) -> None:
        session = self.session()
        drive_input = DriveInput.from_mapping(valid_control())
        with session._lock:
            session._last_input = (drive_input, 100.0)

        command, source, _ = session._command(100.1, camera_stale=True)

        self.assertEqual(source, "camera_deadman")
        self.assertEqual(command.brake, 1.0)
        self.assertEqual(command.throttle, 0.0)

    def test_reverse_is_interlocked_until_vehicle_is_nearly_stopped(self) -> None:
        session = self.session()
        reverse = DriveInput.from_mapping(valid_control(reverse=True, throttle=0.4))
        with session._lock:
            session._last_input = (reverse, 100.0)
            session._telemetry["speed"] = 1.2

        command, source, _ = session._command(100.1, camera_stale=False)

        self.assertEqual(source, "reverse_interlock")
        self.assertEqual(command.throttle, 0.0)
        self.assertEqual(command.brake, 1.0)
        self.assertFalse(command.reverse)

    def test_control_sequences_must_increase_monotonically(self) -> None:
        session = self.session()
        session.submit_control(DriveInput.from_mapping(valid_control(sequence=7)))

        for sequence in (7, 6):
            with (
                self.subTest(sequence=sequence),
                self.assertRaisesRegex(
                    ValueError,
                    "sequence must be newer",
                ),
            ):
                session.submit_control(DriveInput.from_mapping(valid_control(sequence=sequence)))

        snapshot = session.submit_control(DriveInput.from_mapping(valid_control(sequence=8)))
        self.assertEqual(snapshot["status"], "starting")

    def test_emergency_and_stop_requests_are_idempotent(self) -> None:
        session = self.session()

        first_emergency = session.emergency_stop()
        second_emergency = session.emergency_stop()
        self.assertEqual(first_emergency["control_source"], "emergency_stop")
        self.assertEqual(second_emergency["control_source"], "emergency_stop")
        self.assertTrue(second_emergency["deadman_active"])

        first_stop = session.request_stop("operator_stop")
        second_stop = session.request_stop("operator_stop")
        self.assertEqual(first_stop["status"], "stopping")
        self.assertEqual(second_stop["status"], "stopping")
        self.assertEqual(second_stop["stop_reason"], "operator_stop")


class _FakeSession:
    def __init__(self, config: DriveStartConfig, *, workspace: Path) -> None:
        self.config = config
        self.workspace = workspace
        self.session_id = config.run_id
        self.status = "starting"
        self.started = False

    def start(self) -> None:
        self.started = True
        self.status = "running"

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "session_id": self.session_id,
            "run_id": self.session_id,
            "status": self.status,
        }


class _FakeRpc:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs
        self.closed = False

    def __enter__(self) -> _FakeRpc:
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True

    def value_call(self, method: str) -> Any:
        calls: dict[str, Any] = {
            "version": "0.9.16",
            "get_map_info": [
                "/Game/Carla/Maps/Town10HD_Opt",
                [[[-1.0, 2.0, 0.5], [0.0, 90.0, 0.0]]],
            ],
            "get_available_maps": [
                "/Game/Carla/Maps/Town03",
                "/Game/Carla/Maps/Town10HD_Opt",
            ],
            "get_actor_definitions": [
                [
                    1,
                    "vehicle.tesla.model3",
                    ["vehicle"],
                    [
                        ["role_name", 0, "", [], True],
                        ["color", 0, "255,255,255", ["255,0,0", "0,0,255"], True],
                    ],
                ],
                [1, "sensor.camera.rgb", ["sensor"], []],
            ],
        }
        return calls[method]


class DriveSessionManagerTests(_WorkspaceTestCase):
    def manager(self, **overrides: object) -> DriveSessionManager:
        kwargs: dict[str, object] = {
            "workspace": self.workspace,
            "carla_host": CARLA_HOST,
            "carla_port": CARLA_PORT,
            "session_factory": _FakeSession,
        }
        kwargs.update(overrides)
        return DriveSessionManager(**kwargs)

    def test_only_one_active_session_can_start(self) -> None:
        sessions: list[_FakeSession] = []

        def factory(config: DriveStartConfig, *, workspace: Path) -> _FakeSession:
            session = _FakeSession(config, workspace=workspace)
            sessions.append(session)
            return session

        manager = self.manager(session_factory=factory)
        first = manager.start(valid_start())

        self.assertEqual(first["status"], "running")
        self.assertTrue(sessions[0].started)
        with self.assertRaisesRegex(RuntimeError, "already active"):
            manager.start(valid_start(run_id="second-drive"))
        self.assertEqual(len(sessions), 1)

        sessions[0].status = "success"
        second = manager.start(valid_start(run_id="second-drive"))
        self.assertEqual(second["session_id"], "second-drive")
        self.assertEqual(len(sessions), 2)

    def test_catalog_uses_rpc_data_and_exposes_only_vehicle_definitions(self) -> None:
        manager = self.manager(rpc_factory=_FakeRpc)

        catalog = manager.catalog()

        self.assertTrue(catalog["connected"])
        self.assertEqual(catalog["server_version"], "0.9.16")
        self.assertEqual(catalog["map"], "Town10HD_Opt")
        self.assertEqual(catalog["spawn_count"], 1)
        self.assertEqual(
            catalog["vehicles"],
            [
                {
                    "id": "vehicle.tesla.model3",
                    "label": "Tesla · Model3",
                    "colors": ["255,0,0", "0,0,255"],
                }
            ],
        )
        self.assertFalse(catalog["capabilities"]["autopilot"])
        self.assertFalse(catalog["capabilities"]["map_reload"])


class _FakeHttpDriveManager:
    def __init__(self) -> None:
        self.started: dict[str, Any] | None = None
        self.controls: list[dict[str, Any]] = []
        self.shutdown_called = False

    def catalog(self) -> dict[str, Any]:
        return {"connected": True, "vehicles": [], "capabilities": {}}

    def state(self) -> dict[str, Any]:
        return {"status": "idle", "session_id": None}

    def frame(self, view: str) -> tuple[int, bytes]:
        if view != "raw":
            raise ValueError("fake only serves raw")
        return 7, b"fake-jpeg"

    def start(self, raw: dict[str, Any]) -> dict[str, Any]:
        self.started = dict(raw)
        return {"status": "starting", "session_id": str(raw["run_id"])}

    def control(self, raw: dict[str, Any]) -> dict[str, Any]:
        self.controls.append(dict(raw))
        return {"status": "running", "session_id": raw["session_id"]}

    def weather(self, raw: dict[str, Any]) -> dict[str, Any]:
        return {"status": "running", **raw}

    def emergency_stop(self, raw: dict[str, Any]) -> dict[str, Any]:
        return {"status": "running", "control_source": "emergency_stop", **raw}

    def stop(self, raw: dict[str, Any]) -> dict[str, Any]:
        return {"status": "success", **raw}

    def shutdown(self) -> None:
        self.shutdown_called = True


class DriveHttpTests(_WorkspaceTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.server = create_server(
            workspace=self.workspace,
            bind="127.0.0.1",
            port=0,
            sessions_root="operator_sessions",
            carla_host=CARLA_HOST,
            carla_port=CARLA_PORT,
        )
        self.fake = _FakeHttpDriveManager()
        self.server.application.drive = self.fake  # type: ignore[assignment]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)
        self.assertTrue(self.fake.shutdown_called)
        super().tearDown()

    def request(
        self,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        authorized: bool = True,
    ) -> tuple[int, Any, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers: dict[str, str] = {}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if authorized and payload is not None:
            headers["X-Operator-Token"] = self.server.application.token
        request = urllib.request.Request(
            self.base + path,
            data=data,
            headers=headers,
            method="POST" if payload is not None else "GET",
        )
        response = urllib.request.urlopen(request, timeout=5.0)
        return response.status, response.headers, response.read()

    def test_drive_json_and_jpeg_routes(self) -> None:
        status, _, body = self.request("/api/drive/catalog")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["connected"])

        status, headers, body = self.request("/api/drive/frame.jpg?view=raw")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/jpeg")
        self.assertEqual(headers["X-Drive-Frame-Sequence"], "7")
        self.assertEqual(body, b"fake-jpeg")

        status, _, body = self.request(
            "/api/drive/start",
            payload={"run_id": "browser-http-test"},
        )
        self.assertEqual(status, 202)
        self.assertEqual(json.loads(body)["session_id"], "browser-http-test")
        self.assertEqual(self.fake.started, {"run_id": "browser-http-test"})

    def test_drive_mutations_require_operator_token(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request(
                "/api/drive/start",
                payload={"run_id": "unauthorized"},
                authorized=False,
            )
        self.assertEqual(caught.exception.code, 403)


if __name__ == "__main__":
    unittest.main()
