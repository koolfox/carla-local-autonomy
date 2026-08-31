from __future__ import annotations

import json
import re
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from unittest import mock

from carla_vision.operator import drive as drive_module
from carla_vision.operator.drive import (
    DriveSession,
    DriveSessionManager,
    _validate_camera_attachment,
)
from carla_vision.operator.drive_contracts import (
    DriveInput,
    DriveStartConfig,
    weather_payload,
)
from carla_vision.operator.server import create_server

CARLA_HOST = "172.20.10.7"
CARLA_PORT = 2000
STATIC_ROOT = Path(__file__).parents[1] / "carla_vision" / "operator" / "static"
DRIVE_SOURCE = Path(__file__).parents[1] / "carla_vision" / "operator" / "drive.py"


class _StaticHtmlContractParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.elements: dict[str, tuple[str, dict[str, str | None]]] = {}
        self.forms: list[str] = []
        self.details: list[str] = []
        self.nested_forms: list[tuple[str, str]] = []
        self.controls: list[dict[str, Any]] = []
        self._form_stack: list[str] = []
        self._details_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id is not None:
            self.ids.append(element_id)
            self.elements[element_id] = (tag, attributes)

        if tag == "form":
            form_id = element_id or "<anonymous>"
            if self._form_stack:
                self.nested_forms.append((self._form_stack[-1], form_id))
            self.forms.append(form_id)
            self._form_stack.append(form_id)
        elif tag == "details":
            details_id = element_id or "<anonymous>"
            self.details.append(details_id)
            self._details_stack.append(details_id)

        if tag in {"button", "input", "select", "textarea"} and element_id is not None:
            associated_form = attributes.get("form")
            if associated_form is None and self._form_stack:
                associated_form = self._form_stack[-1]
            self.controls.append(
                {
                    "id": element_id,
                    "tag": tag,
                    "form": associated_form,
                    "details": tuple(self._details_stack),
                }
            )

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self._form_stack:
            self._form_stack.pop()
        elif tag == "details" and self._details_stack:
            self._details_stack.pop()


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
        self.assertEqual(config.pedestrian_crossing_factor, 0.2)
        self.assertEqual(config.speed_difference_percent, 12.0)
        self.assertEqual(config.following_distance_metres, 2.0)
        self.assertEqual(config.experiment_preset, "free_drive")
        self.assertFalse(config.manifest_config()["model_output_actuated"])

    def test_world_worker_dynamics_are_retained_in_manifest(self) -> None:
        config = DriveStartConfig.from_mapping(
            valid_start(
                pedestrian_crossing_factor=0.95,
                speed_difference_percent=-30.0,
                following_distance_metres=10.0,
            ),
            workspace=self.workspace,
            expected_host=CARLA_HOST,
            expected_port=CARLA_PORT,
            world_worker_configured=True,
        )

        self.assertEqual(config.pedestrian_crossing_factor, 0.95)
        self.assertEqual(config.speed_difference_percent, -30.0)
        self.assertEqual(config.following_distance_metres, 10.0)
        manifest = config.manifest_config()
        self.assertEqual(manifest["pedestrian_crossing_factor"], 0.95)
        self.assertEqual(manifest["speed_difference_percent"], -30.0)
        self.assertEqual(manifest["following_distance_metres"], 10.0)

    def test_detector_can_be_disabled_without_resolving_weights(self) -> None:
        config = self.config(detector_enabled=False, weights="")

        self.assertIsNone(config.weights)
        self.assertFalse(config.detector_enabled)

    def test_camera_contract_accepts_60_fps_but_rejects_higher_rates(self) -> None:
        config = self.config(camera_fps=60.0)

        self.assertEqual(config.camera_fps, 60.0)
        with self.assertRaisesRegex(ValueError, "camera_fps"):
            self.config(camera_fps=60.1)

    def test_human_experiment_preset_is_retained_in_manifest(self) -> None:
        config = self.config(experiment_preset="perception_review")

        self.assertEqual(config.experiment_preset, "perception_review")
        self.assertEqual(config.manifest_config()["experiment_preset"], "perception_review")

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
            ({"experiment_preset": "make-something-up"}, "experiment_preset"),
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


class CameraAttachmentTests(unittest.TestCase):
    def test_accepts_serialized_camera_parented_to_ego(self) -> None:
        _validate_camera_attachment([50, 49], 49)

    def test_rejects_wrong_or_malformed_parent(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not attached"):
            _validate_camera_attachment([50, 48], 49)
        for malformed in ([], [50], [50, "not-an-actor"]):
            with self.subTest(camera=malformed):
                with self.assertRaisesRegex(RuntimeError, "malformed parent metadata"):
                    _validate_camera_attachment(malformed, 49)


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

    def test_human_moment_is_bounded_and_retained_with_frame_context(self) -> None:
        session = self.session()
        with session._lock:
            session._status = "running"
            session._raw_frame_sequence = 42
            session._overlay_frame_sequence = 40
            session._control_mode = "autopilot"
            session._control_source = "worker_autopilot"

        snapshot = session.mark_human_event("false_detection", "phantom car")

        self.assertEqual(snapshot["human_markers_written"], 1)
        self.assertEqual(snapshot["human_marker_counts"]["false_detection"], 1)
        with session._lock:
            event = session._pending_events[-1]
        self.assertEqual(event["event"], "human_moment_marked")
        self.assertEqual(event["raw_camera_sequence"], 42)
        self.assertEqual(event["detector_sequence"], 40)
        self.assertEqual(event["control_mode"], "autopilot")
        self.assertEqual(event["note"], "phantom car")

        for label, note in (("unknown", None), ("interesting", "x\nsecond line")):
            with self.subTest(label=label, note=note), self.assertRaises(ValueError):
                session.mark_human_event(label, note)

    def test_persistent_stream_waits_for_new_frames_and_reports_fps_and_age(self) -> None:
        session = self.session()
        session._cache_frame("raw", 10, b"first")
        session._cache_frame("raw", 11, b"second")

        sequence, payload = session.wait_for_frame("raw", after_sequence=10, timeout=0.1)
        stream = session.snapshot()["stream"]

        self.assertEqual((sequence, payload), (11, b"second"))
        self.assertGreater(stream["source_fps"], 0.0)
        self.assertFalse(stream["stale"])
        self.assertTrue(stream["raw_video_model_independent"])
        with self.assertRaisesRegex(TimeoutError, "timed out"):
            session.wait_for_frame("raw", after_sequence=11, timeout=0.01)


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
        self.assertEqual(
            catalog["maps"],
            [
                {"id": "Town03", "label": "Town03"},
                {"id": "Town10HD_Opt", "label": "Town10HD_Opt"},
            ],
        )
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

    def test_catalog_reports_local_vision_runtime_instead_of_claiming_advisory(self) -> None:
        runtime = {
            "available": False,
            "torch_importable": True,
            "ultralytics_importable": False,
            "missing": ["Ultralytics"],
        }
        with mock.patch.object(drive_module, "_vision_runtime_status", return_value=runtime):
            catalog = self.manager(rpc_factory=_FakeRpc).catalog()

        self.assertEqual(catalog["vision_runtime"], runtime)
        self.assertFalse(catalog["capabilities"]["model_advisory"])


class StaticDriveConsoleContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        self.script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        self.styles = (STATIC_ROOT / "app.css").read_text(encoding="utf-8")
        self.parser = _StaticHtmlContractParser()
        self.parser.feed(self.html)

    def test_html_has_unique_ids_one_form_and_closed_advanced_settings(self) -> None:
        duplicate_ids = sorted(
            element_id
            for element_id in set(self.parser.ids)
            if self.parser.ids.count(element_id) > 1
        )
        self.assertEqual(duplicate_ids, [])
        self.assertEqual(self.parser.nested_forms, [])
        self.assertIn("drive-start-form", self.parser.forms)
        self.assertEqual(self.parser.details.count("drive-advanced-settings"), 1)

        details_tag, details_attributes = self.parser.elements["drive-advanced-settings"]
        self.assertEqual(details_tag, "details")
        self.assertNotIn("open", details_attributes)

        controls = {item["id"]: item for item in self.parser.controls}
        direct_primary_configuration = {
            "drive-vehicle",
            "drive-color",
            "drive-camera-profile",
            "drive-detector-enabled",
            "drive-record-video",
            "drive-spectator-follow",
        }
        actual_primary = {
            item["id"]
            for item in self.parser.controls
            if item["form"] == "drive-start-form"
            and not item["details"]
            and item["tag"] != "button"
        }
        self.assertEqual(actual_primary, direct_primary_configuration)

        world_configuration = {
            "drive-map-choice",
            "drive-weather",
            "drive-traffic-choice",
            "drive-walkers-choice",
            "drive-props",
            "drive-starting-choice",
        }
        for element_id in world_configuration:
            with self.subTest(element_id=element_id):
                self.assertEqual(controls[element_id]["form"], "drive-start-form")
                self.assertEqual(controls[element_id]["details"], ("garage-world-settings",))

        advanced_configuration = {
            "drive-run-id",
            "drive-host",
            "drive-port",
            "drive-seed",
            "drive-resolution",
            "drive-camera-fps",
            "drive-camera-fov",
            "drive-detector",
            "drive-weights",
            "drive-device",
            "drive-image-size",
            "drive-confidence",
        }
        for element_id in advanced_configuration:
            with self.subTest(element_id=element_id):
                self.assertEqual(controls[element_id]["form"], "drive-start-form")
                self.assertEqual(
                    controls[element_id]["details"],
                    ("drive-advanced-settings",),
                )

        for element_id in ("drive-start", "drive-start-another"):
            with self.subTest(element_id=element_id):
                self.assertEqual(controls[element_id]["form"], "drive-start-form")
                self.assertEqual(controls[element_id]["details"], ())

    def test_every_literal_javascript_id_reference_exists_in_html(self) -> None:
        literal_id_references = set(re.findall(r"\$\(\s*[\"']([^\"']+)[\"']\s*\)", self.script))
        self.assertGreater(len(literal_id_references), 100)
        self.assertEqual(sorted(literal_id_references - set(self.parser.ids)), [])

    def test_detector_runtime_is_visible_gated_and_selects_overlay_on_start(self) -> None:
        note_tag, _ = self.parser.elements["drive-detector-runtime-note"]

        self.assertEqual(note_tag, "small")
        self.assertIn("function driveVisionRuntime()", self.script)
        self.assertIn("detectorToggle.disabled = active || !visionRuntime.available;", self.script)
        self.assertIn("throw new Error(driveVisionRuntimeMessage());", self.script)
        self.assertIn(
            'setDriveView(startConfig.detector_enabled ? "overlay" : "raw");',
            self.script,
        )
        self.assertIn('$("drive-view-raw").addEventListener("click"', self.script)

    def test_drive_camera_defaults_to_balanced_profile_with_60_fps_available(self) -> None:
        tag, attributes = self.parser.elements["drive-resolution"]
        self.assertEqual(tag, "select")
        selected = re.search(
            r'<option\s+value="([^"]+)"\s+selected>',
            self.html[self.html.index('id="drive-resolution"') :],
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected.group(1), "1280x720")
        profile = self.html[self.html.index('id="drive-camera-profile"') :]
        self.assertRegex(profile, r'value="balanced"\s+selected')
        self.assertIn('value="high-refresh"', profile)
        fps_tag, fps_attributes = self.parser.elements["drive-camera-fps"]
        self.assertEqual(fps_tag, "input")
        self.assertEqual(fps_attributes.get("value"), "30")
        self.assertEqual(fps_attributes.get("max"), "60")
        self.assertIn("def _jpeg(image: np.ndarray, quality: int = 92)", DRIVE_SOURCE.read_text())

    def test_browser_uses_one_persistent_drive_stream_without_frame_polling(self) -> None:
        self.assertIn("/api/drive/stream.mjpg", self.script)
        self.assertNotIn("/api/drive/frame.jpg", self.script)
        self.assertNotIn("refreshDriveFrame", self.script)

    def test_high_bandwidth_profiles_require_the_compressed_worker_capabilities(self) -> None:
        for capability in (
            "compressed_camera_relay",
            "persistent_mjpeg_camera_relay",
            "in_memory_jpeg_encoder_available",
            "camera_60_fps",
        ):
            self.assertIn(capability, self.script)
        self.assertIn("option.disabled", self.script)
        self.assertIn('select.value = "compatibility"', self.script)
        self.assertIn("Compatibility mode is enforced", self.script)

    def test_human_experiments_are_in_context_and_markers_are_wired(self) -> None:
        for element_id in (
            "game-experiments",
            "drive-experiment-toggle",
            "experiment-presets",
            "human-marker-count",
            "human-marker-note",
        ):
            with self.subTest(element_id=element_id):
                self.assertIn(element_id, self.parser.elements)
        for preset in (
            "free_drive",
            "manual_handling",
            "autopilot_takeover",
            "perception_review",
            "traffic_stress",
            "adverse_weather",
        ):
            with self.subTest(preset=preset):
                self.assertIn(f'data-experiment-preset="{preset}"', self.html)
        for marker in (
            "false_detection",
            "missed_object",
            "autopilot_issue",
            "scene_issue",
            "interesting",
        ):
            with self.subTest(marker=marker):
                self.assertIn(f'data-human-marker="{marker}"', self.html)
        self.assertIn('request("/api/drive/mark"', self.script)
        self.assertIn("experiment_preset: state.drive.experimentPreset", self.script)
        self.assertIn('openGameDrawer("experiments")', self.script)

    def test_touch_hud_mapping_pointer_lifecycle_and_responsive_visibility(self) -> None:
        expected_controls = {
            "drive-touch-left": "left",
            "drive-touch-right": "right",
            "drive-touch-throttle": "forward",
            "drive-touch-brake": "brake",
            "drive-touch-handbrake": "handBrake",
            "drive-touch-reverse": "reverseModifier",
        }
        for element_id, mapping in expected_controls.items():
            with self.subTest(element_id=element_id):
                tag, attributes = self.parser.elements[element_id]
                self.assertEqual(tag, "button")
                self.assertEqual(attributes.get("type"), "button")
                self.assertEqual(attributes.get("data-drive-control"), mapping)

        for binding in (
            'button.addEventListener("pointerdown", handleDrivePointerDown);',
            'button.addEventListener("pointerup", releaseDrivePointer);',
            'button.addEventListener("pointercancel", releaseDrivePointer);',
            'button.addEventListener("lostpointercapture", releaseDrivePointer);',
            "button.setPointerCapture(event.pointerId);",
            "state.drive.touchPointers.delete(event.pointerId);",
            "bindDriveTouchControls();",
        ):
            with self.subTest(binding=binding):
                self.assertIn(binding, self.script)

        self.assertIn('document.body.classList.toggle("drive-immersive", immersive);', self.script)
        self.assertRegex(
            self.styles,
            re.compile(r"\.drive-touch-controls\s*\{\s*display:\s*none;", re.DOTALL),
        )
        self.assertRegex(
            self.styles,
            re.compile(
                r"@media\s*\(any-pointer:\s*coarse\),\s*\(hover:\s*none\)\s*\{"
                r".*?\.drive-immersive\s+\.drive-touch-controls\s*\{.*?display:\s*flex;",
                re.DOTALL,
            ),
        )
        self.assertRegex(
            self.styles,
            re.compile(
                r"@media\s*\(pointer:\s*fine\)\s+and\s+\(hover:\s*hover\)\s*\{"
                r".*?\.drive-touch-controls\s*\{.*?display:\s*none;",
                re.DOTALL,
            ),
        )
        self.assertIn("touch-action: none;", self.styles)

    def test_keyboard_heartbeat_and_focus_loss_keep_deadman_wired(self) -> None:
        viewport_tag, viewport_attributes = self.parser.elements["drive-viewport"]
        self.assertEqual(viewport_tag, "div")
        self.assertEqual(viewport_attributes.get("tabindex"), "0")

        safety_hooks = (
            'viewport.addEventListener("keydown", (event) => handleDriveKey(event, true));',
            'viewport.addEventListener("keyup", (event) => handleDriveKey(event, false));',
            'viewport.addEventListener("blur", () => releaseDriveControl("Viewport focus lost"));',
            'window.addEventListener("blur", () => releaseDriveControl("Browser focus lost"));',
            'window.addEventListener("pagehide", () => releaseDriveControl("Page closing"));',
            'document.addEventListener("visibilitychange", () => {',
            'if (document.hidden) releaseDriveControl("Page hidden");',
            "window.setInterval(() => void sendDriveControl(), 67);",
            "void sendDriveControl({ safety: true, keepalive: true });",
        )
        for hook in safety_hooks:
            with self.subTest(hook=hook):
                self.assertIn(hook, self.script)

        self.assertIn("function nextDriveControlSequence()", self.script)
        self.assertIn("Date.now() * 1000", self.script)
        self.assertIn("sequence: nextDriveControlSequence()", self.script)

        self.assertRegex(
            self.script,
            re.compile(
                r"const command = safety\s*\?\s*\{\s*throttle:\s*0,\s*steer:\s*0,"
                r"\s*brake:\s*1,\s*hand_brake:\s*false,\s*reverse:\s*false\s*\}",
                re.DOTALL,
            ),
        )


class _FakeHttpDriveManager:
    def __init__(self) -> None:
        self.started: dict[str, Any] | None = None
        self.controls: list[dict[str, Any]] = []
        self.markers: list[dict[str, Any]] = []
        self.shutdown_called = False

    def catalog(self) -> dict[str, Any]:
        return {"connected": True, "vehicles": [], "capabilities": {}}

    def state(self) -> dict[str, Any]:
        return {"status": "idle", "session_id": None}

    def frame(self, view: str) -> tuple[int, bytes]:
        if view != "raw":
            raise ValueError("fake only serves raw")
        return 7, b"fake-jpeg"

    def wait_for_frame(
        self,
        view: str,
        after_sequence: int,
        timeout: float,
    ) -> tuple[int, bytes]:
        del timeout
        if view != "raw":
            raise ValueError("fake only serves raw")
        if after_sequence >= 7:
            raise EOFError("fake stream ended")
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

    def mark(self, raw: dict[str, Any]) -> dict[str, Any]:
        self.markers.append(dict(raw))
        return {"status": "running", "human_markers_written": len(self.markers)}

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

        status, _, body = self.request(
            "/api/drive/mark",
            payload={
                "session_id": "browser-http-test",
                "label": "interesting",
                "note": "human review",
            },
        )
        self.assertEqual(status, 201)
        self.assertEqual(json.loads(body)["human_markers_written"], 1)
        self.assertEqual(self.fake.markers[0]["label"], "interesting")

    def test_drive_mjpeg_route_streams_multipart_without_serial_polling(self) -> None:
        status, headers, body = self.request("/api/drive/stream.mjpg?view=raw")

        self.assertEqual(status, 200)
        self.assertEqual(
            headers["Content-Type"],
            "multipart/x-mixed-replace; boundary=carla-drive",
        )
        self.assertIn(b"X-Drive-Frame-Sequence: 7", body)
        self.assertIn(b"fake-jpeg", body)

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
