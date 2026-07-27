from __future__ import annotations

import argparse
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np

from carla_vision.contracts import Detection, PerceptionResult
from carla_vision.controller import ControlCommand
from carla_vision.risk import HazardPolicy
from carla_vision.runtime import (
    _actor_attributes,
    _detector_config,
    _policy_config,
    _safe_server_version,
    _serialize_detection_log,
    _write_json_line,
    brake_and_verify_stop,
    camera_actor,
    parse_args,
    parse_json_object,
    parse_resolution,
    validate_camera_mount,
)


def serialized_camera(
    *,
    camera_id: int = 25,
    parent_id: int = 24,
    role_name: str = "front",
    width: int = 640,
    height: int = 384,
    fov: str = "90.0",
    sensor_tick: str = "0.1",
    token: list[int] | None = None,
) -> list[object]:
    attributes = [
        ["role_name", 0, role_name],
        ["sensor_tick", 0, sensor_tick],
        ["image_size_x", 0, str(width)],
        ["image_size_y", 0, str(height)],
        ["fov", 0, fov],
    ]
    return [
        camera_id,
        parent_id,
        [123, "sensor.camera.rgb", attributes],
        [],
        [],
        [1] if token is None else token,
    ]


class RuntimeParsingTests(unittest.TestCase):
    def test_parse_resolution_accepts_bounds_and_uppercase_separator(self) -> None:
        self.assertEqual(parse_resolution("320x180"), (320, 180))
        self.assertEqual(parse_resolution("3840X2160"), (3840, 2160))

    def test_parse_resolution_rejects_malformed_and_out_of_range_values(self) -> None:
        for value in ("640", "widex384", "319x180", "3841x2160", "640x2170"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    parse_resolution(value)

    def test_parse_args_normalizes_drive_alias_and_builds_detector_config(self) -> None:
        args = parse_args(
            [
                "--resolution",
                "800x450",
                "--drive",
                "--detector",
                "custom",
                "--detector-factory",
                "example.detector:create",
                "--weights",
                "weights.bin",
                "--camera-fps",
                "20",
                "--duration",
                "5",
            ]
        )

        self.assertEqual(args.resolution, (800, 450))
        self.assertEqual(args.control, "teacher")
        config = _detector_config(args)
        self.assertEqual(config.backend, "custom")
        self.assertEqual(config.factory, "example.detector:create")
        self.assertEqual(config.weights, Path("weights.bin"))

    def test_parse_args_rejects_unreleased_vision_control_without_side_effects(self) -> None:
        with patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--control", "vision"])

    def test_shadow_policy_is_non_actuating_and_resolves_a_strict_config(self) -> None:
        args = parse_args(
            [
                "--shadow-policy",
                "hazard-stop",
                "--policy-device",
                "mps",
                "--policy-options",
                '{"confidence": 0.4}',
            ]
        )
        self.assertEqual(args.control, "none")
        config = _policy_config(args)
        self.assertIsNotNone(config)
        self.assertEqual(config.backend, "hazard-stop")
        self.assertEqual(config.device, "mps")
        self.assertEqual(config.options, {"confidence": 0.4})

    def test_shadow_policy_cli_rejects_incomplete_custom_and_orphan_options(self) -> None:
        for arguments in (
            ["--shadow-policy", "custom"],
            ["--policy-factory", "example:create"],
            ["--policy-options", '{"confidence": 0.4}'],
        ):
            with (
                self.subTest(arguments=arguments),
                patch(
                    "sys.stderr",
                    new=io.StringIO(),
                ),
            ):
                with self.assertRaises(SystemExit):
                    parse_args(arguments)

    def test_parse_json_object_rejects_arrays(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_json_object("[]")

    def test_default_model_references_are_selected_without_loading_a_model(self) -> None:
        yolo = _detector_config(parse_args(["--detector", "yolo"]))
        rtdetr = _detector_config(parse_args(["--detector", "rtdetr"]))

        self.assertEqual(yolo.weights, Path("yolo26n.pt"))
        self.assertEqual(rtdetr.weights, Path("rtdetr-l.pt"))

    def test_model_package_is_mutually_exclusive_with_loose_model_options(self) -> None:
        for option in (
            ["--detector", "rtdetr"],
            ["--weights", "model.pt"],
            ["--detector-factory", "example:create"],
            ["--image-size", "320"],
        ):
            with self.subTest(option=option), patch("sys.stderr", new=io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(["--model-package", "models/example", *option])


class RuntimeCameraTests(unittest.TestCase):
    def test_actor_attributes_extracts_values_and_handles_missing_actor(self) -> None:
        actor = serialized_camera(width=800, height=450, fov="100")
        self.assertEqual(
            _actor_attributes(actor),
            {
                "role_name": "front",
                "sensor_tick": "0.1",
                "image_size_x": "800",
                "image_size_y": "450",
                "fov": "100",
            },
        )
        self.assertEqual(_actor_attributes(None), {})
        self.assertEqual(_actor_attributes([25, 24]), {})

    def test_camera_actor_reuses_matching_front_camera(self) -> None:
        existing = serialized_camera(fov="90.00")
        rpc = Mock()
        rpc.actor.return_value = existing

        with patch("carla_vision.runtime.spawn_front_camera") as spawn:
            actor, was_spawned = camera_actor(
                rpc,
                vehicle_id=24,
                camera_id=25,
                width=640,
                height=384,
                camera_fps=10.0,
                fov=90.0,
            )

        self.assertIs(actor, existing)
        self.assertFalse(was_spawned)
        rpc.actor.assert_called_once_with(25)
        spawn.assert_not_called()

    def test_camera_actor_spawns_when_existing_sensor_tick_mismatches(self) -> None:
        rpc = Mock()
        rpc.actor.return_value = serialized_camera(sensor_tick="0.05")
        spawned = serialized_camera(camera_id=31)

        with patch(
            "carla_vision.runtime.spawn_front_camera",
            return_value=spawned,
        ) as spawn:
            actor, was_spawned = camera_actor(
                rpc,
                vehicle_id=24,
                camera_id=25,
                width=640,
                height=384,
                camera_fps=10.0,
                fov=90.0,
            )

        self.assertIs(actor, spawned)
        self.assertTrue(was_spawned)
        spawn.assert_called_once_with(
            rpc,
            24,
            width=640,
            height=384,
            sensor_tick=0.1,
            fov=90.0,
        )

    def test_camera_actor_spawns_when_existing_camera_contract_mismatches(self) -> None:
        rpc = Mock()
        rpc.actor.return_value = serialized_camera(role_name="rear")
        spawned = serialized_camera(camera_id=31, fov="100.0")

        with patch(
            "carla_vision.runtime.spawn_front_camera",
            return_value=spawned,
        ) as spawn:
            actor, was_spawned = camera_actor(
                rpc,
                vehicle_id=24,
                camera_id=25,
                width=800,
                height=450,
                camera_fps=20.0,
                fov=100.0,
            )

        self.assertIs(actor, spawned)
        self.assertTrue(was_spawned)
        spawn.assert_called_once_with(
            rpc,
            24,
            width=800,
            height=450,
            sensor_tick=0.05,
            fov=100.0,
        )

    def test_validate_camera_mount_accepts_wrapped_rotation_and_rejects_offset(self) -> None:
        rpc = Mock()
        rpc.actor_transform.return_value = [[1.04, 2.0, 3.0], [0.0, -179.7, 0.0]]
        inferred = [[1.0, 2.0, 3.0], [0.0, 179.7, 0.0]]

        with patch(
            "carla_vision.runtime.vehicle_transform_from_front_camera",
            return_value=inferred,
        ):
            validate_camera_mount(rpc, 24, object())
        rpc.actor_transform.assert_called_once_with(24)

        rpc.actor_transform.return_value = [[2.0, 2.0, 3.0], [0.0, 179.7, 0.0]]
        with patch(
            "carla_vision.runtime.vehicle_transform_from_front_camera",
            return_value=inferred,
        ):
            with self.assertRaisesRegex(RuntimeError, "location_error=1.000m"):
                validate_camera_mount(rpc, 24, object())


class RuntimeSerializationTests(unittest.TestCase):
    def test_detection_log_preserves_frame_provenance_risk_and_teacher_decision(self) -> None:
        detection = Detection(
            class_id=0,
            source_class_id=7,
            label="person",
            confidence=0.8,
            xyxy=(100.0, 100.0, 220.0, 170.0),
            attributes={"source": "unit-test"},
        )
        result = PerceptionResult(
            sequence=8,
            carla_frame=108,
            source_timestamp=4.25,
            source_received_monotonic=10.0,
            inference_started_monotonic=10.1,
            completed_monotonic=10.4,
            detections=(detection,),
            source_bgr=np.zeros((180, 320, 3), dtype=np.uint8),
            detector_name="fake-detector",
            source_transform=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            source_fov=90.0,
        )
        risk = HazardPolicy().assess(result.detections, 320, 180)
        command = ControlCommand(throttle=0.2, steer=-0.1, brake=0.0)

        payload = _serialize_detection_log(
            result,
            risk,
            mode="SIMULATOR TEACHER",
            simulator_speed=2.25,
            route_progress="12/100",
            command=command,
            shadow_proposal=None,
            latest_camera_sequence=9,
            latest_camera_timestamp=4.35,
            ego_transform=[[10.0, 20.0, 0.5], [0.0, 90.0, 0.0]],
        )

        self.assertEqual(payload["sequence"], 8)
        self.assertEqual(payload["carla_frame"], 108)
        self.assertAlmostEqual(payload["pipeline_latency_seconds"], 0.4)
        self.assertAlmostEqual(payload["model_inference_seconds"], 0.3)
        self.assertEqual(
            payload["source_camera"],
            {
                "transform": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "fov_degrees": 90.0,
                "width": 320,
                "height": 180,
            },
        )
        self.assertEqual(
            payload["detections"][0],
            {
                "class_id": 0,
                "source_class_id": 7,
                "label": "person",
                "confidence": 0.8,
                "xyxy": [100.0, 100.0, 220.0, 170.0],
                "attributes": {"source": "unit-test"},
                "risk": {
                    "in_driving_corridor": True,
                    "visually_close": True,
                    "hazard": True,
                    "confidence_threshold": 0.35,
                },
            },
        )
        self.assertTrue(payload["hazard"])
        self.assertEqual(
            payload["decision_context"],
            {
                "latest_camera_sequence": 9,
                "latest_camera_timestamp": 4.35,
                "ego_transform_privileged": [[10.0, 20.0, 0.5], [0.0, 90.0, 0.0]],
                "teacher_control": {
                    "throttle": 0.2,
                    "steer": -0.1,
                    "brake": 0.0,
                    "hand_brake": False,
                },
                "vision_shadow": None,
            },
        )

    def test_json_line_is_sorted_unicode_and_single_line(self) -> None:
        stream = io.StringIO()
        payload = {"z": 1, "label": "خودرو", "a": 2}

        _write_json_line(stream, payload)

        serialized = stream.getvalue()
        self.assertEqual(serialized.count("\n"), 1)
        self.assertTrue(serialized.startswith('{"a": 2, "label": "خودرو", "z": 1}'))
        self.assertEqual(json.loads(serialized), payload)

    def test_safe_server_version_converts_value_and_suppresses_rpc_failure(self) -> None:
        rpc = Mock()
        rpc.value_call.return_value = "0.9.16"
        self.assertEqual(_safe_server_version(rpc), "0.9.16")

        rpc.value_call.side_effect = RuntimeError("offline")
        self.assertIsNone(_safe_server_version(rpc))


class RuntimeStopTests(unittest.TestCase):
    def test_brake_and_verify_stop_services_brake_then_parks(self) -> None:
        rpc = Mock()
        rpc.telemetry.side_effect = [
            SimpleNamespace(speed=0.4),
            SimpleNamespace(speed=0.02),
            SimpleNamespace(speed=0.0),
        ]

        with (
            patch("carla_vision.runtime.time.monotonic", side_effect=[10.0, 10.1]),
            patch("carla_vision.runtime.time.sleep") as sleep,
        ):
            telemetry = brake_and_verify_stop(rpc, 24, timeout=2.0)

        self.assertEqual(telemetry.speed, 0.0)
        self.assertEqual(
            rpc.apply_vehicle_control.call_args_list,
            [
                call(24, ControlCommand.service_brake().as_carla()),
                call(24, ControlCommand.parked().as_carla()),
            ],
        )
        sleep.assert_called_once_with(0.08)


if __name__ == "__main__":
    unittest.main()
