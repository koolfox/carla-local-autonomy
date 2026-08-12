from __future__ import annotations

import math
import unittest
from unittest.mock import Mock

from carla_vision.bridge import (
    CarlaError,
    CarlaRpc,
    garage_camera_preset_transform,
    garage_orbit_camera_transform,
    spawn_unparented_rgb_camera,
    spectator_chase_transform,
)


class CarlaSpectatorRpcTests(unittest.TestCase):
    def test_spectator_and_episode_unwrap_the_expected_endpoints(self) -> None:
        rpc = object.__new__(CarlaRpc)
        rpc.value_call = Mock(
            side_effect=[
                [1, 0, [0, "spectator", []]],
                [123456789, [1, 2, 3]],
            ]
        )

        self.assertEqual(rpc.spectator()[0], 1)
        self.assertEqual(rpc.episode_id(), 123456789)
        self.assertEqual(
            rpc.value_call.call_args_list,
            [unittest.mock.call("get_spectator"), unittest.mock.call("get_episode_info")],
        )

    def test_malformed_spectator_is_rejected(self) -> None:
        rpc = object.__new__(CarlaRpc)
        rpc.value_call = Mock(return_value=[])

        with self.assertRaisesRegex(CarlaError, "malformed actor"):
            rpc.spectator()

    def test_set_actor_transform_waits_for_the_void_response(self) -> None:
        rpc = object.__new__(CarlaRpc)
        rpc.void_call = Mock()
        transform = [[1.0, 2.0, 3.0], [-15.0, 90.0, 0.0]]

        rpc.set_actor_transform(7, transform)

        rpc.void_call.assert_called_once_with("set_actor_transform", 7, transform)


class SpectatorChaseTransformTests(unittest.TestCase):
    def test_flat_vehicle_uses_fixed_chase_offset(self) -> None:
        transform = spectator_chase_transform([[10.0, 20.0, 1.0], [0.0, 0.0, 0.0]])

        self.assertEqual(transform, [[3.0, 20.0, 4.0], [-15.0, 0.0, 0.0]])

    def test_yaw_and_pitch_are_respected(self) -> None:
        yawed = spectator_chase_transform([[10.0, 20.0, 1.0], [0.0, 90.0, 0.0]])
        pitched = spectator_chase_transform([[10.0, 20.0, 1.0], [60.0, 0.0, 0.0]])

        self.assertAlmostEqual(yawed[0][0], 10.0, places=6)
        self.assertAlmostEqual(yawed[0][1], 13.0, places=6)
        self.assertAlmostEqual(pitched[0][0], 6.5, places=6)

    def test_invalid_or_nonfinite_transform_is_rejected(self) -> None:
        for transform in (
            [[1.0, 2.0], [0.0, 0.0, 0.0]],
            [[1.0, 2.0, 3.0], [0.0, 0.0]],
            [[1.0, 2.0, float("nan")], [0.0, 0.0, 0.0]],
        ):
            with self.subTest(transform=transform), self.assertRaises(ValueError):
                spectator_chase_transform(transform)


class GarageOrbitCameraTransformTests(unittest.TestCase):
    def test_zero_azimuth_is_in_front_and_points_inward(self) -> None:
        result = garage_orbit_camera_transform(
            [[10.0, 20.0, 0.5], [0.0, 0.0, 0.0]],
            azimuth_degrees=0.0,
        )

        horizontal_radius = 6.5 * math.cos(math.radians(-10.0))
        expected_height = 0.5 + 0.9 - 6.5 * math.sin(math.radians(-10.0))
        self.assertAlmostEqual(result[0][0], 10.0 + horizontal_radius)
        self.assertAlmostEqual(result[0][1], 20.0)
        self.assertAlmostEqual(result[0][2], expected_height)
        self.assertEqual(result[1], [-10.0, 180.0, 0.0])

    def test_azimuth_rotates_toward_vehicle_positive_y(self) -> None:
        result = garage_orbit_camera_transform(
            [[10.0, 20.0, 0.5], [0.0, 0.0, 0.0]],
            azimuth_degrees=90.0,
            pitch_degrees=0.0,
            distance=4.0,
        )

        self.assertAlmostEqual(result[0][0], 10.0)
        self.assertAlmostEqual(result[0][1], 24.0)
        self.assertAlmostEqual(result[0][2], 1.4)
        self.assertEqual(result[1], [0.0, 270.0, 0.0])

    def test_vehicle_yaw_rotates_the_whole_orbit(self) -> None:
        result = garage_orbit_camera_transform(
            [[10.0, 20.0, 0.5], [0.0, 90.0, 0.0]],
            azimuth_degrees=0.0,
            pitch_degrees=0.0,
            distance=4.0,
        )

        self.assertAlmostEqual(result[0][0], 10.0)
        self.assertAlmostEqual(result[0][1], 24.0)
        self.assertEqual(result[1], [0.0, 270.0, 0.0])

    def test_invalid_transform_and_orbit_bounds_are_rejected(self) -> None:
        valid = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        invalid_calls = (
            ([[0.0, 0.0], [0.0, 0.0, 0.0]], {}),
            ([[0.0, 0.0, 0.0], [0.0, float("nan"), 0.0]], {}),
            (valid, {"azimuth_degrees": float("inf")}),
            (valid, {"distance": 3.49}),
            (valid, {"distance": 10.01}),
            (valid, {"pitch_degrees": -25.01}),
            (valid, {"pitch_degrees": 15.01}),
            (valid, {"target_height": -0.01}),
        )
        for transform, overrides in invalid_calls:
            values = {"azimuth_degrees": 0.0, **overrides}
            with self.subTest(transform=transform, values=values), self.assertRaises(ValueError):
                garage_orbit_camera_transform(transform, **values)

    def test_fixed_presets_and_cockpit_respect_vehicle_yaw(self) -> None:
        vehicle = [[10.0, 20.0, 0.5], [0.0, 90.0, 0.0]]

        front = garage_camera_preset_transform(vehicle, "front")
        rear = garage_camera_preset_transform(vehicle, "rear")
        top = garage_camera_preset_transform(vehicle, "top")
        cockpit = garage_camera_preset_transform(vehicle, "cockpit")

        self.assertGreater(front[0][1], 20.0)
        self.assertLess(rear[0][1], 20.0)
        self.assertEqual(front[1], [-8.0, 270.0, 0.0])
        self.assertEqual(rear[1], [-8.0, 450.0, 0.0])
        self.assertEqual(top[1], [-25.0, 270.0, 0.0])
        self.assertAlmostEqual(cockpit[0][0], 10.0)
        self.assertAlmostEqual(cockpit[0][1], 20.35)
        self.assertAlmostEqual(cockpit[0][2], 1.75)
        self.assertEqual(cockpit[1], [0.0, 90.0, 0.0])

    def test_orbit_preset_uses_continuous_payload_and_unknown_is_rejected(self) -> None:
        vehicle = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        direct = garage_orbit_camera_transform(
            vehicle,
            azimuth_degrees=45.0,
            pitch_degrees=-12.0,
            distance=5.0,
        )
        preset = garage_camera_preset_transform(
            vehicle,
            "orbit",
            azimuth_degrees=45.0,
            pitch_degrees=-12.0,
            distance=5.0,
        )

        self.assertEqual(preset, direct)
        with self.assertRaisesRegex(ValueError, "garage camera preset"):
            garage_camera_preset_transform(vehicle, "cinematic")


class UnparentedRgbCameraTests(unittest.TestCase):
    @staticmethod
    def _definitions() -> list[list[object]]:
        attributes = [
            ["role_name", 0, "", [], True, False],
            ["sensor_tick", 1, "0.0", [], True, False],
            ["image_size_x", 2, "800", [], True, False],
            ["image_size_y", 2, "600", [], True, False],
            ["fov", 1, "90.0", [], True, False],
            ["motion_blur_intensity", 1, "0.45", [], True, False],
            ["motion_blur_max_distortion", 1, "0.35", [], True, False],
        ]
        return [[123, "sensor.camera.rgb", ["sensor", "camera", "rgb"], attributes]]

    def test_spawns_world_space_camera_and_returns_serialized_actor(self) -> None:
        token = bytes(range(24))
        serialized = [71, None, [123, "sensor.camera.rgb", []], [], [], token]
        rpc = Mock()
        rpc.value_call.side_effect = [self._definitions(), serialized]
        transform = [[1.0, 2.0, 3.0], [-10.0, 180.0, 0.0]]

        result = spawn_unparented_rgb_camera(
            rpc,
            transform,
            width=1280,
            height=720,
            sensor_tick=0.05,
            fov=70.0,
        )

        self.assertIs(result, serialized)
        self.assertEqual(result[0], 71)
        self.assertEqual(result[5], token)
        calls = rpc.value_call.call_args_list
        self.assertEqual(calls[0], unittest.mock.call("get_actor_definitions"))
        self.assertEqual(calls[1].args[0], "spawn_actor")
        self.assertEqual(calls[1].args[2], transform)
        description = calls[1].args[1]
        attributes = {item[0]: item[2] for item in description[2]}
        self.assertEqual(attributes["role_name"], "garage_preview")
        self.assertEqual(attributes["image_size_x"], "1280")
        self.assertEqual(attributes["image_size_y"], "720")
        self.assertEqual(attributes["sensor_tick"], "0.05")
        self.assertEqual(attributes["fov"], "70.0")
        self.assertEqual(attributes["motion_blur_intensity"], "0.0")

    def test_malformed_stream_token_destroys_spawned_camera(self) -> None:
        malformed = [71, None, [123, "sensor.camera.rgb", []], [], [], b"short"]
        rpc = Mock()
        rpc.value_call.side_effect = [self._definitions(), malformed]

        with self.assertRaisesRegex(CarlaError, "stream token"):
            spawn_unparented_rgb_camera(
                rpc,
                [[1.0, 2.0, 3.0], [-10.0, 180.0, 0.0]],
            )

        rpc.destroy_actor.assert_called_once_with(71)

    def test_transform_and_camera_contract_are_validated_before_spawn(self) -> None:
        for transform, options in (
            ([[1.0, 2.0], [0.0, 0.0, 0.0]], {}),
            ([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], {"width": True}),
            ([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], {"height": 0}),
            ([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], {"sensor_tick": 0.0}),
            ([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], {"fov": 180.0}),
        ):
            rpc = Mock()
            with self.subTest(transform=transform, options=options), self.assertRaises(ValueError):
                spawn_unparented_rgb_camera(rpc, transform, **options)
            rpc.value_call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
