from __future__ import annotations

import unittest
from unittest.mock import Mock

from carla_vision.bridge import CarlaError, CarlaRpc, spectator_chase_transform


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


if __name__ == "__main__":
    unittest.main()
