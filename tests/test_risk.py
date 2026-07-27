from __future__ import annotations

import unittest

from carla_vision.contracts import Detection
from carla_vision.risk import HazardPolicy


class HazardPolicyTests(unittest.TestCase):
    def test_close_road_user_in_corridor_is_hazard(self) -> None:
        detection = Detection(
            class_id=0,
            label="person",
            confidence=0.8,
            xyxy=(250.0, 220.0, 390.0, 380.0),
        )
        assessment = HazardPolicy().assess((detection,), 640, 384)
        self.assertTrue(assessment.hazard)
        self.assertEqual(assessment.hazard_indices, frozenset({0}))

    def test_policy_does_not_treat_detector_output_as_automatic_hazard(self) -> None:
        off_corridor_car = Detection(
            class_id=2,
            label="car",
            confidence=0.99,
            xyxy=(0.0, 100.0, 100.0, 300.0),
        )
        sign = Detection(
            class_id=9,
            label="traffic light",
            confidence=0.99,
            xyxy=(300.0, 10.0, 340.0, 60.0),
        )
        assessment = HazardPolicy().assess((off_corridor_car, sign), 640, 384)
        self.assertFalse(assessment.hazard)


if __name__ == "__main__":
    unittest.main()
