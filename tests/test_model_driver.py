from __future__ import annotations

import numpy as np
import pytest

from carla_vision.model_driver import (
    ModelControl,
    ModelDriverConfig,
    ModelObservation,
    control_from_value,
    create_driving_model,
)


def test_model_observation_copies_and_freezes_image() -> None:
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    observation = ModelObservation(
        frame=1,
        timestamp=0.5,
        image_bgr=image,
        speed_mps=2.0,
        dt_seconds=0.05,
    )
    image[0, 0, 0] = 255
    assert observation.image_bgr[0, 0, 0] == 0
    assert not observation.image_bgr.flags.writeable


def test_model_control_rejects_throttle_and_brake_together() -> None:
    with pytest.raises(ValueError, match="throttle and brake"):
        ModelControl(throttle=0.2, steer=0.0, brake=0.2)


def test_control_from_mapping_and_sequence() -> None:
    assert control_from_value({"throttle": 0.2, "steer": -0.1, "brake": 0.0}) == ModelControl(
        throttle=0.2,
        steer=-0.1,
        brake=0.0,
    )
    assert control_from_value([0.0, 0.25, 0.6]) == ModelControl(
        throttle=0.0,
        steer=0.25,
        brake=0.6,
    )


def test_example_model_factory_contract() -> None:
    model = create_driving_model(
        "carla_vision.model_examples.lane_center:create_driver",
        ModelDriverConfig(options={"target_speed_mps": 3.0}),
    )
    model.reset()
    observation = ModelObservation(
        frame=3,
        timestamp=1.0,
        image_bgr=np.zeros((180, 320, 3), dtype=np.uint8),
        speed_mps=0.0,
        dt_seconds=0.05,
    )
    control = control_from_value(model.predict(observation))
    assert control.brake > 0.0
    model.close()
