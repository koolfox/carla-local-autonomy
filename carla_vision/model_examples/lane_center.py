"""A tiny image-based lane-centering demo for validating the model interface.

This is not a trained autonomous-driving model and is not suitable as a safety
baseline.  It exists only to prove that a front-camera model can close the
control loop in CARLA without receiving map or waypoint objects.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..model_driver import ModelControl, ModelDriverConfig, ModelObservation


class LaneCenterDemo:
    def __init__(self, config: ModelDriverConfig) -> None:
        options: dict[str, Any] = dict(config.options)
        self._target_speed = float(options.pop("target_speed_mps", 4.0))
        self._steer_gain = float(options.pop("steer_gain", 0.75))
        self._smoothing = float(options.pop("smoothing", 0.75))
        self._minimum_pixels = int(options.pop("minimum_pixels", 120))
        if options:
            raise ValueError("unknown lane-center options: " + ", ".join(sorted(options)))
        if self._target_speed <= 0.0:
            raise ValueError("target_speed_mps must be positive")
        if not 0.0 <= self._smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        if self._minimum_pixels <= 0:
            raise ValueError("minimum_pixels must be positive")
        self._steer = 0.0

    def reset(self) -> None:
        self._steer = 0.0

    def predict(self, observation: ModelObservation) -> ModelControl:
        image = observation.image_bgr
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        white = cv2.inRange(hsv, np.array([0, 0, 175]), np.array([180, 80, 255]))
        yellow = cv2.inRange(hsv, np.array([12, 65, 90]), np.array([40, 255, 255]))
        mask = cv2.bitwise_or(white, yellow)

        roi = np.zeros_like(mask)
        polygon = np.array(
            [
                [int(width * 0.08), height - 1],
                [int(width * 0.38), int(height * 0.56)],
                [int(width * 0.62), int(height * 0.56)],
                [int(width * 0.92), height - 1],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(roi, polygon, 255)
        mask = cv2.bitwise_and(mask, roi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        ys, xs = np.nonzero(mask)
        if len(xs) < self._minimum_pixels:
            return ModelControl(throttle=0.0, steer=self._steer, brake=0.65)

        weights = np.square((ys.astype(np.float32) + 1.0) / float(height))
        center_x = float(np.average(xs, weights=weights))
        error = (center_x - (width / 2.0)) / max(width / 2.0, 1.0)
        desired = float(np.clip(self._steer_gain * error, -0.65, 0.65))
        self._steer = self._smoothing * self._steer + (1.0 - self._smoothing) * desired

        speed_error = self._target_speed - observation.speed_mps
        if speed_error < -0.4:
            return ModelControl(throttle=0.0, steer=self._steer, brake=0.25)
        throttle = float(np.clip(0.16 + 0.10 * speed_error, 0.08, 0.42))
        return ModelControl(throttle=throttle, steer=self._steer, brake=0.0)

    def close(self) -> None:
        return None


def create_driver(config: ModelDriverConfig) -> LaneCenterDemo:
    return LaneCenterDemo(config)


__all__ = ["LaneCenterDemo", "create_driver"]
