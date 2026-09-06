"""Checkpoint-backed imitation-driving factory for ``carla-local-drive``."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch

from ..model_driver import ModelControl, ModelDriverConfig, ModelObservation
from ..pytorch_loading import load_weights_only_checkpoint, prepare_module_for_inference
from .model import ImitationControlNet, ImitationModelConfig


class ImitationDrivingPredictor:
    def __init__(self, config: ModelDriverConfig) -> None:
        if config.checkpoint is None:
            raise ValueError("imitation-driving predictor requires a checkpoint")
        self.checkpoint = Path(config.checkpoint).expanduser().resolve(strict=True)
        payload = load_weights_only_checkpoint(self.checkpoint)
        if not isinstance(payload, dict):
            raise ValueError("imitation checkpoint must contain a mapping")
        if payload.get("task") != "behavior_agent_control_imitation":
            raise ValueError("checkpoint task is not behavior_agent_control_imitation")
        self.model_config = ImitationModelConfig.from_dict(dict(payload["model_config"]))
        self.model = ImitationControlNet(self.model_config)
        self.device = prepare_module_for_inference(
            self.model,
            payload["state_dict"],
            device=config.device,
        )
        options = dict(config.options)
        self.steer_gain = float(options.get("steer_gain", 1.0))
        self.throttle_gain = float(options.get("throttle_gain", 1.0))
        self.brake_gain = float(options.get("brake_gain", 1.0))
        self.longitudinal_deadband = float(options.get("longitudinal_deadband", 0.03))
        for name, value in (
            ("steer_gain", self.steer_gain),
            ("throttle_gain", self.throttle_gain),
            ("brake_gain", self.brake_gain),
            ("longitudinal_deadband", self.longitudinal_deadband),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.longitudinal_deadband >= 1.0:
            raise ValueError("longitudinal_deadband must be less than 1")

    def reset(self) -> None:
        return None

    def _preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        image = np.asarray(image_bgr)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("imitation input image must be uint8 HxWx3 BGR")
        height, width = self.model_config.image_size_hw
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return torch.from_numpy(chw).unsqueeze(0).to(self.device)

    def predict(self, observation: ModelObservation) -> ModelControl:
        image = self._preprocess(observation.image_bgr)
        normalized_speed = min(
            float(observation.speed_mps) / self.model_config.speed_scale_mps,
            2.0,
        )
        speed = torch.tensor([[normalized_speed]], dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            outputs = self.model(image, speed)
        steer = float(outputs["steer"][0].cpu()) * self.steer_gain
        longitudinal = float(outputs["longitudinal"][0].cpu())
        steer = float(np.clip(steer, -1.0, 1.0))
        if longitudinal > self.longitudinal_deadband:
            throttle = float(np.clip(longitudinal * self.throttle_gain, 0.0, 1.0))
            brake = 0.0
        elif longitudinal < -self.longitudinal_deadband:
            throttle = 0.0
            brake = float(np.clip(-longitudinal * self.brake_gain, 0.0, 1.0))
        else:
            throttle = 0.0
            brake = 0.0
        return ModelControl(throttle=throttle, steer=steer, brake=brake)

    def close(self) -> None:
        self.model.to("cpu")


def create_driver(config: ModelDriverConfig) -> ImitationDrivingPredictor:
    return ImitationDrivingPredictor(config)


__all__ = ["ImitationDrivingPredictor", "create_driver"]
