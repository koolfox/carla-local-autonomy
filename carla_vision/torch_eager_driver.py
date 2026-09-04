"""Standard eager PyTorch adapter for external driving policies.

This adapter intentionally supports only a normal trusted Python factory model
shape. It does not discover arbitrary modules or own CARLA state. Model
packages decide when this runtime is allowed through the existing registry
contract.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .model_driver import ModelControl, ModelDriverConfig, ModelObservation


class TorchEagerControlDriver:
    """Small adapter around an eager PyTorch module or factory-created object."""

    def __init__(self, config: ModelDriverConfig) -> None:
        if config.checkpoint is None:
            raise ValueError("eager PyTorch driving policy requires a checkpoint")
        try:
            import torch
        except (ImportError, OSError) as error:  # pragma: no cover
            raise RuntimeError("eager PyTorch driving policy requires PyTorch") from error

        self._torch = torch
        self.device = torch.device(config.device)
        self.width = int(config.options.get("width", 320))
        self.height = int(config.options.get("height", 180))
        self.model = torch.load(
            str(config.checkpoint),
            map_location=self.device,
            weights_only=False,
        )
        if isinstance(self.model, dict):
            raise TypeError("eager adapter requires a serialized torch module, not a state dict")
        self.model.to(self.device)
        self.model.eval()

    def reset(self) -> None:
        reset = getattr(self.model, "reset", None)
        if callable(reset):
            reset()

    def _image_tensor(self, observation: ModelObservation) -> Any:
        image = observation.image_bgr
        if image.shape[:2] != (self.height, self.width):
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        array = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return self._torch.from_numpy(array).unsqueeze(0).to(self.device)

    @staticmethod
    def _control_values(values: list[float]) -> tuple[float, float, float]:
        if len(values) != 3:
            raise ValueError("eager PyTorch policy output must contain exactly 3 values")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("eager PyTorch policy output contains non-finite values")
        throttle, steer, brake = values
        if not 0.0 <= throttle <= 1.0:
            raise ValueError("eager PyTorch throttle must be between 0 and 1")
        if not -1.0 <= steer <= 1.0:
            raise ValueError("eager PyTorch steer must be between -1 and 1")
        if not 0.0 <= brake <= 1.0:
            raise ValueError("eager PyTorch brake must be between 0 and 1")
        return throttle, steer, brake

    def predict(self, observation: ModelObservation) -> ModelControl:
        image = self._image_tensor(observation)
        with self._torch.inference_mode():
            output = self.model(image, self._torch.tensor([[observation.speed_mps]], device=self.device))
        if isinstance(output, dict):
            values = [
                float(output["throttle"]),
                float(output["steer"]),
                float(output["brake"]),
            ]
        else:
            values = output.detach().reshape(-1).cpu().tolist()
        throttle, steer, brake = self._control_values(values)
        return ModelControl(throttle=throttle, steer=steer, brake=brake)

    def close(self) -> None:
        self.model = None


def create_driver(config: ModelDriverConfig) -> TorchEagerControlDriver:
    return TorchEagerControlDriver(config)


__all__ = ["TorchEagerControlDriver", "create_driver"]
