"""Eager PyTorch driving model adapter.

The runtime boundary intentionally does not own CARLA state. It receives a
ModelObservation and returns a validated ModelControl.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import cv2
import numpy as np

from .model_driver import ModelControl, ModelDriverConfig, ModelObservation


class TorchEagerControlDriver:
    """Adapter for a trusted eager PyTorch module factory."""

    def __init__(self, config: ModelDriverConfig, model: Any | None = None) -> None:
        if config.checkpoint is None and model is None:
            raise ValueError("eager PyTorch driving policy requires a checkpoint or model")

        try:
            import torch
        except (ImportError, OSError) as error:  # pragma: no cover
            raise RuntimeError("PyTorch is required") from error

        self._torch = torch
        self.device = torch.device(config.device)
        self.width = int(config.options.get("width", 320))
        self.height = int(config.options.get("height", 180))

        if model is None:
            loaded = torch.load(str(config.checkpoint), map_location=self.device, weights_only=False)
            if isinstance(loaded, dict):
                raise TypeError("state_dict checkpoints require an explicit model factory")
            model = loaded

        if not hasattr(model, "to") or not hasattr(model, "eval"):
            raise TypeError("checkpoint must contain a torch module")

        self.model = model.to(self.device)
        self.model.eval()

    def reset(self) -> None:
        reset = getattr(self.model, "reset", None)
        if callable(reset):
            reset()

    def _image_tensor(self, observation: ModelObservation) -> Any:
        image = observation.image_bgr
        if image.shape[:2] != (self.height, self.width):
            image = cv2.resize(image, (self.width, self.height))
        array = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return self._torch.from_numpy(array).unsqueeze(0).to(self.device)

    def predict(self, observation: ModelObservation) -> ModelControl:
        with self._torch.inference_mode():
            output = self.model(
                self._image_tensor(observation),
                self._torch.tensor([[observation.speed_mps]], device=self.device),
            )

        if isinstance(output, dict):
            values = [output["throttle"], output["steer"], output["brake"]]
        else:
            values = output.detach().reshape(-1).cpu().tolist()

        if len(values) != 3 or not all(math.isfinite(float(v)) for v in values):
            raise ValueError("invalid driving model output")

        return ModelControl(
            throttle=float(values[0]),
            steer=float(values[1]),
            brake=float(values[2]),
        )

    def close(self) -> None:
        self.model = None


def create_driver(config: ModelDriverConfig) -> TorchEagerControlDriver:
    return TorchEagerControlDriver(config)


__all__ = ["TorchEagerControlDriver", "create_driver"]
