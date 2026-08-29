"""Standard TorchScript adapter for external end-to-end driving policies.

The adapter is intentionally narrow. It loads only a TorchScript artifact via
``torch.jit.load`` and expects the model to return three already-bounded values
in ``[throttle, steer, brake]`` order. Arbitrary eager PyTorch/state-dict models
must use the explicit trusted ``python_factory`` model-package runtime instead.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

from .model_driver import ModelControl, ModelDriverConfig, ModelObservation


def _number_list(raw: Any, name: str, *, length: int) -> tuple[float, ...]:
    if not isinstance(raw, (list, tuple)) or len(raw) != length:
        raise ValueError(f"{name} must contain exactly {length} numbers")
    values = tuple(float(value) for value in raw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain finite numbers")
    return values


class TorchScriptControlDriver:
    def __init__(self, config: ModelDriverConfig) -> None:
        if config.checkpoint is None:
            raise ValueError("TorchScript driving policy requires a checkpoint")
        options = dict(config.options)
        image = options.get("image")
        if not isinstance(image, Mapping):
            raise ValueError("torchscript_control_v1 requires options.image")
        self.width = int(image.get("width", 0))
        self.height = int(image.get("height", 0))
        if not 32 <= self.width <= 4096 or not 32 <= self.height <= 4096:
            raise ValueError("TorchScript image width/height must be in [32, 4096]")
        self.color = str(image.get("color", "rgb")).strip().lower()
        if self.color not in {"rgb", "bgr"}:
            raise ValueError("TorchScript image color must be rgb or bgr")
        self.mean = _number_list(image.get("mean", [0.0, 0.0, 0.0]), "image.mean", length=3)
        self.std = _number_list(image.get("std", [1.0, 1.0, 1.0]), "image.std", length=3)
        if any(value <= 0.0 for value in self.std):
            raise ValueError("image.std values must be positive")
        speed = options.get("speed", {})
        if not isinstance(speed, Mapping):
            raise ValueError("options.speed must be an object")
        self.speed_enabled = bool(speed.get("enabled", False))
        unit = str(speed.get("unit", "mps")).strip().lower()
        if unit not in {"mps", "kmh"}:
            raise ValueError("speed.unit must be mps or kmh")
        self.speed_unit = unit

        try:
            import torch
        except (ImportError, OSError) as error:  # pragma: no cover - profile dependent
            raise RuntimeError("TorchScript driving policy requires PyTorch") from error
        self._torch = torch
        self.device = torch.device(config.device)
        self.model = torch.jit.load(str(config.checkpoint), map_location=self.device)
        self.model.eval()

    def reset(self) -> None:
        reset = getattr(self.model, "reset", None)
        if callable(reset):
            reset()

    def _image_tensor(self, observation: ModelObservation) -> Any:
        frame = observation.image_bgr
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if self.color == "rgb":
            frame = frame[:, :, ::-1]
        array = np.ascontiguousarray(frame.transpose(2, 0, 1), dtype=np.float32) / 255.0
        mean = np.asarray(self.mean, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(self.std, dtype=np.float32).reshape(3, 1, 1)
        array = (array - mean) / std
        return self._torch.from_numpy(array).unsqueeze(0).to(self.device)

    def predict(self, observation: ModelObservation) -> ModelControl:
        image = self._image_tensor(observation)
        with self._torch.inference_mode():
            if self.speed_enabled:
                speed = observation.speed_mps
                if self.speed_unit == "kmh":
                    speed *= 3.6
                speed_tensor = self._torch.tensor(
                    [[float(speed)]], dtype=self._torch.float32, device=self.device
                )
                output = self.model(image, speed_tensor)
            else:
                output = self.model(image)
        if isinstance(output, (tuple, list)):
            if len(output) != 3:
                raise ValueError("TorchScript policy tuple/list output must contain 3 values")
            values = [float(item.detach().reshape(-1)[0].cpu().item()) for item in output]
        elif hasattr(output, "detach"):
            flat = output.detach().reshape(-1).cpu()
            if int(flat.numel()) != 3:
                raise ValueError("TorchScript policy tensor output must contain exactly 3 values")
            values = [float(value) for value in flat.tolist()]
        else:
            raise TypeError("TorchScript policy must return one tensor or a 3-value tensor tuple/list")
        return ModelControl(throttle=values[0], steer=values[1], brake=values[2])

    def close(self) -> None:
        self.model = None


def create_driver(config: ModelDriverConfig) -> TorchScriptControlDriver:
    return TorchScriptControlDriver(config)


__all__ = ["TorchScriptControlDriver", "create_driver"]
