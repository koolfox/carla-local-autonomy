"""Checkpoint-backed RGB-only temporal voxel predictor factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ..contracts import CameraVoxelPrediction, VoxelGridSpec
from ..models import CameraVoxelModelConfig
from ..training.model import TemporalVoxelModelConfig, TemporalVoxelNet


def _spec_from_checkpoint(payload: dict[str, Any]) -> VoxelGridSpec:
    grid = dict(payload["grid_spec"])
    grid.pop("shape_zyx", None)
    return VoxelGridSpec(**grid)


class TemporalBaselinePredictor:
    def __init__(self, config: CameraVoxelModelConfig) -> None:
        if config.checkpoint is None:
            raise ValueError("temporal voxel predictor requires a checkpoint")
        checkpoint = Path(config.checkpoint).expanduser().resolve(strict=True)
        self.device = torch.device(config.device)
        payload = torch.load(checkpoint, map_location=self.device, weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("temporal voxel checkpoint must contain a mapping")
        self.spec = _spec_from_checkpoint(payload)
        self.model_config = TemporalVoxelModelConfig.from_dict(dict(payload["model_config"]))
        self.model = TemporalVoxelNet(self.model_config).to(self.device)
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.eval()
        self.history_frames = int(payload["history_frames"])
        self.image_size = tuple(int(value) for value in payload["image_size_hw"])
        self.horizons_s = tuple(float(value) for value in payload["horizons_s"])
        self.checkpoint = checkpoint

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("RGB history frames must be uint8 HxWx3 arrays")
        height, width = self.image_size
        resized = cv2.resize(array, (width, height), interpolation=cv2.INTER_AREA)
        chw = np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return torch.from_numpy(chw)

    def predict(
        self,
        rgb_history: tuple[np.ndarray, ...],
        spec: VoxelGridSpec,
    ) -> CameraVoxelPrediction:
        if spec != self.spec:
            raise ValueError("runtime voxel-grid spec does not match checkpoint")
        if len(rgb_history) < self.history_frames:
            raise ValueError(
                f"predictor requires {self.history_frames} RGB frames, got {len(rgb_history)}"
            )
        selected = rgb_history[-self.history_frames :]
        tensor = torch.stack([self._preprocess(frame) for frame in selected])
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.inference_mode():
            outputs = self.model(tensor)
            occupancy = torch.sigmoid(outputs["occupancy_logits"])[0]
            semantics = outputs["semantic_logits"][0]
        return CameraVoxelPrediction(
            occupancy_probability=occupancy.cpu().numpy().astype(np.float32),
            horizons_s=self.horizons_s,
            semantic_logits=semantics.cpu().numpy().astype(np.float32),
            metadata={
                "model": "TemporalVoxelNet",
                "checkpoint": str(self.checkpoint),
                "rgb_only": True,
            },
        )


def create_predictor(config: CameraVoxelModelConfig) -> TemporalBaselinePredictor:
    return TemporalBaselinePredictor(config)
