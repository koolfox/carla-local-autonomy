from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from carla_vision.voxel.contracts import VoxelGridSpec
from carla_vision.voxel.model_examples.temporal_baseline import create_predictor
from carla_vision.voxel.models import CameraVoxelModelConfig
from carla_vision.voxel.training.model import TemporalVoxelModelConfig, TemporalVoxelNet


def test_checkpoint_factory_preserves_rgb_only_contract(tmp_path: Path) -> None:
    spec = VoxelGridSpec(
        x_min=0.0,
        x_max=8.0,
        y_min=-4.0,
        y_max=4.0,
        z_min=-1.0,
        z_max=3.0,
        resolution=1.0,
    )
    config = TemporalVoxelModelConfig(
        horizons_s=(0.0, 1.0),
        grid_shape_zyx=spec.shape,
        semantic_classes=7,
        encoder_channels=32,
        recurrent_channels=48,
        seed_channels=32,
        decoder_channels=16,
    )
    model = TemporalVoxelNet(config)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model_config": config.as_dict(),
            "state_dict": model.state_dict(),
            "grid_spec": spec.as_dict(),
            "history_frames": 3,
            "image_size_hw": [48, 64],
            "horizons_s": [0.0, 1.0],
        },
        checkpoint,
    )
    predictor = create_predictor(
        CameraVoxelModelConfig(checkpoint=checkpoint, device="cpu")
    )
    history = tuple(np.zeros((60, 80, 3), dtype=np.uint8) for _ in range(3))
    prediction = predictor.predict(history, spec)
    assert prediction.occupancy_probability.shape == (2, *spec.shape)
    assert prediction.semantic_logits is not None
    assert prediction.metadata == {
        "model": "TemporalVoxelNet",
        "checkpoint": str(checkpoint.resolve()),
        "rgb_only": True,
    }
