from __future__ import annotations

import numpy as np
import pytest

from carla_vision.model_driver import ModelDriverConfig, ModelObservation


torch = pytest.importorskip("torch")

from carla_vision.torch_eager_driver import create_driver


class Policy(torch.nn.Module):
    def forward(self, image, speed):
        assert image.shape == (1, 3, 180, 320)
        assert speed.shape == (1, 1)
        return torch.tensor([[0.1, 0.0, 0.0]], device=image.device)


def observation():
    return ModelObservation(
        frame=1,
        timestamp=1.0,
        image_bgr=np.zeros((180, 320, 3), dtype=np.uint8),
        speed_mps=2.0,
        dt_seconds=0.05,
    )


def test_driver_runs_trusted_module():
    driver = create_driver(ModelDriverConfig(), model=Policy())
    result = driver.predict(observation())
    assert result.throttle == pytest.approx(0.1)


def test_driver_rejects_missing_model():
    with pytest.raises(ValueError):
        create_driver(ModelDriverConfig())
