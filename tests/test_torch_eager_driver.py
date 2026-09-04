from __future__ import annotations

import numpy as np
import pytest

from carla_vision.model_driver import ModelDriverConfig, ModelObservation


torch = pytest.importorskip("torch")

from carla_vision.torch_eager_driver import create_driver


class _Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.reset_called = False

    def reset(self):
        self.reset_called = True

    def forward(self, image, speed):
        assert image.shape[1:] == (3, 180, 320)
        assert speed.shape == (1, 1)
        return torch.tensor([[0.1, 0.0, 0.0]], device=image.device)


class _InvalidPolicy(torch.nn.Module):
    def forward(self, image, speed):
        return torch.tensor([[1.5, 0.0, 0.0]], device=image.device)


def _observation() -> ModelObservation:
    return ModelObservation(
        frame=1,
        timestamp=1.0,
        image_bgr=np.zeros((180, 320, 3), dtype=np.uint8),
        speed_mps=2.0,
        dt_seconds=0.05,
    )


def test_eager_driver_runs_module_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "policy.pt"
    torch.save(_Policy(), checkpoint)

    driver = create_driver(
        ModelDriverConfig(checkpoint=checkpoint, options={"width": 320, "height": 180})
    )
    driver.reset()

    control = driver.predict(_observation())

    assert control.throttle == pytest.approx(0.1)
    assert driver.model.reset_called is True
    driver.close()


def test_eager_driver_rejects_state_dict(tmp_path) -> None:
    checkpoint = tmp_path / "weights.pt"
    torch.save({}, checkpoint)

    with pytest.raises(TypeError, match="state dict"):
        create_driver(ModelDriverConfig(checkpoint=checkpoint))


def test_eager_driver_rejects_invalid_control_output(tmp_path) -> None:
    checkpoint = tmp_path / "invalid.pt"
    torch.save(_InvalidPolicy(), checkpoint)

    driver = create_driver(ModelDriverConfig(checkpoint=checkpoint))
    try:
        with pytest.raises(ValueError, match="throttle"):
            driver.predict(_observation())
    finally:
        driver.close()


def test_eager_driver_requires_checkpoint() -> None:
    with pytest.raises(ValueError, match="checkpoint"):
        create_driver(ModelDriverConfig(checkpoint=None))
