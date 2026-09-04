from __future__ import annotations

import numpy as np
import pytest

from carla_vision.model_driver import ModelDriverConfig, ModelObservation


torch = pytest.importorskip("torch")

from carla_vision.torch_eager_driver import create_driver


class _Policy(torch.nn.Module):
    def forward(self, image, speed):
        assert image.shape[1:] == (3, 180, 320)
        assert speed.shape == (1, 1)
        return torch.tensor([[0.1, 0.0, 0.0]], device=image.device)


def test_eager_driver_runs_module_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "policy.pt"
    torch.save(_Policy(), checkpoint)

    driver = create_driver(
        ModelDriverConfig(checkpoint=checkpoint, options={"width": 320, "height": 180})
    )
    driver.reset()

    control = driver.predict(
        ModelObservation(
            frame=1,
            timestamp=1.0,
            image_bgr=np.zeros((180, 320, 3), dtype=np.uint8),
            speed_mps=2.0,
            dt_seconds=0.05,
        )
    )

    assert control.throttle == pytest.approx(0.1)
    driver.close()


def test_eager_driver_rejects_state_dict(tmp_path) -> None:
    checkpoint = tmp_path / "weights.pt"
    torch.save({}, checkpoint)

    with pytest.raises(TypeError, match="state dict"):
        create_driver(ModelDriverConfig(checkpoint=checkpoint))
