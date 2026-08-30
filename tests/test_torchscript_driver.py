from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from carla_vision.model_driver import ModelDriverConfig, ModelObservation
from carla_vision.torchscript_driver import create_driver

torch = pytest.importorskip("torch")


class _ImageOnlyPolicy(torch.nn.Module):
    def forward(self, image):
        mean = image.mean()
        return torch.stack((mean * 0.0 + 0.25, mean * 0.0 - 0.1, mean * 0.0 + 0.0))


class _SpeedPolicy(torch.nn.Module):
    def forward(self, image, speed):
        mean = image.mean()
        speed_value = speed.reshape(-1)[0]
        return torch.stack((mean * 0.0 + 0.2, speed_value * 0.01, mean * 0.0 + 0.0))


class _InvalidPolicy(torch.nn.Module):
    def forward(self, image):
        mean = image.mean()
        return torch.stack((mean * 0.0 + 1.5, mean * 0.0, mean * 0.0))


def _observation(speed_mps: float = 5.0) -> ModelObservation:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    return ModelObservation(
        frame=1,
        timestamp=1.5,
        image_bgr=image,
        speed_mps=speed_mps,
        dt_seconds=0.05,
    )


def _save_script(module: torch.nn.Module, path: Path) -> None:
    torch.jit.script(module).save(str(path))


def test_torchscript_image_only_policy_executes_through_standard_control_contract(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "policy.pt"
    _save_script(_ImageOnlyPolicy(), checkpoint)
    driver = create_driver(
        ModelDriverConfig(
            checkpoint=checkpoint,
            device="cpu",
            options={
                "image": {
                    "width": 64,
                    "height": 48,
                    "color": "rgb",
                    "mean": [0.0, 0.0, 0.0],
                    "std": [1.0, 1.0, 1.0],
                },
                "speed": {"enabled": False, "unit": "mps"},
            },
        )
    )
    try:
        result = driver.predict(_observation())
    finally:
        driver.close()

    assert result.throttle == pytest.approx(0.25)
    assert result.steer == pytest.approx(-0.1)
    assert result.brake == pytest.approx(0.0)


def test_torchscript_policy_can_receive_explicit_speed_input(tmp_path: Path) -> None:
    checkpoint = tmp_path / "speed.pt"
    _save_script(_SpeedPolicy(), checkpoint)
    driver = create_driver(
        ModelDriverConfig(
            checkpoint=checkpoint,
            device="cpu",
            options={
                "image": {"width": 64, "height": 48, "color": "bgr"},
                "speed": {"enabled": True, "unit": "kmh"},
            },
        )
    )
    try:
        result = driver.predict(_observation(speed_mps=5.0))
    finally:
        driver.close()

    assert result.throttle == pytest.approx(0.2)
    assert result.steer == pytest.approx(0.18)
    assert result.brake == pytest.approx(0.0)


def test_torchscript_invalid_control_is_rejected_by_model_control_validation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "invalid.pt"
    _save_script(_InvalidPolicy(), checkpoint)
    driver = create_driver(
        ModelDriverConfig(
            checkpoint=checkpoint,
            device="cpu",
            options={"image": {"width": 64, "height": 48}},
        )
    )
    try:
        with pytest.raises(ValueError, match="throttle"):
            driver.predict(_observation())
    finally:
        driver.close()


def test_torchscript_runtime_rejects_non_boolean_speed_contract_before_loading(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"not-loaded")

    with pytest.raises(TypeError, match="speed.enabled must be a boolean"):
        create_driver(
            ModelDriverConfig(
                checkpoint=checkpoint,
                device="cpu",
                options={
                    "image": {"width": 64, "height": 48},
                    "speed": {"enabled": 1, "unit": "mps"},
                },
            )
        )
