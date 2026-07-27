"""Compact RGB-plus-speed control network for BehaviorAgent imitation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ImitationModelConfig:
    image_size_hw: tuple[int, int] = (192, 320)
    speed_scale_mps: float = 20.0
    encoder_channels: int = 48
    hidden_dim: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        height, width = self.image_size_hw
        if height <= 0 or width <= 0:
            raise ValueError("image_size_hw dimensions must be positive")
        if self.encoder_channels <= 0 or self.hidden_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if not math.isfinite(self.speed_scale_mps) or self.speed_scale_mps <= 0.0:
            raise ValueError("speed_scale_mps must be finite and positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["image_size_hw"] = list(self.image_size_hw)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ImitationModelConfig":
        values = dict(payload)
        values["image_size_hw"] = tuple(int(value) for value in values["image_size_hw"])
        return cls(**values)


class ConvNormAct(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, *, stride: int) -> None:
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=5 if input_channels == 3 else 3,
                stride=stride,
                padding=2 if input_channels == 3 else 1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )


class ImitationControlNet(nn.Module):
    """Predict steering and signed longitudinal command from RGB and speed."""

    def __init__(self, config: ImitationModelConfig) -> None:
        super().__init__()
        self.config = config
        channels = config.encoder_channels
        self.image_encoder = nn.Sequential(
            ConvNormAct(3, channels, stride=2),
            ConvNormAct(channels, channels * 2, stride=2),
            ConvNormAct(channels * 2, channels * 3, stride=2),
            ConvNormAct(channels * 3, channels * 4, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.speed_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 32),
            nn.SiLU(inplace=True),
        )
        fused_dim = channels * 4 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.steer_head = nn.Linear(config.hidden_dim, 1)
        self.longitudinal_head = nn.Linear(config.hidden_dim, 1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, image: torch.Tensor, speed: torch.Tensor) -> dict[str, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape (B, 3, H, W)")
        if speed.ndim == 1:
            speed = speed[:, None]
        if speed.ndim != 2 or speed.shape[1] != 1 or speed.shape[0] != image.shape[0]:
            raise ValueError("speed must have shape (B, 1) and match image batch")
        image_features = self.image_encoder(image)
        speed_features = self.speed_encoder(speed)
        features = self.fusion(torch.cat((image_features, speed_features), dim=1))
        steer = torch.tanh(self.steer_head(features)).squeeze(1)
        longitudinal = torch.tanh(self.longitudinal_head(features)).squeeze(1)
        return {"steer": steer, "longitudinal": longitudinal}


__all__ = ["ImitationControlNet", "ImitationModelConfig"]
