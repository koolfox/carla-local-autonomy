"""Small temporal RGB-only voxel occupancy baseline."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class TemporalVoxelModelConfig:
    horizons_s: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
    grid_shape_zyx: tuple[int, int, int] = (14, 100, 100)
    semantic_classes: int = 7
    encoder_channels: int = 64
    recurrent_channels: int = 128
    seed_channels: int = 64
    decoder_channels: int = 24

    def __post_init__(self) -> None:
        horizons = tuple(float(value) for value in self.horizons_s)
        if not horizons or any(not math.isfinite(value) or value < 0 for value in horizons):
            raise ValueError("model horizons must be finite and non-negative")
        if tuple(sorted(horizons)) != horizons or len(set(horizons)) != len(horizons):
            raise ValueError("model horizons must be unique and increasing")
        if len(self.grid_shape_zyx) != 3 or any(value <= 0 for value in self.grid_shape_zyx):
            raise ValueError("grid_shape_zyx must contain three positive dimensions")
        for name in (
            "semantic_classes",
            "encoder_channels",
            "recurrent_channels",
            "seed_channels",
            "decoder_channels",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        object.__setattr__(self, "horizons_s", horizons)
        object.__setattr__(self, "grid_shape_zyx", tuple(int(v) for v in self.grid_shape_zyx))

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["horizons_s"] = list(self.horizons_s)
        payload["grid_shape_zyx"] = list(self.grid_shape_zyx)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TemporalVoxelModelConfig":
        values = dict(payload)
        values["horizons_s"] = tuple(float(v) for v in values["horizons_s"])
        values["grid_shape_zyx"] = tuple(int(v) for v in values["grid_shape_zyx"])
        return cls(**values)


class _Conv2dBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 2) -> None:
        groups = 8 if out_channels % 8 == 0 else 4
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )


class _Conv3dBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        groups = 8 if out_channels % 8 == 0 else 4
        super().__init__(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )


class TemporalVoxelNet(nn.Module):
    """Encode RGB history, fuse it with a GRU, and decode a dense 3D world state."""

    seed_shape = (2, 4, 4)

    def __init__(self, config: TemporalVoxelModelConfig) -> None:
        super().__init__()
        self.config = config
        mid = max(16, config.encoder_channels // 2)
        self.frame_encoder = nn.Sequential(
            _Conv2dBlock(3, 16),
            _Conv2dBlock(16, mid),
            _Conv2dBlock(mid, config.encoder_channels),
            nn.AdaptiveAvgPool2d(1),
        )
        self.temporal = nn.GRU(
            input_size=config.encoder_channels,
            hidden_size=config.recurrent_channels,
            batch_first=True,
        )
        seed_values = config.seed_channels * math.prod(self.seed_shape)
        self.seed_projection = nn.Linear(config.recurrent_channels, seed_values)
        decoder_mid = max(config.decoder_channels, config.seed_channels // 2)
        self.decoder = nn.Sequential(
            _Conv3dBlock(config.seed_channels, decoder_mid),
            _Conv3dBlock(decoder_mid, config.decoder_channels),
        )
        horizon_count = len(config.horizons_s)
        self.occupancy_head = nn.Conv3d(config.decoder_channels, horizon_count, kernel_size=1)
        self.semantic_head = nn.Conv3d(
            config.decoder_channels,
            horizon_count * config.semantic_classes,
            kernel_size=1,
        )

    def forward(self, rgb_history: torch.Tensor) -> dict[str, torch.Tensor]:
        if rgb_history.ndim != 5 or rgb_history.shape[2] != 3:
            raise ValueError("rgb_history must have shape (B, T, 3, H, W)")
        if not torch.is_floating_point(rgb_history):
            raise ValueError("rgb_history must be a floating-point tensor")
        batch, timesteps, channels, height, width = rgb_history.shape
        frames = rgb_history.reshape(batch * timesteps, channels, height, width)
        encoded = self.frame_encoder(frames).flatten(1)
        encoded = encoded.reshape(batch, timesteps, -1)
        _, hidden = self.temporal(encoded)
        seed = self.seed_projection(hidden[-1])
        seed = seed.reshape(batch, self.config.seed_channels, *self.seed_shape)
        features = self.decoder(seed)
        features = F.interpolate(
            features,
            size=self.config.grid_shape_zyx,
            mode="trilinear",
            align_corners=False,
        )
        occupancy_logits = self.occupancy_head(features)
        semantic = self.semantic_head(features)
        horizon_count = len(self.config.horizons_s)
        semantic_logits = semantic.reshape(
            batch,
            horizon_count,
            self.config.semantic_classes,
            *self.config.grid_shape_zyx,
        )
        return {
            "occupancy_logits": occupancy_logits,
            "semantic_logits": semantic_logits,
        }

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_temporal_voxel_model(
    *,
    horizons_s: Sequence[float],
    grid_shape_zyx: Sequence[int],
    semantic_classes: int = 7,
    options: dict[str, Any] | None = None,
) -> TemporalVoxelNet:
    values = dict(options or {})
    config = TemporalVoxelModelConfig(
        horizons_s=tuple(float(v) for v in horizons_s),
        grid_shape_zyx=tuple(int(v) for v in grid_shape_zyx),
        semantic_classes=int(semantic_classes),
        **values,
    )
    return TemporalVoxelNet(config)
