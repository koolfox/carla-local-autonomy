"""Temporal RGB-only occupancy model with an auxiliary dynamic voxel-flow head."""

from __future__ import annotations

import torch
from torch import nn

from .model import TemporalVoxelModelConfig, TemporalVoxelNet


class TemporalVoxelFlowNet(nn.Module):
    """Keep the existing occupancy model intact and add current-frame flow prediction.

    The auxiliary head receives the horizon occupancy logits as dense 3D features.
    It predicts instantaneous xyz velocity in metres/second for the source frame.
    Flow supervision is sparse and restricted to dynamic teacher voxels.
    """

    def __init__(self, config: TemporalVoxelModelConfig) -> None:
        super().__init__()
        self.config = config
        self.base = TemporalVoxelNet(config)
        horizon_count = len(config.horizons_s)
        hidden = max(8, horizon_count * 4)
        self.flow_head = nn.Sequential(
            nn.Conv3d(horizon_count, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, 3, kernel_size=1),
        )

    def forward(self, rgb_history: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.base(rgb_history)
        outputs["flow_mps"] = self.flow_head(outputs["occupancy_logits"])
        return outputs

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
