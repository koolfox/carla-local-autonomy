"""Dataset wrapper that adds sparse privileged voxel-flow supervision."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import TemporalVoxelDataset

Partition = Literal["train", "val", "test"]


class TemporalVoxelFlowDataset(Dataset[dict[str, Any]]):
    """Wrap :class:`TemporalVoxelDataset` and require anchor-frame flow labels."""

    def __init__(
        self,
        roots: Sequence[Path | str],
        *,
        partition: Partition,
        history_frames: int = 4,
        horizons_s: Sequence[float] = (0.0, 0.5, 1.0, 2.0),
        image_size: tuple[int, int] = (192, 320),
        split_seed: int = 17,
        val_fraction: float = 0.2,
        test_fraction: float = 0.1,
        max_samples: int | None = None,
    ) -> None:
        self.base = TemporalVoxelDataset(
            roots,
            partition=partition,
            history_frames=history_frames,
            horizons_s=horizons_s,
            image_size=image_size,
            split_seed=split_seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            max_samples=None,
        )
        eligible: list[int] = []
        for index, reference in enumerate(self.base.samples):
            episode = self.base.episodes[reference.episode_index]
            anchor = episode.records[reference.anchor_index]
            relative = anchor.get("teacher_flow")
            if not relative:
                continue
            if (episode.root / str(relative)).is_file():
                eligible.append(index)
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            eligible = eligible[:max_samples]
        self.indices = tuple(eligible)
        self.spec = self.base.spec
        self.partition = self.base.partition
        self.horizons_s = self.base.horizons_s
        self.history_frames = self.base.history_frames
        self.image_size = self.base.image_size

    @property
    def episodes(self):
        return self.base.episodes

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_index = self.indices[index]
        sample = dict(self.base[base_index])
        reference = self.base.samples[base_index]
        episode = self.base.episodes[reference.episode_index]
        anchor = episode.records[reference.anchor_index]
        path = episode.root / str(anchor["teacher_flow"])
        with np.load(path, allow_pickle=False) as payload:
            velocity = np.asarray(payload["velocity_mps"], dtype=np.float32)
            valid = np.asarray(payload["valid"], dtype=bool)
        if velocity.shape != (3, *self.spec.shape):
            raise ValueError(f"teacher flow shape does not match grid spec: {path}")
        if valid.shape != self.spec.shape:
            raise ValueError(f"teacher flow valid-mask shape does not match grid spec: {path}")
        if not np.isfinite(velocity).all():
            raise ValueError(f"teacher flow contains non-finite values: {path}")
        sample["flow_mps"] = torch.from_numpy(velocity.copy())
        sample["flow_valid"] = torch.from_numpy(valid.copy())
        sample["flow_path"] = str(path)
        return sample
