from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from carla_vision.voxel.training.dataset import (
    COMPACT_SEMANTIC_IGNORE,
    TemporalVoxelDataset,
    compact_semantic_labels,
    stable_partition,
)


def _write_episode(root: Path, *, episode_id: str = "ep-1", route_id: str = "route-a") -> None:
    (root / "rgb").mkdir(parents=True)
    (root / "teacher_voxels").mkdir()
    records = []
    futures = []
    for frame in range(1, 7):
        stem = f"{frame:08d}"
        image = np.full((24, 32, 3), frame * 10, dtype=np.uint8)
        assert cv2.imwrite(str(root / "rgb" / f"{stem}.png"), image)
        occupancy = np.zeros((2, 4, 4), dtype=np.int8)
        occupancy[0, 1, min(frame - 1, 3)] = 1
        occupancy[1, :, :] = -1
        semantics = np.full((2, 4, 4), 255, dtype=np.uint8)
        semantics[occupancy == 1] = 10
        np.savez_compressed(
            root / "teacher_voxels" / f"{stem}.npz",
            occupancy=occupancy,
            semantics=semantics,
        )
        records.append(
            {
                "frame": frame,
                "timestamp": frame * 0.5,
                "rgb": f"rgb/{stem}.png",
                "teacher_voxel": f"teacher_voxels/{stem}.npz",
            }
        )
        futures.append(
            {
                "frame": frame,
                "future_targets": {
                    "0.000": frame,
                    "0.500": frame + 1 if frame < 6 else None,
                },
            }
        )
    sequence = {
        "grid": {
            "x_min": 0.0,
            "x_max": 4.0,
            "y_min": -2.0,
            "y_max": 2.0,
            "z_min": -1.0,
            "z_max": 1.0,
            "resolution": 1.0,
            "shape_zyx": [2, 4, 4],
        },
        "records": records,
        "future_targets": futures,
    }
    manifest = {"episode_id": episode_id, "route_id": route_id, "seed": 5}
    (root / "sequence.json").write_text(json.dumps(sequence), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_dataset_loads_rgb_history_and_future_targets(tmp_path: Path) -> None:
    root = tmp_path / "episode"
    _write_episode(root)
    dataset = TemporalVoxelDataset(
        [root],
        partition="train",
        history_frames=2,
        horizons_s=(0.0, 0.5),
        image_size=(16, 20),
        val_fraction=0.0,
        test_fraction=0.0,
    )
    assert len(dataset) == 4
    sample = dataset[0]
    assert tuple(sample["rgb_history"].shape) == (2, 3, 16, 20)
    assert tuple(sample["occupancy"].shape) == (2, 2, 4, 4)
    assert tuple(sample["semantics"].shape) == (2, 2, 4, 4)
    assert int(sample["semantics"][sample["occupancy"] == 1][0]) == 3
    assert sample["episode_id"] == "ep-1"


def test_compact_semantics_ignore_free_and_unknown() -> None:
    occupancy = np.asarray([0, 1, 1, -1], dtype=np.int8)
    tags = np.asarray([7, 10, 4, 1], dtype=np.uint8)
    compact = compact_semantic_labels(tags, occupancy)
    assert compact.tolist() == [COMPACT_SEMANTIC_IGNORE, 3, 4, COMPACT_SEMANTIC_IGNORE]


def test_stable_partition_keeps_same_route_seed_together() -> None:
    first = stable_partition("route-a|5", split_seed=9)
    second = stable_partition("route-a|5", split_seed=9)
    assert first == second
