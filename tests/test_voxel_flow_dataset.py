from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from carla_vision.voxel.training.flow_dataset import TemporalVoxelFlowDataset


def _write_episode(root: Path) -> None:
    (root / "rgb").mkdir(parents=True)
    (root / "teacher_voxels").mkdir()
    (root / "teacher_flow").mkdir()
    grid = {
        "x_min": 0.0,
        "x_max": 4.0,
        "y_min": -2.0,
        "y_max": 2.0,
        "z_min": -1.0,
        "z_max": 1.0,
        "resolution": 1.0,
    }
    shape = (2, 4, 4)
    records = []
    future = []
    for frame in range(4):
        name = f"{frame:08d}"
        image = np.full((12, 16, 3), frame * 10, dtype=np.uint8)
        assert cv2.imwrite(str(root / "rgb" / f"{name}.png"), image)
        occupancy = np.zeros(shape, dtype=np.int8)
        occupancy[0, 1, 1] = 1
        semantics = np.full(shape, 255, dtype=np.uint8)
        semantics[0, 1, 1] = 10
        np.savez_compressed(
            root / "teacher_voxels" / f"{name}.npz",
            occupancy=occupancy,
            semantics=semantics,
        )
        flow_path = None
        if frame < 3:
            velocity = np.zeros((3, *shape), dtype=np.float32)
            velocity[0, 0, 1, 1] = 2.0
            valid = np.zeros(shape, dtype=bool)
            valid[0, 1, 1] = True
            np.savez_compressed(
                root / "teacher_flow" / f"{name}.npz",
                velocity_mps=velocity,
                valid=valid,
            )
            flow_path = f"teacher_flow/{name}.npz"
        records.append(
            {
                "frame": frame,
                "timestamp": frame * 0.5,
                "rgb": f"rgb/{name}.png",
                "teacher_voxel": f"teacher_voxels/{name}.npz",
                "teacher_flow": flow_path,
            }
        )
        future.append(
            {
                "frame": frame,
                "future_targets": {"0.000": frame},
            }
        )
    (root / "sequence.json").write_text(
        json.dumps({"grid": grid, "records": records, "future_targets": future}),
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps({"episode_id": root.name, "route_id": root.name, "seed": 1}),
        encoding="utf-8",
    )


def test_flow_dataset_filters_anchor_without_next_frame_label(tmp_path: Path) -> None:
    root = tmp_path / "episode"
    _write_episode(root)
    dataset = TemporalVoxelFlowDataset(
        [root],
        partition="train",
        history_frames=1,
        horizons_s=(0.0,),
        image_size=(12, 16),
        val_fraction=0.0,
        test_fraction=0.0,
    )
    assert len(dataset) == 3
    sample = dataset[0]
    assert tuple(sample["flow_mps"].shape) == (3, 2, 4, 4)
    assert int(sample["flow_valid"].sum()) == 1
