from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from carla_vision.imitation.dataset import BehaviorImitationDataset, split_summary


def _build_dataset(root: Path, *, groups: int = 12) -> Path:
    samples = []
    for index in range(groups):
        sample_id = f"{index + 1:08d}"
        image_path = root / "images" / "train" / f"{sample_id}.png"
        metadata_path = root / "metadata" / f"{sample_id}.json"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        image = np.zeros((16, 24, 3), dtype=np.uint8)
        image[:, :12, 1] = 40 + index
        image[:, 12:, 2] = 120
        assert cv2.imwrite(str(image_path), image)
        steer = -0.4 + 0.8 * index / max(groups - 1, 1)
        metadata = {
            "context": {
                "control_mode": "behavior_agent_teacher",
                "map_family": "Town10HD_Opt",
                "scenario_recipe_id": f"recipe-{index % 3}",
                "group": {"static_layout_seed": index % 4},
                "route": {
                    "route_id": f"route-ep-{index}-leg-000-d{index + 1:03d}",
                    "route_seed": 1000 + index,
                    "start_spawn_index": index,
                    "destination_spawn_index": index + 1,
                },
                "privileged_teacher_control": {
                    "source": "BehaviorAgent.run_step",
                    "throttle": 0.3 if index % 4 else 0.0,
                    "steer": steer,
                    "brake": 0.5 if index % 4 == 0 else 0.0,
                },
                "privileged_evaluation": {
                    "velocity_mps": {"speed": float(2 + index % 6)}
                },
            }
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        samples.append(
            {
                "sample_id": sample_id,
                "episode_id": f"ep-{index}",
                "carla_frame": 100 + index,
                "rgb": {"path": image_path.relative_to(root).as_posix()},
                "metadata": {"path": metadata_path.relative_to(root).as_posix()},
            }
        )
    (root / "dataset.json").write_text(
        json.dumps({"dataset_id": root.name, "samples": samples}),
        encoding="utf-8",
    )
    return root


def test_route_groups_do_not_leak_across_splits(tmp_path: Path) -> None:
    root = _build_dataset(tmp_path / "teacher", groups=40)
    common = {
        "roots": [root],
        "image_size": (16, 24),
        "split_seed": 17,
        "val_fraction": 0.2,
        "test_fraction": 0.2,
        "verify": False,
    }
    datasets = {
        "train": BehaviorImitationDataset(partition="train", **common),
        "val": BehaviorImitationDataset(partition="val", **common),
        "test": BehaviorImitationDataset(partition="test", **common),
    }
    summary = split_summary(datasets)
    assert sum(summary["sample_counts"].values()) == 40
    assert summary["route_group_overlap"] == {}
    assert all(summary["route_group_counts"][name] > 0 for name in datasets)


def test_augmentation_is_reproducible_for_one_sample(tmp_path: Path) -> None:
    root = _build_dataset(tmp_path / "teacher", groups=2)
    dataset = BehaviorImitationDataset(
        [root],
        partition="train",
        image_size=(16, 24),
        val_fraction=0.0,
        test_fraction=0.0,
        augment=True,
        augmentation_seed=55,
        brightness=0.2,
        contrast=0.2,
        verify=False,
    )
    first = dataset[0]
    repeated = dataset[0]
    assert torch.equal(first["image"], repeated["image"])
    assert torch.equal(first["target"], repeated["target"])


def test_horizontal_flip_inverts_teacher_steer(tmp_path: Path) -> None:
    root = _build_dataset(tmp_path / "teacher", groups=2)
    plain = BehaviorImitationDataset(
        [root],
        partition="train",
        image_size=(16, 24),
        val_fraction=0.0,
        test_fraction=0.0,
        augment=False,
        verify=False,
    )
    flipped = BehaviorImitationDataset(
        [root],
        partition="train",
        image_size=(16, 24),
        val_fraction=0.0,
        test_fraction=0.0,
        augment=True,
        brightness=0.0,
        contrast=0.0,
        horizontal_flip_probability=1.0,
        verify=False,
    )
    assert flipped[0]["target"][0].item() == -plain[0]["target"][0].item()
    assert not torch.equal(flipped[0]["image"], plain[0]["image"])
