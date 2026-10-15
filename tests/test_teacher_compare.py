from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from test_dataset_writer import camera_frame

from carla_vision.dataset.sync import SynchronizedFramePair
from carla_vision.dataset.writer import DatasetWriter
from carla_vision.native.teacher_compare import compare_teacher_datasets


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_dataset(
    root: Path,
    *,
    frame_origin: int,
    time_origin: float,
    steer_second: float = 0.1,
) -> Path:
    with DatasetWriter(
        root.parent,
        dataset_id=root.name,
        carla_endpoint={"host": "fixture", "port": 2000},
        carla_version="0.9.16",
        carla_map="Town10HD_Opt",
        minimum_pixels=1,
    ) as writer:
        for index, (offset, seconds, steer) in enumerate(
            ((0, 0.0, 0.0), (2, 0.1, steer_second)), 1
        ):
            frame, timestamp = frame_origin + offset, time_origin + seconds
            pixels = np.full((8, 12, 4), index * 30, np.uint8)
            mask = np.zeros_like(pixels)
            pair = SynchronizedFramePair(
                camera_frame(index, frame, pixels, timestamp=timestamp),
                camera_frame(index, frame, mask, timestamp=timestamp),
                0,
                0,
            )
            writer.add_pair(
                pair,
                split="train",
                scenario_id="fixture",
                episode_id="ep-a",
                context={
                    "control_mode": "behavior_agent_teacher",
                    "route": {
                        "route_id": "route-ep-a-leg-000-d001",
                        "leg_index": 0,
                        "destination_spawn_index": 1,
                        "destination_world_transform": {"x": 30.0, "y": 0.0, "z": 0.0},
                    },
                    "privileged_teacher_control": {
                        "source": "BehaviorAgent.run_step",
                        "carla_frame": frame,
                        "throttle": 0.2,
                        "steer": steer,
                        "brake": 0.0,
                    },
                    "privileged_evaluation": {"velocity_mps": {"speed": 2.0 + index}},
                },
            )
        writer.set_release_metadata(
            {"control_mode": "behavior_agent_teacher", "episode_ids": ["ep-a"]}
        )
    return root


def test_comparison_matches_relative_frames_and_timestamps(tmp_path: Path) -> None:
    left = _build_dataset(tmp_path / "left", frame_origin=100, time_origin=5.0)
    right = _build_dataset(tmp_path / "right", frame_origin=500, time_origin=20.0)
    report = compare_teacher_datasets(left, right)
    assert report["status"] == "matched"
    assert report["compared_sample_count"] == 2
    assert report["mismatch_count"] == 0


def test_comparison_detects_teacher_control_difference(tmp_path: Path) -> None:
    left = _build_dataset(tmp_path / "left", frame_origin=100, time_origin=5.0)
    right = _build_dataset(
        tmp_path / "right",
        frame_origin=500,
        time_origin=20.0,
        steer_second=0.3,
    )
    report = compare_teacher_datasets(left, right, control_tolerance=1e-5)
    assert report["status"] == "failed"
    assert any(item.get("field") == "steer" for item in report["mismatches"])
