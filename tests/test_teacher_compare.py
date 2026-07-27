from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

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
    samples = []
    checksum_lines = []
    for index, (frame_offset, time_offset, steer) in enumerate(
        ((0, 0.0, 0.0), (2, 0.1, steer_second)),
        start=1,
    ):
        sample_id = f"{index:08d}"
        image_path = root / "images" / "train" / f"{sample_id}.png"
        metadata_path = root / "metadata" / f"{sample_id}.json"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((8, 12, 3), index * 30, dtype=np.uint8)
        assert cv2.imwrite(str(image_path), image)
        frame = frame_origin + frame_offset
        timestamp = time_origin + time_offset
        metadata = {
            "carla_frame": frame,
            "simulation_timestamp_seconds": timestamp,
            "camera": {"width": 12, "height": 8},
            "synchronization": {"exact_carla_frame_match": True},
            "context": {
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
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        image_rel = image_path.relative_to(root).as_posix()
        metadata_rel = metadata_path.relative_to(root).as_posix()
        samples.append(
            {
                "sample_id": sample_id,
                "carla_frame": frame,
                "source_timestamp": timestamp,
                "episode_id": "ep-a",
                "rgb": {
                    "path": image_rel,
                    "sha256": _sha(image_path),
                    "size_bytes": image_path.stat().st_size,
                },
                "metadata": {
                    "path": metadata_rel,
                    "sha256": _sha(metadata_path),
                    "size_bytes": metadata_path.stat().st_size,
                },
            }
        )
        checksum_lines.extend(
            (f"{_sha(image_path)}  {image_rel}", f"{_sha(metadata_path)}  {metadata_rel}")
        )
    dataset = {
        "status": "complete",
        "dataset_id": root.name,
        "sample_count": len(samples),
        "samples": samples,
        "release_metadata": {
            "control_mode": "behavior_agent_teacher",
            "episode_ids": ["ep-a"],
        },
    }
    (root / "dataset.json").write_text(json.dumps(dataset), encoding="utf-8")
    (root / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
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
