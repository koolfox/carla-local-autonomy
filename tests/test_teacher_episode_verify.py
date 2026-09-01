from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from carla_vision.native.teacher_verify import verify_teacher_dataset
from carla_vision.navigation_intent import (
    NAVIGATION_INTENT_SCHEMA_VERSION,
    NavigationCommand,
    NavigationIntent,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_dataset(root: Path) -> Path:
    image_path = root / "images" / "train" / "00000001.png"
    metadata_path = root / "metadata" / "00000001.json"
    image_path.parent.mkdir(parents=True)
    metadata_path.parent.mkdir(parents=True)
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[:, 4:12] = 127
    assert cv2.imwrite(str(image_path), image)
    metadata = {
        "carla_frame": 100,
        "simulation_timestamp_seconds": 5.0,
        "camera": {"width": 16, "height": 12},
        "synchronization": {"exact_carla_frame_match": True},
        "context": {
            "control_mode": "behavior_agent_teacher",
            "route": {
                "route_id": "route-ep-a-leg-000-d001",
                "destination_spawn_index": 1,
                "destination_world_transform": {
                    "x": 10.0,
                    "y": 0.0,
                    "z": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "roll": 0.0,
                },
            },
            "privileged_teacher_control": {
                "source": "BehaviorAgent.run_step",
                "carla_frame": 100,
                "throttle": 0.2,
                "steer": 0.0,
                "brake": 0.0,
            },
            "privileged_evaluation": {"velocity_mps": {"speed": 2.5}},
        },
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    image_rel = image_path.relative_to(root).as_posix()
    metadata_rel = metadata_path.relative_to(root).as_posix()
    dataset = {
        "status": "complete",
        "dataset_id": "teacher-test",
        "sample_count": 1,
        "samples": [
            {
                "sample_id": "00000001",
                "carla_frame": 100,
                "source_timestamp": 5.0,
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
        ],
        "release_metadata": {
            "control_mode": "behavior_agent_teacher",
            "episode_ids": ["ep-a"],
        },
    }
    (root / "dataset.json").write_text(json.dumps(dataset), encoding="utf-8")
    (root / "checksums.sha256").write_text(
        f"{_sha(image_path)}  {image_rel}\n{_sha(metadata_path)}  {metadata_rel}\n",
        encoding="utf-8",
    )
    return root


def test_teacher_dataset_verifier_passes_complete_dataset(tmp_path: Path) -> None:
    report = verify_teacher_dataset(_build_dataset(tmp_path / "dataset"))
    assert report["status"] == "passed"
    assert report["sample_count"] == 1
    assert report["episode_count"] == 1
    assert report["route_leg_count"] == 1


def test_teacher_dataset_verifier_detects_missing_image(tmp_path: Path) -> None:
    root = _build_dataset(tmp_path / "dataset")
    (root / "images" / "train" / "00000001.png").unlink()
    report = verify_teacher_dataset(root)
    assert report["status"] == "failed"
    assert any("missing" in error for error in report["errors"])


def test_teacher_dataset_verifier_detects_incomplete_route_metadata(tmp_path: Path) -> None:
    root = _build_dataset(tmp_path / "dataset")
    metadata_path = root / "metadata" / "00000001.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata["context"]["route"]["destination_world_transform"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    dataset = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    dataset["samples"][0]["metadata"]["sha256"] = _sha(metadata_path)
    (root / "dataset.json").write_text(json.dumps(dataset), encoding="utf-8")
    image_path = root / "images" / "train" / "00000001.png"
    (root / "checksums.sha256").write_text(
        f"{_sha(image_path)}  images/train/00000001.png\n"
        f"{_sha(metadata_path)}  metadata/00000001.json\n",
        encoding="utf-8",
    )
    report = verify_teacher_dataset(root)
    assert report["status"] == "failed"
    assert any("destination_world_transform" in error for error in report["errors"])


def test_teacher_dataset_verifier_rejects_missing_declared_navigation_intent(
    tmp_path: Path,
) -> None:
    root = _build_dataset(tmp_path / "dataset")
    dataset_path = root / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["release_metadata"]["navigation_intent"] = {
        "schema_version": NAVIGATION_INTENT_SCHEMA_VERSION,
        "available_for_every_sample": True,
        "privileged": True,
        "runtime_model_input": False,
        "strict_rgb_only_input": False,
        "route_conditioned_vision_input": True,
    }
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    report = verify_teacher_dataset(root)

    assert report["status"] == "failed"
    assert report["navigation_intent_count"] == 0
    assert any("found 0/1" in error for error in report["errors"])


def test_teacher_dataset_verifier_accepts_exact_frame_navigation_teacher_label(
    tmp_path: Path,
) -> None:
    root = _build_dataset(tmp_path / "dataset")
    metadata_path = root / "metadata" / "00000001.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["context"]["navigation_intent"] = NavigationIntent(
        source_frame_id=100,
        command=NavigationCommand.FOLLOW_LANE,
        direction=(1.0, 0.0),
        target_point_m=(20.0, 0.0),
        distance_to_maneuver_m=0.0,
        route_polyline_m=((0.0, 0.0), (20.0, 0.0)),
        route_id="route-ep-a-leg-000-d001",
        source="carla_global_route_planner_via_behavior_agent",
        confidence=1.0,
        privileged=True,
    ).as_dict()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    dataset_path = root / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["samples"][0]["metadata"]["sha256"] = _sha(metadata_path)
    dataset["samples"][0]["metadata"]["size_bytes"] = metadata_path.stat().st_size
    dataset["release_metadata"]["navigation_intent"] = {
        "schema_version": NAVIGATION_INTENT_SCHEMA_VERSION,
        "available_for_every_sample": True,
        "privileged": True,
        "runtime_model_input": False,
        "strict_rgb_only_input": False,
        "route_conditioned_vision_input": True,
    }
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    image_path = root / "images" / "train" / "00000001.png"
    (root / "checksums.sha256").write_text(
        f"{_sha(image_path)}  images/train/00000001.png\n"
        f"{_sha(metadata_path)}  metadata/00000001.json\n",
        encoding="utf-8",
    )

    report = verify_teacher_dataset(root)

    assert report["status"] == "passed"
    assert report["navigation_intent_count"] == 1
