from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from carla_vision.voxel.benchmark import benchmark_voxel_runs


def build_voxel_benchmark_run(root: Path) -> Path:
    (root / "teacher_voxels").mkdir(parents=True)
    (root / "predicted_voxels").mkdir(parents=True)
    (root / "rgb").mkdir(parents=True)
    grid = {
        "x_min": 0.0,
        "x_max": 2.0,
        "y_min": 0.0,
        "y_max": 2.0,
        "z_min": 0.0,
        "z_max": 1.0,
        "resolution": 1.0,
        "shape_zyx": [1, 2, 2],
    }
    current = np.zeros((1, 2, 2), dtype=np.int8)
    current[0, 0, 0] = 1
    future = np.zeros((1, 2, 2), dtype=np.int8)
    future[0, 0, 1] = 1
    semantics_current = np.full(current.shape, 255, dtype=np.uint8)
    semantics_current[current == 1] = 10
    semantics_future = np.full(future.shape, 255, dtype=np.uint8)
    semantics_future[future == 1] = 10
    np.savez_compressed(
        root / "teacher_voxels" / "00000001.npz",
        occupancy=current,
        semantics=semantics_current,
    )
    np.savez_compressed(
        root / "teacher_voxels" / "00000002.npz",
        occupancy=future,
        semantics=semantics_future,
    )
    probabilities = np.full((2, 1, 2, 2), 0.05, dtype=np.float32)
    probabilities[0, 0, 0, 0] = 0.95
    probabilities[1, 0, 0, 1] = 0.95
    np.savez_compressed(
        root / "predicted_voxels" / "00000001.npz",
        occupancy_probability=probabilities,
        horizons_s=np.asarray([0.0, 1.0], dtype=np.float32),
        semantic_logits=np.asarray([], dtype=np.float32),
    )
    records = [
        {
            "frame": 1,
            "timestamp": 0.0,
            "rgb": "rgb/00000001.png",
            "teacher_voxel": "teacher_voxels/00000001.npz",
            "prediction": "predicted_voxels/00000001.npz",
        },
        {
            "frame": 2,
            "timestamp": 1.0,
            "rgb": "rgb/00000002.png",
            "teacher_voxel": "teacher_voxels/00000002.npz",
            "prediction": None,
        },
    ]
    sequence = {
        "grid": grid,
        "records": records,
        "future_targets": [
            {"frame": 1, "future_targets": {"0.000": 1, "1.000": 2}},
            {"frame": 2, "future_targets": {"0.000": 2, "1.000": None}},
        ],
    }
    manifest = {
        "episode_id": "episode-benchmark",
        "route_id": "route-1",
        "seed": 17,
    }
    (root / "sequence.json").write_text(json.dumps(sequence), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_benchmark_aligns_future_predictions_and_beats_persistence(tmp_path: Path) -> None:
    root = build_voxel_benchmark_run(tmp_path / "run")
    report = benchmark_voxel_runs([root], threshold=0.5, uncertainty_band=0.08)
    assert report["status"] == "complete"
    assert report["evaluated_prediction_frame_count"] == 1
    assert report["current_occupied_iou"] == 1.0
    assert report["future_occupied_iou_1s"] == 1.0
    assert report["future_persistence_iou_1s"] == 0.0
    assert report["future_model_minus_persistence_iou_1s"] == 1.0
    assert report["horizons"]["1.000"]["free_precision"] == 1.0
    assert report["mean_brier_score"] < 0.01


def test_benchmark_reports_uncertain_predictions(tmp_path: Path) -> None:
    root = build_voxel_benchmark_run(tmp_path / "run")
    prediction_path = root / "predicted_voxels" / "00000001.npz"
    with np.load(prediction_path, allow_pickle=False) as payload:
        horizons = payload["horizons_s"]
    uncertain = np.full((2, 1, 2, 2), 0.5, dtype=np.float32)
    np.savez_compressed(
        prediction_path,
        occupancy_probability=uncertain,
        horizons_s=horizons,
        semantic_logits=np.asarray([], dtype=np.float32),
    )
    report = benchmark_voxel_runs([root], threshold=0.5, uncertainty_band=0.08)
    assert report["maximum_uncertain_voxel_fraction"] == 1.0
    assert report["mean_brier_score"] == 0.25
