from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from carla_vision.voxel.benchmark import benchmark_voxel_runs
from carla_vision.voxel.evidence_builder import build_voxel_readiness_evidence


def _build_run(root: Path) -> Path:
    (root / "teacher_voxels").mkdir(parents=True)
    (root / "predicted_voxels").mkdir(parents=True)
    (root / "rgb").mkdir(parents=True)
    for frame in (1, 2):
        (root / "rgb" / f"{frame:08d}.png").write_bytes(b"rgb-placeholder")
    grid = {
        "x_min": 0.0,
        "x_max": 2.0,
        "y_min": 0.0,
        "y_max": 2.0,
        "z_min": 0.0,
        "z_max": 1.0,
        "resolution": 1.0,
    }
    current = np.zeros((1, 2, 2), dtype=np.int8)
    current[0, 0, 0] = 1
    future = np.zeros((1, 2, 2), dtype=np.int8)
    future[0, 0, 1] = 1
    semantics = np.full(current.shape, 255, dtype=np.uint8)
    np.savez_compressed(
        root / "teacher_voxels" / "00000001.npz",
        occupancy=current,
        semantics=semantics,
    )
    np.savez_compressed(
        root / "teacher_voxels" / "00000002.npz",
        occupancy=future,
        semantics=semantics,
    )
    probability = np.full((2, 1, 2, 2), 0.05, dtype=np.float32)
    probability[0, 0, 0, 0] = 0.95
    probability[1, 0, 0, 1] = 0.95
    np.savez_compressed(
        root / "predicted_voxels" / "00000001.npz",
        occupancy_probability=probability,
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
    (root / "sequence.json").write_text(
        json.dumps(
            {
                "grid": grid,
                "records": records,
                "future_targets": [
                    {"frame": 1, "future_targets": {"0.000": 1, "1.000": 2}},
                    {"frame": 2, "future_targets": {"0.000": 2, "1.000": None}},
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps({"episode_id": "episode-a", "route_id": "route-a", "seed": 17}),
        encoding="utf-8",
    )
    return root


def _shadow(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "read_only": True,
                "actuation_enabled": False,
                "control_calls": 0,
                "records": 2,
                "errors": 0,
                "stop_reason": "frames",
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "prediction_latency_ms": 20.0,
            "uncertain_voxel_fraction": 0.10,
            "actuation_applied": False,
        },
        {
            "prediction_latency_ms": 30.0,
            "uncertain_voxel_fraction": 0.20,
            "actuation_applied": False,
        },
    ]
    (root / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return root


def _checkpoint(path: Path) -> Path:
    torch.save(
        {
            "model_config": {"example": True},
            "state_dict": {},
            "grid_spec": {
                "x_min": 0.0,
                "x_max": 2.0,
                "y_min": 0.0,
                "y_max": 2.0,
                "z_min": 0.0,
                "z_max": 1.0,
                "resolution": 1.0,
            },
            "history_frames": 4,
            "horizons_s": [0.0, 1.0],
        },
        path,
    )
    return path


def test_evidence_builder_binds_checkpoint_benchmark_and_shadow(tmp_path: Path) -> None:
    run_root = _build_run(tmp_path / "voxel-run")
    benchmark = benchmark_voxel_runs([run_root])
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
    training_summary = {
        "splits": {"train": ["episode-a"], "val": [], "test": []},
        "best_mean_occupied_iou": 0.8,
        "flow_head": False,
    }
    training_summary_path = tmp_path / "training-summary.json"
    training_summary_path.write_text(json.dumps(training_summary), encoding="utf-8")
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt")
    shadow = _shadow(tmp_path / "shadow")

    evidence = build_voxel_readiness_evidence(
        checkpoint=checkpoint,
        training_summary_path=training_summary_path,
        training_dataset_roots=[run_root],
        benchmark_report_path=benchmark_path,
        shadow_roots=[shadow],
        issue_7_acceptance_complete=False,
        operator_review_complete=False,
    )
    assert evidence["issue_7_acceptance_complete"] is False
    assert evidence["operator_review_complete"] is False
    assert len(evidence["model"]["checkpoint_sha256"]) == 64
    assert evidence["model"]["rgb_only_runtime_contract"] is True
    assert evidence["training"]["dataset_verification_status"] == "passed"
    assert evidence["training"]["route_group_leakage"] is False
    assert evidence["validation"]["future_occupied_iou_1s"] == 1.0
    assert evidence["validation"]["future_model_minus_persistence_iou_1s"] == 1.0
    assert evidence["validation"]["p95_latency_ms"] == 29.5
    assert evidence["shadow"]["actuation_enabled"] is False
    assert evidence["shadow"]["control_calls"] == 0


def test_evidence_builder_reports_split_overlap(tmp_path: Path) -> None:
    run_root = _build_run(tmp_path / "voxel-run")
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(benchmark_voxel_runs([run_root])),
        encoding="utf-8",
    )
    training_summary_path = tmp_path / "training-summary.json"
    training_summary_path.write_text(
        json.dumps(
            {
                "splits": {
                    "train": ["episode-a"],
                    "val": ["episode-a"],
                    "test": [],
                }
            }
        ),
        encoding="utf-8",
    )
    evidence = build_voxel_readiness_evidence(
        checkpoint=_checkpoint(tmp_path / "checkpoint.pt"),
        training_summary_path=training_summary_path,
        training_dataset_roots=[run_root],
        benchmark_report_path=benchmark_path,
        shadow_roots=[_shadow(tmp_path / "shadow")],
        issue_7_acceptance_complete=False,
        operator_review_complete=False,
    )
    assert evidence["training"]["route_group_leakage"] is True
    assert evidence["training"]["route_group_overlap"] == {
        "train:val": ["episode-a"]
    }
