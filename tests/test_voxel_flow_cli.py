from __future__ import annotations

from carla_vision.voxel.flow_capture import build_parser as build_capture_parser
from carla_vision.voxel.flow_capture import run as run_capture
from carla_vision.voxel.training.flow_runner import build_parser as build_train_parser
from carla_vision.voxel.training.flow_runner import run as run_train


def test_flow_capture_dry_run_does_not_require_carla() -> None:
    args = build_capture_parser().parse_args(["--dry-run", "--frames", "3"])
    summary = run_capture(args)
    assert summary["mode"] == "teacher-flow"
    assert summary["rgb_only_inference"] is True
    assert "optical_flow" in summary["privileged_teacher_sensors"]


def test_flow_training_synthetic_dry_run() -> None:
    args = build_train_parser().parse_args(
        [
            "--dry-run",
            "--device",
            "cpu",
            "--image-size",
            "32x48",
            "--encoder-channels",
            "16",
            "--recurrent-channels",
            "16",
            "--seed-channels",
            "16",
            "--decoder-channels",
            "8",
        ]
    )
    summary = run_train(args)
    assert summary["synthetic"] is True
    assert summary["flow_shape"][1] == 3
    assert summary["flow_valid_voxels"] > 0
