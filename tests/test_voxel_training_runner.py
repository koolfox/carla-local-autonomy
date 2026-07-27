from __future__ import annotations

from carla_vision.voxel.training.runner import build_parser, run


def test_training_runner_synthetic_dry_run() -> None:
    args = build_parser().parse_args(
        [
            "--dry-run",
            "--device",
            "cpu",
            "--history-frames",
            "2",
            "--horizons",
            "0,1",
            "--image-size",
            "48x64",
            "--encoder-channels",
            "32",
            "--recurrent-channels",
            "48",
            "--seed-channels",
            "32",
            "--decoder-channels",
            "16",
        ]
    )
    summary = run(args)
    assert summary["dry_run"] is True
    assert summary["synthetic"] is True
    assert summary["occupancy_shape"] == [1, 2, 4, 8, 8]
