from __future__ import annotations

import pytest

from carla_vision.local_drive import _dry_run_payload, parse_args


def test_local_drive_defaults_to_looping_behavior_agent() -> None:
    args = parse_args(["--dry-run"])
    assert args.host == "127.0.0.1"
    assert args.driver == "behavior"
    assert args.vehicles == 60
    assert args.walkers == 30
    assert args.loop_destinations is True


def test_model_driver_requires_factory() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--driver", "model", "--dry-run"])


def test_dry_run_describes_scene_without_importing_carla() -> None:
    args = parse_args(
        [
            "--dry-run",
            "--driver",
            "model",
            "--model-factory",
            "carla_vision.model_examples.lane_center:create_driver",
            "--vehicles",
            "12",
            "--walkers",
            "7",
        ]
    )
    payload = _dry_run_payload(args)
    assert payload["connects_to"] == "127.0.0.1:2000"
    assert "spawn up to 12 NPC vehicles" in payload["expected_actions"]
    assert "spawn up to 7 walkers" in payload["expected_actions"]
