from __future__ import annotations

import pytest

from carla_vision.voxel.local_drive_actuation import _dry_run_payload, parse_args


def test_normal_local_drive_path_keeps_voxel_actuation_disabled() -> None:
    args = parse_args(["--dry-run"])
    assert args.enable_voxel_actuation is False
    payload = _dry_run_payload(args)
    assert "voxel_actuation" not in payload
    assert payload["driver"] == "behavior"


def test_voxel_actuation_requires_explicit_acknowledgement() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--dry-run",
                "--enable-voxel-actuation",
                "--voxel-predictor-factory",
                "example.module:create_predictor",
            ]
        )


def test_voxel_actuation_requires_behavior_driver() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--dry-run",
                "--driver",
                "traffic-manager",
                "--enable-voxel-actuation",
                "--acknowledge-voxel-actuation",
                "--voxel-predictor-factory",
                "example.module:create_predictor",
            ]
        )


def test_voxel_actuation_dry_run_is_explicit_and_default_off() -> None:
    args = parse_args(
        [
            "--dry-run",
            "--enable-voxel-actuation",
            "--acknowledge-voxel-actuation",
            "--voxel-predictor-factory",
            "example.module:create_predictor",
            "--voxel-max-speed-kmh",
            "30",
        ]
    )
    payload = _dry_run_payload(args)
    voxel = payload["voxel_actuation"]
    assert voxel["enabled"] is True
    assert voxel["operator_acknowledged"] is True
    assert voxel["longitudinal_control"] == "BehaviorAgent"
    assert voxel["steering_control"] == "voxel_planner"
    assert voxel["supervisor"]["maximum_speed_mps"] == pytest.approx(30.0 / 3.6)
