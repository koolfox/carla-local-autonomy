from __future__ import annotations

from spawn_vehicle_bridge_and_run import build_runtime_command, parse_args


def test_wrapper_spectator_follow_is_opt_in_and_propagated() -> None:
    default_args = parse_args(["--run-id", "wrapper-default", "--dry-run"])
    follow_args = parse_args(["--run-id", "wrapper-follow", "--dry-run", "--spectator-follow"])

    default_command = build_runtime_command(default_args, vehicle_id=42)
    follow_command = build_runtime_command(follow_args, vehicle_id=42)

    assert "--spectator-follow" not in default_command
    assert follow_command.count("--spectator-follow") == 1
