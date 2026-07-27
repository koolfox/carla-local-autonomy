from __future__ import annotations

from types import SimpleNamespace

import pytest

from carla_vision.native.teacher_routes import (
    build_route_leg,
    choose_destination_index,
    serialize_behavior_control,
    validate_behavior_sample_context,
)


def _transform(x: float, y: float) -> SimpleNamespace:
    return SimpleNamespace(
        location=SimpleNamespace(x=x, y=y, z=0.0),
        rotation=SimpleNamespace(pitch=0.0, yaw=0.0, roll=0.0),
    )


def test_destination_selection_is_deterministic_and_distant() -> None:
    spawn_points = [_transform(0, 0), _transform(10, 0), _transform(60, 0), _transform(0, 80)]
    selected = choose_destination_index(
        spawn_points,
        current_location=spawn_points[0].location,
        current_spawn_index=0,
        route_seed=1234,
        leg_index=0,
        minimum_distance_m=40.0,
    )
    repeated = choose_destination_index(
        spawn_points,
        current_location=spawn_points[0].location,
        current_spawn_index=0,
        route_seed=1234,
        leg_index=0,
        minimum_distance_m=40.0,
    )
    assert selected == repeated
    assert selected in {2, 3}


def test_route_leg_contains_explicit_destination_metadata() -> None:
    spawn_points = [_transform(0, 0), _transform(30, 40)]
    leg = build_route_leg(
        episode_id="ep-001",
        leg_index=2,
        route_seed=9,
        start_spawn_index=0,
        destination_spawn_index=1,
        spawn_points=spawn_points,
    ).as_dict()
    assert leg["route_id"] == "route-ep-001-leg-002-d001"
    assert leg["destination_spawn_index"] == 1
    assert leg["straight_line_distance_m"] == pytest.approx(50.0)
    assert leg["destination_world_transform"]["x"] == 30.0


def test_control_serialization_and_context_validation() -> None:
    control = SimpleNamespace(
        throttle=0.3,
        steer=-0.2,
        brake=0.0,
        hand_brake=False,
        reverse=False,
        manual_gear_shift=False,
        gear=1,
    )
    serialized = serialize_behavior_control(
        control,
        carla_frame=88,
        route_id="route-ep-001-leg-000-d001",
    )
    context = {
        "control_mode": "behavior_agent_teacher",
        "route": {
            "route_id": serialized["route_id"],
            "destination_spawn_index": 1,
            "destination_world_transform": {"x": 1.0, "y": 2.0, "z": 0.0},
        },
        "privileged_teacher_control": serialized,
    }
    assert validate_behavior_sample_context(context, carla_frame=88) == []


def test_invalid_control_is_rejected() -> None:
    control = SimpleNamespace(throttle=1.2, steer=0.0, brake=0.0)
    with pytest.raises(ValueError, match="throttle"):
        serialize_behavior_control(control, carla_frame=1, route_id="route")
