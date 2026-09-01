from __future__ import annotations

import pytest

from carla_vision.navigation_intent import (
    CONDITIONAL_IMITATION_COMMAND_ID,
    NAVIGATION_COMMAND_ORDER,
    NavigationCommand,
    NavigationIntent,
    command_from_road_option,
)


def _intent(command: NavigationCommand = NavigationCommand.LEFT) -> NavigationIntent:
    return NavigationIntent(
        source_frame_id=42,
        command=command,
        direction=(0.8, -0.6),
        target_point_m=(20.0, -15.0),
        distance_to_maneuver_m=12.5,
        route_polyline_m=((0.0, 0.0), (8.0, 0.0), (20.0, -15.0)),
        route_id="route-episode-1-leg-000-d012",
        source="carla_global_route_planner_via_behavior_agent",
        confidence=1.0,
        privileged=True,
    )


def test_navigation_intent_round_trips_as_carla_independent_json() -> None:
    original = _intent()
    payload = original.as_dict()
    restored = NavigationIntent.from_mapping(payload)

    assert restored == original
    assert payload["source_frame"] == {
        "kind": "carla_world_frame",
        "id": 42,
        "exact": True,
    }
    assert payload["conditional_imitation_learning"]["core_command_id"] == 3
    assert CONDITIONAL_IMITATION_COMMAND_ID[NavigationCommand.FOLLOW_LANE] == 2


def test_model_features_are_stable_and_horizontal_mirror_is_semantic() -> None:
    original = _intent()
    mirrored = original.mirrored()

    assert mirrored.command is NavigationCommand.RIGHT
    assert mirrored.direction == pytest.approx((0.8, 0.6))
    assert mirrored.target_point_m == pytest.approx((20.0, 15.0))
    assert mirrored.route_polyline_m[-1] == pytest.approx((20.0, 15.0))
    features = original.model_features()
    assert len(features) == len(NAVIGATION_COMMAND_ORDER) + 3
    assert features[original.command_index] == 1.0
    assert sum(features[: len(NAVIGATION_COMMAND_ORDER)]) == 1.0


@pytest.mark.parametrize(
    ("road_option", "expected"),
    [
        (2, NavigationCommand.RIGHT),
        (3, NavigationCommand.STRAIGHT),
        (4, NavigationCommand.FOLLOW_LANE),
        (5, NavigationCommand.CHANGE_LANE_LEFT),
        ("LANEFOLLOW", NavigationCommand.FOLLOW_LANE),
    ],
)
def test_official_carla_road_options_map_without_importing_carla(
    road_option: int | str,
    expected: NavigationCommand,
) -> None:
    assert command_from_road_option(road_option) is expected


def test_navigation_intent_rejects_non_unit_or_non_exact_input() -> None:
    with pytest.raises(ValueError, match="unit vector"):
        NavigationIntent(
            source_frame_id=1,
            command=NavigationCommand.FOLLOW_LANE,
            direction=(2.0, 0.0),
            target_point_m=(5.0, 0.0),
            distance_to_maneuver_m=0.0,
            route_polyline_m=(),
            route_id="route",
            source="teacher",
            confidence=1.0,
            privileged=True,
        )

    payload = _intent().as_dict()
    payload["source_frame"]["exact"] = False
    with pytest.raises(ValueError, match="exact CARLA world frame"):
        NavigationIntent.from_mapping(payload)
