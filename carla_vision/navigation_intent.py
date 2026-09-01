"""Versioned, CARLA-independent route intent for driving models and artifacts.

CARLA route planners are privileged teachers.  This module deliberately knows
nothing about CARLA objects so retained datasets, PyTorch loaders, runtimes and
frontends can share one factual contract without importing PythonAPI.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

NAVIGATION_INTENT_SCHEMA_VERSION = "1.0"
NAVIGATION_FEATURE_SCHEMA_VERSION = "1.0"


class NavigationCommand(StrEnum):
    FOLLOW_LANE = "follow_lane"
    LEFT = "left"
    RIGHT = "right"
    STRAIGHT = "straight"
    CHANGE_LANE_LEFT = "change_lane_left"
    CHANGE_LANE_RIGHT = "change_lane_right"
    STOP = "stop"


NAVIGATION_COMMAND_ORDER: tuple[NavigationCommand, ...] = tuple(NavigationCommand)
CONDITIONAL_IMITATION_COMMAND_ID: dict[NavigationCommand, int] = {
    NavigationCommand.FOLLOW_LANE: 2,
    NavigationCommand.LEFT: 3,
    NavigationCommand.RIGHT: 4,
    NavigationCommand.STRAIGHT: 5,
}
_MIRRORED_COMMAND = {
    NavigationCommand.LEFT: NavigationCommand.RIGHT,
    NavigationCommand.RIGHT: NavigationCommand.LEFT,
    NavigationCommand.CHANGE_LANE_LEFT: NavigationCommand.CHANGE_LANE_RIGHT,
    NavigationCommand.CHANGE_LANE_RIGHT: NavigationCommand.CHANGE_LANE_LEFT,
}


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _pair(value: Any, name: str) -> tuple[float, float]:
    if isinstance(value, Mapping):
        return (
            _finite(value.get("forward_m", value.get("forward")), f"{name}.forward"),
            _finite(value.get("right_m", value.get("right")), f"{name}.right"),
        )
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise TypeError(f"{name} must contain forward and right coordinates")
    return _finite(value[0], f"{name}[0]"), _finite(value[1], f"{name}[1]")


@dataclass(frozen=True, slots=True)
class NavigationIntent:
    """One route command associated with an exact source frame.

    Coordinates use an ego-centric ground plane: +forward points through the
    bonnet and +right points toward the passenger side.  The privileged flag is
    part of the data, not documentation, so route-conditioned and strict RGB
    experiments cannot be confused silently.
    """

    source_frame_id: int
    command: NavigationCommand
    direction: tuple[float, float]
    target_point_m: tuple[float, float]
    distance_to_maneuver_m: float
    route_polyline_m: tuple[tuple[float, float], ...]
    route_id: str
    source: str
    confidence: float
    privileged: bool

    def __post_init__(self) -> None:
        if isinstance(self.source_frame_id, bool) or not isinstance(self.source_frame_id, int):
            raise TypeError("source_frame_id must be an integer")
        if self.source_frame_id < 0:
            raise ValueError("source_frame_id must be a non-negative integer")
        if not isinstance(self.command, NavigationCommand):
            object.__setattr__(self, "command", NavigationCommand(str(self.command)))
        direction = _pair(self.direction, "direction")
        magnitude = math.hypot(*direction)
        if not math.isclose(magnitude, 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError("direction must be a unit vector")
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "target_point_m", _pair(self.target_point_m, "target_point_m"))
        distance = _finite(self.distance_to_maneuver_m, "distance_to_maneuver_m")
        if distance < 0.0:
            raise ValueError("distance_to_maneuver_m must be non-negative")
        object.__setattr__(self, "distance_to_maneuver_m", distance)
        if len(self.route_polyline_m) > 64:
            raise ValueError("route_polyline_m may contain at most 64 points")
        object.__setattr__(
            self,
            "route_polyline_m",
            tuple(_pair(point, "route_polyline_m point") for point in self.route_polyline_m),
        )
        if not self.route_id.strip():
            raise ValueError("route_id must not be empty")
        if not self.source.strip():
            raise ValueError("source must not be empty")
        confidence = _finite(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        if not isinstance(self.privileged, bool):
            raise TypeError("privileged must be a boolean")

    @property
    def command_index(self) -> int:
        return NAVIGATION_COMMAND_ORDER.index(self.command)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NAVIGATION_INTENT_SCHEMA_VERSION,
            "source_frame": {
                "kind": "carla_world_frame",
                "id": self.source_frame_id,
                "exact": True,
            },
            "command": self.command.value,
            "direction": {
                "forward": self.direction[0],
                "right": self.direction[1],
            },
            "target_point": {
                "forward_m": self.target_point_m[0],
                "right_m": self.target_point_m[1],
            },
            "distance_to_maneuver_m": self.distance_to_maneuver_m,
            "route_polyline": [
                {"forward_m": point[0], "right_m": point[1]}
                for point in self.route_polyline_m
            ],
            "route_id": self.route_id,
            "source": self.source,
            "confidence": self.confidence,
            "privileged": self.privileged,
            "conditional_imitation_learning": {
                "core_command_id": CONDITIONAL_IMITATION_COMMAND_ID.get(self.command),
                "extension": self.command not in CONDITIONAL_IMITATION_COMMAND_ID,
            },
            "experiment_tracks": {
                "strict_rgb_only_input": False,
                "route_conditioned_vision_input": True,
            },
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> NavigationIntent:
        if raw.get("schema_version") != NAVIGATION_INTENT_SCHEMA_VERSION:
            raise ValueError(
                "navigation intent schema_version must be "
                f"{NAVIGATION_INTENT_SCHEMA_VERSION!r}"
            )
        source_frame = raw.get("source_frame")
        if not isinstance(source_frame, Mapping):
            raise TypeError("navigation intent source_frame must be an object")
        if source_frame.get("kind") != "carla_world_frame" or source_frame.get("exact") is not True:
            raise ValueError("navigation intent must declare an exact CARLA world frame")
        source_frame_id = source_frame.get("id")
        if isinstance(source_frame_id, bool) or not isinstance(source_frame_id, int):
            raise TypeError("navigation intent source_frame.id must be an integer")
        route_polyline = raw.get("route_polyline", ())
        if not isinstance(route_polyline, Sequence) or isinstance(
            route_polyline, (str, bytes)
        ):
            raise TypeError("navigation intent route_polyline must be an array")
        return cls(
            source_frame_id=source_frame_id,
            command=NavigationCommand(str(raw.get("command", ""))),
            direction=_pair(raw.get("direction"), "direction"),
            target_point_m=_pair(raw.get("target_point"), "target_point"),
            distance_to_maneuver_m=_finite(
                raw.get("distance_to_maneuver_m"), "distance_to_maneuver_m"
            ),
            route_polyline_m=tuple(
                _pair(point, "route_polyline point") for point in route_polyline
            ),
            route_id=str(raw.get("route_id", "")),
            source=str(raw.get("source", "")),
            confidence=_finite(raw.get("confidence"), "confidence"),
            privileged=raw.get("privileged"),
        )

    def mirrored(self) -> NavigationIntent:
        """Return the semantically correct intent for a horizontal image flip."""

        return NavigationIntent(
            source_frame_id=self.source_frame_id,
            command=_MIRRORED_COMMAND.get(self.command, self.command),
            direction=(self.direction[0], -self.direction[1]),
            target_point_m=(self.target_point_m[0], -self.target_point_m[1]),
            distance_to_maneuver_m=self.distance_to_maneuver_m,
            route_polyline_m=tuple((forward, -right) for forward, right in self.route_polyline_m),
            route_id=self.route_id,
            source=self.source,
            confidence=self.confidence,
            privileged=self.privileged,
        )

    def model_features(self, *, distance_scale_m: float = 100.0) -> tuple[float, ...]:
        """Return a stable one-hot route feature vector for PyTorch adapters."""

        scale = _finite(distance_scale_m, "distance_scale_m")
        if scale <= 0.0:
            raise ValueError("distance_scale_m must be positive")
        one_hot = [0.0] * len(NAVIGATION_COMMAND_ORDER)
        one_hot[self.command_index] = 1.0
        return tuple(
            [
                *one_hot,
                self.direction[0],
                self.direction[1],
                min(self.distance_to_maneuver_m / scale, 2.0),
            ]
        )


def command_from_road_option(value: Any) -> NavigationCommand:
    """Map CARLA ``RoadOption`` names/values without importing CARLA."""

    name = (
        value.strip().upper()
        if isinstance(value, str)
        else str(getattr(value, "name", "")).strip().upper()
    )
    if not name:
        try:
            name = {
                -1: "VOID",
                1: "LEFT",
                2: "RIGHT",
                3: "STRAIGHT",
                4: "LANEFOLLOW",
                5: "CHANGELANELEFT",
                6: "CHANGELANERIGHT",
            }[int(value)]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"unsupported CARLA RoadOption: {value!r}") from error
    mapping = {
        "VOID": NavigationCommand.FOLLOW_LANE,
        "LEFT": NavigationCommand.LEFT,
        "RIGHT": NavigationCommand.RIGHT,
        "STRAIGHT": NavigationCommand.STRAIGHT,
        "LANEFOLLOW": NavigationCommand.FOLLOW_LANE,
        "CHANGELANELEFT": NavigationCommand.CHANGE_LANE_LEFT,
        "CHANGELANERIGHT": NavigationCommand.CHANGE_LANE_RIGHT,
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise ValueError(f"unsupported CARLA RoadOption name: {name!r}") from error


__all__ = [
    "NAVIGATION_COMMAND_ORDER",
    "NAVIGATION_FEATURE_SCHEMA_VERSION",
    "NAVIGATION_INTENT_SCHEMA_VERSION",
    "CONDITIONAL_IMITATION_COMMAND_ID",
    "NavigationCommand",
    "NavigationIntent",
    "command_from_road_option",
]
