"""Deterministic route metadata and control serialization for BehaviorAgent teachers."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

BEHAVIOR_TEACHER_CONTROL_SCHEMA_VERSION = "1.0"
BEHAVIOR_ROUTE_SCHEMA_VERSION = "1.0"


def _location_xyz(value: Any) -> tuple[float, float, float]:
    location = getattr(value, "location", value)
    return (
        float(location.x),
        float(location.y),
        float(getattr(location, "z", 0.0)),
    )


def _transform_dict(transform: Any) -> dict[str, float]:
    location = transform.location
    rotation = transform.rotation
    return {
        "x": float(location.x),
        "y": float(location.y),
        "z": float(location.z),
        "pitch": float(rotation.pitch),
        "yaw": float(rotation.yaw),
        "roll": float(rotation.roll),
    }


def _distance(left: Any, right: Any) -> float:
    lx, ly, lz = _location_xyz(left)
    rx, ry, rz = _location_xyz(right)
    return math.sqrt((lx - rx) ** 2 + (ly - ry) ** 2 + (lz - rz) ** 2)


@dataclass(frozen=True, slots=True)
class BehaviorRouteLeg:
    """One deterministic BehaviorAgent route leg between map spawn points."""

    route_id: str
    episode_id: str
    leg_index: int
    route_seed: int
    start_spawn_index: int
    destination_spawn_index: int
    start_world_transform: dict[str, float]
    destination_world_transform: dict[str, float]
    straight_line_distance_m: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BEHAVIOR_ROUTE_SCHEMA_VERSION,
            **asdict(self),
        }


def choose_destination_index(
    spawn_points: Sequence[Any],
    *,
    current_location: Any,
    current_spawn_index: int,
    route_seed: int,
    leg_index: int,
    minimum_distance_m: float,
) -> int:
    """Choose a deterministic, preferably distant destination spawn point."""

    if len(spawn_points) < 2:
        raise ValueError("at least two spawn points are required for a route")
    if not 0 <= current_spawn_index < len(spawn_points):
        raise ValueError("current_spawn_index is outside spawn_points")
    if minimum_distance_m < 0.0 or not math.isfinite(minimum_distance_m):
        raise ValueError("minimum_distance_m must be finite and non-negative")
    if leg_index < 0:
        raise ValueError("leg_index must be non-negative")

    candidates = [
        (index, _distance(current_location, transform.location))
        for index, transform in enumerate(spawn_points)
        if index != current_spawn_index
    ]
    eligible = [item for item in candidates if item[1] >= minimum_distance_m]
    pool = eligible or [max(candidates, key=lambda item: (item[1], -item[0]))]
    pool = sorted(pool, key=lambda item: item[0])
    derived_seed = (int(route_seed) * 1_000_003 + int(leg_index) * 97_409) % ((2**31) - 1)
    return random.Random(derived_seed).choice(pool)[0]


def build_route_leg(
    *,
    episode_id: str,
    leg_index: int,
    route_seed: int,
    start_spawn_index: int,
    destination_spawn_index: int,
    spawn_points: Sequence[Any],
) -> BehaviorRouteLeg:
    if not episode_id.strip():
        raise ValueError("episode_id must not be empty")
    if not 0 <= start_spawn_index < len(spawn_points):
        raise ValueError("start_spawn_index is outside spawn_points")
    if not 0 <= destination_spawn_index < len(spawn_points):
        raise ValueError("destination_spawn_index is outside spawn_points")
    start = spawn_points[start_spawn_index]
    destination = spawn_points[destination_spawn_index]
    return BehaviorRouteLeg(
        route_id=f"route-{episode_id}-leg-{leg_index:03d}-d{destination_spawn_index:03d}",
        episode_id=episode_id,
        leg_index=leg_index,
        route_seed=int(route_seed),
        start_spawn_index=int(start_spawn_index),
        destination_spawn_index=int(destination_spawn_index),
        start_world_transform=_transform_dict(start),
        destination_world_transform=_transform_dict(destination),
        straight_line_distance_m=_distance(start.location, destination.location),
    )


def serialize_behavior_control(
    control: Any,
    *,
    carla_frame: int,
    route_id: str,
) -> dict[str, Any]:
    """Serialize the exact BehaviorAgent command applied before one CARLA tick."""

    if carla_frame < 0:
        raise ValueError("carla_frame must be non-negative")
    if not route_id.strip():
        raise ValueError("route_id must not be empty")
    bounded = {
        "throttle": (float(control.throttle), 0.0, 1.0),
        "steer": (float(control.steer), -1.0, 1.0),
        "brake": (float(control.brake), 0.0, 1.0),
    }
    for name, (value, minimum, maximum) in bounded.items():
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"BehaviorAgent control {name} must be in [{minimum}, {maximum}]")
    return {
        "schema_version": BEHAVIOR_TEACHER_CONTROL_SCHEMA_VERSION,
        "privileged": True,
        "purpose": "offline_teacher_action_target_only",
        "source": "BehaviorAgent.run_step",
        "applied_before_world_tick": True,
        "carla_frame": int(carla_frame),
        "route_id": route_id,
        "throttle": bounded["throttle"][0],
        "steer": bounded["steer"][0],
        "brake": bounded["brake"][0],
        "hand_brake": bool(getattr(control, "hand_brake", False)),
        "reverse": bool(getattr(control, "reverse", False)),
        "manual_gear_shift": bool(getattr(control, "manual_gear_shift", False)),
        "gear": int(getattr(control, "gear", 0)),
    }


def validate_behavior_sample_context(context: Mapping[str, Any], *, carla_frame: int) -> list[str]:
    """Return route/control validation errors for one DatasetWriter sample context."""

    errors: list[str] = []
    if context.get("control_mode") != "behavior_agent_teacher":
        errors.append("context.control_mode must be behavior_agent_teacher")
    route = context.get("route")
    if not isinstance(route, Mapping):
        errors.append("context.route must be an object")
    else:
        for name in ("route_id", "destination_spawn_index", "destination_world_transform"):
            if name not in route:
                errors.append(f"context.route.{name} is required")
        if not isinstance(route.get("route_id"), str) or not route.get("route_id", "").strip():
            errors.append("context.route.route_id must be a non-empty string")
    control = context.get("privileged_teacher_control")
    if not isinstance(control, Mapping):
        errors.append("context.privileged_teacher_control must be an object")
    else:
        if control.get("source") != "BehaviorAgent.run_step":
            errors.append("teacher control source must be BehaviorAgent.run_step")
        if control.get("carla_frame") != carla_frame:
            errors.append("teacher control carla_frame must match sample")
        for name, minimum, maximum in (
            ("throttle", 0.0, 1.0),
            ("steer", -1.0, 1.0),
            ("brake", 0.0, 1.0),
        ):
            value = control.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                errors.append(f"teacher control {name} must be finite")
            elif not minimum <= float(value) <= maximum:
                errors.append(f"teacher control {name} is outside [{minimum}, {maximum}]")
    return errors


__all__ = [
    "BEHAVIOR_ROUTE_SCHEMA_VERSION",
    "BEHAVIOR_TEACHER_CONTROL_SCHEMA_VERSION",
    "BehaviorRouteLeg",
    "build_route_leg",
    "choose_destination_index",
    "serialize_behavior_control",
    "validate_behavior_sample_context",
]
