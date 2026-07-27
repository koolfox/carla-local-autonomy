"""Read-only closed-loop CARLA driving evaluator.

The evaluator attaches event sensors to an existing hero vehicle and observes
route progress, controls, traffic lights, distance, collisions, and lane
invasions. It never sends a vehicle command and does not own ``world.tick()``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CLOSED_LOOP_EVALUATION_SCHEMA_VERSION = "1.0"


def _load_carla() -> Any:
    try:
        return importlib.import_module("carla")
    except ImportError as error:
        raise RuntimeError(
            "closed-loop evaluation requires the version-matched CARLA PythonAPI"
        ) from error


def _load_global_route_planner() -> type[Any]:
    try:
        module = importlib.import_module("agents.navigation.global_route_planner")
    except ImportError as error:
        raise RuntimeError(
            "GlobalRoutePlanner is unavailable. Add CARLA/PythonAPI/carla to PYTHONPATH."
        ) from error
    planner_type = getattr(module, "GlobalRoutePlanner", None)
    if planner_type is None:
        raise RuntimeError("global_route_planner has no GlobalRoutePlanner")
    return planner_type


def _speed_mps(vehicle: Any) -> float:
    velocity = vehicle.get_velocity()
    return math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2)


def _distance_xyz(left: Any, right: Any) -> float:
    return math.sqrt(
        (float(left.x) - float(right.x)) ** 2
        + (float(left.y) - float(right.y)) ** 2
        + (float(left.z) - float(right.z)) ** 2
    )


def _location_dict(location: Any) -> dict[str, float]:
    return {"x": float(location.x), "y": float(location.y), "z": float(location.z)}


def _ensure_output(path: Path) -> Path:
    root = path.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"evaluation output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@dataclass(slots=True)
class RouteProgressTracker:
    """Monotonic progress over a sampled polyline route."""

    points_xy: np.ndarray
    _cumulative_m: np.ndarray = field(init=False, repr=False)
    _best_index: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        points = np.asarray(self.points_xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
            raise ValueError("points_xy must have shape (N, 2) with at least two points")
        if not np.isfinite(points).all():
            raise ValueError("route points must be finite")
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(segment_lengths)))
        if cumulative[-1] <= 0.0:
            raise ValueError("route length must be positive")
        self.points_xy = points
        self._cumulative_m = cumulative

    @property
    def route_length_m(self) -> float:
        return float(self._cumulative_m[-1])

    @property
    def best_index(self) -> int:
        return self._best_index

    @property
    def completion(self) -> float:
        return float(self._cumulative_m[self._best_index] / self._cumulative_m[-1])

    def update(self, x: float, y: float) -> float:
        point = np.asarray([x, y], dtype=np.float64)
        if not np.isfinite(point).all():
            raise ValueError("vehicle location must be finite")
        start = max(0, self._best_index - 5)
        distances = np.linalg.norm(self.points_xy[start:] - point, axis=1)
        nearest = start + int(np.argmin(distances))
        self._best_index = max(self._best_index, nearest)
        return self.completion


@dataclass(slots=True)
class RedLightMonitor:
    """Approximate red-light passage from CARLA trigger-volume observations."""

    stop_speed_mps: float = 0.3
    active_light_id: int | None = field(default=None, init=False)
    minimum_speed_mps: float = field(default=math.inf, init=False)
    violations: int = field(default=0, init=False)

    def update(
        self,
        *,
        at_traffic_light: bool,
        light_id: int | None,
        light_state: str | None,
        speed_mps: float,
    ) -> bool:
        if speed_mps < 0.0 or not math.isfinite(speed_mps):
            raise ValueError("speed_mps must be finite and non-negative")
        state = (light_state or "").casefold()
        if at_traffic_light and light_id is not None and state == "red":
            if self.active_light_id != light_id:
                self.active_light_id = light_id
                self.minimum_speed_mps = speed_mps
            else:
                self.minimum_speed_mps = min(self.minimum_speed_mps, speed_mps)
            return False
        if self.active_light_id is None:
            return False
        released_by_green = at_traffic_light and state and state != "red"
        violation = not released_by_green and self.minimum_speed_mps > self.stop_speed_mps
        if violation:
            self.violations += 1
        self.active_light_id = None
        self.minimum_speed_mps = math.inf
        return violation


@dataclass(slots=True)
class EvaluationEvents:
    collision_debounce_frames: int = 10
    collision_count: int = field(default=0, init=False)
    collision_intensity_sum: float = field(default=0.0, init=False)
    lane_invasion_count: int = field(default=0, init=False)
    full_brake_intervention_count: int = field(default=0, init=False)
    _last_collision_frame_by_actor: dict[int, int] = field(default_factory=dict, init=False)
    _full_brake_active: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def collision_callback(self, event: Any) -> None:
        other_actor = getattr(event, "other_actor", None)
        actor_id = int(getattr(other_actor, "id", -1))
        frame = int(event.frame)
        impulse = event.normal_impulse
        intensity = math.sqrt(
            float(impulse.x) ** 2 + float(impulse.y) ** 2 + float(impulse.z) ** 2
        )
        with self._lock:
            previous = self._last_collision_frame_by_actor.get(actor_id)
            self._last_collision_frame_by_actor[actor_id] = frame
            if previous is not None and frame - previous <= self.collision_debounce_frames:
                self.collision_intensity_sum += intensity
                return
            self.collision_count += 1
            self.collision_intensity_sum += intensity

    def lane_invasion_callback(self, event: Any) -> None:
        with self._lock:
            self.lane_invasion_count += 1

    def observe_control(
        self,
        *,
        throttle: float,
        brake: float,
        speed_mps: float,
        threshold: float,
        minimum_speed_mps: float,
    ) -> bool:
        full_brake = (
            brake >= threshold and throttle <= 0.05 and speed_mps >= minimum_speed_mps
        )
        started = full_brake and not self._full_brake_active
        if started:
            self.full_brake_intervention_count += 1
        self._full_brake_active = full_brake
        return started

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            return {
                "collision_count": self.collision_count,
                "collision_intensity_sum": self.collision_intensity_sum,
                "lane_invasion_count": self.lane_invasion_count,
                "full_brake_intervention_count": self.full_brake_intervention_count,
            }


def _find_vehicle(world: Any, *, actor_id: int | None, role_name: str) -> Any:
    actors = world.get_actors()
    if actor_id is not None:
        actor = actors.find(actor_id)
        if actor is None or not str(actor.type_id).startswith("vehicle."):
            raise RuntimeError(f"vehicle actor {actor_id} was not found")
        return actor
    matches = [
        actor
        for actor in actors.filter("vehicle.*")
        if str(actor.attributes.get("role_name", "")) == role_name
    ]
    if not matches:
        raise RuntimeError(f"no vehicle with role_name={role_name!r} was found")
    return sorted(matches, key=lambda actor: int(actor.id))[0]


def _choose_destination(
    world: Any,
    vehicle: Any,
    *,
    destination_index: int | None,
    destination_seed: int,
) -> tuple[Any, int]:
    spawn_points = list(world.get_map().get_spawn_points())
    if not spawn_points:
        raise RuntimeError("active map has no vehicle spawn points")
    if destination_index is not None:
        if not 0 <= destination_index < len(spawn_points):
            raise ValueError("destination index is outside map spawn points")
        return spawn_points[destination_index], destination_index
    location = vehicle.get_location()
    candidates = sorted(
        range(len(spawn_points)),
        key=lambda index: _distance_xyz(location, spawn_points[index].location),
        reverse=True,
    )
    pool = candidates[: max(1, min(12, len(candidates)))]
    index = random.Random(destination_seed).choice(pool)
    return spawn_points[index], index


def _route_points(world: Any, start: Any, destination: Any, sampling_resolution: float) -> np.ndarray:
    planner_type = _load_global_route_planner()
    planner = planner_type(world.get_map(), sampling_resolution)
    route = planner.trace_route(start, destination)
    points = [
        (float(waypoint.transform.location.x), float(waypoint.transform.location.y))
        for waypoint, _road_option in route
    ]
    if len(points) < 2:
        raise RuntimeError("GlobalRoutePlanner returned an empty or trivial route")
    return np.asarray(points, dtype=np.float64)


def _spawn_event_sensor(world: Any, carla: Any, vehicle: Any, blueprint_id: str) -> Any:
    blueprint = world.get_blueprint_library().find(blueprint_id)
    if blueprint is None:
        raise RuntimeError(f"sensor blueprint is unavailable: {blueprint_id}")
    return world.spawn_actor(carla.Transform(), blueprint=blueprint, attach_to=vehicle)


def _traffic_light_state(vehicle: Any) -> tuple[bool, int | None, str | None]:
    at_light = bool(vehicle.is_at_traffic_light())
    if not at_light:
        return False, None, None
    light = vehicle.get_traffic_light()
    if light is None:
        return True, None, None
    return True, int(light.id), str(light.state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Observe an existing CARLA hero and record closed-loop driving metrics."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--actor-id", type=int)
    parser.add_argument("--role-name", default="hero")
    parser.add_argument("--driver-label", required=False, default="unknown")
    parser.add_argument("--run-label", required=False, default="closed-loop-run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--destination-index", type=int)
    parser.add_argument("--destination-seed", type=int, default=17)
    parser.add_argument("--route-sampling-resolution", type=float, default=2.0)
    parser.add_argument("--destination-threshold-m", type=float, default=3.0)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--status-every", type=float, default=1.0)
    parser.add_argument("--red-light-stop-speed-mps", type=float, default=0.3)
    parser.add_argument("--full-brake-threshold", type=float, default=0.95)
    parser.add_argument("--full-brake-minimum-speed-mps", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("runs/closed-loop-evaluation"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    for name in (
        "timeout",
        "route_sampling_resolution",
        "destination_threshold_m",
        "status_every",
    ):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    if args.duration < 0.0 or args.max_frames < 0:
        raise ValueError("duration and max_frames must be non-negative")
    if args.duration == 0.0 and args.max_frames == 0 and not args.dry_run:
        raise ValueError("a real run requires a positive duration or max_frames")
    if not 0.0 <= args.full_brake_threshold <= 1.0:
        raise ValueError("full_brake_threshold must be in [0, 1]")
    if args.full_brake_minimum_speed_mps < 0.0:
        raise ValueError("full_brake_minimum_speed_mps must be non-negative")


def _dry_run_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": CLOSED_LOOP_EVALUATION_SCHEMA_VERSION,
        "dry_run": True,
        "read_only_vehicle_control": True,
        "control_calls": 0,
        "connects_to": f"{args.host}:{args.port}",
        "role_name": args.role_name,
        "actor_id": args.actor_id,
        "driver_label": args.driver_label,
        "destination_index": args.destination_index,
        "would_spawn_only_event_sensors": [
            "sensor.other.collision",
            "sensor.other.lane_invasion",
        ],
        "would_own_world_tick": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    if args.dry_run:
        payload = _dry_run_payload(args)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return payload

    carla = _load_carla()
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    vehicle = _find_vehicle(world, actor_id=args.actor_id, role_name=args.role_name)
    destination_transform, destination_index = _choose_destination(
        world,
        vehicle,
        destination_index=args.destination_index,
        destination_seed=args.destination_seed,
    )
    route_points = _route_points(
        world,
        vehicle.get_location(),
        destination_transform.location,
        args.route_sampling_resolution,
    )
    tracker = RouteProgressTracker(route_points)
    events = EvaluationEvents()
    red_lights = RedLightMonitor(stop_speed_mps=args.red_light_stop_speed_mps)
    output = _ensure_output(args.output)
    _write_json(
        output / "route.json",
        {
            "schema_version": CLOSED_LOOP_EVALUATION_SCHEMA_VERSION,
            "destination_index": destination_index,
            "destination": _location_dict(destination_transform.location),
            "route_length_m": tracker.route_length_m,
            "points_xy": route_points.tolist(),
        },
    )

    collision_sensor = _spawn_event_sensor(
        world, carla, vehicle, "sensor.other.collision"
    )
    lane_sensor = _spawn_event_sensor(
        world, carla, vehicle, "sensor.other.lane_invasion"
    )
    collision_sensor.listen(events.collision_callback)
    lane_sensor.listen(events.lane_invasion_callback)
    sensors = (collision_sensor, lane_sensor)
    started = time.monotonic()
    status_at = started
    frame_count = 0
    distance_m = 0.0
    previous_location = vehicle.get_location()
    stop_reason = "duration"
    telemetry_path = output / "telemetry.jsonl"
    try:
        with telemetry_path.open("w", encoding="utf-8") as telemetry:
            while True:
                elapsed = time.monotonic() - started
                if args.duration > 0.0 and elapsed >= args.duration:
                    break
                if args.max_frames > 0 and frame_count >= args.max_frames:
                    stop_reason = "max_frames"
                    break
                snapshot = world.wait_for_tick(seconds=args.timeout)
                frame_count += 1
                location = vehicle.get_location()
                distance_m += _distance_xyz(previous_location, location)
                previous_location = location
                speed = _speed_mps(vehicle)
                completion = tracker.update(float(location.x), float(location.y))
                distance_to_destination = _distance_xyz(
                    location, destination_transform.location
                )
                control = vehicle.get_control()
                full_brake_started = events.observe_control(
                    throttle=float(control.throttle),
                    brake=float(control.brake),
                    speed_mps=speed,
                    threshold=args.full_brake_threshold,
                    minimum_speed_mps=args.full_brake_minimum_speed_mps,
                )
                at_light, light_id, light_state = _traffic_light_state(vehicle)
                red_violation = red_lights.update(
                    at_traffic_light=at_light,
                    light_id=light_id,
                    light_state=light_state,
                    speed_mps=speed,
                )
                event_snapshot = events.snapshot()
                record = {
                    "schema_version": CLOSED_LOOP_EVALUATION_SCHEMA_VERSION,
                    "frame": int(snapshot.frame),
                    "simulation_timestamp_seconds": float(snapshot.timestamp.elapsed_seconds),
                    "elapsed_wall_seconds": elapsed,
                    "location": _location_dict(location),
                    "speed_mps": speed,
                    "distance_traveled_m": distance_m,
                    "route_completion": completion,
                    "distance_to_destination_m": distance_to_destination,
                    "control": {
                        "throttle": float(control.throttle),
                        "steer": float(control.steer),
                        "brake": float(control.brake),
                    },
                    "traffic_light": {
                        "at_light": at_light,
                        "actor_id": light_id,
                        "state": light_state,
                        "violation_observed": red_violation,
                    },
                    "full_brake_started": full_brake_started,
                    "events": event_snapshot,
                    "read_only_vehicle_control": True,
                    "control_calls": 0,
                }
                telemetry.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                telemetry.flush()
                if distance_to_destination <= args.destination_threshold_m:
                    tracker._best_index = len(tracker.points_xy) - 1
                    stop_reason = "destination_reached"
                    break
                now = time.monotonic()
                if now >= status_at:
                    print(
                        f"frame={snapshot.frame} speed={speed * 3.6:.1f}km/h "
                        f"route={completion * 100:.1f}% collisions="
                        f"{event_snapshot['collision_count']} lanes="
                        f"{event_snapshot['lane_invasion_count']}",
                        flush=True,
                    )
                    status_at = now + args.status_every
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    finally:
        for sensor in sensors:
            try:
                sensor.stop()
            except RuntimeError:
                pass
            try:
                sensor.destroy()
            except RuntimeError:
                pass

    event_snapshot = events.snapshot()
    distance_km = distance_m / 1000.0
    summary = {
        "schema_version": CLOSED_LOOP_EVALUATION_SCHEMA_VERSION,
        "run_label": args.run_label,
        "driver_label": args.driver_label,
        "seed": args.seed,
        "server_version": str(client.get_server_version()),
        "client_version": str(client.get_client_version()),
        "map": str(world.get_map().name),
        "ego_actor_id": int(vehicle.id),
        "destination_index": destination_index,
        "destination": _location_dict(destination_transform.location),
        "stop_reason": stop_reason,
        "frame_count": frame_count,
        "elapsed_wall_seconds": time.monotonic() - started,
        "distance_traveled_m": distance_m,
        "route_length_m": tracker.route_length_m,
        "route_completion": tracker.completion,
        "destination_reached": stop_reason == "destination_reached",
        **event_snapshot,
        "red_light_violation_count": red_lights.violations,
        "collisions_per_km": event_snapshot["collision_count"] / max(distance_km, 1e-6),
        "lane_invasions_per_km": event_snapshot["lane_invasion_count"]
        / max(distance_km, 1e-6),
        "read_only_vehicle_control": True,
        "control_calls": 0,
        "red_light_metric_note": (
            "Approximate: counts leaving a CARLA red-light trigger without slowing below "
            f"{args.red_light_stop_speed_mps:.3f} m/s."
        ),
        "full_brake_metric_note": (
            "Counts observed transitions into high brake while moving; the observer cannot "
            "distinguish model safety brakes from intentional braking."
        ),
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


__all__ = [
    "CLOSED_LOOP_EVALUATION_SCHEMA_VERSION",
    "EvaluationEvents",
    "RedLightMonitor",
    "RouteProgressTracker",
    "build_parser",
    "main",
    "run",
]


if __name__ == "__main__":
    raise SystemExit(main())
