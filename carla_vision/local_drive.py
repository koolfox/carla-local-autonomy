"""Run a complete local CARLA traffic scene with a new ego vehicle.

The runner supports three ego drivers:

``traffic-manager``
    CARLA Traffic Manager controls the ego and all NPC vehicles.
``behavior``
    CARLA's Python ``BehaviorAgent`` follows changing destinations and reacts
    to vehicles, walkers, traffic lights, and speed limits.
``model``
    A user-supplied front-camera model directly predicts throttle, steering,
    and braking through :mod:`carla_vision.model_driver`.

CARLA is imported lazily so ``--dry-run`` and unit tests work without the
version-matched PythonAPI installed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .model_driver import (
    DrivingModel,
    ModelControl,
    ModelDriverConfig,
    ModelObservation,
    control_from_value,
    create_driving_model,
)

EXPECTED_CARLA_VERSION = "0.9.16"


def parse_json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return parsed


def parse_resolution(value: str) -> tuple[int, int]:
    parts = value.lower().split("x", maxsplit=1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must use WIDTHxHEIGHT")
    try:
        width, height = (int(item) for item in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("resolution values must be integers") from error
    if not 320 <= width <= 3840 or not 180 <= height <= 2160:
        raise argparse.ArgumentTypeError("resolution must be between 320x180 and 3840x2160")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Connect to a local CARLA 0.9.16 server, spawn an ego vehicle, "
            "populate traffic/walkers, and drive through the scene."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--expected-version", default=EXPECTED_CARLA_VERSION)
    parser.add_argument("--allow-version-mismatch", action="store_true")
    parser.add_argument(
        "--map",
        default="current",
        help="CARLA map to load, or 'current' to keep the active map.",
    )

    parser.add_argument(
        "--driver",
        choices=("traffic-manager", "behavior", "model"),
        default="behavior",
    )
    parser.add_argument(
        "--behavior",
        choices=("cautious", "normal", "aggressive"),
        default="normal",
    )
    parser.add_argument("--ego-blueprint", default="vehicle.tesla.model3")
    parser.add_argument("--ego-spawn-index", type=int)
    parser.add_argument("--ego-target-speed-kmh", type=float, default=35.0)
    parser.add_argument("--destination-index", type=int)
    parser.add_argument(
        "--loop-destinations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Choose another destination whenever BehaviorAgent finishes.",
    )

    parser.add_argument("--vehicles", type=int, default=60)
    parser.add_argument("--walkers", type=int, default=30)
    parser.add_argument("--vehicle-filter", default="vehicle.*")
    parser.add_argument("--walker-filter", default="walker.pedestrian.*")
    parser.add_argument("--safe-vehicles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--traffic-distance", type=float, default=2.0)
    parser.add_argument(
        "--traffic-speed-difference",
        type=float,
        default=12.0,
        help="Percentage slower than the speed limit.",
    )
    parser.add_argument("--pedestrian-crossing", type=float, default=0.20)
    parser.add_argument("--running-walkers", type=float, default=0.05)
    parser.add_argument("--automatic-lights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hybrid-physics", action="store_true")
    parser.add_argument("--respawn-dormant-vehicles", action="store_true")

    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--asynchronous", action="store_true")
    parser.add_argument(
        "--duration",
        type=float,
        default=180.0,
        help="Seconds to run; use 0 to continue until Ctrl+C.",
    )
    parser.add_argument("--spectator-follow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--status-every", type=float, default=1.0)

    parser.add_argument("--camera-resolution", type=parse_resolution, default=(640, 384))
    parser.add_argument("--camera-fov", type=float, default=90.0)
    parser.add_argument("--camera-x", type=float, default=1.5)
    parser.add_argument("--camera-z", type=float, default=1.7)
    parser.add_argument("--camera-pitch", type=float, default=0.0)
    parser.add_argument("--camera-timeout", type=float, default=1.0)
    parser.add_argument("--model-factory")
    parser.add_argument("--model-checkpoint")
    parser.add_argument("--model-device", default="cpu")
    parser.add_argument("--model-options", type=parse_json_object, default={})
    parser.add_argument("--max-model-speed-kmh", type=float, default=45.0)
    parser.add_argument("--max-steer-rate", type=float, default=2.5)
    parser.add_argument("--max-model-errors", type=int, default=3)

    parser.add_argument(
        "--keep-actors",
        action="store_true",
        help="Leave spawned actors alive when the script exits.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or not 1 <= args.tm_port <= 65535:
        parser.error("ports must be in [1, 65535]")
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    if args.vehicles < 0 or args.walkers < 0:
        parser.error("--vehicles and --walkers must be non-negative")
    if args.fps <= 0.0 or args.fps > 100.0:
        parser.error("--fps must be in (0, 100]")
    if args.duration < 0.0:
        parser.error("--duration must be non-negative")
    if args.status_every <= 0.0:
        parser.error("--status-every must be positive")
    if args.ego_target_speed_kmh <= 0.0:
        parser.error("--ego-target-speed-kmh must be positive")
    if args.traffic_distance <= 0.0:
        parser.error("--traffic-distance must be positive")
    if not 0.0 <= args.pedestrian_crossing <= 1.0:
        parser.error("--pedestrian-crossing must be in [0, 1]")
    if not 0.0 <= args.running_walkers <= 1.0:
        parser.error("--running-walkers must be in [0, 1]")
    if not 30.0 <= args.camera_fov <= 150.0:
        parser.error("--camera-fov must be between 30 and 150")
    if args.camera_timeout <= 0.0:
        parser.error("--camera-timeout must be positive")
    if args.max_model_speed_kmh <= 0.0:
        parser.error("--max-model-speed-kmh must be positive")
    if args.max_steer_rate <= 0.0:
        parser.error("--max-steer-rate must be positive")
    if args.max_model_errors <= 0:
        parser.error("--max-model-errors must be positive")
    if args.driver == "model" and not args.model_factory:
        parser.error("--driver model requires --model-factory module:callable")
    if args.driver != "model" and any(
        value for value in (args.model_factory, args.model_checkpoint, args.model_options)
    ):
        parser.error("model options require --driver model")
    return args


def _dry_run_payload(args: argparse.Namespace) -> dict[str, Any]:
    actions = [
        "connect to the CARLA server",
        "spawn one new hero vehicle",
        f"spawn up to {args.vehicles} NPC vehicles",
        f"spawn up to {args.walkers} walkers",
        f"drive the hero with {args.driver}",
    ]
    if args.map != "current":
        actions.insert(1, f"load map {args.map}")
    if args.driver == "model":
        actions.append(f"attach a front RGB camera and load {args.model_factory}")
    return {
        "schema_version": "1.0",
        "connects_to": f"{args.host}:{args.port}",
        "traffic_manager": f"{args.host}:{args.tm_port}",
        "map": args.map,
        "driver": args.driver,
        "synchronous": not args.asynchronous,
        "expected_actions": actions,
    }


def _load_carla() -> Any:
    try:
        return importlib.import_module("carla")
    except ImportError as error:
        raise RuntimeError(
            "CARLA PythonAPI is not importable. Install or expose the PythonAPI "
            "that ships with the same CARLA server release."
        ) from error


def _speed_mps(vehicle: Any) -> float:
    velocity = vehicle.get_velocity()
    return math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2)


def _map_basename(name: str) -> str:
    return str(name).replace("\\", "/").rstrip("/").rsplit("/", maxsplit=1)[-1]


def _choose_attribute(blueprint: Any, name: str, rng: random.Random) -> None:
    if not blueprint.has_attribute(name):
        return
    values = list(blueprint.get_attribute(name).recommended_values)
    if values:
        blueprint.set_attribute(name, rng.choice(values))


def _vehicle_blueprints(library: Any, pattern: str, safe_only: bool) -> list[Any]:
    blueprints = sorted(library.filter(pattern), key=lambda item: item.id)
    if not safe_only:
        return blueprints
    safe: list[Any] = []
    for blueprint in blueprints:
        if blueprint.has_attribute("base_type"):
            if str(blueprint.get_attribute("base_type")) == "car":
                safe.append(blueprint)
        elif not any(token in blueprint.id for token in ("bike", "motorcycle", "microlino")):
            safe.append(blueprint)
    return safe or blueprints


def _spawn_ego(
    world: Any,
    blueprint_id: str,
    spawn_points: list[Any],
    requested_index: int | None,
    rng: random.Random,
) -> tuple[Any, int]:
    library = world.get_blueprint_library()
    blueprint = library.find(blueprint_id)
    if blueprint is None:
        raise RuntimeError(f"ego blueprint not found: {blueprint_id}")
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "hero")
    _choose_attribute(blueprint, "color", rng)

    if requested_index is not None:
        if not 0 <= requested_index < len(spawn_points):
            raise ValueError("--ego-spawn-index is outside available spawn points")
        indices = [requested_index]
    else:
        indices = list(range(len(spawn_points)))
        rng.shuffle(indices)

    for index in indices:
        actor = world.try_spawn_actor(blueprint, spawn_points[index])
        if actor is not None:
            return actor, index
    raise RuntimeError("could not spawn the ego vehicle at any map spawn point")


def _spawn_npc_vehicles(
    world: Any,
    traffic_manager: Any,
    spawn_points: list[Any],
    excluded_index: int,
    count: int,
    args: argparse.Namespace,
    rng: random.Random,
) -> list[Any]:
    if count == 0:
        return []
    blueprints = _vehicle_blueprints(
        world.get_blueprint_library(), args.vehicle_filter, args.safe_vehicles
    )
    if not blueprints:
        raise RuntimeError("no NPC vehicle blueprints matched the configured filter")

    indices = [index for index in range(len(spawn_points)) if index != excluded_index]
    rng.shuffle(indices)
    actors: list[Any] = []
    for index in indices[:count]:
        blueprint = rng.choice(blueprints)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "autopilot")
        _choose_attribute(blueprint, "color", rng)
        _choose_attribute(blueprint, "driver_id", rng)
        actor = world.try_spawn_actor(blueprint, spawn_points[index])
        if actor is None:
            continue
        actor.set_autopilot(True, traffic_manager.get_port())
        if args.automatic_lights:
            traffic_manager.update_vehicle_lights(actor, True)
        actors.append(actor)
    return actors


def _walker_speed(blueprint: Any, running: bool) -> float:
    if not blueprint.has_attribute("speed"):
        return 2.8 if running else 1.4
    values = list(blueprint.get_attribute("speed").recommended_values)
    if not values:
        return 2.8 if running else 1.4
    index = 2 if running and len(values) > 2 else min(1, len(values) - 1)
    try:
        return float(values[index])
    except ValueError:
        return 2.8 if running else 1.4


def _spawn_walkers(
    world: Any,
    count: int,
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[list[Any], list[Any]]:
    if count == 0:
        return [], []
    library = world.get_blueprint_library()
    walker_blueprints = sorted(library.filter(args.walker_filter), key=lambda item: item.id)
    controller_blueprint = library.find("controller.ai.walker")
    if not walker_blueprints or controller_blueprint is None:
        raise RuntimeError("walker or walker-controller blueprints are unavailable")

    walkers: list[Any] = []
    controllers: list[Any] = []
    speeds: list[float] = []
    for _ in range(count):
        location = world.get_random_location_from_navigation()
        if location is None:
            continue
        transform = importlib.import_module("carla").Transform(location)
        blueprint = rng.choice(walker_blueprints)
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "false")
        running = rng.random() < args.running_walkers
        walker = world.try_spawn_actor(blueprint, transform)
        if walker is None:
            continue
        controller = world.try_spawn_actor(controller_blueprint, importlib.import_module("carla").Transform(), attach_to=walker)
        if controller is None:
            walker.destroy()
            continue
        walkers.append(walker)
        controllers.append(controller)
        speeds.append(_walker_speed(blueprint, running))

    if not args.asynchronous:
        world.tick()
    for controller, speed in zip(controllers, speeds, strict=True):
        controller.start()
        destination = world.get_random_location_from_navigation()
        if destination is not None:
            controller.go_to_location(destination)
        controller.set_max_speed(speed)
    return walkers, controllers


@dataclass(frozen=True)
class CameraFrame:
    frame: int
    timestamp: float
    bgr: np.ndarray


class LatestCamera:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: CameraFrame | None = None
        self._closed = False

    def callback(self, image: Any) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3].copy()
        frame = CameraFrame(int(image.frame), float(image.timestamp), array)
        with self._condition:
            if not self._closed:
                self._latest = frame
                self._condition.notify_all()

    def wait_after(self, frame: int, timeout: float) -> CameraFrame:
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._closed:
                if self._latest is not None and self._latest.frame > frame:
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("front-camera frame timeout")
                self._condition.wait(remaining)
        raise RuntimeError("camera stream is closed")

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


def _spawn_camera(carla: Any, world: Any, ego: Any, args: argparse.Namespace, stream: LatestCamera) -> Any:
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    if blueprint is None:
        raise RuntimeError("sensor.camera.rgb blueprint is unavailable")
    width, height = args.camera_resolution
    blueprint.set_attribute("image_size_x", str(width))
    blueprint.set_attribute("image_size_y", str(height))
    blueprint.set_attribute("fov", str(args.camera_fov))
    blueprint.set_attribute("sensor_tick", str(1.0 / args.fps))
    transform = carla.Transform(
        carla.Location(x=args.camera_x, z=args.camera_z),
        carla.Rotation(pitch=args.camera_pitch),
    )
    camera = world.spawn_actor(blueprint, transform, attach_to=ego)
    camera.listen(stream.callback)
    return camera


def _new_destination(
    spawn_points: list[Any],
    ego: Any,
    rng: random.Random,
    requested_index: int | None,
) -> tuple[Any, int]:
    if requested_index is not None:
        if not 0 <= requested_index < len(spawn_points):
            raise ValueError("--destination-index is outside available spawn points")
        return spawn_points[requested_index].location, requested_index

    location = ego.get_location()
    candidates = list(range(len(spawn_points)))
    rng.shuffle(candidates)
    candidates.sort(
        key=lambda index: spawn_points[index].location.distance(location), reverse=True
    )
    index = rng.choice(candidates[: max(1, min(12, len(candidates)))])
    return spawn_points[index].location, index


def _create_behavior_agent(
    ego: Any,
    args: argparse.Namespace,
    spawn_points: list[Any],
    rng: random.Random,
) -> tuple[Any, int]:
    try:
        module = importlib.import_module("agents.navigation.behavior_agent")
    except ImportError as error:
        raise RuntimeError(
            "BehaviorAgent is unavailable. Add CARLA/PythonAPI/carla to PYTHONPATH "
            "so the agents package can be imported."
        ) from error
    agent = module.BehaviorAgent(ego, behavior=args.behavior)
    if hasattr(agent, "set_target_speed"):
        agent.set_target_speed(args.ego_target_speed_kmh)
    destination, index = _new_destination(
        spawn_points, ego, rng, args.destination_index
    )
    agent.set_destination(destination)
    return agent, index


def _model_vehicle_control(
    carla: Any,
    proposal: ModelControl,
    *,
    previous_steer: float,
    dt_seconds: float,
    max_steer_rate: float,
    speed_kmh: float,
    max_speed_kmh: float,
) -> tuple[Any, float, str]:
    maximum_delta = max_steer_rate * dt_seconds
    steer = float(np.clip(proposal.steer, previous_steer - maximum_delta, previous_steer + maximum_delta))
    throttle = proposal.throttle
    brake = proposal.brake
    reason = "model"
    if speed_kmh >= max_speed_kmh and throttle > 0.0:
        throttle = 0.0
        brake = max(brake, 0.25)
        reason = "model:speed-limit"
    control = carla.VehicleControl(
        throttle=float(throttle),
        steer=steer,
        brake=float(brake),
        hand_brake=bool(proposal.hand_brake),
        reverse=bool(proposal.reverse),
    )
    return control, steer, reason


def _follow_spectator(carla: Any, spectator: Any, ego: Any) -> None:
    transform = ego.get_transform()
    forward = transform.get_forward_vector()
    location = transform.location - carla.Location(
        x=forward.x * 7.0,
        y=forward.y * 7.0,
    )
    location.z += 3.0
    spectator.set_transform(
        carla.Transform(
            location,
            carla.Rotation(pitch=-15.0, yaw=transform.rotation.yaw),
        )
    )


@dataclass
class SpawnedActors:
    ego: Any | None = None
    camera: Any | None = None
    vehicles: list[Any] = field(default_factory=list)
    walkers: list[Any] = field(default_factory=list)
    walker_controllers: list[Any] = field(default_factory=list)


def _destroy(actor: Any) -> None:
    try:
        actor.destroy()
    except Exception:
        pass


def _cleanup(
    world: Any,
    traffic_manager: Any,
    actors: SpawnedActors,
    camera_stream: LatestCamera | None,
    original_settings: Any,
    changed_settings: bool,
    keep_actors: bool,
) -> None:
    if camera_stream is not None:
        camera_stream.close()
    if actors.camera is not None:
        try:
            actors.camera.stop()
        except Exception:
            pass
    for controller in actors.walker_controllers:
        try:
            controller.stop()
        except Exception:
            pass
    if not keep_actors:
        for actor in [
            actors.camera,
            *actors.walker_controllers,
            *actors.walkers,
            *actors.vehicles,
            actors.ego,
        ]:
            if actor is not None:
                _destroy(actor)
    try:
        traffic_manager.set_synchronous_mode(False)
    except Exception:
        pass
    if changed_settings:
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass


def _check_version(client_version: str, server_version: str, args: argparse.Namespace) -> None:
    expected = args.expected_version.strip()
    if args.allow_version_mismatch or not expected:
        return
    if expected not in client_version or expected not in server_version:
        raise RuntimeError(
            "CARLA version mismatch: "
            f"expected {expected}, client={client_version}, server={server_version}. "
            "Use the matching PythonAPI or pass --allow-version-mismatch explicitly."
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run:
        payload = _dry_run_payload(args)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return payload

    carla = _load_carla()
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    client_version = str(client.get_client_version())
    server_version = str(client.get_server_version())
    _check_version(client_version, server_version, args)

    world = client.get_world()
    if args.map != "current" and _map_basename(world.get_map().name) != _map_basename(args.map):
        world = client.load_world(args.map)

    original_settings = world.get_settings()
    changed_settings = not args.asynchronous
    if changed_settings:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / args.fps
        settings.no_rendering_mode = False
        world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_synchronous_mode(not args.asynchronous)
    traffic_manager.set_random_device_seed(args.seed)
    traffic_manager.set_global_distance_to_leading_vehicle(args.traffic_distance)
    traffic_manager.global_percentage_speed_difference(args.traffic_speed_difference)
    if args.hybrid_physics:
        traffic_manager.set_hybrid_physics_mode(True)
        traffic_manager.set_hybrid_physics_radius(70.0)
    if args.respawn_dormant_vehicles:
        traffic_manager.set_respawn_dormant_vehicles(True)
    world.set_pedestrians_seed(args.seed + 1)
    world.set_pedestrians_cross_factor(args.pedestrian_crossing)

    rng = random.Random(args.seed)
    actors = SpawnedActors()
    camera_stream: LatestCamera | None = None
    driving_model: DrivingModel | None = None
    behavior_agent: Any | None = None
    destination_index: int | None = None
    stop_reason = "duration"
    started = time.monotonic()
    status_at = started
    previous_tick = started
    previous_model_frame = -1
    previous_steer = 0.0
    consecutive_model_errors = 0
    last_control_reason = "starting"

    try:
        spawn_points = list(world.get_map().get_spawn_points())
        if not spawn_points:
            raise RuntimeError("active CARLA map has no vehicle spawn points")
        actors.ego, ego_spawn_index = _spawn_ego(
            world, args.ego_blueprint, spawn_points, args.ego_spawn_index, rng
        )
        actors.vehicles = _spawn_npc_vehicles(
            world,
            traffic_manager,
            spawn_points,
            ego_spawn_index,
            args.vehicles,
            args,
            rng,
        )
        actors.walkers, actors.walker_controllers = _spawn_walkers(
            world, args.walkers, args, rng
        )

        if args.driver == "traffic-manager":
            actors.ego.set_autopilot(True, traffic_manager.get_port())
            if args.automatic_lights:
                traffic_manager.update_vehicle_lights(actors.ego, True)
            if hasattr(traffic_manager, "set_desired_speed"):
                traffic_manager.set_desired_speed(actors.ego, args.ego_target_speed_kmh)
            last_control_reason = "traffic-manager"
        elif args.driver == "behavior":
            behavior_agent, destination_index = _create_behavior_agent(
                actors.ego, args, spawn_points, rng
            )
            last_control_reason = f"behavior:{args.behavior}"
        else:
            camera_stream = LatestCamera()
            actors.camera = _spawn_camera(carla, world, actors.ego, args, camera_stream)
            driving_model = create_driving_model(
                args.model_factory,
                ModelDriverConfig(
                    checkpoint=Path(args.model_checkpoint) if args.model_checkpoint else None,
                    device=args.model_device,
                    options=args.model_options,
                ),
            )
            driving_model.reset()
            actors.ego.apply_control(carla.VehicleControl(brake=1.0))
            last_control_reason = f"model:{args.model_factory}"

        print(
            f"Connected CARLA server={server_version} client={client_version} "
            f"map={_map_basename(world.get_map().name)}"
        )
        print(
            f"Spawned ego={actors.ego.id} NPC={len(actors.vehicles)} "
            f"walkers={len(actors.walkers)} driver={args.driver}"
        )

        while True:
            loop_started = time.monotonic()
            elapsed = loop_started - started
            if args.duration > 0.0 and elapsed >= args.duration:
                break

            if args.asynchronous:
                world.wait_for_tick(seconds=args.timeout)
            else:
                world.tick()

            now = time.monotonic()
            dt = max(1.0 / args.fps, min(0.25, now - previous_tick))
            previous_tick = now
            speed_mps = _speed_mps(actors.ego)
            speed_kmh = speed_mps * 3.6

            if args.driver == "behavior":
                if behavior_agent.done():
                    if not args.loop_destinations:
                        actors.ego.apply_control(carla.VehicleControl(brake=1.0))
                        stop_reason = "destination_complete"
                        break
                    destination, destination_index = _new_destination(
                        spawn_points, actors.ego, rng, args.destination_index
                    )
                    behavior_agent.set_destination(destination)
                actors.ego.apply_control(behavior_agent.run_step(debug=False))

            elif args.driver == "model":
                assert camera_stream is not None and driving_model is not None
                try:
                    frame = camera_stream.wait_after(previous_model_frame, args.camera_timeout)
                    previous_model_frame = frame.frame
                    proposal = control_from_value(
                        driving_model.predict(
                            ModelObservation(
                                frame=frame.frame,
                                timestamp=frame.timestamp,
                                image_bgr=frame.bgr,
                                speed_mps=speed_mps,
                                dt_seconds=dt,
                            )
                        )
                    )
                    consecutive_model_errors = 0
                    control, previous_steer, last_control_reason = _model_vehicle_control(
                        carla,
                        proposal,
                        previous_steer=previous_steer,
                        dt_seconds=dt,
                        max_steer_rate=args.max_steer_rate,
                        speed_kmh=speed_kmh,
                        max_speed_kmh=args.max_model_speed_kmh,
                    )
                except Exception as error:
                    consecutive_model_errors += 1
                    control = carla.VehicleControl(brake=1.0)
                    previous_steer = 0.0
                    last_control_reason = f"safety-brake:{type(error).__name__}"
                    print(f"WARNING: {error}; applying full brake", file=sys.stderr)
                    if consecutive_model_errors >= args.max_model_errors:
                        actors.ego.apply_control(control)
                        stop_reason = "model_error_limit"
                        break
                actors.ego.apply_control(control)

            if args.spectator_follow:
                _follow_spectator(carla, world.get_spectator(), actors.ego)

            if now >= status_at:
                location = actors.ego.get_location()
                destination_text = (
                    f" destination={destination_index}" if destination_index is not None else ""
                )
                print(
                    f"t={elapsed:6.1f}s speed={speed_kmh:5.1f}km/h "
                    f"x={location.x:8.1f} y={location.y:8.1f} "
                    f"control={last_control_reason}{destination_text}",
                    flush=True,
                )
                status_at = now + args.status_every

    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    finally:
        if driving_model is not None:
            try:
                driving_model.close()
            except Exception as error:
                print(f"WARNING: model close failed: {error}", file=sys.stderr)
        _cleanup(
            world,
            traffic_manager,
            actors,
            camera_stream,
            original_settings,
            changed_settings,
            args.keep_actors,
        )

    result = {
        "schema_version": "1.0",
        "server_version": server_version,
        "client_version": client_version,
        "map": _map_basename(world.get_map().name),
        "driver": args.driver,
        "ego_actor_id": int(actors.ego.id) if actors.ego is not None else None,
        "npc_vehicle_count": len(actors.vehicles),
        "walker_count": len(actors.walkers),
        "elapsed_seconds": time.monotonic() - started,
        "stop_reason": stop_reason,
        "actors_kept": bool(args.keep_actors),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
