"""Native CARLA 0.9.16 deterministic multi-episode dataset worker.

This module intentionally imports ``carla`` only for a real collection run.
Plan verification and ``--dry-run`` therefore work on machines where the
version-matched native CARLA PythonAPI wheel is unavailable.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..dataset.sync import SynchronizedFramePair
from ..dataset.writer import DATASET_PARTITIONS, DatasetWriter
from ..scenarios.contracts import ScenarioRecipe, TransformRecipe
from ..scenarios.planner import EpisodePlan
from ..scenarios.splits import canonical_map_family
from ..scenarios.verified_plan import (
    VerifiedScenarioPlan,
    load_verified_scenario_plan,
)
from .synchronization import (
    NativeSensorQueue,
    compose_relative_transform,
    image_to_bridge_frame,
)

NATIVE_WORKER_SCHEMA_VERSION = "1.1"
PRIVILEGED_TEACHER_CONTROL_SCHEMA_VERSION = "1.0"


def _load_carla_module(python_api_path: str | None) -> Any:
    if python_api_path:
        resolved = Path(python_api_path).expanduser().resolve(strict=True)
        sys.path.insert(0, str(resolved))
    try:
        return importlib.import_module("carla")
    except ImportError as error:
        raise RuntimeError(
            "the native worker requires the version-matched CARLA 0.9.16 "
            "PythonAPI wheel/egg; provide it through --carla-python-api or PYTHONPATH"
        ) from error


def select_episodes(
    plan: VerifiedScenarioPlan,
    *,
    episode_ids: Sequence[str] = (),
    partitions: Sequence[str] = (),
    max_episodes: int | None = None,
) -> tuple[EpisodePlan, ...]:
    requested_ids = tuple(dict.fromkeys(str(value) for value in episode_ids))
    requested_partitions = tuple(dict.fromkeys(str(value) for value in partitions))
    invalid_partitions = sorted(set(requested_partitions) - DATASET_PARTITIONS)
    if invalid_partitions:
        raise ValueError("unknown partitions: " + ", ".join(invalid_partitions))
    available_ids = {episode.episode_id for episode in plan.episodes}
    missing = sorted(set(requested_ids) - available_ids)
    if missing:
        raise ValueError("requested episode IDs are absent from the plan: " + ", ".join(missing))
    selected = tuple(
        episode
        for episode in plan.episodes
        if (not requested_ids or episode.episode_id in requested_ids)
        and (not requested_partitions or episode.split.partition in requested_partitions)
    )
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        selected = selected[:max_episodes]
    if not selected:
        raise ValueError("episode selection is empty")
    return selected


def _transform_recipe(transform: Any) -> TransformRecipe:
    return TransformRecipe(
        x=float(transform.location.x),
        y=float(transform.location.y),
        z=float(transform.location.z),
        pitch=float(transform.rotation.pitch),
        yaw=float(transform.rotation.yaw),
        roll=float(transform.rotation.roll),
    )


def _carla_transform(carla: Any, transform: TransformRecipe) -> Any:
    return carla.Transform(
        carla.Location(x=transform.x, y=transform.y, z=transform.z),
        carla.Rotation(
            pitch=transform.pitch,
            yaw=transform.yaw,
            roll=transform.roll,
        ),
    )


def _actor_record(actor: Any, *, role: str) -> dict[str, Any]:
    transform = _transform_recipe(actor.get_transform())
    return {
        "actor_id": int(actor.id),
        "type_id": str(actor.type_id),
        "role": role,
        "attributes": dict(sorted((str(k), str(v)) for k, v in actor.attributes.items())),
        "world_transform": transform.as_dict(),
    }


def _set_blueprint_attribute(blueprint: Any, name: str, value: Any) -> None:
    if blueprint.has_attribute(name):
        blueprint.set_attribute(name, str(value).lower() if isinstance(value, bool) else str(value))


def _filter_generation(
    blueprint_library: Any,
    pattern: str,
    generation: str,
) -> list[Any]:
    blueprints = sorted(blueprint_library.filter(pattern), key=lambda item: item.id)
    if generation.casefold() == "all" or len(blueprints) <= 1:
        return blueprints
    try:
        generation_number = int(generation)
    except ValueError as error:
        raise ValueError(f"invalid CARLA actor generation {generation!r}") from error
    return [
        blueprint
        for blueprint in blueprints
        if blueprint.has_attribute("generation")
        and int(blueprint.get_attribute("generation")) == generation_number
    ]


def _choose_blueprint(
    blueprints: Sequence[Any],
    rng: random.Random,
    *,
    role_name: str,
) -> Any:
    if not blueprints:
        raise RuntimeError("CARLA blueprint filter returned no candidates")
    blueprint = rng.choice(blueprints)
    _set_blueprint_attribute(blueprint, "role_name", role_name)
    for attribute_name in ("color", "driver_id"):
        if blueprint.has_attribute(attribute_name):
            values = list(blueprint.get_attribute(attribute_name).recommended_values)
            if values:
                blueprint.set_attribute(attribute_name, rng.choice(values))
    return blueprint


def _response_ids(
    responses: Sequence[Any],
    *,
    operation: str,
    strict: bool,
) -> tuple[list[int], list[dict[str, Any]]]:
    actor_ids: list[int] = []
    failures: list[dict[str, Any]] = []
    for index, response in enumerate(responses):
        if response.error:
            failures.append(
                {
                    "operation": operation,
                    "request_index": index,
                    "error": str(response.error),
                }
            )
        else:
            actor_ids.append(int(response.actor_id))
    if strict and failures:
        raise RuntimeError(
            f"{operation} failed for {len(failures)} request(s): "
            + "; ".join(item["error"] for item in failures)
        )
    return actor_ids, failures


def _map_basename(map_name: str) -> str:
    return str(map_name).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _privileged_ego_state(ego: Any) -> dict[str, Any]:
    transform = _transform_recipe(ego.get_transform())
    velocity = ego.get_velocity()
    acceleration = ego.get_acceleration()
    angular_velocity = ego.get_angular_velocity()
    traffic_light_id: int | None = None
    traffic_light_state: str | None = None
    if bool(ego.is_at_traffic_light()):
        light = ego.get_traffic_light()
        if light is not None:
            traffic_light_id = int(light.id)
            traffic_light_state = str(light.state)
    return {
        "privileged": True,
        "purpose": "teacher_and_evaluation_only",
        "world_transform": transform.as_dict(),
        "velocity_mps": {
            "x": float(velocity.x),
            "y": float(velocity.y),
            "z": float(velocity.z),
            "speed": math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2),
        },
        "acceleration_mps2": {
            "x": float(acceleration.x),
            "y": float(acceleration.y),
            "z": float(acceleration.z),
        },
        "angular_velocity_deg_s": {
            "x": float(angular_velocity.x),
            "y": float(angular_velocity.y),
            "z": float(angular_velocity.z),
        },
        "traffic_light_actor_id": traffic_light_id,
        "traffic_light_state": traffic_light_state,
    }


def _privileged_teacher_control(
    ego: Any,
    *,
    carla_frame: int,
) -> dict[str, Any]:
    """Serialize the teacher action applied to the captured CARLA frame."""

    control = ego.get_control()
    bounded = {
        "throttle": (float(control.throttle), 0.0, 1.0),
        "steer": (float(control.steer), -1.0, 1.0),
        "brake": (float(control.brake), 0.0, 1.0),
    }
    for name, (value, minimum, maximum) in bounded.items():
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise RuntimeError(
                f"CARLA teacher control {name} must be finite and in [{minimum}, {maximum}]"
            )
    return {
        "schema_version": PRIVILEGED_TEACHER_CONTROL_SCHEMA_VERSION,
        "privileged": True,
        "purpose": "offline_teacher_action_target_only",
        "source": "carla.Vehicle.get_control",
        "carla_frame": int(carla_frame),
        "throttle": bounded["throttle"][0],
        "steer": bounded["steer"][0],
        "brake": bounded["brake"][0],
        "hand_brake": bool(control.hand_brake),
        "reverse": bool(control.reverse),
        "manual_gear_shift": bool(control.manual_gear_shift),
        "gear": int(control.gear),
    }


@dataclass
class EpisodeActors:
    ego_id: int | None = None
    rgb_sensor_id: int | None = None
    teacher_sensor_id: int | None = None
    traffic_vehicle_ids: list[int] = field(default_factory=list)
    walker_ids: list[int] = field(default_factory=list)
    walker_controller_ids: list[int] = field(default_factory=list)
    prop_ids: list[int] = field(default_factory=list)
    additional_sensor_ids: list[int] = field(default_factory=list)

    def destruction_order(self) -> list[int]:
        values = [
            self.rgb_sensor_id,
            self.teacher_sensor_id,
            *self.additional_sensor_ids,
            *self.walker_controller_ids,
            *self.walker_ids,
            *self.traffic_vehicle_ids,
            *self.prop_ids,
            self.ego_id,
        ]
        return [int(value) for value in values if value is not None]


class NativeCarlaSession:
    """Exclusive synchronous CARLA session for one sequence of episodes."""

    def __init__(
        self,
        carla: Any,
        *,
        host: str,
        port: int,
        timeout: float,
        expected_version: str,
        traffic_manager_port: int,
        sensor_timeout: float,
    ) -> None:
        self.carla = carla
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.expected_version = expected_version
        self.traffic_manager_port = traffic_manager_port
        self.sensor_timeout = sensor_timeout
        self.world: Any | None = None
        self.traffic_manager: Any | None = None
        self.server_version = str(self.client.get_server_version())
        self.client_version = str(self.client.get_client_version())
        if self.server_version != expected_version:
            raise RuntimeError(
                f"CARLA server version {self.server_version!r} does not match "
                f"scenario plan {expected_version!r}"
            )
        if self.client_version != expected_version:
            raise RuntimeError(
                f"CARLA PythonAPI version {self.client_version!r} does not match "
                f"scenario plan {expected_version!r}"
            )

    def _configure_world(self, episode: EpisodePlan) -> tuple[Any, Any]:
        recipe = episode.recipe
        print(f"Loading CARLA map {recipe.map_name}", flush=True)
        world = self.client.load_world(recipe.map_name, reset_settings=True)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = recipe.fixed_delta_seconds
        settings.no_rendering_mode = False
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = max(
            1,
            math.ceil(recipe.fixed_delta_seconds / settings.max_substep_delta_time),
        )
        world.apply_settings(settings)
        print("Reloading map with synchronous capture settings", flush=True)
        world = self.client.reload_world(False)
        self.world = world
        actual_map = _map_basename(world.get_map().name)
        if actual_map != _map_basename(recipe.map_name):
            raise RuntimeError(
                f"loaded CARLA map {actual_map!r}, expected {_map_basename(recipe.map_name)!r}"
            )
        applied = world.get_settings()
        if not applied.synchronous_mode or not math.isclose(
            float(applied.fixed_delta_seconds),
            recipe.fixed_delta_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError("CARLA did not retain synchronous fixed-delta settings")

        traffic_manager = self.client.get_trafficmanager(self.traffic_manager_port)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(episode.seeds.values["traffic_manager"])
        traffic_manager.set_global_distance_to_leading_vehicle(
            recipe.traffic.global_distance_to_leading_vehicle
        )
        traffic_manager.global_percentage_speed_difference(
            recipe.traffic.global_speed_difference_percent
        )
        self.traffic_manager = traffic_manager
        world.set_pedestrians_seed(episode.seeds.values["walkers"])
        world.set_pedestrians_cross_factor(recipe.traffic.pedestrian_crossing_factor)
        self._set_weather(recipe)
        print("CARLA map, Traffic Manager and weather ready", flush=True)
        return world, traffic_manager

    def _set_weather(self, recipe: ScenarioRecipe) -> None:
        if self.world is None:
            raise RuntimeError("CARLA world is not configured")
        weather = self.carla.WeatherParameters()
        for name, value in recipe.weather.as_dict().items():
            if name in {"weather_id", "light"}:
                continue
            if not hasattr(weather, name):
                raise RuntimeError(f"CARLA {self.server_version} WeatherParameters lacks {name!r}")
            setattr(weather, name, float(value))
        self.world.set_weather(weather)

    def _spawn_ego(
        self,
        episode: EpisodePlan,
        actors: EpisodeActors,
        spawn_points: Sequence[Any],
        blueprint_library: Any,
    ) -> Any:
        recipe = episode.recipe
        if recipe.ego_spawn_index >= len(spawn_points):
            raise RuntimeError(
                f"ego spawn index {recipe.ego_spawn_index} exceeds "
                f"{len(spawn_points)} map spawn points"
            )
        blueprint = blueprint_library.find(recipe.ego_blueprint)
        _set_blueprint_attribute(blueprint, "role_name", "hero")
        command = self.carla.command.SpawnActor(
            blueprint,
            spawn_points[recipe.ego_spawn_index],
        ).then(
            self.carla.command.SetAutopilot(
                self.carla.command.FutureActor,
                True,
                self.traffic_manager_port,
            )
        )
        actor_ids, _ = _response_ids(
            self.client.apply_batch_sync([command], True),
            operation="spawn_ego",
            strict=True,
        )
        actors.ego_id = actor_ids[0]
        ego = self.world.get_actor(actors.ego_id)
        if ego is None:
            raise RuntimeError("spawned ego actor could not be retrieved")
        return ego

    def _spawn_props(
        self,
        episode: EpisodePlan,
        actors: EpisodeActors,
        blueprint_library: Any,
        ego_start: TransformRecipe,
    ) -> list[dict[str, Any]]:
        commands: list[Any] = []
        placements: list[dict[str, Any]] = []
        for prop in episode.recipe.props:
            blueprint = blueprint_library.find(prop.blueprint_id)
            world_transform = (
                compose_relative_transform(ego_start, prop.transform)
                if prop.relative_to == "ego_start"
                else prop.transform
            )
            commands.append(
                self.carla.command.SpawnActor(
                    blueprint,
                    _carla_transform(self.carla, world_transform),
                )
            )
            placements.append(
                {
                    "blueprint_id": prop.blueprint_id,
                    "relative_to": prop.relative_to,
                    "resolved_world_transform": world_transform.as_dict(),
                }
            )
        if commands:
            actor_ids, _ = _response_ids(
                self.client.apply_batch_sync(commands, True),
                operation="spawn_props",
                strict=True,
            )
            actors.prop_ids.extend(actor_ids)
        return placements

    def _spawn_traffic(
        self,
        episode: EpisodePlan,
        actors: EpisodeActors,
        spawn_points: Sequence[Any],
        blueprint_library: Any,
    ) -> list[dict[str, Any]]:
        recipe = episode.recipe
        rng = random.Random(episode.seeds.values["vehicles"])
        candidates = [
            point for index, point in enumerate(spawn_points) if index != recipe.ego_spawn_index
        ]
        rng.shuffle(candidates)
        requested = recipe.traffic.vehicle_count
        attempted_points = candidates[:requested]
        blueprints = _filter_generation(
            blueprint_library,
            recipe.traffic.vehicle_filter,
            recipe.traffic.vehicle_generation,
        )
        commands = []
        for point in attempted_points:
            blueprint = _choose_blueprint(
                blueprints,
                rng,
                role_name="scenario_traffic",
            )
            commands.append(
                self.carla.command.SpawnActor(blueprint, point).then(
                    self.carla.command.SetAutopilot(
                        self.carla.command.FutureActor,
                        True,
                        self.traffic_manager_port,
                    )
                )
            )
        actor_ids: list[int] = []
        failures: list[dict[str, Any]] = []
        if commands:
            actor_ids, failures = _response_ids(
                self.client.apply_batch_sync(commands, True),
                operation="spawn_traffic_vehicle",
                strict=False,
            )
            actors.traffic_vehicle_ids.extend(actor_ids)
            if recipe.traffic.automatic_vehicle_lights:
                for actor_id in actor_ids:
                    actor = self.world.get_actor(actor_id)
                    if actor is not None:
                        self.traffic_manager.update_vehicle_lights(actor, True)
        if requested > len(candidates):
            failures.append(
                {
                    "operation": "spawn_traffic_vehicle",
                    "error": (
                        f"requested {requested} vehicles but map provides only "
                        f"{len(candidates)} non-ego spawn points"
                    ),
                }
            )
        return failures

    def _spawn_walkers(
        self,
        episode: EpisodePlan,
        actors: EpisodeActors,
        blueprint_library: Any,
    ) -> list[dict[str, Any]]:
        recipe = episode.recipe
        requested = recipe.traffic.walker_count
        if requested == 0:
            return []
        rng = random.Random(episode.seeds.values["walkers"])
        crossing_rng = random.Random(episode.seeds.values["walker_crossing"])
        blueprints = _filter_generation(
            blueprint_library,
            recipe.traffic.walker_filter,
            recipe.traffic.walker_generation,
        )
        spawn_commands: list[Any] = []
        speeds: list[float] = []
        failures: list[dict[str, Any]] = []
        for request_index in range(requested):
            location = self.world.get_random_location_from_navigation()
            if location is None:
                failures.append(
                    {
                        "operation": "sample_walker_spawn",
                        "request_index": request_index,
                        "error": "navigation mesh returned no location",
                    }
                )
                continue
            blueprint = _choose_blueprint(
                blueprints,
                rng,
                role_name="scenario_walker",
            )
            _set_blueprint_attribute(blueprint, "is_invincible", False)
            recommended_speeds = (
                list(blueprint.get_attribute("speed").recommended_values)
                if blueprint.has_attribute("speed")
                else []
            )
            running = crossing_rng.random() < 0.1
            speed_index = 2 if running else 1
            speed = (
                float(recommended_speeds[min(speed_index, len(recommended_speeds) - 1)])
                if recommended_speeds
                else (2.5 if running else 1.4)
            )
            speeds.append(speed)
            spawn_commands.append(
                self.carla.command.SpawnActor(
                    blueprint,
                    self.carla.Transform(location),
                )
            )
        walker_responses = (
            self.client.apply_batch_sync(spawn_commands, True) if spawn_commands else []
        )
        walker_ids, spawn_failures = _response_ids(
            walker_responses,
            operation="spawn_walker",
            strict=False,
        )
        failures.extend(spawn_failures)
        successful_walkers = [
            (int(response.actor_id), speed)
            for speed, response in zip(speeds, walker_responses, strict=True)
            if not response.error
        ]
        actors.walker_ids.extend(walker_ids)
        controller_blueprint = blueprint_library.find("controller.ai.walker")
        controller_commands = [
            self.carla.command.SpawnActor(
                controller_blueprint,
                self.carla.Transform(),
                walker_id,
            )
            for walker_id, _ in successful_walkers
        ]
        controller_responses = (
            self.client.apply_batch_sync(controller_commands, True) if controller_commands else []
        )
        controller_ids, controller_failures = _response_ids(
            controller_responses,
            operation="spawn_walker_controller",
            strict=False,
        )
        failures.extend(controller_failures)
        actors.walker_controller_ids.extend(controller_ids)
        successful_controllers = [
            (int(response.actor_id), speed)
            for (_, speed), response in zip(
                successful_walkers,
                controller_responses,
                strict=True,
            )
            if not response.error
        ]
        for index, (controller_id, speed) in enumerate(successful_controllers):
            controller = self.world.get_actor(controller_id)
            if controller is None:
                failures.append(
                    {
                        "operation": "start_walker_controller",
                        "request_index": index,
                        "error": "controller actor could not be retrieved",
                    }
                )
                continue
            target = self.world.get_random_location_from_navigation()
            if target is None:
                failures.append(
                    {
                        "operation": "start_walker_controller",
                        "request_index": index,
                        "error": "navigation mesh returned no target",
                    }
                )
                continue
            controller.start()
            controller.go_to_location(target)
            controller.set_max_speed(speed)
        return failures

    def _camera_blueprint(
        self,
        blueprint_library: Any,
        sensor_type: str,
        episode: EpisodePlan,
        *,
        role_name: str,
    ) -> Any:
        camera = episode.recipe.camera
        blueprint = blueprint_library.find(sensor_type)
        values = {
            "role_name": role_name,
            "image_size_x": camera.width,
            "image_size_y": camera.height,
            "fov": camera.fov_degrees,
            "sensor_tick": camera.sensor_tick_seconds,
            "gamma": camera.gamma,
            "enable_postprocess_effects": camera.enable_postprocess_effects,
            "motion_blur_intensity": 0.0,
            "motion_blur_max_distortion": 0.0,
        }
        for name, value in values.items():
            _set_blueprint_attribute(blueprint, name, value)
        return blueprint

    def _spawn_cameras(
        self,
        episode: EpisodePlan,
        actors: EpisodeActors,
        blueprint_library: Any,
    ) -> tuple[Any, Any, NativeSensorQueue, NativeSensorQueue]:
        rgb_blueprint = self._camera_blueprint(
            blueprint_library,
            "sensor.camera.rgb",
            episode,
            role_name="front_rgb_model_input",
        )
        teacher_blueprint = self._camera_blueprint(
            blueprint_library,
            "sensor.camera.instance_segmentation",
            episode,
            role_name="front_instance_teacher",
        )
        mount = _carla_transform(self.carla, episode.recipe.camera.mount)
        commands = [
            self.carla.command.SpawnActor(rgb_blueprint, mount, actors.ego_id),
            self.carla.command.SpawnActor(teacher_blueprint, mount, actors.ego_id),
        ]
        sensor_ids, _ = _response_ids(
            self.client.apply_batch_sync(commands, False),
            operation="spawn_cameras",
            strict=True,
        )
        actors.rgb_sensor_id, actors.teacher_sensor_id = sensor_ids
        rgb_sensor = self.world.get_actor(actors.rgb_sensor_id)
        teacher_sensor = self.world.get_actor(actors.teacher_sensor_id)
        if rgb_sensor is None or teacher_sensor is None:
            raise RuntimeError("spawned camera actor could not be retrieved")
        rgb_queue = NativeSensorQueue("front RGB")
        teacher_queue = NativeSensorQueue("instance teacher")
        rgb_sensor.listen(rgb_queue.callback)
        teacher_sensor.listen(teacher_queue.callback)
        return rgb_sensor, teacher_sensor, rgb_queue, teacher_queue

    def _actor_inventory(self, actors: EpisodeActors) -> list[dict[str, Any]]:
        role_by_id = {
            actors.ego_id: "ego_teacher_driven_vehicle",
            actors.rgb_sensor_id: "front_rgb_model_input",
            actors.teacher_sensor_id: "front_instance_teacher",
            **{actor_id: "background_vehicle" for actor_id in actors.traffic_vehicle_ids},
            **{actor_id: "walker" for actor_id in actors.walker_ids},
            **{actor_id: "walker_controller" for actor_id in actors.walker_controller_ids},
            **{actor_id: "static_prop" for actor_id in actors.prop_ids},
        }
        inventory = []
        for actor_id in actors.destruction_order():
            actor = self.world.get_actor(actor_id)
            if actor is not None:
                inventory.append(_actor_record(actor, role=role_by_id.get(actor_id, "unknown")))
        return sorted(inventory, key=lambda item: item["actor_id"])

    def _cleanup_episode(
        self,
        actors: EpisodeActors,
        sensor_actors: Sequence[Any],
    ) -> dict[str, Any]:
        errors: list[dict[str, Any]] = []
        for sensor in sensor_actors:
            try:
                sensor.stop()
            except BaseException as error:
                errors.append(
                    {
                        "operation": "stop_sensor",
                        "actor_id": int(sensor.id),
                        "error": str(error),
                    }
                )
        for controller_id in actors.walker_controller_ids:
            actor = self.world.get_actor(controller_id)
            if actor is not None:
                try:
                    actor.stop()
                except BaseException as error:
                    errors.append(
                        {
                            "operation": "stop_walker_controller",
                            "actor_id": controller_id,
                            "error": str(error),
                        }
                    )
        actor_ids = actors.destruction_order()
        if actor_ids:
            responses = self.client.apply_batch_sync(
                [self.carla.command.DestroyActor(actor_id) for actor_id in actor_ids],
                True,
            )
            for actor_id, response in zip(actor_ids, responses, strict=True):
                if response.error:
                    errors.append(
                        {
                            "operation": "destroy_actor",
                            "actor_id": actor_id,
                            "error": str(response.error),
                        }
                    )
        destroy_error_count = sum(error["operation"] == "destroy_actor" for error in errors)
        return {
            "requested_actor_ids": actor_ids,
            "destroyed_actor_count": len(actor_ids) - destroy_error_count,
            "errors": errors,
            "success": not errors,
        }

    def run_episode(
        self,
        episode: EpisodePlan,
        writer: DatasetWriter,
    ) -> dict[str, Any]:
        started = time.monotonic()
        actors = EpisodeActors()
        sensors: list[Any] = []
        primary_error: BaseException | None = None
        result: dict[str, Any] | None = None
        cleanup: dict[str, Any] = {}
        try:
            world, _ = self._configure_world(episode)
            blueprint_library = world.get_blueprint_library()
            spawn_points = list(world.get_map().get_spawn_points())
            ego = self._spawn_ego(
                episode,
                actors,
                spawn_points,
                blueprint_library,
            )
            ego_start = _transform_recipe(ego.get_transform())
            prop_placements = self._spawn_props(
                episode,
                actors,
                blueprint_library,
                ego_start,
            )
            spawn_failures = self._spawn_traffic(
                episode,
                actors,
                spawn_points,
                blueprint_library,
            )
            spawn_failures.extend(
                self._spawn_walkers(
                    episode,
                    actors,
                    blueprint_library,
                )
            )
            rgb_sensor, teacher_sensor, rgb_queue, teacher_queue = self._spawn_cameras(
                episode,
                actors,
                blueprint_library,
            )
            sensors.extend((rgb_sensor, teacher_sensor))
            actor_inventory = self._actor_inventory(actors)

            recipe = episode.recipe
            sensor_tick_multiple = round(
                recipe.camera.sensor_tick_seconds / recipe.fixed_delta_seconds
            )
            if recipe.capture.warmup_ticks % sensor_tick_multiple:
                raise RuntimeError(
                    "warmup_ticks must preserve the camera sensor phase for exact-frame capture"
                )
            if recipe.capture.capture_every_ticks % sensor_tick_multiple:
                raise RuntimeError(
                    "capture_every_ticks must be a multiple of the camera sensor period"
                )
            for _ in range(sensor_tick_multiple):
                world.tick()
            first_rgb = rgb_queue.get_next(self.sensor_timeout)
            first_teacher = teacher_queue.get_next(self.sensor_timeout)
            first_rgb_frame = int(first_rgb.image.frame)
            first_teacher_frame = int(first_teacher.image.frame)
            if first_rgb_frame != first_teacher_frame:
                raise RuntimeError(
                    "co-located RGB and instance cameras started on different "
                    f"CARLA frames: {first_rgb_frame} and {first_teacher_frame}"
                )
            sensor_phase_frame = first_rgb_frame
            for _ in range(recipe.capture.warmup_ticks):
                world.tick()

            first_capture_frame: int | None = None
            alignment_ticks = 0
            for _ in range(sensor_tick_multiple):
                candidate = int(world.tick())
                alignment_ticks += 1
                if (candidate - sensor_phase_frame) % sensor_tick_multiple == 0:
                    first_capture_frame = candidate
                    break
            if first_capture_frame is None:
                raise RuntimeError("could not align collection to the camera sensor phase")

            sample_ids: list[str] = []
            captured_frames: list[int] = []
            annotation_count = 0
            first_timestamp: float | None = None
            last_timestamp: float | None = None
            for tick_index in range(recipe.capture.duration_ticks):
                world_frame = first_capture_frame if tick_index == 0 else int(world.tick())
                if tick_index % recipe.capture.capture_every_ticks:
                    continue
                if (world_frame - sensor_phase_frame) % sensor_tick_multiple:
                    raise RuntimeError(
                        f"capture frame {world_frame} is outside the camera sensor phase"
                    )
                rgb_queued = rgb_queue.get_exact(world_frame, self.sensor_timeout)
                teacher_queued = teacher_queue.get_exact(
                    world_frame,
                    self.sensor_timeout,
                )
                rgb_frame = image_to_bridge_frame(
                    rgb_queued,
                    fov_degrees=recipe.camera.fov_degrees,
                    sensor_type=0,
                )
                teacher_frame = image_to_bridge_frame(
                    teacher_queued,
                    fov_degrees=recipe.camera.fov_degrees,
                    sensor_type=1,
                )
                pair = SynchronizedFramePair(
                    rgb=rgb_frame,
                    teacher=teacher_frame,
                    rgb_skipped=rgb_queue.discarded,
                    teacher_skipped=teacher_queue.discarded,
                )
                privileged_evaluation = _privileged_ego_state(ego)
                privileged_teacher_control = _privileged_teacher_control(
                    ego,
                    carla_frame=world_frame,
                )
                sample = writer.add_pair(
                    pair,
                    split=episode.split.partition,
                    scenario_id=episode.scenario_id,
                    episode_id=episode.episode_id,
                    context={
                        "map": world.get_map().name,
                        "map_family": canonical_map_family(recipe.map_name),
                        "weather_recipe_id": recipe.weather.weather_id,
                        "light": recipe.weather.light,
                        "route_region_id": recipe.route_region_id,
                        "scenario_recipe_id": recipe.recipe_id,
                        "group": episode.group.as_dict(),
                        "control_mode": "traffic_manager_teacher",
                        "teacher_uses_privileged_simulator_state": True,
                        "runtime_sensor_contract": "front_monocular_rgb_only",
                        "actors": {
                            "ego": actors.ego_id,
                            "rgb_camera": actors.rgb_sensor_id,
                            "instance_teacher": actors.teacher_sensor_id,
                        },
                        "privileged_evaluation": privileged_evaluation,
                        "privileged_teacher_control": privileged_teacher_control,
                    },
                )
                sample_ids.append(sample.sample_id)
                captured_frames.append(world_frame)
                annotation_count += sample.annotation_count
                first_timestamp = (
                    rgb_frame.timestamp if first_timestamp is None else first_timestamp
                )
                last_timestamp = rgb_frame.timestamp

            expected_samples = (
                1 + (recipe.capture.duration_ticks - 1) // recipe.capture.capture_every_ticks
            )
            if len(sample_ids) != expected_samples:
                raise RuntimeError(
                    f"captured {len(sample_ids)} samples; expected {expected_samples}"
                )
            result = {
                "schema_version": NATIVE_WORKER_SCHEMA_VERSION,
                "status": "complete",
                "episode": episode.as_dict(),
                "carla": {
                    "client_version": self.client_version,
                    "server_version": self.server_version,
                    "map": world.get_map().name,
                    "world_settings": {
                        "synchronous_mode": True,
                        "fixed_delta_seconds": recipe.fixed_delta_seconds,
                        "single_tick_owner": True,
                        "sensor_tick_multiple": sensor_tick_multiple,
                        "sensor_phase_frame": sensor_phase_frame,
                        "capture_alignment_ticks": alignment_ticks,
                    },
                },
                "samples": {
                    "count": len(sample_ids),
                    "sample_ids": sample_ids,
                    "carla_frames": captured_frames,
                    "annotation_count": annotation_count,
                    "first_simulation_timestamp_seconds": first_timestamp,
                    "last_simulation_timestamp_seconds": last_timestamp,
                    "rgb_queue_received": rgb_queue.received,
                    "rgb_queue_discarded": rgb_queue.discarded,
                    "teacher_queue_received": teacher_queue.received,
                    "teacher_queue_discarded": teacher_queue.discarded,
                },
                "actors": {
                    "inventory_before_cleanup": actor_inventory,
                    "prop_placements": prop_placements,
                    "spawn_failures": spawn_failures,
                    "requested": {
                        "traffic_vehicles": recipe.traffic.vehicle_count,
                        "walkers": recipe.traffic.walker_count,
                        "props": len(recipe.props),
                    },
                    "spawned": {
                        "traffic_vehicles": len(actors.traffic_vehicle_ids),
                        "walkers": len(actors.walker_ids),
                        "walker_controllers": len(actors.walker_controller_ids),
                        "props": len(actors.prop_ids),
                    },
                },
                "elapsed_wall_seconds_before_cleanup": time.monotonic() - started,
            }
        except BaseException as error:
            primary_error = error
        finally:
            cleanup = self._cleanup_episode(actors, sensors)

        if primary_error is not None:
            if not cleanup.get("success", False):
                primary_error.add_note(
                    "CARLA episode cleanup also failed: " + json.dumps(cleanup, ensure_ascii=False)
                )
            raise primary_error
        if not cleanup.get("success", False):
            raise RuntimeError(
                "CARLA episode cleanup failed: " + json.dumps(cleanup, ensure_ascii=False)
            )
        if result is None:
            raise RuntimeError("native episode completed without a result")
        result["actors"]["cleanup"] = cleanup
        result["elapsed_wall_seconds"] = time.monotonic() - started
        return result

    def restore_asynchronous_mode(self) -> None:
        errors: list[str] = []
        if self.traffic_manager is not None:
            try:
                self.traffic_manager.set_synchronous_mode(False)
            except BaseException as error:
                errors.append(f"Traffic Manager restore: {error}")
        if self.world is not None:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                settings.no_rendering_mode = False
                self.world.apply_settings(settings)
            except BaseException as error:
                errors.append(f"world restore: {error}")
        if errors:
            raise RuntimeError("; ".join(errors))


def _dry_run_summary(
    plan: VerifiedScenarioPlan,
    episodes: Sequence[EpisodePlan],
) -> dict[str, Any]:
    return {
        "mode": "dry_run",
        "scenario_plan": dict(plan.reference),
        "suite_id": plan.suite.suite_id,
        "carla_version": plan.suite.carla_version,
        "selected_episode_count": len(episodes),
        "selected_partition_counts": dict(
            sorted(Counter(episode.split.partition for episode in episodes).items())
        ),
        "selected_map_families": sorted(
            {canonical_map_family(episode.recipe.map_name) for episode in episodes}
        ),
        "planned_capture_count": sum(
            1
            + (episode.recipe.capture.duration_ticks - 1)
            // episode.recipe.capture.capture_every_ticks
            for episode in episodes
        ),
        "would_mutate_simulator": True,
        "destructive_scope": (
            "Each selected episode loads/reloads its map, destroying existing "
            "world actors, then becomes the sole synchronous tick owner."
        ),
    }


def collect_native(args: argparse.Namespace) -> dict[str, Any]:
    plan = load_verified_scenario_plan(args.scenario_plan)
    episodes = select_episodes(
        plan,
        episode_ids=args.episode_id,
        partitions=args.partition,
        max_episodes=args.max_episodes,
    )
    if args.dry_run:
        result = _dry_run_summary(plan, episodes)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    if not args.acknowledge_exclusive_tick_owner:
        raise RuntimeError(
            "real native collection requires --acknowledge-exclusive-tick-owner; "
            "map loading destroys existing actors and the worker must be the only "
            "client that calls world.tick()"
        )

    thresholds = {
        (
            episode.recipe.capture.minimum_visible_pixels,
            episode.recipe.capture.minimum_box_width,
            episode.recipe.capture.minimum_box_height,
        )
        for episode in episodes
    }
    if len(thresholds) != 1:
        raise RuntimeError(
            "one dataset release requires identical label-filter thresholds "
            "across all selected episodes"
        )
    minimum_pixels, minimum_box_width, minimum_box_height = next(iter(thresholds))
    carla = _load_carla_module(args.carla_python_api)
    session = NativeCarlaSession(
        carla,
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        expected_version=plan.suite.carla_version,
        traffic_manager_port=plan.suite.traffic_manager_port,
        sensor_timeout=args.sensor_timeout,
    )
    episode_results: list[dict[str, Any]] = []
    restored = False
    with DatasetWriter(
        args.datasets_root,
        dataset_id=args.dataset_id,
        cli_args=vars(args),
        carla_endpoint={"host": args.host, "port": args.port},
        carla_version=session.server_version,
        carla_map=f"scenario-suite:{plan.suite.suite_id}",
        config={
            "collector": "native_official_pythonapi",
            "worker_schema_version": NATIVE_WORKER_SCHEMA_VERSION,
            "scenario_plan": dict(plan.reference),
            "selected_episode_ids": [episode.episode_id for episode in episodes],
            "single_tick_owner_acknowledged": True,
            "synchronization": "synchronous_fixed_delta_exact_sensor_frame",
            "control_mode": "traffic_manager_teacher",
            "teacher_uses_privileged_simulator_state": True,
            "runtime_sensor_contract": "front_monocular_rgb_only",
        },
        repository_root=Path.cwd(),
        minimum_pixels=minimum_pixels,
        minimum_box_width=minimum_box_width,
        minimum_box_height=minimum_box_height,
    ) as writer:
        try:
            writer.add_auxiliary_json(
                "provenance/scenario_plan_reference.json",
                dict(plan.reference),
                role="scenario_plan_reference",
                metadata={"scenario_plan_run_id": plan.run_id},
            )
            writer.add_auxiliary_json(
                "provenance/selected_episodes.json",
                {
                    "schema_version": NATIVE_WORKER_SCHEMA_VERSION,
                    "scenario_plan_run_id": plan.run_id,
                    "episodes": [episode.as_dict() for episode in episodes],
                },
                role="selected_episode_plans",
                metadata={"episode_count": len(episodes)},
            )
            for index, episode in enumerate(episodes, start=1):
                print(
                    f"episode={index}/{len(episodes)} id={episode.episode_id} "
                    f"partition={episode.split.partition} map={episode.recipe.map_name}",
                    flush=True,
                )
                episode_result = session.run_episode(episode, writer)
                writer.add_auxiliary_json(
                    f"episodes/{episode.episode_id}.json",
                    episode_result,
                    role="native_episode_provenance",
                    metadata={
                        "episode_id": episode.episode_id,
                        "partition": episode.split.partition,
                        "sample_count": episode_result["samples"]["count"],
                    },
                )
                episode_results.append(episode_result)
            session.restore_asynchronous_mode()
            restored = True
            writer.set_release_metadata(
                {
                    "collector": "native_official_pythonapi",
                    "scenario_plan": dict(plan.reference),
                    "episode_count": len(episode_results),
                    "episode_ids": [result["episode"]["episode_id"] for result in episode_results],
                    "partition_counts": dict(
                        sorted(
                            Counter(
                                result["episode"]["partition"] for result in episode_results
                            ).items()
                        )
                    ),
                    "sample_count": sum(result["samples"]["count"] for result in episode_results),
                    "annotation_count": sum(
                        result["samples"]["annotation_count"] for result in episode_results
                    ),
                    "privileged_teacher_control": {
                        "schema_version": PRIVILEGED_TEACHER_CONTROL_SCHEMA_VERSION,
                        "available_for_every_sample": True,
                        "source": "carla.Vehicle.get_control",
                        "sample_alignment": (
                            "queried_after_exact_sensor_frame_before_sample_write"
                        ),
                        "runtime_model_input": False,
                    },
                    "world_restored_to_asynchronous_mode": True,
                }
            )
        except BaseException as error:
            if not restored:
                try:
                    session.restore_asynchronous_mode()
                except BaseException as restore_error:
                    error.add_note(f"CARLA asynchronous-mode restore also failed: {restore_error}")
            raise

    result = {
        "dataset_id": args.dataset_id,
        "dataset_dir": str((Path(args.datasets_root) / args.dataset_id).resolve()),
        "scenario_plan_run_id": plan.run_id,
        "episode_count": len(episode_results),
        "sample_count": sum(result["samples"]["count"] for result in episode_results),
        "annotation_count": sum(
            result["samples"]["annotation_count"] for result in episode_results
        ),
        "server_version": session.server_version,
        "client_version": session.client_version,
        "world_restored_to_asynchronous_mode": restored,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an integrity-verified scenario plan through the official CARLA "
            "0.9.16 PythonAPI in synchronous fixed-delta mode"
        )
    )
    parser.add_argument("--scenario-plan", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--datasets-root", default="datasets")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sensor-timeout", type=float, default=10.0)
    parser.add_argument("--carla-python-api")
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument(
        "--partition",
        action="append",
        choices=tuple(sorted(DATASET_PARTITIONS)),
        default=[],
    )
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--acknowledge-exclusive-tick-owner",
        action="store_true",
        help=(
            "required for a real run; acknowledges that map reload destroys "
            "existing actors and this worker must be the sole world.tick owner"
        ),
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    if args.sensor_timeout <= 0.0:
        parser.error("--sensor-timeout must be positive")
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    collect_native(parse_args(argv))
    return 0


__all__ = [
    "NATIVE_WORKER_SCHEMA_VERSION",
    "PRIVILEGED_TEACHER_CONTROL_SCHEMA_VERSION",
    "NativeCarlaSession",
    "collect_native",
    "main",
    "parse_args",
    "select_episodes",
]


if __name__ == "__main__":
    raise SystemExit(main())
