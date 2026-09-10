"""Deterministic BehaviorAgent teacher episode collection for CARLA 0.9.16.

This collector reuses the verified scenario plans, exact sensor synchronization,
and immutable DatasetWriter format used by :mod:`carla_vision.native.worker`.
CARLA and its ``agents`` package are imported only for a real collection run.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..dataset.camera_rig import resolve_camera_rig
from ..dataset.sync import SynchronizedFramePair
from ..dataset.writer import DATASET_PARTITIONS, DatasetWriter
from ..scenarios.planner import EpisodePlan
from ..scenarios.splits import canonical_map_family
from ..scenarios.verified_plan import VerifiedScenarioPlan, load_verified_scenario_plan
from .camera_rig import TeacherCameraRig, read_frame_bundle
from .synchronization import SensorFrameError, image_to_bridge_frame
from .teacher_routes import (
    BEHAVIOR_TEACHER_CONTROL_SCHEMA_VERSION,
    NAVIGATION_INTENT_SCHEMA_VERSION,
    build_route_leg,
    choose_destination_index,
    navigation_intent_from_plan,
    serialize_behavior_control,
)
from .teacher_verify import verify_teacher_dataset
from .worker import (
    NATIVE_WORKER_SCHEMA_VERSION,
    EpisodeActors,
    NativeCarlaSession,
    _actor_record,
    _load_carla_module,
    _privileged_ego_state,
    _response_ids,
    _set_blueprint_attribute,
    _transform_recipe,
    select_episodes,
)

BEHAVIOR_TEACHER_WORKER_SCHEMA_VERSION = "1.0"


def _load_behavior_agent() -> type[Any]:
    try:
        module = importlib.import_module("agents.navigation.behavior_agent")
    except ImportError as error:
        raise RuntimeError(
            "BehaviorAgent is unavailable. Add CARLA/PythonAPI/carla to PYTHONPATH; "
            "installing only the carla wheel may not include the agents package."
        ) from error
    agent_type = getattr(module, "BehaviorAgent", None)
    if agent_type is None:
        raise RuntimeError("agents.navigation.behavior_agent has no BehaviorAgent")
    return agent_type


def _distance_between_locations(left: Any, right: Any) -> float:
    return math.sqrt(
        (float(left.x) - float(right.x)) ** 2
        + (float(left.y) - float(right.y)) ** 2
        + (float(left.z) - float(right.z)) ** 2
    )


class BehaviorRouteController:
    """Own deterministic route legs and apply BehaviorAgent controls before ticks."""

    def __init__(
        self,
        ego: Any,
        spawn_points: Sequence[Any],
        episode: EpisodePlan,
        *,
        behavior: str,
        target_speed_kmh: float,
        minimum_route_distance_m: float,
    ) -> None:
        if not spawn_points:
            raise ValueError("spawn_points must not be empty")
        self.ego = ego
        self.spawn_points = tuple(spawn_points)
        self.episode = episode
        self.behavior = behavior
        self.target_speed_kmh = float(target_speed_kmh)
        self.minimum_route_distance_m = float(minimum_route_distance_m)
        self.route_seed = int(episode.seeds.values["route"])
        self._agent = _load_behavior_agent()(ego, behavior=behavior)
        if hasattr(self._agent, "set_target_speed"):
            self._agent.set_target_speed(self.target_speed_kmh)
        self._current_spawn_index = int(episode.recipe.ego_spawn_index)
        self._leg_index = -1
        self._current_leg: dict[str, Any] | None = None
        self._history: list[dict[str, Any]] = []
        self._start_next_leg(completed_before_frame=None)

    def _start_next_leg(self, *, completed_before_frame: int | None) -> None:
        if self._current_leg is not None:
            self._current_leg["completed"] = True
            self._current_leg["completed_before_frame"] = completed_before_frame
            self._current_spawn_index = int(self._current_leg["destination_spawn_index"])
        self._leg_index += 1
        destination_index = choose_destination_index(
            self.spawn_points,
            current_location=self.ego.get_location(),
            current_spawn_index=self._current_spawn_index,
            route_seed=self.route_seed,
            leg_index=self._leg_index,
            minimum_distance_m=self.minimum_route_distance_m,
        )
        leg = build_route_leg(
            episode_id=self.episode.episode_id,
            leg_index=self._leg_index,
            route_seed=self.route_seed,
            start_spawn_index=self._current_spawn_index,
            destination_spawn_index=destination_index,
            spawn_points=self.spawn_points,
        ).as_dict()
        leg.update(
            {
                "behavior": self.behavior,
                "target_speed_kmh": self.target_speed_kmh,
                "completed": False,
                "completed_before_frame": None,
            }
        )
        self._agent.set_destination(self.spawn_points[destination_index].location)
        self._current_leg = leg
        self._history.append(leg)

    def apply_before_tick(self, *, current_world_frame: int) -> Any:
        if bool(self._agent.done()):
            self._start_next_leg(completed_before_frame=current_world_frame)
        control = self._agent.run_step(debug=False)
        self.ego.apply_control(control)
        return control

    def sample_route_context(self, *, carla_frame: int) -> dict[str, Any]:
        if self._current_leg is None:
            raise RuntimeError("route controller has no current leg")
        destination_index = int(self._current_leg["destination_spawn_index"])
        remaining = _distance_between_locations(
            self.ego.get_location(), self.spawn_points[destination_index].location
        )
        return {
            **dict(self._current_leg),
            "remaining_straight_line_distance_m": remaining,
            "sample_carla_frame": int(carla_frame),
        }

    def sample_navigation_intent(self, *, carla_frame: int) -> dict[str, Any]:
        if self._current_leg is None:
            raise RuntimeError("route controller has no current leg")
        local_planner = self._agent.get_local_planner()
        return navigation_intent_from_plan(
            ego_transform=self.ego.get_transform(),
            plan=list(local_planner.get_plan()),
            carla_frame=carla_frame,
            route_id=str(self._current_leg["route_id"]),
        ).as_dict()

    @property
    def history(self) -> list[dict[str, Any]]:
        return [dict(leg) for leg in self._history]


class BehaviorTeacherSession(NativeCarlaSession):
    """Native CARLA session whose ego is controlled by BehaviorAgent."""

    def __init__(
        self,
        carla: Any,
        *,
        behavior: str,
        target_speed_kmh: float,
        minimum_route_distance_m: float,
        camera_rig: str = "front",
        camera_rig_config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(carla, **kwargs)
        self.behavior = behavior
        self.target_speed_kmh = target_speed_kmh
        self.minimum_route_distance_m = minimum_route_distance_m
        self.camera_rig = camera_rig
        self.camera_rig_config = camera_rig_config

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
        actor_ids, _ = _response_ids(
            self.client.apply_batch_sync(
                [
                    self.carla.command.SpawnActor(
                        blueprint,
                        spawn_points[recipe.ego_spawn_index],
                    )
                ],
                True,
            ),
            operation="spawn_behavior_teacher_ego",
            strict=True,
        )
        actors.ego_id = actor_ids[0]
        ego = self.world.get_actor(actors.ego_id)
        if ego is None:
            raise RuntimeError("spawned BehaviorAgent ego actor could not be retrieved")
        return ego

    def _actor_inventory(self, actors: EpisodeActors) -> list[dict[str, Any]]:
        role_by_id = {
            actors.ego_id: "ego_behavior_agent_teacher_vehicle",
            actors.rgb_sensor_id: "front_rgb_model_input",
            actors.teacher_sensor_id: "front_instance_teacher",
            **{actor_id: "additional_rgb_model_input" for actor_id in actors.additional_sensor_ids},
            **{actor_id: "background_vehicle" for actor_id in actors.traffic_vehicle_ids},
            **{actor_id: "walker" for actor_id in actors.walker_ids},
            **{actor_id: "walker_controller" for actor_id in actors.walker_controller_ids},
            **{actor_id: "static_prop" for actor_id in actors.prop_ids},
        }
        inventory: list[dict[str, Any]] = []
        for actor_id in actors.destruction_order():
            actor = self.world.get_actor(actor_id)
            if actor is not None:
                inventory.append(_actor_record(actor, role=role_by_id.get(actor_id, "unknown")))
        return sorted(inventory, key=lambda item: item["actor_id"])

    def run_episode(self, episode: EpisodePlan, writer: DatasetWriter) -> dict[str, Any]:
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
            ego = self._spawn_ego(episode, actors, spawn_points, blueprint_library)
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
                self._spawn_walkers(episode, actors, blueprint_library)
            )
            recipes = resolve_camera_rig(
                episode.recipe.camera, preset=self.camera_rig, config=self.camera_rig_config
            )
            rig = TeacherCameraRig(recipes) if len(recipes) > 1 else None
            if rig is None:
                rgb_sensor, teacher_sensor, rgb_queue, teacher_queue = self._spawn_cameras(
                    episode, actors, blueprint_library,
                )
                sensors.extend((rgb_sensor, teacher_sensor))
            else:
                rig.spawn(self, episode, actors, sensors)
                rgb_queue, teacher_queue = rig.queues["front"], rig.queues["front_teacher"]
            controller = BehaviorRouteController(
                ego,
                spawn_points,
                episode,
                behavior=self.behavior,
                target_speed_kmh=self.target_speed_kmh,
                minimum_route_distance_m=self.minimum_route_distance_m,
            )
            actor_inventory = self._actor_inventory(actors)
            recipe = episode.recipe
            sensor_tick_multiple = round(
                recipe.camera.sensor_tick_seconds / recipe.fixed_delta_seconds
            )
            if sensor_tick_multiple <= 0:
                raise RuntimeError("camera sensor period must be at least one world tick")
            if recipe.capture.warmup_ticks % sensor_tick_multiple:
                raise RuntimeError(
                    "warmup_ticks must preserve the camera sensor phase for exact-frame capture"
                )
            if recipe.capture.capture_every_ticks % sensor_tick_multiple:
                raise RuntimeError(
                    "capture_every_ticks must be a multiple of the camera sensor period"
                )

            def controlled_tick() -> tuple[int, Any, dict[str, Any], dict[str, Any]]:
                snapshot = world.get_snapshot()
                control = controller.apply_before_tick(current_world_frame=int(snapshot.frame))
                frame = int(world.tick())
                return (
                    frame,
                    control,
                    controller.sample_route_context(carla_frame=frame),
                    controller.sample_navigation_intent(carla_frame=frame),
                )

            for _ in range(sensor_tick_multiple):
                controlled_tick()
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
            if rig is not None:
                read_frame_bundle(
                    {name: sensor_queue for name, sensor_queue in rig.queues.items()
                     if name not in {"front", "front_teacher"}},
                    sensor_phase_frame, self.sensor_timeout,
                )
            for _ in range(recipe.capture.warmup_ticks):
                controlled_tick()

            first_capture: tuple[int, Any, dict[str, Any], dict[str, Any]] | None = None
            alignment_ticks = 0
            for _ in range(sensor_tick_multiple):
                candidate = controlled_tick()
                alignment_ticks += 1
                if (candidate[0] - sensor_phase_frame) % sensor_tick_multiple == 0:
                    first_capture = candidate
                    break
            if first_capture is None:
                raise RuntimeError("could not align collection to the camera sensor phase")

            sample_ids: list[str] = []
            captured_frames: list[int] = []
            route_ids: list[str] = []
            navigation_commands: Counter[str] = Counter()
            annotation_count = 0
            first_timestamp: float | None = None
            last_timestamp: float | None = None
            for tick_index in range(recipe.capture.duration_ticks):
                world_frame, applied_control, route_context, navigation_intent = (
                    first_capture if tick_index == 0 else controlled_tick()
                )
                if tick_index % recipe.capture.capture_every_ticks:
                    continue
                if (world_frame - sensor_phase_frame) % sensor_tick_multiple:
                    raise RuntimeError(
                        f"capture frame {world_frame} is outside the camera sensor phase"
                    )
                view_frames = None
                if rig is None:
                    rgb_queued = rgb_queue.get_exact(world_frame, self.sensor_timeout)
                    teacher_queued = teacher_queue.get_exact(world_frame, self.sensor_timeout)
                else:
                    try:
                        bundle = read_frame_bundle(rig.queues, world_frame, self.sensor_timeout)
                    except SensorFrameError as error:
                        writer.add_auxiliary_json(
                            f"episodes/{episode.episode_id}/capture_failure.json",
                            {"carla_frame": world_frame, "error": str(error),
                             "expected_cameras": list(recipes), "queues": rig.diagnostics()},
                            role="rgb_capture_failure",
                        )
                        raise
                    rgb_queued, teacher_queued = bundle["front"], bundle["front_teacher"]
                    view_frames = {
                        name: image_to_bridge_frame(
                            bundle[name], fov_degrees=camera.fov_degrees, sensor_type=0
                        ) for name, camera in recipes.items() if name != "front"
                    }
                rgb_frame = image_to_bridge_frame(
                    rgb_queued,
                    fov_degrees=recipe.camera.fov_degrees,
                    sensor_type=0,
                )
                if view_frames is not None:
                    view_frames["front"] = rgb_frame
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
                route_id = str(route_context["route_id"])
                teacher_control = serialize_behavior_control(
                    applied_control,
                    carla_frame=world_frame,
                    route_id=route_id,
                )
                sample = writer.add_pair(
                    pair,
                    rgb_views=view_frames,
                    camera_calibrations=rig.calibrations if rig is not None else None,
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
                        "control_mode": "behavior_agent_teacher",
                        "teacher_uses_privileged_simulator_state": True,
                        "runtime_sensor_contract": "front_monocular_rgb_only",
                        "route": route_context,
                        "navigation_intent": navigation_intent,
                        "actors": {
                            "ego": actors.ego_id,
                            "rgb_camera": actors.rgb_sensor_id,
                            "instance_teacher": actors.teacher_sensor_id,
                        },
                        "privileged_evaluation": _privileged_ego_state(ego),
                        "privileged_teacher_control": teacher_control,
                    },
                )
                sample_ids.append(sample.sample_id)
                captured_frames.append(world_frame)
                route_ids.append(route_id)
                navigation_commands[str(navigation_intent["command"])] += 1
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
                "schema_version": BEHAVIOR_TEACHER_WORKER_SCHEMA_VERSION,
                "native_worker_schema_version": NATIVE_WORKER_SCHEMA_VERSION,
                "rgb_camera_ids": list(recipes),
                "rgb_calibrations": rig.calibrations if rig is not None else {},
                "rgb_queues": rig.diagnostics() if rig is not None else {},
                "status": "complete",
                "episode": episode.as_dict(),
                "teacher": {
                    "control_mode": "behavior_agent_teacher",
                    "behavior": self.behavior,
                    "target_speed_kmh": self.target_speed_kmh,
                    "control_schema_version": BEHAVIOR_TEACHER_CONTROL_SCHEMA_VERSION,
                    "navigation_intent_schema_version": NAVIGATION_INTENT_SCHEMA_VERSION,
                    "control_alignment": "BehaviorAgent.run_step applied before captured world.tick",
                },
                "route": {
                    "route_seed": int(episode.seeds.values["route"]),
                    "minimum_route_distance_m": self.minimum_route_distance_m,
                    "leg_count": len(controller.history),
                    "legs": controller.history,
                },
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
                    "route_ids": route_ids,
                    "navigation_command_counts": dict(sorted(navigation_commands.items())),
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
                    "CARLA episode cleanup also failed: "
                    + json.dumps(cleanup, ensure_ascii=False)
                )
            raise primary_error
        if not cleanup.get("success", False):
            raise RuntimeError(
                "CARLA episode cleanup failed: " + json.dumps(cleanup, ensure_ascii=False)
            )
        if result is None:
            raise RuntimeError("BehaviorAgent episode completed without a result")
        result["actors"]["cleanup"] = cleanup
        result["elapsed_wall_seconds"] = time.monotonic() - started
        return result


def _dry_run_summary(
    plan: VerifiedScenarioPlan,
    episodes: Sequence[EpisodePlan],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "mode": "dry_run",
        "collector": "behavior_agent_teacher",
        "scenario_plan": dict(plan.reference),
        "suite_id": plan.suite.suite_id,
        "carla_version": plan.suite.carla_version,
        "behavior": args.behavior,
        "target_speed_kmh": args.target_speed_kmh,
        "minimum_route_distance_m": args.minimum_route_distance_m,
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
        "would_verify_dataset_after_collection": bool(args.verify_after_collection),
        "destructive_scope": (
            "Each selected episode loads/reloads its map, destroys existing actors, "
            "and becomes the sole synchronous tick owner."
        ),
    }


def collect_behavior_teacher(args: argparse.Namespace) -> dict[str, Any]:
    plan = load_verified_scenario_plan(args.scenario_plan)
    episodes = select_episodes(
        plan,
        episode_ids=args.episode_id,
        partitions=args.partition,
        max_episodes=args.max_episodes,
    )
    rig_config = None
    if args.camera_rig_config:
        rig_config = json.loads(Path(args.camera_rig_config).read_text(encoding="utf-8"))
        if not isinstance(rig_config, Mapping):
            raise ValueError("camera rig config must be a JSON object")
    resolved_rigs = {
        episode.episode_id: {
            name: camera.as_dict() for name, camera in resolve_camera_rig(
                episode.recipe.camera, preset=args.camera_rig, config=rig_config
            ).items()
        } for episode in episodes
    }
    if args.dry_run:
        result = _dry_run_summary(plan, episodes, args)
        result["camera_rigs"] = resolved_rigs
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    if not args.acknowledge_exclusive_tick_owner:
        raise RuntimeError(
            "real BehaviorAgent collection requires --acknowledge-exclusive-tick-owner; "
            "map loading destroys existing actors and this worker must be the only tick owner"
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
            "one dataset release requires identical label-filter thresholds across episodes"
        )
    minimum_pixels, minimum_box_width, minimum_box_height = next(iter(thresholds))
    carla = _load_carla_module(args.carla_python_api)
    session = BehaviorTeacherSession(
        carla,
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        expected_version=plan.suite.carla_version,
        traffic_manager_port=plan.suite.traffic_manager_port,
        sensor_timeout=args.sensor_timeout,
        behavior=args.behavior,
        target_speed_kmh=args.target_speed_kmh,
        minimum_route_distance_m=args.minimum_route_distance_m,
        camera_rig=args.camera_rig,
        camera_rig_config=rig_config,
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
            "collector": "behavior_agent_teacher_official_pythonapi",
            "worker_schema_version": BEHAVIOR_TEACHER_WORKER_SCHEMA_VERSION,
            "scenario_plan": dict(plan.reference),
            "selected_episode_ids": [episode.episode_id for episode in episodes],
            "single_tick_owner_acknowledged": True,
            "synchronization": "synchronous_fixed_delta_exact_sensor_frame",
            "control_mode": "behavior_agent_teacher",
            "behavior": args.behavior,
            "target_speed_kmh": args.target_speed_kmh,
            "teacher_uses_privileged_simulator_state": True,
            "runtime_sensor_contract": "front_monocular_rgb_only",
            "camera_rigs": resolved_rigs,
            "navigation_intent": {
                "schema_version": NAVIGATION_INTENT_SCHEMA_VERSION,
                "source": "carla_global_route_planner_via_behavior_agent",
                "available_for_every_sample": True,
                "privileged": True,
                "runtime_model_input": False,
                "strict_rgb_only_input": False,
                "route_conditioned_vision_input": True,
            },
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
                    "schema_version": BEHAVIOR_TEACHER_WORKER_SCHEMA_VERSION,
                    "scenario_plan_run_id": plan.run_id,
                    "control_mode": "behavior_agent_teacher",
                    "episodes": [episode.as_dict() for episode in episodes],
                },
                role="selected_behavior_teacher_episode_plans",
                metadata={"episode_count": len(episodes)},
            )
            for index, episode in enumerate(episodes, start=1):
                print(
                    f"episode={index}/{len(episodes)} id={episode.episode_id} "
                    f"partition={episode.split.partition} map={episode.recipe.map_name} "
                    f"teacher=BehaviorAgent:{args.behavior}",
                    flush=True,
                )
                episode_result = session.run_episode(episode, writer)
                writer.add_auxiliary_json(
                    f"episodes/{episode.episode_id}.json",
                    episode_result,
                    role="behavior_teacher_episode_provenance",
                    metadata={
                        "episode_id": episode.episode_id,
                        "partition": episode.split.partition,
                        "sample_count": episode_result["samples"]["count"],
                        "route_leg_count": episode_result["route"]["leg_count"],
                    },
                )
                episode_results.append(episode_result)
            session.restore_asynchronous_mode()
            restored = True
            writer.set_release_metadata(
                {
                    "collector": "behavior_agent_teacher_official_pythonapi",
                    "control_mode": "behavior_agent_teacher",
                    "behavior": args.behavior,
                    "target_speed_kmh": args.target_speed_kmh,
                    "scenario_plan": dict(plan.reference),
                    "episode_count": len(episode_results),
                    "episode_ids": [
                        result["episode"]["episode_id"] for result in episode_results
                    ],
                    "partition_counts": dict(
                        sorted(
                            Counter(
                                result["episode"]["partition"] for result in episode_results
                            ).items()
                        )
                    ),
                    "sample_count": sum(
                        result["samples"]["count"] for result in episode_results
                    ),
                    "annotation_count": sum(
                        result["samples"]["annotation_count"] for result in episode_results
                    ),
                    "route_leg_count": sum(
                        result["route"]["leg_count"] for result in episode_results
                    ),
                    "privileged_teacher_control": {
                        "schema_version": BEHAVIOR_TEACHER_CONTROL_SCHEMA_VERSION,
                        "available_for_every_sample": True,
                        "source": "BehaviorAgent.run_step",
                        "sample_alignment": "applied before exact captured world.tick",
                        "runtime_model_input": False,
                    },
                    "navigation_intent": {
                        "schema_version": NAVIGATION_INTENT_SCHEMA_VERSION,
                        "available_for_every_sample": True,
                        "source": "carla_global_route_planner_via_behavior_agent",
                        "privileged": True,
                        "runtime_model_input": False,
                        "strict_rgb_only_input": False,
                        "route_conditioned_vision_input": True,
                        "command_counts": dict(
                            sorted(
                                sum(
                                    (
                                        Counter(
                                            result["samples"]["navigation_command_counts"]
                                        )
                                        for result in episode_results
                                    ),
                                    Counter(),
                                ).items()
                            )
                        ),
                    },
                    "world_restored_to_asynchronous_mode": True,
                }
            )
        except BaseException as error:
            if not restored:
                try:
                    session.restore_asynchronous_mode()
                except BaseException as restore_error:
                    error.add_note(
                        f"CARLA asynchronous-mode restore also failed: {restore_error}"
                    )
            raise

    dataset_dir = (Path(args.datasets_root) / args.dataset_id).resolve()
    verification = (
        verify_teacher_dataset(dataset_dir) if args.verify_after_collection else None
    )
    if verification is not None and verification["status"] != "passed":
        raise RuntimeError(
            "BehaviorAgent teacher dataset verification failed: "
            + json.dumps(verification["errors"], ensure_ascii=False)
        )
    result = {
        "dataset_id": args.dataset_id,
        "dataset_dir": str(dataset_dir),
        "scenario_plan_run_id": plan.run_id,
        "control_mode": "behavior_agent_teacher",
        "behavior": args.behavior,
        "episode_count": len(episode_results),
        "sample_count": sum(result["samples"]["count"] for result in episode_results),
        "annotation_count": sum(
            result["samples"]["annotation_count"] for result in episode_results
        ),
        "route_leg_count": sum(
            result["route"]["leg_count"] for result in episode_results
        ),
        "server_version": session.server_version,
        "client_version": session.client_version,
        "world_restored_to_asynchronous_mode": restored,
        "verification": verification,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect deterministic route-driven CARLA episodes with BehaviorAgent "
            "controls aligned to exact RGB frames."
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
    rig_options = parser.add_mutually_exclusive_group()
    rig_options.add_argument(
        "--camera-rig", choices=("front", "front-three"), default="front",
        help="RGB collection rig; front-three adds left/right views without changing Drive",
    )
    rig_options.add_argument(
        "--camera-rig-config", help="JSON rig with additional RGB camera mounts and FOVs",
    )
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument(
        "--partition",
        action="append",
        choices=tuple(sorted(DATASET_PARTITIONS)),
        default=[],
    )
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument(
        "--behavior",
        choices=("cautious", "normal", "aggressive"),
        default="normal",
    )
    parser.add_argument("--target-speed-kmh", type=float, default=35.0)
    parser.add_argument("--minimum-route-distance-m", type=float, default=40.0)
    parser.add_argument(
        "--verify-after-collection",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--acknowledge-exclusive-tick-owner",
        action="store_true",
        help=(
            "required for a real run; acknowledges map reload/destruction and "
            "exclusive ownership of world.tick()"
        ),
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    if args.sensor_timeout <= 0.0:
        parser.error("--sensor-timeout must be positive")
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes must be positive")
    if args.target_speed_kmh <= 0.0:
        parser.error("--target-speed-kmh must be positive")
    if args.minimum_route_distance_m < 0.0:
        parser.error("--minimum-route-distance-m must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    collect_behavior_teacher(parse_args(argv))
    return 0


__all__ = [
    "BEHAVIOR_TEACHER_WORKER_SCHEMA_VERSION",
    "BehaviorRouteController",
    "BehaviorTeacherSession",
    "collect_behavior_teacher",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    raise SystemExit(main())
