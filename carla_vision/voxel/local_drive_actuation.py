"""Opt-in voxel steering integration for the existing ``carla-local-drive`` CLI.

Normal invocations delegate to :mod:`carla_vision.local_drive`.  The voxel path
is entered only with ``--enable-voxel-actuation`` and explicit operator
acknowledgement.  BehaviorAgent keeps longitudinal control while the voxel
planner may replace steering.  Any rejected/invalid voxel decision applies a
real full brake through CARLA.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from carla_vision import local_drive as base

from .actuation_runtime import VoxelActuationRuntime
from .actuation_supervisor import VoxelActuationSupervisorPolicy
from .contracts import VoxelGridSpec
from .models import CameraVoxelModelConfig, create_camera_voxel_predictor
from .shadow import ShadowPlannerConfig

VOXEL_LOCAL_DRIVE_SCHEMA_VERSION = "1.0"


def _parse_steering_values(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "voxel steering values must be comma-separated numbers"
        ) from error
    if not values or any(not math.isfinite(item) or not -1.0 <= item <= 1.0 for item in values):
        raise argparse.ArgumentTypeError("voxel steering values must be finite and in [-1, 1]")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    group = parser.add_argument_group("voxel actuation (opt-in)")
    group.add_argument("--enable-voxel-actuation", action="store_true")
    group.add_argument(
        "--acknowledge-voxel-actuation",
        action="store_true",
        help="explicitly acknowledge that voxel predictions will steer the CARLA ego vehicle",
    )
    group.add_argument("--voxel-predictor-factory")
    group.add_argument("--voxel-predictor-checkpoint", type=Path)
    group.add_argument("--voxel-predictor-device", default="cpu")
    group.add_argument("--voxel-predictor-options", type=base.parse_json_object, default={})
    group.add_argument("--voxel-readiness-report", type=Path)
    group.add_argument("--voxel-history-frames", type=int, default=4)
    group.add_argument(
        "--voxel-steering-values",
        type=_parse_steering_values,
        default=(-0.6, -0.3, 0.0, 0.3, 0.6),
    )
    group.add_argument("--voxel-trajectory-steps", type=int, default=12)
    group.add_argument("--voxel-trajectory-dt", type=float, default=0.25)
    group.add_argument("--voxel-wheelbase", type=float, default=2.8)
    group.add_argument("--voxel-ego-radius", type=float, default=1.2)
    group.add_argument("--voxel-uncertainty-band", type=float, default=0.08)
    group.add_argument("--voxel-max-collision-risk", type=float, default=0.20)
    group.add_argument("--voxel-max-uncertain-fraction", type=float, default=0.45)
    group.add_argument("--voxel-max-latency-ms", type=float, default=120.0)
    group.add_argument("--voxel-max-prediction-age", type=float, default=0.20)
    group.add_argument("--voxel-max-speed-kmh", type=float, default=35.0)
    group.add_argument("--voxel-max-absolute-steering", type=float, default=0.65)
    group.add_argument("--voxel-minimum-horizon", type=float, default=1.0)
    group.add_argument("--voxel-max-errors", type=int, default=3)
    group.add_argument("--voxel-x-min", type=float, default=0.0)
    group.add_argument("--voxel-x-max", type=float, default=50.0)
    group.add_argument("--voxel-y-min", type=float, default=-25.0)
    group.add_argument("--voxel-y-max", type=float, default=25.0)
    group.add_argument("--voxel-z-min", type=float, default=-2.0)
    group.add_argument("--voxel-z-max", type=float, default=5.0)
    group.add_argument("--voxel-resolution", type=float, default=0.5)
    return parser


def _validate_base_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
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


def _validate_voxel_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    voxel_options_present = any(
        (
            args.acknowledge_voxel_actuation,
            args.voxel_predictor_factory,
            args.voxel_predictor_checkpoint,
            args.voxel_predictor_options,
            args.voxel_readiness_report,
        )
    )
    if not args.enable_voxel_actuation:
        if voxel_options_present:
            parser.error("voxel actuation options require --enable-voxel-actuation")
        return
    if args.driver != "behavior":
        parser.error("--enable-voxel-actuation currently requires --driver behavior")
    if not args.acknowledge_voxel_actuation:
        parser.error("voxel actuation requires --acknowledge-voxel-actuation")
    if not args.voxel_predictor_factory:
        parser.error("voxel actuation requires --voxel-predictor-factory module:callable")
    if args.voxel_history_frames <= 0 or args.voxel_trajectory_steps <= 0:
        parser.error("voxel history/trajectory steps must be positive")
    if args.voxel_trajectory_dt <= 0 or args.voxel_wheelbase <= 0 or args.voxel_ego_radius < 0:
        parser.error("voxel trajectory geometry values are invalid")
    if not 0.0 <= args.voxel_uncertainty_band < 0.5:
        parser.error("--voxel-uncertainty-band must be in [0, 0.5)")
    if not 0.0 <= args.voxel_max_collision_risk <= 1.0:
        parser.error("--voxel-max-collision-risk must be in [0, 1]")
    if not 0.0 <= args.voxel_max_uncertain_fraction <= 1.0:
        parser.error("--voxel-max-uncertain-fraction must be in [0, 1]")
    if args.voxel_max_latency_ms <= 0 or args.voxel_max_prediction_age <= 0:
        parser.error("voxel latency/age limits must be positive")
    if args.voxel_max_speed_kmh <= 0 or args.voxel_minimum_horizon <= 0:
        parser.error("voxel speed/horizon limits must be positive")
    if not 0.0 < args.voxel_max_absolute_steering <= 1.0:
        parser.error("--voxel-max-absolute-steering must be in (0, 1]")
    if args.voxel_max_errors <= 0:
        parser.error("--voxel-max-errors must be positive")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_base_args(parser, args)
    _validate_voxel_args(parser, args)
    return args


def _grid_spec(args: argparse.Namespace) -> VoxelGridSpec:
    return VoxelGridSpec(
        x_min=args.voxel_x_min,
        x_max=args.voxel_x_max,
        y_min=args.voxel_y_min,
        y_max=args.voxel_y_max,
        z_min=args.voxel_z_min,
        z_max=args.voxel_z_max,
        resolution=args.voxel_resolution,
    )


def _load_readiness_report(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("voxel readiness report must contain a JSON object")
    return payload


def _planner_config(args: argparse.Namespace) -> ShadowPlannerConfig:
    return ShadowPlannerConfig(
        history_frames=args.voxel_history_frames,
        steering_values=args.voxel_steering_values,
        trajectory_steps=args.voxel_trajectory_steps,
        trajectory_dt_s=args.voxel_trajectory_dt,
        wheelbase_m=args.voxel_wheelbase,
        ego_radius_m=args.voxel_ego_radius,
        uncertainty_band=args.voxel_uncertainty_band,
    )


def _supervisor_policy(args: argparse.Namespace) -> VoxelActuationSupervisorPolicy:
    return VoxelActuationSupervisorPolicy(
        maximum_prediction_age_s=args.voxel_max_prediction_age,
        maximum_prediction_latency_ms=args.voxel_max_latency_ms,
        maximum_uncertain_voxel_fraction=args.voxel_max_uncertain_fraction,
        maximum_collision_risk=args.voxel_max_collision_risk,
        maximum_speed_mps=args.voxel_max_speed_kmh / 3.6,
        maximum_absolute_steering=args.voxel_max_absolute_steering,
        maximum_steering_rate_per_s=args.max_steer_rate,
        minimum_horizon_s=args.voxel_minimum_horizon,
    )


def _dry_run_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload = base._dry_run_payload(args)
    if not args.enable_voxel_actuation:
        return payload
    return {
        **payload,
        "schema_version": VOXEL_LOCAL_DRIVE_SCHEMA_VERSION,
        "voxel_actuation": {
            "enabled": True,
            "operator_acknowledged": True,
            "driver": "behavior",
            "longitudinal_control": "BehaviorAgent",
            "steering_control": "voxel_planner",
            "emergency_brake": "full CARLA brake on rejected/invalid voxel decision",
            "predictor_factory": args.voxel_predictor_factory,
            "predictor_checkpoint": (
                str(args.voxel_predictor_checkpoint) if args.voxel_predictor_checkpoint else None
            ),
            "readiness_report": (
                str(args.voxel_readiness_report) if args.voxel_readiness_report else None
            ),
            "grid": _grid_spec(args).as_dict(),
            "planner": asdict(_planner_config(args)),
            "supervisor": asdict(_supervisor_policy(args)),
        },
    }


def _run_voxel(args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run:
        payload = _dry_run_payload(args)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return payload

    carla = base._load_carla()
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    client_version = str(client.get_client_version())
    server_version = str(client.get_server_version())
    base._check_version(client_version, server_version, args)

    world = client.get_world()
    if args.map != "current" and base._map_basename(world.get_map().name) != base._map_basename(
        args.map
    ):
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
    actors = base.SpawnedActors()
    camera_stream = base.LatestCamera()
    behavior_agent: Any | None = None
    voxel_runtime: VoxelActuationRuntime | None = None
    destination_index: int | None = None
    stop_reason = "duration"
    started = time.monotonic()
    status_at = started
    previous_tick = started
    previous_camera_frame = -1
    previous_voxel_steering = 0.0
    consecutive_voxel_errors = 0
    voxel_control_calls = 0
    voxel_emergency_brakes = 0
    last_control_reason = "voxel:starting"
    last_voxel_result: dict[str, Any] | None = None

    try:
        spawn_points = list(world.get_map().get_spawn_points())
        if not spawn_points:
            raise RuntimeError("active CARLA map has no vehicle spawn points")
        actors.ego, ego_spawn_index = base._spawn_ego(
            world, args.ego_blueprint, spawn_points, args.ego_spawn_index, rng
        )
        actors.vehicles = base._spawn_npc_vehicles(
            world,
            traffic_manager,
            spawn_points,
            ego_spawn_index,
            args.vehicles,
            args,
            rng,
        )
        actors.walkers, actors.walker_controllers = base._spawn_walkers(
            world, args.walkers, args, rng
        )
        behavior_agent, destination_index = base._create_behavior_agent(
            actors.ego, args, spawn_points, rng
        )
        actors.camera = base._spawn_camera(carla, world, actors.ego, args, camera_stream)

        predictor = create_camera_voxel_predictor(
            args.voxel_predictor_factory,
            CameraVoxelModelConfig(
                checkpoint=args.voxel_predictor_checkpoint,
                device=args.voxel_predictor_device,
                options=args.voxel_predictor_options,
            ),
        )
        voxel_runtime = VoxelActuationRuntime(
            predictor,
            _grid_spec(args),
            planner_config=_planner_config(args),
            supervisor_policy=_supervisor_policy(args),
            readiness_report=_load_readiness_report(args.voxel_readiness_report),
        )
        voxel_runtime.reset()
        actors.ego.apply_control(carla.VehicleControl(brake=1.0))
        voxel_control_calls += 1

        print(
            f"Connected CARLA server={server_version} client={client_version} "
            f"map={base._map_basename(world.get_map().name)}"
        )
        print(
            f"Spawned ego={actors.ego.id} NPC={len(actors.vehicles)} "
            f"walkers={len(actors.walkers)} driver=behavior voxel_actuation=enabled"
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
            speed_mps = base._speed_mps(actors.ego)
            speed_kmh = speed_mps * 3.6

            if behavior_agent.done():
                if not args.loop_destinations:
                    actors.ego.apply_control(carla.VehicleControl(brake=1.0))
                    voxel_control_calls += 1
                    voxel_emergency_brakes += 1
                    stop_reason = "destination_complete"
                    break
                destination, destination_index = base._new_destination(
                    spawn_points, actors.ego, rng, args.destination_index
                )
                behavior_agent.set_destination(destination)

            base_control = behavior_agent.run_step(debug=False)
            try:
                frame = camera_stream.wait_after(previous_camera_frame, args.camera_timeout)
                previous_camera_frame = frame.frame
                rgb = frame.bgr[:, :, ::-1].copy()
                result = voxel_runtime.step(
                    frame=frame.frame,
                    timestamp=frame.timestamp,
                    rgb=rgb,
                    speed_mps=speed_mps,
                    previous_steering=previous_voxel_steering,
                    dt_s=dt,
                )
                last_voxel_result = result.as_dict()
                consecutive_voxel_errors = 0
                if result.decision.emergency_brake or not result.decision.actuation_authorized:
                    control = carla.VehicleControl(brake=1.0)
                    previous_voxel_steering = 0.0
                    voxel_emergency_brakes += 1
                    last_control_reason = f"voxel:emergency-brake:{result.decision.reason}"
                else:
                    base_control.steer = float(result.decision.steering)
                    control = base_control
                    previous_voxel_steering = float(result.decision.steering)
                    last_control_reason = f"voxel:{result.decision.reason}"
            except Exception as error:
                consecutive_voxel_errors += 1
                control = carla.VehicleControl(brake=1.0)
                previous_voxel_steering = 0.0
                voxel_emergency_brakes += 1
                last_control_reason = f"voxel:emergency-brake:{type(error).__name__}"
                print(f"WARNING: voxel actuation failed: {error}; applying full brake", file=sys.stderr)
                if consecutive_voxel_errors >= args.voxel_max_errors:
                    actors.ego.apply_control(control)
                    voxel_control_calls += 1
                    stop_reason = "voxel_error_limit"
                    break

            actors.ego.apply_control(control)
            voxel_control_calls += 1

            if args.spectator_follow:
                base._follow_spectator(carla, world.get_spectator(), actors.ego)

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
        if voxel_runtime is not None:
            try:
                voxel_runtime.close()
            except Exception as error:
                print(f"WARNING: voxel predictor close failed: {error}", file=sys.stderr)
        base._cleanup(
            world,
            traffic_manager,
            actors,
            camera_stream,
            original_settings,
            changed_settings,
            args.keep_actors,
        )

    result = {
        "schema_version": VOXEL_LOCAL_DRIVE_SCHEMA_VERSION,
        "server_version": server_version,
        "client_version": client_version,
        "map": base._map_basename(world.get_map().name),
        "driver": "behavior",
        "voxel_actuation_enabled": True,
        "operator_acknowledged": True,
        "ego_actor_id": int(actors.ego.id) if actors.ego is not None else None,
        "npc_vehicle_count": len(actors.vehicles),
        "walker_count": len(actors.walkers),
        "elapsed_seconds": time.monotonic() - started,
        "stop_reason": stop_reason,
        "voxel_control_calls": voxel_control_calls,
        "voxel_emergency_brakes": voxel_emergency_brakes,
        "last_control_reason": last_control_reason,
        "last_voxel_result": last_voxel_result,
        "actors_kept": bool(args.keep_actors),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.enable_voxel_actuation:
        return base.run(args)
    return _run_voxel(args)


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
