"""Pilot CARLA RGB dataset collection with privileged instance labels."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..bridge import (
    CarlaCameraStream,
    CarlaRpc,
    spawn_camera,
    vehicle_transform_from_front_camera,
)
from ..controller import ControlCommand, PurePursuitController
from ..runtime import (
    _safe_server_version,
    brake_and_verify_stop,
    camera_actor,
    parse_resolution,
    validate_camera_mount,
)
from ..watchdog import SafeActuator
from .sync import ExactFramePairer
from .writer import DATASET_PARTITIONS, DatasetWriter


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect synchronized CARLA RGB and privileged instance labels "
            "into a checksum-indexed COCO/YOLO dataset"
        )
    )
    parser.add_argument("--host", default="172.20.10.7")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--vehicle-id", type=int, default=24)
    parser.add_argument("--rgb-camera-id", type=int, default=25)
    parser.add_argument("--resolution", type=parse_resolution, default=(1280, 720))
    parser.add_argument("--camera-fps", type=float, default=10.0)
    parser.add_argument("--camera-fov", type=float, default=90.0)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--pair-timeout", type=float, default=8.0)
    parser.add_argument("--minimum-pixels", type=int, default=16)
    parser.add_argument("--minimum-box-width", type=int, default=2)
    parser.add_argument("--minimum-box-height", type=int, default=2)
    parser.add_argument(
        "--control",
        choices=("none", "teacher"),
        default="none",
        help="teacher uses privileged simulator pose and is never a runtime policy",
    )
    parser.add_argument("--cruise-speed", type=float, default=2.0)
    parser.add_argument("--datasets-root", default="datasets")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument(
        "--split",
        choices=tuple(sorted(DATASET_PARTITIONS)),
        default="unassigned",
    )
    parser.add_argument("--scenario-id", default="scn-pilot")
    parser.add_argument("--episode-id", default="ep-pilot-r00")
    args = parser.parse_args(argv)

    if args.samples <= 0 or args.samples > 1_000_000:
        parser.error("--samples must be in [1, 1000000]")
    if args.sample_every <= 0:
        parser.error("--sample-every must be positive")
    if not 0.0 < args.camera_fps <= 60.0:
        parser.error("--camera-fps must be in (0, 60]")
    if not 30.0 <= args.camera_fov <= 150.0:
        parser.error("--camera-fov must be between 30 and 150")
    if args.pair_timeout <= 0.0:
        parser.error("--pair-timeout must be positive")
    for field in (
        "minimum_pixels",
        "minimum_box_width",
        "minimum_box_height",
    ):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if not 0.0 < args.cruise_speed <= 5.0:
        parser.error("--cruise-speed must be in (0, 5]")
    return args


def _map_name(rpc: CarlaRpc) -> str:
    return str(rpc.value_call("get_map_info")[0])


def _safe_destroy(
    rpc: CarlaRpc,
    actor_id: int | None,
    cleanup_errors: list[dict[str, Any]],
) -> None:
    if actor_id is None:
        return
    try:
        rpc.destroy_actor(actor_id)
    except BaseException as error:
        cleanup_errors.append(
            {
                "operation": "destroy_actor",
                "actor_id": actor_id,
                "type": type(error).__qualname__,
                "message": str(error),
            }
        )


def collect(args: argparse.Namespace) -> dict[str, Any]:
    width, height = args.resolution
    camera_tick = 1.0 / args.camera_fps
    with CarlaRpc(args.host, args.port, timeout=3.0) as rpc:
        vehicle = rpc.actor(args.vehicle_id)
        if vehicle is None or not vehicle[2][1].startswith("vehicle."):
            raise RuntimeError(f"vehicle actor {args.vehicle_id} was not found")
        map_name = _map_name(rpc)
        if args.control == "teacher" and "Town10HD" not in map_name:
            raise RuntimeError(
                "the built-in simulator-teacher route currently supports "
                f"Town10HD only, got {map_name}"
            )
        carla_version = _safe_server_version(rpc) or "unknown"
        rgb_actor: list[Any] | None = None
        rgb_spawned = False
        teacher_actor_id: int | None = None
        actuator: SafeActuator | None = None
        cleanup_errors: list[dict[str, Any]] = []
        final_speed: float | None = None
        started = time.monotonic()

        try:
            rgb_actor, rgb_spawned = camera_actor(
                rpc,
                vehicle_id=args.vehicle_id,
                camera_id=args.rgb_camera_id,
                width=width,
                height=height,
                camera_fps=args.camera_fps,
                fov=args.camera_fov,
            )
            teacher_actor = spawn_camera(
                rpc,
                args.vehicle_id,
                "sensor.camera.instance_segmentation",
                role_name="front_instance_teacher",
                width=width,
                height=height,
                sensor_tick=camera_tick,
                fov=args.camera_fov,
            )
            teacher_actor_id = int(teacher_actor[0])
            print(
                f"dataset={args.dataset_id} vehicle={args.vehicle_id} "
                f"rgb={rgb_actor[0]} teacher={teacher_actor_id} map={map_name}",
                flush=True,
            )

            pairer = ExactFramePairer()
            controller = PurePursuitController(cruise_speed=args.cruise_speed)
            if args.control == "teacher":
                actuator = SafeActuator(
                    args.host,
                    args.port,
                    args.vehicle_id,
                    heartbeat_timeout=0.25,
                )

            with (
                CarlaCameraStream(args.host, rgb_actor[5]) as rgb_stream,
                CarlaCameraStream(args.host, teacher_actor[5]) as teacher_stream,
                DatasetWriter(
                    args.datasets_root,
                    dataset_id=args.dataset_id,
                    cli_args=vars(args),
                    carla_endpoint={"host": args.host, "port": args.port},
                    carla_version=carla_version,
                    carla_map=map_name,
                    config={
                        **vars(args),
                        "resolution": [width, height],
                        "rgb_sensor": "sensor.camera.rgb",
                        "teacher_sensor": ("sensor.camera.instance_segmentation"),
                        "synchronization": "exact_carla_frame",
                        "teacher_uses_privileged_pose": (args.control == "teacher"),
                    },
                    repository_root=Path.cwd(),
                    minimum_pixels=args.minimum_pixels,
                    minimum_box_width=args.minimum_box_width,
                    minimum_box_height=args.minimum_box_height,
                ) as writer,
            ):
                pair_count = 0
                retained = 0
                annotation_count = 0
                route_progress = "-"
                stop_reason = "sample_target_reached"
                previous_control_time = time.monotonic()

                while retained < args.samples:
                    pair = pairer.next_pair(
                        rgb_stream,
                        teacher_stream,
                        timeout=args.pair_timeout,
                    )
                    pair_count += 1
                    if args.control == "teacher":
                        if actuator is None or not actuator.alive:
                            raise RuntimeError("safe actuator stopped unexpectedly")
                        if pair_count == 1:
                            validate_camera_mount(
                                rpc,
                                args.vehicle_id,
                                pair.rgb,
                            )
                        now = time.monotonic()
                        telemetry = rpc.telemetry(args.vehicle_id)
                        state = controller.compute(
                            vehicle_transform_from_front_camera(pair.rgb),
                            max(0.0, abs(telemetry.speed)),
                            now - previous_control_time,
                        )
                        route_progress = f"{state.nearest_index}/{len(controller.route) - 1}"
                        command = (
                            state.command
                            if state.valid and not state.done
                            else ControlCommand.service_brake(steer=state.command.steer)
                        )
                        actuator.send(command)
                        previous_control_time = now
                        if state.done:
                            stop_reason = "route_complete"
                            break

                    if (pair_count - 1) % args.sample_every != 0:
                        continue
                    sample = writer.add_pair(
                        pair,
                        split=args.split,
                        scenario_id=args.scenario_id,
                        episode_id=args.episode_id,
                        context={
                            "map": map_name,
                            "control_mode": args.control,
                            "route_progress": route_progress,
                            "rgb_camera_actor_id": int(rgb_actor[0]),
                            "teacher_camera_actor_id": teacher_actor_id,
                        },
                    )
                    retained += 1
                    annotation_count += sample.annotation_count
                    print(
                        f"sample={sample.sample_id} frame={pair.carla_frame} "
                        f"annotations={sample.annotation_count} "
                        f"retained={retained}/{args.samples}",
                        flush=True,
                    )

                writer.set_release_metadata(
                    {
                        "requested_samples": args.samples,
                        "retained_samples": retained,
                        "observed_exact_pairs": pair_count,
                        "annotation_count": annotation_count,
                        "rgb_frames_skipped_during_pairing": (pairer.total_rgb_skipped),
                        "teacher_frames_skipped_during_pairing": (pairer.total_teacher_skipped),
                        "control_mode": args.control,
                        "stop_reason": stop_reason,
                        "elapsed_wall_seconds": time.monotonic() - started,
                        "actors": {
                            "vehicle": args.vehicle_id,
                            "rgb_camera": int(rgb_actor[0]),
                            "teacher_camera": teacher_actor_id,
                        },
                    }
                )
                if retained != args.samples:
                    raise RuntimeError(
                        f"retained {retained} of {args.samples} samples before {stop_reason}"
                    )

            result = {
                "dataset_id": args.dataset_id,
                "dataset_dir": str((Path(args.datasets_root) / args.dataset_id).resolve()),
                "samples": retained,
                "annotations": annotation_count,
                "exact_pairs_observed": pair_count,
                "rgb_pairing_skips": pairer.total_rgb_skipped,
                "teacher_pairing_skips": pairer.total_teacher_skipped,
                "carla_version": carla_version,
                "map": map_name,
                "control_mode": args.control,
                "stop_reason": stop_reason,
            }
        finally:
            if actuator is not None:
                try:
                    actuator.stop()
                except BaseException as error:
                    cleanup_errors.append(
                        {
                            "operation": "stop_actuator",
                            "type": type(error).__qualname__,
                            "message": str(error),
                        }
                    )
            if args.control == "teacher":
                try:
                    telemetry = brake_and_verify_stop(
                        rpc,
                        args.vehicle_id,
                    )
                    final_speed = float(telemetry.speed)
                except BaseException as error:
                    cleanup_errors.append(
                        {
                            "operation": "brake_and_verify_stop",
                            "type": type(error).__qualname__,
                            "message": str(error),
                        }
                    )
            _safe_destroy(rpc, teacher_actor_id, cleanup_errors)
            if rgb_spawned and rgb_actor is not None:
                _safe_destroy(rpc, int(rgb_actor[0]), cleanup_errors)

        if cleanup_errors:
            raise RuntimeError(
                "dataset collection cleanup failed: "
                + "; ".join(item["message"] for item in cleanup_errors)
            )
        result["final_simulator_speed_mps"] = final_speed
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    collect(args)
    return 0


__all__ = ["collect", "main", "parse_args"]
