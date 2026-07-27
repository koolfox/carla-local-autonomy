#!/usr/bin/env python3
"""
Spawn a CARLA vehicle through this project's lightweight MessagePack bridge,
then launch the existing model-neutral carla_vision runtime.

Designed for the Apple-silicon client:
- does NOT import the official `carla` Python package;
- uses carla_vision.bridge.CarlaRpc;
- derives safe Town10HD spawn candidates from PurePursuitController.route;
- lets runtime.py create and later destroy the front RGB camera;
- launches RT-DETR/YOLO/custom detector with live viewer and recording;
- optionally enables privileged low-speed teacher motion;
- always attempts to park and destroy only the vehicle created by this wrapper.

Vision-policy proposals remain shadow-only and are never actuated.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from carla_vision.bridge import CarlaError, CarlaRpc
from carla_vision.controller import ControlCommand, PurePursuitController


def parse_resolution(value: str) -> tuple[int, int]:
    parts = value.lower().split("x", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must be WIDTHxHEIGHT")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution values must be integers") from exc
    if width < 320 or height < 180:
        raise argparse.ArgumentTypeError("resolution is too small")
    return width, height


def parse_json_object(value: str) -> dict[str, Any]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Spawn a CARLA vehicle over the lightweight RPC bridge and run "
            "the existing CARLA Vision runtime against it."
        )
    )

    parser.add_argument("--host", default="172.20.10.7")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--rpc-timeout", type=float, default=5.0)
    parser.add_argument("--expected-map", default="Town10HD_Opt")

    parser.add_argument(
        "--vehicle-blueprint",
        default="vehicle.tesla.model3",
        help="Exact CARLA vehicle blueprint ID.",
    )
    parser.add_argument("--role-name", default="hero")
    parser.add_argument(
        "--color",
        help="Optional blueprint color such as '255,0,0'.",
    )

    parser.add_argument(
        "--spawn-x",
        type=float,
        help="Explicit spawn X. Supply X, Y and yaw together.",
    )
    parser.add_argument("--spawn-y", type=float)
    parser.add_argument("--spawn-z", type=float, default=0.8)
    parser.add_argument("--spawn-yaw", type=float)
    parser.add_argument("--spawn-pitch", type=float, default=0.0)
    parser.add_argument("--spawn-roll", type=float, default=0.0)
    parser.add_argument(
        "--route-stride",
        type=int,
        default=20,
        help="Try every Nth teacher-route point when deriving spawn candidates.",
    )
    parser.add_argument(
        "--max-spawn-attempts",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--detector",
        choices=("rtdetr", "yolo", "custom"),
        default="rtdetr",
    )
    parser.add_argument("--weights", default="rtdetr-l.pt")
    parser.add_argument("--detector-factory")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.20)

    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default=(640, 384),
    )
    parser.add_argument("--camera-fps", type=float, default=10.0)
    parser.add_argument("--camera-fov", type=float, default=90.0)
    parser.add_argument(
        "--view",
        choices=("none", "overlay", "split"),
        default="split",
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--cruise-speed", type=float, default=2.0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--no-video", action="store_true")

    parser.add_argument(
        "--shadow-policy",
        choices=("none", "hazard-stop", "custom"),
        default="hazard-stop",
    )
    parser.add_argument("--policy-factory")
    parser.add_argument("--policy-checkpoint")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument(
        "--policy-options",
        type=parse_json_object,
        default={"confidence": 0.35, "close_bottom": 0.72},
    )

    parser.add_argument(
        "--no-teacher",
        action="store_true",
        help="Run perception/shadow only and do not move the vehicle.",
    )
    parser.add_argument(
        "--acknowledge-teacher-motion",
        action="store_true",
        help="Required unless --no-teacher is used.",
    )
    parser.add_argument(
        "--keep-vehicle",
        action="store_true",
        help="Leave the spawned vehicle in CARLA after runtime exits.",
    )
    parser.add_argument(
        "--spawn-only",
        action="store_true",
        help="Spawn and print the vehicle ID without launching the runtime.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print candidate/command information only.",
    )

    args = parser.parse_args()

    explicit = [args.spawn_x, args.spawn_y, args.spawn_yaw]
    if any(value is not None for value in explicit) and not all(
        value is not None for value in explicit
    ):
        parser.error("--spawn-x, --spawn-y and --spawn-yaw must be supplied together")
    if args.route_stride <= 0:
        parser.error("--route-stride must be positive")
    if args.max_spawn_attempts <= 0:
        parser.error("--max-spawn-attempts must be positive")
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.camera_fps <= 0:
        parser.error("--camera-fps must be positive")
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be in [0, 1]")
    if args.detector == "custom" and not args.detector_factory:
        parser.error("--detector custom requires --detector-factory")
    if args.shadow_policy == "custom" and not args.policy_factory:
        parser.error("--shadow-policy custom requires --policy-factory")
    if not args.no_teacher and not args.acknowledge_teacher_motion and not args.dry_run:
        parser.error(
            "teacher mode moves the vehicle; add --acknowledge-teacher-motion "
            "or use --no-teacher"
        )

    return args


def actor_description(
    definition: list[Any],
    overrides: dict[str, str],
) -> list[Any]:
    """Serialize ActorDescription exactly like bridge._actor_description."""

    uid, actor_id, _tags, attributes = definition
    values: list[list[Any]] = []
    available = {str(item[0]) for item in attributes}

    unknown = sorted(set(overrides) - available)
    if unknown:
        print(
            "WARNING: blueprint does not expose attributes: "
            + ", ".join(unknown),
            file=sys.stderr,
        )

    for attr_id, attr_type, value, _recommended, _modifiable, _restricted in attributes:
        values.append(
            [
                attr_id,
                attr_type,
                overrides.get(str(attr_id), value),
            ]
        )
    return [uid, actor_id, values]


def find_definition(
    rpc: CarlaRpc,
    blueprint_id: str,
) -> list[Any]:
    definitions = rpc.value_call("get_actor_definitions")
    definition = next(
        (item for item in definitions if str(item[1]) == blueprint_id),
        None,
    )
    if definition is None:
        available = sorted(
            str(item[1])
            for item in definitions
            if str(item[1]).startswith("vehicle.")
        )
        preview = ", ".join(available[:12])
        raise RuntimeError(
            f"vehicle blueprint {blueprint_id!r} is unavailable. "
            f"First available vehicles: {preview}"
        )
    return definition


def get_xy(value: Any) -> tuple[float, float]:
    """Extract X/Y from common route-point representations."""

    if isinstance(value, dict):
        for x_key, y_key in (("x", "y"), ("X", "Y")):
            if x_key in value and y_key in value:
                return float(value[x_key]), float(value[y_key])
        for key in ("location", "position", "point", "transform"):
            if key in value:
                return get_xy(value[key])

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) >= 2:
            try:
                return float(value[0]), float(value[1])
            except (TypeError, ValueError):
                pass
        if value:
            return get_xy(value[0])

    for x_name, y_name in (("x", "y"), ("X", "Y")):
        if hasattr(value, x_name) and hasattr(value, y_name):
            return float(getattr(value, x_name)), float(getattr(value, y_name))

    for name in ("location", "position", "point", "transform"):
        if hasattr(value, name):
            return get_xy(getattr(value, name))

    raise TypeError(f"cannot extract route X/Y from {value!r}")


def route_spawn_candidates(
    *,
    z: float,
    pitch: float,
    roll: float,
    stride: int,
    limit: int,
) -> list[list[list[float]]]:
    """
    Build vehicle transforms from the existing teacher route.

    Each transform uses a route point for location and the direction to the
    next point for yaw. This keeps the vehicle aligned with the controller's
    supported Town10HD loop.
    """

    controller = PurePursuitController(cruise_speed=2.0)
    route = list(controller.route)
    if len(route) < 2:
        raise RuntimeError("teacher route has fewer than two points")

    indices = list(range(0, len(route), stride))
    if indices[-1] != len(route) - 1:
        indices.append(len(route) - 1)

    candidates: list[list[list[float]]] = []
    seen: set[tuple[int, int, int]] = set()

    for index in indices:
        current = get_xy(route[index])
        next_index = (index + 1) % len(route)
        following = get_xy(route[next_index])

        dx = following[0] - current[0]
        dy = following[1] - current[1]
        if math.hypot(dx, dy) < 1e-4:
            continue

        yaw = math.degrees(math.atan2(dy, dx))
        key = (
            round(current[0] * 100),
            round(current[1] * 100),
            round(yaw * 10),
        )
        if key in seen:
            continue
        seen.add(key)

        candidates.append(
            [
                [float(current[0]), float(current[1]), float(z)],
                [float(pitch), float(yaw), float(roll)],
            ]
        )
        if len(candidates) >= limit:
            break

    if not candidates:
        raise RuntimeError("no usable vehicle transform was derived from the route")
    return candidates


def explicit_spawn_candidate(args: argparse.Namespace) -> list[list[float]] | None:
    if args.spawn_x is None:
        return None
    return [
        [float(args.spawn_x), float(args.spawn_y), float(args.spawn_z)],
        [
            float(args.spawn_pitch),
            float(args.spawn_yaw),
            float(args.spawn_roll),
        ],
    ]


def spawn_vehicle(
    rpc: CarlaRpc,
    *,
    blueprint_id: str,
    role_name: str,
    color: str | None,
    candidates: Iterable[list[list[float]]],
) -> tuple[list[Any], list[list[float]], int]:
    definition = find_definition(rpc, blueprint_id)
    overrides = {"role_name": role_name}
    if color is not None:
        overrides["color"] = color
    description = actor_description(definition, overrides)

    errors: list[str] = []
    for attempt, transform in enumerate(candidates, start=1):
        try:
            actor = rpc.value_call(
                "spawn_actor",
                description,
                transform,
            )
            if not isinstance(actor, list) or not actor:
                raise RuntimeError(f"malformed spawned actor: {actor!r}")
            actor_id = int(actor[0])
            verified = rpc.actor(actor_id)
            if (
                verified is None
                or len(verified) < 3
                or not str(verified[2][1]).startswith("vehicle.")
            ):
                raise RuntimeError(
                    f"spawned actor {actor_id} did not verify as a vehicle"
                )
            return actor, transform, attempt
        except Exception as exc:
            errors.append(f"attempt {attempt}: {exc}")
            print(
                f"Spawn attempt {attempt} failed at {transform}: {exc}",
                file=sys.stderr,
            )

    raise RuntimeError(
        "vehicle could not be spawned at any candidate:\n  "
        + "\n  ".join(errors)
    )


def build_runtime_command(
    args: argparse.Namespace,
    vehicle_id: int,
) -> list[str]:
    width, height = args.resolution

    command = [
        sys.executable,
        "-m",
        "carla_vision.runtime",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--vehicle-id",
        str(vehicle_id),
        # Actor ID zero is intentionally absent. runtime.camera_actor will
        # create a matching front RGB camera and destroy it during finalization.
        "--camera-id",
        "0",
        "--resolution",
        f"{width}x{height}",
        "--camera-fps",
        str(args.camera_fps),
        "--camera-fov",
        str(args.camera_fov),
        "--expected-map",
        args.expected_map,
        "--detector",
        args.detector,
        "--device",
        args.device,
        "--image-size",
        str(args.image_size),
        "--confidence",
        str(args.confidence),
        "--view",
        args.view,
        "--duration",
        str(args.duration),
        "--runs-root",
        args.runs_root,
        "--run-id",
        args.run_id,
    ]

    if args.weights:
        command.extend(["--weights", args.weights])
    if args.detector_factory:
        command.extend(["--detector-factory", args.detector_factory])

    if args.no_teacher:
        command.extend(["--control", "none"])
    else:
        command.extend(
            [
                "--control",
                "teacher",
                "--cruise-speed",
                str(args.cruise_speed),
            ]
        )

    if args.shadow_policy != "none":
        command.extend(
            [
                "--shadow-policy",
                args.shadow_policy,
                "--policy-device",
                args.policy_device,
                "--policy-options",
                json.dumps(
                    args.policy_options,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
            ]
        )
        if args.policy_factory:
            command.extend(["--policy-factory", args.policy_factory])
        if args.policy_checkpoint:
            command.extend(["--policy-checkpoint", args.policy_checkpoint])

    if args.no_video:
        command.append("--no-video")

    return command


def park_vehicle(rpc: CarlaRpc, vehicle_id: int) -> None:
    """Best-effort service brake followed by parked control."""

    deadline = time.monotonic() + 4.0
    try:
        telemetry = rpc.telemetry(vehicle_id)
        while abs(telemetry.speed) >= 0.05 and time.monotonic() < deadline:
            rpc.apply_vehicle_control(
                vehicle_id,
                ControlCommand.service_brake().as_carla(),
            )
            time.sleep(0.08)
            telemetry = rpc.telemetry(vehicle_id)
        rpc.apply_vehicle_control(
            vehicle_id,
            ControlCommand.parked().as_carla(),
        )
    except Exception as exc:
        print(f"WARNING: vehicle parking failed: {exc}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    runs_path = Path(args.runs_root) / args.run_id

    if runs_path.exists() and not args.dry_run and not args.spawn_only:
        print(
            f"ERROR: run output already exists: {runs_path}\n"
            "Use a new --run-id; failed IDs must not be reused.",
            file=sys.stderr,
        )
        return 2

    explicit = explicit_spawn_candidate(args)
    if explicit is not None:
        candidates = [explicit]
    else:
        candidates = route_spawn_candidates(
            z=args.spawn_z,
            pitch=args.spawn_pitch,
            roll=args.spawn_roll,
            stride=args.route_stride,
            limit=args.max_spawn_attempts,
        )

    print(f"Prepared {len(candidates)} spawn candidate(s).")
    print(f"First candidate: {candidates[0]}")

    if args.dry_run:
        example_command = build_runtime_command(args, vehicle_id=999999)
        print("\nDRY RUN — no CARLA mutation performed.")
        print("Runtime command template:")
        print(shlex.join(example_command))
        return 0

    vehicle_id: int | None = None
    child: subprocess.Popen[str] | None = None
    rpc: CarlaRpc | None = None

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal child
        print(f"\nReceived signal {signum}; stopping runtime...", file=sys.stderr)
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        rpc = CarlaRpc(args.host, args.port, timeout=args.rpc_timeout)

        server_version = str(rpc.value_call("version"))
        map_info = rpc.value_call("get_map_info")
        map_name = str(map_info[0])
        actual_map = map_name.rsplit("/", 1)[-1].removesuffix(".umap")
        expected_map = args.expected_map.rsplit("/", 1)[-1].removesuffix(".umap")

        print(f"CARLA server version: {server_version}")
        print(f"CARLA map:            {map_name}")

        if server_version != "0.9.16":
            raise RuntimeError(
                f"expected CARLA 0.9.16, server reports {server_version}"
            )
        if actual_map != expected_map:
            raise RuntimeError(
                f"expected map {expected_map}, got {actual_map}; "
                "the wrapper will not reload the world"
            )

        actor, used_transform, attempt = spawn_vehicle(
            rpc,
            blueprint_id=args.vehicle_blueprint,
            role_name=args.role_name,
            color=args.color,
            candidates=candidates,
        )
        vehicle_id = int(actor[0])

        print("\nVehicle created successfully:")
        print(f"  vehicle_id = {vehicle_id}")
        print(f"  blueprint  = {args.vehicle_blueprint}")
        print(f"  transform  = {used_transform}")
        print(f"  attempt    = {attempt}")

        # Start from a safely parked state. The runtime teacher will take over.
        rpc.apply_vehicle_control(
            vehicle_id,
            ControlCommand.parked().as_carla(),
        )

        if args.spawn_only:
            if args.keep_vehicle:
                print("Vehicle left in CARLA because --keep-vehicle was supplied.")
                vehicle_id = None
                return 0
            input("Press Enter to destroy the vehicle and exit...")
            return 0

        command = build_runtime_command(args, vehicle_id)
        print("\nLaunching runtime:")
        print(shlex.join(command))
        print()

        child = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=os.environ.copy(),
            text=True,
        )
        return_code = child.wait()
        child = None

        if return_code != 0:
            print(
                f"ERROR: CARLA Vision runtime exited with code {return_code}.",
                file=sys.stderr,
            )
        return return_code

    except KeyboardInterrupt:
        print("\nInterrupted by operator.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                child.kill()

        if rpc is not None and vehicle_id is not None:
            park_vehicle(rpc, vehicle_id)
            if args.keep_vehicle:
                print(f"Keeping vehicle actor {vehicle_id} in CARLA.")
            else:
                try:
                    rpc.destroy_actor(vehicle_id)
                    print(f"Destroyed vehicle actor {vehicle_id}.")
                except Exception as exc:
                    print(
                        f"WARNING: could not destroy vehicle {vehicle_id}: {exc}",
                        file=sys.stderr,
                    )

        if rpc is not None:
            rpc.close()


if __name__ == "__main__":
    raise SystemExit(main())
