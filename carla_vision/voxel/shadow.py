"""Read-only voxel-planner shadow process for a running CARLA hero vehicle."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import VoxelGridSpec
from .models import CameraVoxelModelConfig, create_camera_voxel_predictor
from .planner import generate_constant_curvature_trajectories, select_best_trajectory
from .planning_occupancy import build_planning_occupancy_evidence

SHADOW_SCHEMA_VERSION = "1.0"


def _parse_json(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return parsed


def _parse_resolution(value: str) -> tuple[int, int]:
    parts = value.lower().split("x", maxsplit=1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must use WIDTHxHEIGHT")
    try:
        width, height = (int(item) for item in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("resolution values must be integers") from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution must be positive")
    return width, height


def _parse_steering(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "steering values must be comma-separated numbers"
        ) from error
    if not values or any(not math.isfinite(item) or not -1.0 <= item <= 1.0 for item in values):
        raise argparse.ArgumentTypeError("steering values must be finite and in [-1, 1]")
    return values


@dataclass(frozen=True, slots=True)
class ShadowPlannerConfig:
    history_frames: int = 4
    steering_values: tuple[float, ...] = (-0.6, -0.3, 0.0, 0.3, 0.6)
    trajectory_steps: int = 12
    trajectory_dt_s: float = 0.25
    wheelbase_m: float = 2.8
    ego_radius_m: float = 1.2
    collision_weight: float = 100.0
    unknown_weight: float = 4.0
    curvature_weight: float = 1.0
    progress_weight: float = 1.0
    uncertainty_band: float = 0.08

    def __post_init__(self) -> None:
        if self.history_frames <= 0 or self.trajectory_steps <= 0:
            raise ValueError("history_frames and trajectory_steps must be positive")
        if self.trajectory_dt_s <= 0 or self.wheelbase_m <= 0 or self.ego_radius_m < 0:
            raise ValueError("trajectory geometry values are invalid")
        if not 0.0 <= self.uncertainty_band < 0.5:
            raise ValueError("uncertainty_band must be in [0, 0.5)")


class VoxelPlannerShadow:
    """Pure observer: predicts and scores paths but has no vehicle-control API."""

    def __init__(self, predictor: Any, spec: VoxelGridSpec, config: ShadowPlannerConfig) -> None:
        self.predictor = predictor
        self.spec = spec
        self.config = config
        self.history: deque[np.ndarray] = deque(maxlen=config.history_frames)

    def observe(
        self,
        *,
        frame: int,
        timestamp: float,
        rgb: np.ndarray,
        speed_mps: float,
    ) -> dict[str, Any] | None:
        image = np.asarray(rgb)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("shadow RGB frame must be uint8 HxWx3")
        if not math.isfinite(speed_mps) or speed_mps < 0:
            raise ValueError("speed_mps must be finite and non-negative")
        self.history.append(np.ascontiguousarray(image.copy()))
        if len(self.history) < self.config.history_frames:
            return None
        started = time.perf_counter()
        evidence = build_planning_occupancy_evidence(
            self.predictor.predict(tuple(self.history), self.spec),
            self.spec,
            uncertainty_band=self.config.uncertainty_band,
        )
        candidates = generate_constant_curvature_trajectories(
            self.config.steering_values,
            speed_mps=speed_mps,
            wheelbase_m=self.config.wheelbase_m,
            dt_s=self.config.trajectory_dt_s,
            steps=self.config.trajectory_steps,
        )
        best, scores = select_best_trajectory(
            candidates,
            evidence.planning_grid,
            spec=self.spec,
            ego_radius_m=self.config.ego_radius_m,
            collision_weight=self.config.collision_weight,
            unknown_weight=self.config.unknown_weight,
            curvature_weight=self.config.curvature_weight,
            progress_weight=self.config.progress_weight,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        best_index = next(
            index for index, candidate in enumerate(candidates) if candidate is best
        )
        return {
            "schema_version": SHADOW_SCHEMA_VERSION,
            "frame": int(frame),
            "timestamp": float(timestamp),
            "speed_mps": float(speed_mps),
            "prediction_latency_ms": latency_ms,
            "horizons_s": list(evidence.horizons_s),
            "selected_steering": float(best.steering),
            "selected_score": float(scores[best_index].total),
            "candidates": [
                {
                    "steering": float(candidate.steering),
                    **asdict(score),
                }
                for candidate, score in zip(candidates, scores, strict=True)
            ],
            "uncertain_voxel_fraction": evidence.unknown_fraction,
            "actuation_applied": False,
        }


@dataclass(frozen=True, slots=True)
class CameraPacket:
    frame: int
    timestamp: float
    rgb: np.ndarray


class LatestCamera:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: CameraPacket | None = None
        self._closed = False

    def callback(self, image: Any) -> None:
        bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            (image.height, image.width, 4)
        )
        rgb = bgra[:, :, :3][:, :, ::-1].copy()
        packet = CameraPacket(int(image.frame), float(image.timestamp), rgb)
        with self._condition:
            if not self._closed:
                self._latest = packet
                self._condition.notify_all()

    def wait_after(self, frame: int, timeout: float) -> CameraPacket:
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._closed:
                if self._latest is not None and self._latest.frame > frame:
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("voxel shadow camera timeout")
                self._condition.wait(remaining)
        raise RuntimeError("camera stream is closed")

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


def _load_carla() -> Any:
    try:
        return importlib.import_module("carla")
    except ImportError as error:
        raise RuntimeError("voxel shadow requires the matching CARLA PythonAPI") from error


def _find_vehicle(world: Any, actor_id: int | None, role_name: str) -> Any:
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
    return min(matches, key=lambda actor: int(actor.id))


def _spawn_camera(
    carla: Any,
    world: Any,
    vehicle: Any,
    args: argparse.Namespace,
    stream: LatestCamera,
) -> Any:
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
    camera = world.spawn_actor(blueprint, transform, attach_to=vehicle)
    camera.listen(stream.callback)
    return camera


def _speed_mps(vehicle: Any) -> float:
    velocity = vehicle.get_velocity()
    return math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2)


def _ensure_output(path: Path) -> Path:
    root = path.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"voxel shadow output is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _grid_from_args(args: argparse.Namespace) -> VoxelGridSpec:
    return VoxelGridSpec(
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        z_min=args.z_min,
        z_max=args.z_max,
        resolution=args.voxel_resolution,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a read-only voxel planner beside a CARLA hero vehicle."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--actor-id", type=int)
    parser.add_argument("--role-name", default="hero")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("runs/voxel-planner-shadow"))
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--camera-resolution", type=_parse_resolution, default=(640, 384))
    parser.add_argument("--camera-fov", type=float, default=90.0)
    parser.add_argument("--camera-x", type=float, default=1.5)
    parser.add_argument("--camera-z", type=float, default=1.7)
    parser.add_argument("--camera-pitch", type=float, default=0.0)
    parser.add_argument("--predictor-factory")
    parser.add_argument("--predictor-checkpoint", type=Path)
    parser.add_argument("--predictor-device", default="cpu")
    parser.add_argument("--predictor-options", type=_parse_json, default={})
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument(
        "--steering-values",
        type=_parse_steering,
        default=(-0.6, -0.3, 0.0, 0.3, 0.6),
    )
    parser.add_argument("--trajectory-steps", type=int, default=12)
    parser.add_argument("--trajectory-dt", type=float, default=0.25)
    parser.add_argument("--wheelbase", type=float, default=2.8)
    parser.add_argument("--ego-radius", type=float, default=1.2)
    parser.add_argument("--collision-weight", type=float, default=100.0)
    parser.add_argument("--unknown-weight", type=float, default=4.0)
    parser.add_argument("--curvature-weight", type=float, default=1.0)
    parser.add_argument("--progress-weight", type=float, default=1.0)
    parser.add_argument("--uncertainty-band", type=float, default=0.08)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=50.0)
    parser.add_argument("--y-min", type=float, default=-25.0)
    parser.add_argument("--y-max", type=float, default=25.0)
    parser.add_argument("--z-min", type=float, default=-2.0)
    parser.add_argument("--z-max", type=float, default=5.0)
    parser.add_argument("--voxel-resolution", type=float, default=0.5)
    parser.add_argument("--status-every", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    if args.timeout <= 0 or args.fps <= 0 or args.frames <= 0:
        raise ValueError("timeout, fps, and frames must be positive")
    if args.duration < 0 or args.status_every <= 0:
        raise ValueError("duration must be non-negative and status_every positive")
    if not args.dry_run and not args.predictor_factory:
        raise ValueError("live shadow requires --predictor-factory")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    spec = _grid_from_args(args)
    config = ShadowPlannerConfig(
        history_frames=args.history_frames,
        steering_values=args.steering_values,
        trajectory_steps=args.trajectory_steps,
        trajectory_dt_s=args.trajectory_dt,
        wheelbase_m=args.wheelbase,
        ego_radius_m=args.ego_radius,
        collision_weight=args.collision_weight,
        unknown_weight=args.unknown_weight,
        curvature_weight=args.curvature_weight,
        progress_weight=args.progress_weight,
        uncertainty_band=args.uncertainty_band,
    )
    resolved = {
        "schema_version": SHADOW_SCHEMA_VERSION,
        "actuation_enabled": False,
        "control_calls": 0,
        "host": args.host,
        "port": args.port,
        "role_name": args.role_name,
        "actor_id": args.actor_id,
        "predictor_factory": args.predictor_factory,
        "grid": spec.as_dict(),
        "planner": asdict(config),
    }
    if args.dry_run:
        print(json.dumps({**resolved, "dry_run": True}, indent=2, sort_keys=True))
        return {**resolved, "dry_run": True}

    predictor = create_camera_voxel_predictor(
        args.predictor_factory,
        CameraVoxelModelConfig(
            checkpoint=args.predictor_checkpoint,
            device=args.predictor_device,
            options=args.predictor_options,
        ),
    )
    planner = VoxelPlannerShadow(predictor, spec, config)
    output = _ensure_output(args.output)
    _write_json(output / "manifest.json", resolved)
    records_path = output / "records.jsonl"
    carla = _load_carla()
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    vehicle = _find_vehicle(world, args.actor_id, args.role_name)
    stream = LatestCamera()
    camera = _spawn_camera(carla, world, vehicle, args, stream)
    previous_frame = -1
    started = time.monotonic()
    status_at = started
    observed_frames = 0
    record_count = 0
    error_count = 0
    latest_record: dict[str, Any] | None = None
    try:
        with records_path.open("a", encoding="utf-8") as handle:
            while observed_frames < args.frames:
                elapsed = time.monotonic() - started
                if args.duration > 0 and elapsed >= args.duration:
                    break
                packet = stream.wait_after(previous_frame, args.timeout)
                previous_frame = packet.frame
                observed_frames += 1
                try:
                    record = planner.observe(
                        frame=packet.frame,
                        timestamp=packet.timestamp,
                        rgb=packet.rgb,
                        speed_mps=_speed_mps(vehicle),
                    )
                    if record is None:
                        continue
                    control = vehicle.get_control()
                    location = vehicle.get_location()
                    record["observed_vehicle_control"] = {
                        "throttle": float(control.throttle),
                        "steer": float(control.steer),
                        "brake": float(control.brake),
                    }
                    record["vehicle_location"] = {
                        "x": float(location.x),
                        "y": float(location.y),
                        "z": float(location.z),
                    }
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                    record_count += 1
                    latest_record = record
                except Exception as error:
                    error_count += 1
                    error_record = {
                        "schema_version": SHADOW_SCHEMA_VERSION,
                        "frame": packet.frame,
                        "timestamp": packet.timestamp,
                        "error": f"{type(error).__name__}: {error}",
                        "actuation_applied": False,
                    }
                    handle.write(json.dumps(error_record, sort_keys=True) + "\n")
                    handle.flush()
                now = time.monotonic()
                if now >= status_at:
                    selected = None if latest_record is None else latest_record["selected_steering"]
                    print(
                        f"shadow observed={observed_frames} records={record_count} "
                        f"errors={error_count} "
                        f"selected_steer={selected} actuation=false",
                        flush=True,
                    )
                    status_at = now + args.status_every
    finally:
        stream.close()
        try:
            camera.stop()
        except RuntimeError:
            pass
        camera.destroy()

    summary = {
        **resolved,
        "dry_run": False,
        "vehicle_actor_id": int(vehicle.id),
        "observed_frames": observed_frames,
        "records": record_count,
        "errors": error_count,
        "elapsed_seconds": time.monotonic() - started,
        "output": str(output),
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(build_parser().parse_args(argv))
    except Exception as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
