"""Live CARLA camera-to-voxel test harness and dataset recorder."""

from __future__ import annotations

import argparse
import importlib
import json
import queue
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .contracts import VoxelGridSpec, validate_camera_voxel_prediction
from .geometry import (
    decode_carla_depth_bgra,
    occupancy_bev,
    occupied_iou,
    raycast_voxel_grid,
    rgb_from_bgra,
    semantic_tags_from_bgra,
)
from .models import CameraVoxelModelConfig, create_camera_voxel_predictor

VOXEL_CAPTURE_SCHEMA_VERSION = "1.0"


def _load_carla_module() -> Any:
    try:
        return importlib.import_module("carla")
    except ImportError as error:
        raise RuntimeError(
            "carla-voxel-test requires the version-matched CARLA PythonAPI for a live run"
        ) from error


def _parse_json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must decode to a JSON object")
    return parsed


def _parse_horizons(value: str) -> tuple[float, ...]:
    try:
        horizons = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("forecast horizons must be comma-separated numbers") from error
    if not horizons or any(item < 0 for item in horizons):
        raise argparse.ArgumentTypeError("forecast horizons must be non-negative")
    if tuple(sorted(horizons)) != horizons or len(set(horizons)) != len(horizons):
        raise argparse.ArgumentTypeError("forecast horizons must be unique and increasing")
    return horizons


def _transform_dict(transform: Any) -> dict[str, float]:
    return {
        "x": float(transform.location.x),
        "y": float(transform.location.y),
        "z": float(transform.location.z),
        "pitch": float(transform.rotation.pitch),
        "yaw": float(transform.rotation.yaw),
        "roll": float(transform.rotation.roll),
    }


def _grid_spec_from_args(args: argparse.Namespace) -> VoxelGridSpec:
    return VoxelGridSpec(
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        z_min=args.z_min,
        z_max=args.z_max,
        resolution=args.voxel_resolution,
    )


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
        raise RuntimeError(
            f"no vehicle with role_name={role_name!r}; run carla-local-drive or pass --actor-id"
        )
    return sorted(matches, key=lambda actor: int(actor.id))[0]


def _camera_blueprint(
    world: Any,
    sensor_id: str,
    *,
    width: int,
    height: int,
    fov_deg: float,
) -> Any:
    blueprint = world.get_blueprint_library().find(sensor_id)
    blueprint.set_attribute("image_size_x", str(width))
    blueprint.set_attribute("image_size_y", str(height))
    blueprint.set_attribute("fov", str(fov_deg))
    blueprint.set_attribute("sensor_tick", "0.0")
    return blueprint


def _spawn_camera(
    world: Any,
    carla: Any,
    vehicle: Any,
    sensor_id: str,
    args: argparse.Namespace,
) -> Any:
    blueprint = _camera_blueprint(
        world,
        sensor_id,
        width=args.width,
        height=args.height,
        fov_deg=args.fov,
    )
    transform = carla.Transform(
        carla.Location(x=args.camera_x, y=args.camera_y, z=args.camera_z),
        carla.Rotation(pitch=args.camera_pitch, yaw=args.camera_yaw, roll=args.camera_roll),
    )
    return world.spawn_actor(
        blueprint,
        transform,
        attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid,
    )


def _next_synchronized_packets(
    sensor_queues: Sequence[queue.Queue[Any]],
    buffers: Sequence[dict[int, Any]],
    *,
    timeout_s: float,
) -> tuple[Any, ...]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for index, sensor_queue in enumerate(sensor_queues):
            remaining = max(0.001, deadline - time.monotonic())
            try:
                packet = sensor_queue.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            buffers[index][int(packet.frame)] = packet
        common = set(buffers[0])
        for buffer in buffers[1:]:
            common.intersection_update(buffer)
        if common:
            frame = min(common)
            packets = tuple(buffer.pop(frame) for buffer in buffers)
            for buffer in buffers:
                stale = [item for item in buffer if item < frame]
                for item in stale:
                    buffer.pop(item, None)
            return packets
    raise TimeoutError("timed out waiting for synchronized CARLA camera frames")


def _ensure_output_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"voxel output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    for child in ("rgb", "teacher_voxels", "predicted_voxels", "bev", "metadata"):
        (root / child).mkdir(exist_ok=True)
    return root


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _future_target_map(
    records: Sequence[dict[str, Any]],
    horizons_s: Sequence[float],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        targets: dict[str, int | None] = {}
        timestamp = float(record["timestamp"])
        for horizon in horizons_s:
            target_frame: int | None = None
            for future in records[index:]:
                if float(future["timestamp"]) + 1e-9 >= timestamp + horizon:
                    target_frame = int(future["frame"])
                    break
            targets[f"{horizon:.3f}"] = target_frame
        result.append({"frame": int(record["frame"]), "future_targets": targets})
    return result


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write image {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and evaluate ego-centric voxels from CARLA camera streams."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--actor-id", type=int)
    parser.add_argument("--role-name", default="hero")
    parser.add_argument("--mode", choices=("teacher", "rgb-only"), default="teacher")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("runs/voxel-test"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--camera-x", type=float, default=1.5)
    parser.add_argument("--camera-y", type=float, default=0.0)
    parser.add_argument("--camera-z", type=float, default=1.7)
    parser.add_argument("--camera-pitch", type=float, default=-5.0)
    parser.add_argument("--camera-yaw", type=float, default=0.0)
    parser.add_argument("--camera-roll", type=float, default=0.0)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=50.0)
    parser.add_argument("--y-min", type=float, default=-25.0)
    parser.add_argument("--y-max", type=float, default=25.0)
    parser.add_argument("--z-min", type=float, default=-2.0)
    parser.add_argument("--z-max", type=float, default=5.0)
    parser.add_argument("--voxel-resolution", type=float, default=0.5)
    parser.add_argument("--pixel-stride", type=int, default=8)
    parser.add_argument("--ray-step-m", type=float)
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument(
        "--forecast-horizons",
        type=_parse_horizons,
        default=(0.0, 0.5, 1.0, 2.0),
    )
    parser.add_argument("--predictor-factory")
    parser.add_argument("--predictor-checkpoint", type=Path)
    parser.add_argument("--predictor-device", default="cpu")
    parser.add_argument("--predictor-options", type=_parse_json_object, default={})
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validated_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.port <= 0 or args.frames <= 0:
        raise ValueError("port and frames must be positive")
    if args.timeout <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("timeout and camera dimensions must be positive")
    if args.pixel_stride <= 0 or args.history_frames <= 0:
        raise ValueError("pixel stride and history frames must be positive")
    if not 0.0 < args.prediction_threshold < 1.0:
        raise ValueError("prediction threshold must be in (0, 1)")
    if args.mode == "rgb-only" and not args.predictor_factory:
        raise ValueError("--mode rgb-only requires --predictor-factory")
    _grid_spec_from_args(args)
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    args = _validated_args(args)
    spec = _grid_spec_from_args(args)
    resolved = {
        "schema_version": VOXEL_CAPTURE_SCHEMA_VERSION,
        "mode": args.mode,
        "host": args.host,
        "port": args.port,
        "frames": args.frames,
        "camera": {"width": args.width, "height": args.height, "fov_deg": args.fov},
        "grid": spec.as_dict(),
        "pixel_stride": args.pixel_stride,
        "forecast_horizons_s": list(args.forecast_horizons),
        "predictor_factory": args.predictor_factory,
        "privileged_teacher_sensors": args.mode == "teacher",
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return resolved

    predictor = None
    if args.predictor_factory:
        predictor = create_camera_voxel_predictor(
            args.predictor_factory,
            CameraVoxelModelConfig(
                checkpoint=args.predictor_checkpoint,
                device=args.predictor_device,
                options=args.predictor_options,
            ),
        )

    carla = _load_carla_module()
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    vehicle = _find_vehicle(world, actor_id=args.actor_id, role_name=args.role_name)
    output_root = _ensure_output_root(args.output)
    sensor_ids = ["sensor.camera.rgb"]
    if args.mode == "teacher":
        sensor_ids.extend(("sensor.camera.depth", "sensor.camera.semantic_segmentation"))
    sensors: list[Any] = []
    queues: list[queue.Queue[Any]] = []
    records: list[dict[str, Any]] = []
    rgb_history: deque[np.ndarray] = deque(maxlen=args.history_frames)
    iou_values: list[float] = []
    try:
        for sensor_id in sensor_ids:
            sensor = _spawn_camera(world, carla, vehicle, sensor_id, args)
            sensor_queue: queue.Queue[Any] = queue.Queue(maxsize=8)

            def callback(image: Any, target: queue.Queue[Any] = sensor_queue) -> None:
                try:
                    target.put_nowait(image)
                except queue.Full:
                    try:
                        target.get_nowait()
                    except queue.Empty:
                        pass
                    target.put_nowait(image)

            sensor.listen(callback)
            sensors.append(sensor)
            queues.append(sensor_queue)
        buffers = [dict() for _ in queues]
        while len(records) < args.frames:
            packets = _next_synchronized_packets(queues, buffers, timeout_s=args.timeout)
            rgb_packet = packets[0]
            frame = int(rgb_packet.frame)
            timestamp = float(rgb_packet.timestamp)
            rgb = rgb_from_bgra(rgb_packet.raw_data, args.width, args.height)
            rgb_history.append(rgb.copy())
            stem = f"{frame:08d}"
            _save_rgb(output_root / "rgb" / f"{stem}.png", rgb)
            record: dict[str, Any] = {
                "frame": frame,
                "timestamp": timestamp,
                "rgb": f"rgb/{stem}.png",
                "sensor_transform": _transform_dict(rgb_packet.transform),
                "teacher_voxel": None,
                "prediction": None,
                "occupied_iou": None,
            }
            teacher_occupancy: np.ndarray | None = None
            if args.mode == "teacher":
                depth_packet, semantic_packet = packets[1], packets[2]
                depth = decode_carla_depth_bgra(
                    depth_packet.raw_data,
                    args.width,
                    args.height,
                )
                semantic = semantic_tags_from_bgra(
                    semantic_packet.raw_data,
                    args.width,
                    args.height,
                )
                teacher_occupancy, teacher_semantics = raycast_voxel_grid(
                    depth,
                    semantic,
                    spec=spec,
                    fov_deg=args.fov,
                    pixel_stride=args.pixel_stride,
                    ray_step_m=args.ray_step_m,
                )
                teacher_path = output_root / "teacher_voxels" / f"{stem}.npz"
                np.savez_compressed(
                    teacher_path,
                    occupancy=teacher_occupancy,
                    semantics=teacher_semantics,
                )
                bev_path = output_root / "bev" / f"{stem}.png"
                if not cv2.imwrite(str(bev_path), occupancy_bev(teacher_occupancy)):
                    raise RuntimeError(f"failed to write BEV image {bev_path}")
                record["teacher_voxel"] = f"teacher_voxels/{stem}.npz"
                record["bev"] = f"bev/{stem}.png"
            if predictor is not None and len(rgb_history) == args.history_frames:
                prediction = validate_camera_voxel_prediction(
                    predictor.predict(tuple(rgb_history), spec),
                    spec,
                )
                prediction_path = output_root / "predicted_voxels" / f"{stem}.npz"
                np.savez_compressed(
                    prediction_path,
                    occupancy_probability=prediction.occupancy_probability,
                    horizons_s=np.asarray(prediction.horizons_s, dtype=np.float32),
                    semantic_logits=prediction.semantic_logits,
                )
                record["prediction"] = f"predicted_voxels/{stem}.npz"
                if teacher_occupancy is not None:
                    score = occupied_iou(
                        prediction.occupancy_probability[0],
                        teacher_occupancy,
                        threshold=args.prediction_threshold,
                    )
                    iou_values.append(score)
                    record["occupied_iou"] = score
            _write_json(output_root / "metadata" / f"{stem}.json", record)
            records.append(record)
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

    sequence = {
        "schema_version": VOXEL_CAPTURE_SCHEMA_VERSION,
        "grid": spec.as_dict(),
        "records": records,
        "future_targets": _future_target_map(records, args.forecast_horizons),
    }
    _write_json(output_root / "sequence.json", sequence)
    summary = {
        **resolved,
        "output": str(output_root),
        "vehicle_actor_id": int(vehicle.id),
        "captured_frames": len(records),
        "prediction_frames": sum(record["prediction"] is not None for record in records),
        "mean_current_occupied_iou": (
            float(np.mean(iou_values)) if iou_values else None
        ),
        "teacher_note": (
            "depth and semantic cameras are privileged label/evaluation sources only"
            if args.mode == "teacher"
            else None
        ),
    }
    _write_json(output_root / "manifest.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
