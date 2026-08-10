"""Capture RGB occupancy episodes with privileged dynamic voxel-flow labels."""

from __future__ import annotations

import argparse
import json
import queue
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from .capture import (
    VOXEL_CAPTURE_SCHEMA_VERSION,
    _ensure_output_root,
    _find_vehicle,
    _future_target_map,
    _grid_spec_from_args,
    _load_carla_module,
    _next_synchronized_packets,
    _save_rgb,
    _spawn_camera,
    _transform_dict,
    _validated_args,
    _write_json,
)
from .capture import build_parser as build_base_parser
from .flow_teacher import build_dynamic_voxel_flow, decode_carla_optical_flow
from .geometry import (
    decode_carla_depth_bgra,
    occupancy_bev,
    raycast_voxel_grid,
    rgb_from_bgra,
    semantic_tags_from_bgra,
)

FLOW_CAPTURE_SCHEMA_VERSION = "1.0"


def _transform_matrix(transform: Any) -> np.ndarray:
    if not hasattr(transform, "get_matrix"):
        raise RuntimeError("CARLA transform.get_matrix() is required for voxel-flow labels")
    matrix = np.asarray(transform.get_matrix(), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise RuntimeError("CARLA returned an invalid camera transform matrix")
    return matrix


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser()
    parser.description = "Capture privileged CARLA occupancy plus dynamic voxel-flow teacher labels."
    parser.set_defaults(mode="teacher")
    parser.add_argument("--flow-pixel-stride", type=int, default=4)
    parser.add_argument("--max-flow-speed-mps", type=float, default=60.0)
    return parser


def _validate_args(args: argparse.Namespace) -> argparse.Namespace:
    _validated_args(args)
    if args.mode != "teacher":
        raise ValueError("voxel-flow capture requires --mode teacher")
    if args.predictor_factory:
        raise ValueError("voxel-flow capture is teacher-only; evaluate checkpoints separately")
    if args.flow_pixel_stride <= 0:
        raise ValueError("flow-pixel-stride must be positive")
    if not np.isfinite(args.max_flow_speed_mps) or args.max_flow_speed_mps <= 0:
        raise ValueError("max-flow-speed-mps must be finite and positive")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    args = _validate_args(args)
    spec = _grid_spec_from_args(args)
    resolved = {
        "schema_version": VOXEL_CAPTURE_SCHEMA_VERSION,
        "flow_schema_version": FLOW_CAPTURE_SCHEMA_VERSION,
        "mode": "teacher-flow",
        "host": args.host,
        "port": args.port,
        "frames": args.frames,
        "camera": {"width": args.width, "height": args.height, "fov_deg": args.fov},
        "grid": spec.as_dict(),
        "pixel_stride": args.pixel_stride,
        "flow_pixel_stride": args.flow_pixel_stride,
        "max_flow_speed_mps": args.max_flow_speed_mps,
        "forecast_horizons_s": list(args.forecast_horizons),
        "privileged_teacher_sensors": ["depth", "semantic_segmentation", "optical_flow"],
        "rgb_only_inference": True,
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return resolved

    carla = _load_carla_module()
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    vehicle = _find_vehicle(world, actor_id=args.actor_id, role_name=args.role_name)
    output_root = _ensure_output_root(args.output)
    flow_root = output_root / "teacher_flow"
    flow_root.mkdir(exist_ok=True)

    sensor_ids = [
        "sensor.camera.rgb",
        "sensor.camera.depth",
        "sensor.camera.semantic_segmentation",
        "sensor.camera.optical_flow",
    ]
    sensors: list[Any] = []
    queues: list[queue.Queue[Any]] = []
    records: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    flow_valid_voxels = 0
    flow_labeled_frames = 0
    flow_direction_counts = {"1": 0, "-1": 0}
    static_residuals: list[float] = []

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
            rgb_packet, depth_packet, semantic_packet, optical_packet = _next_synchronized_packets(
                queues, buffers, timeout_s=args.timeout
            )
            frame = int(rgb_packet.frame)
            timestamp = float(rgb_packet.timestamp)
            stem = f"{frame:08d}"
            rgb = rgb_from_bgra(rgb_packet.raw_data, args.width, args.height)
            depth = decode_carla_depth_bgra(depth_packet.raw_data, args.width, args.height)
            semantics = semantic_tags_from_bgra(
                semantic_packet.raw_data, args.width, args.height
            )
            optical = decode_carla_optical_flow(
                optical_packet.raw_data, args.width, args.height
            )
            _save_rgb(output_root / "rgb" / f"{stem}.png", rgb)
            occupancy, teacher_semantics = raycast_voxel_grid(
                depth,
                semantics,
                spec=spec,
                fov_deg=args.fov,
                pixel_stride=args.pixel_stride,
                ray_step_m=args.ray_step_m,
            )
            teacher_path = output_root / "teacher_voxels" / f"{stem}.npz"
            np.savez_compressed(
                teacher_path,
                occupancy=occupancy,
                semantics=teacher_semantics,
            )
            bev_path = output_root / "bev" / f"{stem}.png"
            if not cv2.imwrite(str(bev_path), occupancy_bev(occupancy)):
                raise RuntimeError(f"failed to write BEV image {bev_path}")

            record: dict[str, Any] = {
                "frame": frame,
                "timestamp": timestamp,
                "rgb": f"rgb/{stem}.png",
                "sensor_transform": _transform_dict(rgb_packet.transform),
                "teacher_voxel": f"teacher_voxels/{stem}.npz",
                "teacher_flow": None,
                "teacher_flow_stats": None,
                "prediction": None,
                "occupied_iou": None,
                "bev": f"bev/{stem}.png",
            }
            records.append(record)
            _write_json(output_root / "metadata" / f"{stem}.json", record)

            if previous is not None:
                dt_s = timestamp - float(previous["timestamp"])
                flow, valid, stats = build_dynamic_voxel_flow(
                    previous["depth"],
                    depth,
                    previous["semantics"],
                    previous["optical"],
                    source_to_world=previous["matrix"],
                    target_to_world=_transform_matrix(depth_packet.transform),
                    spec=spec,
                    fov_deg=args.fov,
                    dt_s=dt_s,
                    pixel_stride=args.flow_pixel_stride,
                    max_speed_mps=args.max_flow_speed_mps,
                )
                previous_record = records[-2]
                previous_stem = f"{int(previous_record['frame']):08d}"
                flow_path = flow_root / f"{previous_stem}.npz"
                np.savez_compressed(
                    flow_path,
                    velocity_mps=flow,
                    valid=valid,
                    dt_s=np.float32(stats.dt_s),
                    optical_flow_direction_sign=np.int8(stats.optical_flow_direction_sign),
                )
                previous_record["teacher_flow"] = f"teacher_flow/{previous_stem}.npz"
                previous_record["teacher_flow_stats"] = stats.as_dict()
                _write_json(
                    output_root / "metadata" / f"{previous_stem}.json",
                    previous_record,
                )
                flow_labeled_frames += 1
                flow_valid_voxels += stats.valid_voxels
                flow_direction_counts[str(stats.optical_flow_direction_sign)] += 1
                if stats.static_median_residual_m is not None:
                    static_residuals.append(stats.static_median_residual_m)

            previous = {
                "timestamp": timestamp,
                "depth": depth.copy(),
                "semantics": semantics.copy(),
                "optical": optical.copy(),
                "matrix": _transform_matrix(depth_packet.transform),
            }
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
        "flow_schema_version": FLOW_CAPTURE_SCHEMA_VERSION,
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
        "flow_labeled_frames": flow_labeled_frames,
        "flow_valid_voxels": flow_valid_voxels,
        "flow_direction_counts": flow_direction_counts,
        "mean_static_correspondence_residual_m": (
            float(np.mean(static_residuals)) if static_residuals else None
        ),
        "teacher_note": (
            "depth, semantic and optical-flow cameras are privileged training/evaluation sources only"
        ),
        "last_frame_flow_note": "the final frame has no next-frame target and therefore no teacher_flow",
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
