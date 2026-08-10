"""Privileged camera teacher for dynamic voxel velocity labels.

The deployable model never imports this module.  It combines CARLA depth,
semantic segmentation, optical flow and camera poses to build supervision for
moving vehicles and pedestrians.  Output flow is an instantaneous velocity in
source-camera coordinates (x-forward, y-right, z-up), stored per occupied voxel.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .contracts import VoxelGridSpec
from .geometry import camera_intrinsics

DYNAMIC_SEMANTIC_TAGS = (4, 10)  # pedestrian, vehicle


@dataclass(frozen=True, slots=True)
class VoxelFlowTeacherStats:
    dt_s: float
    optical_flow_direction_sign: int
    sampled_points: int
    matched_points: int
    dynamic_points: int
    valid_voxels: int
    static_median_residual_m: float | None

    def as_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def decode_carla_optical_flow(
    raw_data: bytes | bytearray | memoryview,
    width: int,
    height: int,
) -> np.ndarray:
    """Decode CARLA ``sensor.camera.optical_flow`` into normalized XY flow.

    CARLA stores two float32 values per pixel.  Documentation defines each
    component in approximately [-2, 2]; multiplying X by image width and Y by
    image height converts them to pixel displacement units.
    """

    if width <= 0 or height <= 0:
        raise ValueError("optical-flow image dimensions must be positive")
    values = np.frombuffer(raw_data, dtype=np.float32)
    expected = width * height * 2
    if values.size != expected:
        raise ValueError(f"expected {expected} optical-flow floats, received {values.size}")
    flow = values.reshape(height, width, 2).copy()
    if not np.isfinite(flow).all():
        raise ValueError("CARLA optical-flow payload contains non-finite values")
    return flow


def _validate_matrix(value: np.ndarray | Sequence[Sequence[float]], name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 transformation matrix")
    return matrix


def _backproject_selected(
    depth_m: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    fov_deg: float,
) -> np.ndarray:
    height, width = depth_m.shape
    intrinsics = camera_intrinsics(width, height, fov_deg)
    focal = float(intrinsics[0, 0])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    depth = depth_m[rows, cols].astype(np.float64)
    x = depth
    y = (cols.astype(np.float64) - cx) * depth / focal
    z = -(rows.astype(np.float64) - cy) * depth / focal
    return np.stack((x, y, z), axis=1)


def _transform_points(points_xyz: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float64)
    homogeneous = np.concatenate((points_xyz.astype(np.float64), ones), axis=1)
    return (matrix @ homogeneous.T).T[:, :3]


def _candidate_correspondences(
    depth_source: np.ndarray,
    depth_target: np.ndarray,
    semantic_source: np.ndarray,
    flow_normalized: np.ndarray,
    *,
    source_to_world: np.ndarray,
    target_to_world: np.ndarray,
    fov_deg: float,
    pixel_stride: int,
    sign: int,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = depth_source.shape
    rows, cols = np.mgrid[0:height:pixel_stride, 0:width:pixel_stride]
    rows = rows.reshape(-1).astype(np.int32)
    cols = cols.reshape(-1).astype(np.int32)
    source_depth = depth_source[rows, cols]
    valid = np.isfinite(source_depth) & (source_depth >= 0.1) & (source_depth <= max_depth_m)
    if not np.any(valid):
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty, np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.int32)
    rows = rows[valid]
    cols = cols[valid]
    source_depth = source_depth[valid]
    normalized = flow_normalized[rows, cols]
    target_cols_f = cols.astype(np.float64) + sign * normalized[:, 0] * width
    target_rows_f = rows.astype(np.float64) + sign * normalized[:, 1] * height
    target_cols = np.rint(target_cols_f).astype(np.int32)
    target_rows = np.rint(target_rows_f).astype(np.int32)
    inside = (
        (target_cols >= 0)
        & (target_cols < width)
        & (target_rows >= 0)
        & (target_rows < height)
    )
    rows = rows[inside]
    cols = cols[inside]
    target_rows = target_rows[inside]
    target_cols = target_cols[inside]
    if rows.size == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty, np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.int32)
    target_depth = depth_target[target_rows, target_cols]
    depth_ok = np.isfinite(target_depth) & (target_depth >= 0.1) & (target_depth <= max_depth_m)
    rows = rows[depth_ok]
    cols = cols[depth_ok]
    target_rows = target_rows[depth_ok]
    target_cols = target_cols[depth_ok]
    if rows.size == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty, np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.int32)

    source_points = _backproject_selected(depth_source, rows, cols, fov_deg=fov_deg)
    target_points = _backproject_selected(depth_target, target_rows, target_cols, fov_deg=fov_deg)
    source_world = _transform_points(source_points, source_to_world)
    target_world = _transform_points(target_points, target_to_world)
    displacement_world = target_world - source_world
    source_rotation = source_to_world[:3, :3]
    displacement_source = displacement_world @ source_rotation
    tags = semantic_source[rows, cols].astype(np.uint8)
    source_linear = rows.astype(np.int64) * width + cols.astype(np.int64)
    return (
        source_points.astype(np.float32),
        displacement_source.astype(np.float32),
        tags,
        source_linear.astype(np.int32),
    )


def build_dynamic_voxel_flow(
    depth_source_m: np.ndarray,
    depth_target_m: np.ndarray,
    semantic_source: np.ndarray,
    optical_flow_normalized: np.ndarray,
    *,
    source_to_world: np.ndarray | Sequence[Sequence[float]],
    target_to_world: np.ndarray | Sequence[Sequence[float]],
    spec: VoxelGridSpec,
    fov_deg: float,
    dt_s: float,
    pixel_stride: int = 4,
    dynamic_semantic_tags: Sequence[int] = DYNAMIC_SEMANTIC_TAGS,
    max_speed_mps: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, VoxelFlowTeacherStats]:
    """Lift privileged optical flow into a sparse dynamic voxel-velocity target.

    Two optical-flow directions are evaluated.  The sign with the smallest
    median world-space residual on non-dynamic pixels is selected, which makes
    the teacher robust to renderer/API direction conventions.  Static-camera
    ego motion is removed by comparing source and target points in world space.
    """

    source = np.asarray(depth_source_m, dtype=np.float32)
    target = np.asarray(depth_target_m, dtype=np.float32)
    semantics = np.asarray(semantic_source, dtype=np.uint8)
    optical = np.asarray(optical_flow_normalized, dtype=np.float32)
    if source.ndim != 2 or target.shape != source.shape or semantics.shape != source.shape:
        raise ValueError("source depth, target depth and source semantics must share HxW shape")
    if optical.shape != (*source.shape, 2):
        raise ValueError("optical_flow_normalized must have shape (H, W, 2)")
    if not np.isfinite(optical).all():
        raise ValueError("optical flow must contain finite values")
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    if pixel_stride <= 0:
        raise ValueError("pixel_stride must be positive")
    if not math.isfinite(max_speed_mps) or max_speed_mps <= 0:
        raise ValueError("max_speed_mps must be finite and positive")
    source_matrix = _validate_matrix(source_to_world, "source_to_world")
    target_matrix = _validate_matrix(target_to_world, "target_to_world")
    max_depth = math.sqrt(
        max(abs(spec.x_min), abs(spec.x_max)) ** 2
        + max(abs(spec.y_min), abs(spec.y_max)) ** 2
        + max(abs(spec.z_min), abs(spec.z_max)) ** 2
    )
    candidates: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    residuals: dict[int, float | None] = {}
    dynamic_tags = np.asarray(tuple(int(value) for value in dynamic_semantic_tags), dtype=np.uint8)
    for sign in (1, -1):
        candidate = _candidate_correspondences(
            source,
            target,
            semantics,
            optical,
            source_to_world=source_matrix,
            target_to_world=target_matrix,
            fov_deg=fov_deg,
            pixel_stride=pixel_stride,
            sign=sign,
            max_depth_m=max_depth,
        )
        candidates[sign] = candidate
        _, displacement, tags, _ = candidate
        static = (tags != 0) & ~np.isin(tags, dynamic_tags)
        residuals[sign] = (
            float(np.median(np.linalg.norm(displacement[static], axis=1)))
            if np.any(static)
            else None
        )
    comparable = {sign: value for sign, value in residuals.items() if value is not None}
    chosen_sign = min(comparable, key=comparable.get) if comparable else 1
    points, displacement, tags, _ = candidates[chosen_sign]
    velocity = displacement / float(dt_s) if displacement.size else displacement
    dynamic = np.isin(tags, dynamic_tags)
    speed = np.linalg.norm(velocity, axis=1) if velocity.size else np.empty(0, dtype=np.float32)
    usable = dynamic & np.isfinite(speed) & (speed <= max_speed_mps)
    points = points[usable]
    velocity = velocity[usable]

    flow = np.zeros((3, *spec.shape), dtype=np.float32)
    valid_grid = np.zeros(spec.shape, dtype=bool)
    if points.size:
        indices, inside = spec.metric_to_indices(points)
        indices = indices[inside]
        velocity = velocity[inside]
        if indices.size:
            linear = np.ravel_multi_index(indices.T, spec.shape)
            order = np.argsort(linear, kind="stable")
            linear = linear[order]
            velocity = velocity[order]
            starts = np.flatnonzero(np.r_[True, linear[1:] != linear[:-1]])
            ends = np.r_[starts[1:], linear.size]
            flat_valid = valid_grid.reshape(-1)
            flat_flow = flow.reshape(3, -1)
            for start, end in zip(starts, ends, strict=True):
                voxel = int(linear[start])
                flat_flow[:, voxel] = np.median(velocity[start:end], axis=0)
                flat_valid[voxel] = True

    sampled_points = int(math.ceil(source.shape[0] / pixel_stride) * math.ceil(source.shape[1] / pixel_stride))
    stats = VoxelFlowTeacherStats(
        dt_s=float(dt_s),
        optical_flow_direction_sign=int(chosen_sign),
        sampled_points=sampled_points,
        matched_points=int(candidates[chosen_sign][0].shape[0]),
        dynamic_points=int(np.count_nonzero(usable)),
        valid_voxels=int(np.count_nonzero(valid_grid)),
        static_median_residual_m=residuals[chosen_sign],
    )
    return flow, valid_grid, stats
