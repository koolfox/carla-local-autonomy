"""Geometry utilities for lifting aligned CARLA camera frames into voxels."""

from __future__ import annotations

import math

import numpy as np

from .contracts import FREE, OCCUPIED, SEMANTIC_UNKNOWN, UNKNOWN, VoxelGridSpec

_DEPTH_DENOMINATOR = float(256**3 - 1)


def carla_rotation_matrix(*, roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Pure CARLA Euler geometry: x forward, y right, z up; angles in degrees.

    Positive pitch raises forward; positive roll turns right down. This helper
    has no simulator access and grants no model access to pose information.
    """
    if not all(math.isfinite(value) for value in (roll, pitch, yaw)):
        raise ValueError("camera rotation must contain finite angles")
    roll, pitch, yaw = map(math.radians, (roll, pitch, yaw))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
         [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
         [sp, -cp * sr, cp * cr]],
        dtype=np.float64,
    )


def camera_intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("camera dimensions must be positive")
    if not math.isfinite(fov_deg) or not 0.0 < fov_deg < 180.0:
        raise ValueError("camera FOV must be finite and in (0, 180)")
    focal = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return np.asarray(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _bgra_image(raw_data: bytes | bytearray | memoryview, width: int, height: int) -> np.ndarray:
    expected = width * height * 4
    image = np.frombuffer(raw_data, dtype=np.uint8)
    if image.size != expected:
        raise ValueError(f"expected {expected} BGRA bytes, received {image.size}")
    return image.reshape((height, width, 4))


def decode_carla_depth_bgra(
    raw_data: bytes | bytearray | memoryview,
    width: int,
    height: int,
) -> np.ndarray:
    """Decode CARLA's 24-bit depth camera into metres."""

    bgra = _bgra_image(raw_data, width, height).astype(np.float32)
    blue = bgra[:, :, 0]
    green = bgra[:, :, 1]
    red = bgra[:, :, 2]
    normalized = (red + green * 256.0 + blue * 65536.0) / _DEPTH_DENOMINATOR
    return normalized * 1000.0


def semantic_tags_from_bgra(
    raw_data: bytes | bytearray | memoryview,
    width: int,
    height: int,
) -> np.ndarray:
    """Read CARLA semantic class IDs from the red channel."""

    return _bgra_image(raw_data, width, height)[:, :, 2].copy()


def rgb_from_bgra(
    raw_data: bytes | bytearray | memoryview,
    width: int,
    height: int,
) -> np.ndarray:
    bgra = _bgra_image(raw_data, width, height)
    return bgra[:, :, [2, 1, 0]].copy()


def backproject_depth(
    depth_m: np.ndarray,
    *,
    fov_deg: float,
    pixel_stride: int = 1,
    min_depth_m: float = 0.1,
    max_depth_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Back-project depth into CARLA camera coordinates.

    Returned points follow x-forward, y-right and z-up. Pixel arrays contain
    the sampled source rows and columns, enabling aligned semantic lookup.
    """

    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m must be a 2D array")
    if pixel_stride <= 0:
        raise ValueError("pixel_stride must be positive")
    height, width = depth.shape
    intrinsics = camera_intrinsics(width, height, fov_deg)
    focal = float(intrinsics[0, 0])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    rows, cols = np.mgrid[0:height:pixel_stride, 0:width:pixel_stride]
    sampled_depth = depth[rows, cols]
    valid = np.isfinite(sampled_depth) & (sampled_depth >= min_depth_m)
    if max_depth_m is not None:
        if max_depth_m <= min_depth_m:
            raise ValueError("max_depth_m must be greater than min_depth_m")
        valid &= sampled_depth <= max_depth_m
    sampled_depth = sampled_depth[valid]
    sampled_rows = rows[valid].astype(np.int32)
    sampled_cols = cols[valid].astype(np.int32)
    x = sampled_depth
    y = (sampled_cols.astype(np.float32) - cx) * sampled_depth / focal
    z = -(sampled_rows.astype(np.float32) - cy) * sampled_depth / focal
    return np.stack((x, y, z), axis=1), sampled_rows, sampled_cols


def _assign_semantic_majority(
    semantics: np.ndarray,
    indices_zyx: np.ndarray,
    tags: np.ndarray,
) -> None:
    if indices_zyx.size == 0:
        return
    shape = semantics.shape
    linear = np.ravel_multi_index(indices_zyx.T, shape)
    order = np.argsort(linear, kind="stable")
    linear = linear[order]
    tags = np.asarray(tags, dtype=np.uint8)[order]
    starts = np.flatnonzero(np.r_[True, linear[1:] != linear[:-1]])
    ends = np.r_[starts[1:], linear.size]
    flat_semantics = semantics.reshape(-1)
    for start, end in zip(starts, ends, strict=True):
        group = tags[start:end]
        counts = np.bincount(group, minlength=256)
        flat_semantics[linear[start]] = np.uint8(np.argmax(counts))


def raycast_voxel_grid(
    depth_m: np.ndarray,
    semantic_tags: np.ndarray | None,
    *,
    spec: VoxelGridSpec,
    fov_deg: float,
    pixel_stride: int = 8,
    ray_step_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create visibility-aware free/occupied/unknown voxels from camera labels.

    This is a privileged CARLA teacher-label generator. It is not a deployable
    camera-only predictor: depth and semantics must be replaced by a learned RGB
    model for real inference.
    """

    depth = np.asarray(depth_m, dtype=np.float32)
    if semantic_tags is not None:
        semantic = np.asarray(semantic_tags, dtype=np.uint8)
        if semantic.shape != depth.shape:
            raise ValueError("semantic_tags must match the depth image shape")
    else:
        semantic = None
    step = float(ray_step_m if ray_step_m is not None else spec.resolution)
    if not math.isfinite(step) or step <= 0:
        raise ValueError("ray_step_m must be finite and positive")
    max_depth = math.sqrt(
        max(abs(spec.x_min), abs(spec.x_max)) ** 2
        + max(abs(spec.y_min), abs(spec.y_max)) ** 2
        + max(abs(spec.z_min), abs(spec.z_max)) ** 2
    )
    surfaces, rows, cols = backproject_depth(
        depth,
        fov_deg=fov_deg,
        pixel_stride=pixel_stride,
        max_depth_m=max_depth,
    )
    occupancy = np.full(spec.shape, UNKNOWN, dtype=np.int8)
    semantics = np.full(spec.shape, SEMANTIC_UNKNOWN, dtype=np.uint8)
    if surfaces.size == 0:
        return occupancy, semantics

    for surface in surfaces:
        surface_x = float(surface[0])
        if surface_x <= step:
            continue
        fractions = np.arange(step, surface_x, step, dtype=np.float32) / surface_x
        free_points = surface[None, :] * fractions[:, None]
        free_indices, valid = spec.metric_to_indices(free_points)
        valid_indices = free_indices[valid]
        if valid_indices.size:
            z_index, y_index, x_index = valid_indices.T
            unknown = occupancy[z_index, y_index, x_index] == UNKNOWN
            occupancy[
                z_index[unknown],
                y_index[unknown],
                x_index[unknown],
            ] = FREE

    surface_indices, surface_valid = spec.metric_to_indices(surfaces)
    surface_indices = surface_indices[surface_valid]
    if surface_indices.size:
        z_index, y_index, x_index = surface_indices.T
        occupancy[z_index, y_index, x_index] = OCCUPIED
        if semantic is not None:
            tags = semantic[rows[surface_valid], cols[surface_valid]]
            _assign_semantic_majority(semantics, surface_indices, tags)
    return occupancy, semantics


def occupancy_bev(occupancy_zyx: np.ndarray) -> np.ndarray:
    """Render a top-down uint8 diagnostic image from a voxel grid."""

    occupancy = np.asarray(occupancy_zyx, dtype=np.int8)
    if occupancy.ndim != 3:
        raise ValueError("occupancy_zyx must have shape (Z, Y, X)")
    occupied = np.any(occupancy == OCCUPIED, axis=0)
    observed_free = np.any(occupancy == FREE, axis=0) & ~occupied
    image = np.zeros(occupied.shape, dtype=np.uint8)
    image[observed_free] = 96
    image[occupied] = 255
    return np.flipud(image)


def occupied_iou(
    prediction_probability: np.ndarray,
    target_occupancy: np.ndarray,
    *,
    threshold: float = 0.5,
) -> float:
    prediction = np.asarray(prediction_probability, dtype=np.float32)
    target = np.asarray(target_occupancy, dtype=np.int8)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target grids must have identical shapes")
    known = target != UNKNOWN
    predicted_occupied = prediction >= threshold
    target_occupied = target == OCCUPIED
    intersection = np.count_nonzero(predicted_occupied & target_occupied & known)
    union = np.count_nonzero((predicted_occupied | target_occupied) & known)
    return 1.0 if union == 0 else float(intersection / union)
