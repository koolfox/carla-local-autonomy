"""Bounded RGB geometry views and a separate, privileged route annotation.

The model result is never changed by the teacher. A waypoint projection is a
map reference, not a road mask, an occlusion test, or a safe-to-drive corridor.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .geometry import camera_intrinsics, carla_rotation_matrix

_NEAR = 0.1
_MAX_CELLS = 2048
_GEOMETRY = (220, 196, 92)  # BGR; deliberately different from the teacher.
_TEACHER = (223, 105, 238)
_TEXT = (226, 230, 232)
_CORNERS = np.array([[-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1],
                     [-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1]], float)
_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
          (0, 4), (1, 5), (2, 6), (3, 7))


def project_waypoint_teacher(
    teacher: dict[str, Any], *, transform: tuple[float, ...], width: int,
    height: int, fov: float, source_frame: int,
) -> dict[str, Any]:
    """Project world reference points with the source RGB's own camera pose.

    This matches the geometry to the image, not the asynchronous Traffic
    Manager decision to a past frame. Keep both facts explicit in saved labels.
    """
    if isinstance(source_frame, bool) or not isinstance(source_frame, int) or source_frame < 0:
        raise ValueError("source_frame must be a non-negative integer")
    if teacher.get("source_frame") not in (None, source_frame):
        raise ValueError("waypoint teacher source_frame does not match source RGB")
    if teacher.get("coordinate_frame") != "carla_world_metres":
        raise ValueError("waypoint teacher must use carla_world_metres")
    if teacher.get("teacher_only") is not True:
        raise ValueError("waypoint reference must be declared teacher_only")
    pose = np.asarray(transform, dtype=np.float64)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("source camera transform must contain six finite values")
    if (isinstance(width, bool) or isinstance(height, bool)
            or not isinstance(width, int) or not isinstance(height, int)
            or width * height > 16_777_216):
        raise ValueError("source camera dimensions must be bounded positive integers")
    intrinsics = camera_intrinsics(width, height, fov)
    supplied = teacher.get("points", [])
    if not isinstance(supplied, (list, tuple)) or len(supplied) > 64:
        raise ValueError("waypoint teacher must contain at most 64 points")
    try:
        world = np.array([[p[axis] for axis in ("x", "y", "z")] for p in supplied],
                         dtype=np.float64).reshape(-1, 3)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("waypoint points must contain finite x, y, z coordinates") from error
    if not np.isfinite(world).all():
        raise ValueError("waypoint points must contain finite x, y, z coordinates")
    rotation = carla_rotation_matrix(pitch=pose[3], yaw=pose[4], roll=pose[5])
    camera = (world - pose[:3]) @ rotation
    uv = _camera_pixels(camera, intrinsics)
    in_front = camera[:, 0] > _NEAR
    in_bounds = (in_front & (uv[:, 0] >= 0) & (uv[:, 0] < width)
                 & (uv[:, 1] >= 0) & (uv[:, 1] < height))
    return {
        **teacher,
        "source_frame": source_frame,
        "projection_frame": source_frame,
        "projection_frame_matched": True,
        "route_frame_matched": False,
        "teacher_only": True,
        "input_to_model": False,
        "occlusion_checked": False,
        "camera_points_xyz": camera.tolist(),
        "image_points_uv": [point.tolist() if valid else None
                            for point, valid in zip(uv, in_front, strict=True)],
        "image_in_bounds": in_bounds.tolist(),
        "source_camera_transform": pose.tolist(),
        "calibration": {"width": width, "height": height, "fov": float(fov),
                        "intrinsics": intrinsics.tolist(),
                        "axes": "x_forward_y_right_z_up"},
    }


def _camera_pixels(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    depth = np.maximum(points[:, 0], _NEAR)
    return np.column_stack((intrinsics[0, 2] + intrinsics[0, 0] * points[:, 1] / depth,
                            intrinsics[1, 2] - intrinsics[1, 1] * points[:, 2] / depth))


def _label(image: np.ndarray, text: str, xy: tuple[int, int], *, scale: float = 0.5,
           color: tuple[int, int, int] = _TEXT) -> None:
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _source_image(source: np.ndarray, result: Any, frame: int) -> np.ndarray:
    if (source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3
            or min(source.shape[:2]) <= 0):
        raise ValueError("source image must be uint8 HxWx3 BGR")
    model_frame = result.metadata.get("source_frame")
    if model_frame is not None and model_frame != frame:
        raise ValueError("voxel result is not from this source frame")
    if result.metadata.get("coordinate_frame", "camera") != "camera":
        raise ValueError("live voxel views require camera-local model geometry")
    return source.copy()


def _cells(result: Any) -> np.ndarray:
    """Deterministic bounded occupied cells; never draw free/unknown as objects."""
    indices = np.argwhere(result.occupancy == 1)
    if len(indices) > _MAX_CELLS:
        indices = indices[np.linspace(0, len(indices) - 1, _MAX_CELLS, dtype=int)]
    spec = result.spec
    points = np.column_stack((spec.x_min + (indices[:, 2] + 0.5) * spec.resolution,
                              spec.y_min + (indices[:, 1] + 0.5) * spec.resolution,
                              spec.z_min + (indices[:, 0] + 0.5) * spec.resolution))
    return points[np.argsort(-points[:, 0], kind="stable")]


def _teacher_points(teacher: dict[str, Any] | None, frame: int) -> np.ndarray:
    if teacher is None:
        return np.empty((0, 3))
    if teacher.get("projection_frame") != frame or teacher.get("teacher_only") is not True:
        raise ValueError("teacher projection is not associated with this source frame")
    points = np.asarray(teacher.get("camera_points_xyz", []), dtype=float).reshape(-1, 3)
    if len(points) > 64 or not np.isfinite(points).all():
        raise ValueError("teacher projection must contain at most 64 finite points")
    return points


def _clipped_line(image: np.ndarray, a: np.ndarray, b: np.ndarray,
                  color: tuple[int, int, int], thickness: int = 1) -> None:
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return
    # OpenCV accepts int32. Bound extreme off-camera projections before clipping.
    endpoints = np.clip(np.rint([a, b]), -1_000_000, 1_000_000).astype(int)
    visible, start, end = cv2.clipLine((0, 0, image.shape[1], image.shape[0]),
                                      tuple(endpoints[0]), tuple(endpoints[1]))
    if visible:
        cv2.line(image, start, end, color, thickness, cv2.LINE_AA)


def _camera_segment(a: np.ndarray, b: np.ndarray, intrinsics: np.ndarray):
    """Clip a consecutive segment at the camera near plane, before perspective."""
    if a[0] <= _NEAR and b[0] <= _NEAR:
        return None
    a, b = a.copy(), b.copy()
    if a[0] <= _NEAR:
        a += (b - a) * ((_NEAR - a[0]) / (b[0] - a[0]))
    elif b[0] <= _NEAR:
        b += (a - b) * ((_NEAR - b[0]) / (a[0] - b[0]))
    return _camera_pixels(np.asarray([a, b]), intrinsics)


def render_voxel_overlay(source_bgr: np.ndarray, result: Any, *, frame: int, fov: float,
                         teacher: dict[str, Any] | None = None) -> np.ndarray:
    """Draw predicted occupied cells on their exact RGB; raw input is untouched."""
    image = _source_image(source_bgr, result, frame)
    height, width = image.shape[:2]
    intrinsics = camera_intrinsics(width, height, fov)
    geometry = image.copy()
    for point in _cells(result):
        corners = point + _CORNERS * (result.spec.resolution / 2)
        # A cube crossing the camera origin would fill most of the screen.
        if np.min(corners[:, 0]) <= _NEAR:
            continue
        uv = _camera_pixels(corners, intrinsics)
        if (uv[:, 0].max() < 0 or uv[:, 0].min() >= width
                or uv[:, 1].max() < 0 or uv[:, 1].min() >= height):
            continue
        for a, b in _EDGES:
            _clipped_line(geometry, uv[a], uv[b], _GEOMETRY)
    image = cv2.addWeighted(geometry, 0.52, image, 0.48, 0)
    route = _teacher_points(teacher, frame)
    for a, b in zip(route[:-1], route[1:], strict=False):
        segment = _camera_segment(a, b, intrinsics)
        if segment is not None:
            _clipped_line(image, *segment, (40, 22, 46), 6)
            _clipped_line(image, *segment, _TEACHER, 3)
    # No sidebars, opaque panels or source image resizing in the camera view.
    _label(image, f"RGB geometry | frame {frame}", (12, 23), color=_GEOMETRY)
    if len(route):
        _label(image, "CARLA waypoint teacher (not model)", (12, 44), color=_TEACHER)
    return image


def render_voxel_view(source_bgr: np.ndarray, result: Any, *, frame: int,
                      teacher: dict[str, Any] | None = None) -> np.ndarray:
    """Compact camera-local 3D surface view; fixed metric scale within a run."""
    _source_image(source_bgr, result, frame)
    image = np.full((720, 1280, 3), (27, 30, 33), dtype=np.uint8)
    spec = result.spec
    # Fit the configured volume, never the current predictions or changing route.
    bounds = np.array([[x, y, z] for x in (spec.x_min, spec.x_max)
                       for y in (spec.y_min, spec.y_max) for z in (spec.z_min, spec.z_max)])

    def plane(points: np.ndarray) -> np.ndarray:
        x, y, z = np.asarray(points).T
        return np.column_stack((0.94 * y - 0.24 * x, -0.78 * x - z))

    projected = plane(bounds)
    lower, upper = projected.min(axis=0), projected.max(axis=0)
    scale = min(1180 / (upper[0] - lower[0]), 520 / (upper[1] - lower[1]))
    offset = np.array([640, 366]) - (lower + upper) * scale / 2

    def project(points: np.ndarray) -> np.ndarray:
        return plane(points) * scale + offset

    for x in np.arange(max(0, spec.x_min), spec.x_max + 0.001, 5):
        a, b = project(np.array([[x, spec.y_min, 0], [x, spec.y_max, 0]]))
        _clipped_line(image, a, b, (53, 58, 62))
        _label(image, f"{x:g}m", tuple(np.rint(b + [4, -4]).astype(int)), scale=0.38)
    for y in np.arange(np.ceil(spec.y_min / 5) * 5, spec.y_max + 0.001, 5):
        a, b = project(np.array([[max(0, spec.x_min), y, 0], [spec.x_max, y, 0]]))
        _clipped_line(image, a, b, (53, 58, 62))
    for point in _cells(result):
        box = np.rint(project(point + _CORNERS * spec.resolution / 2)).astype(np.int32)
        lift = float(np.clip((point[2] - spec.z_min) / (spec.z_max - spec.z_min), 0, 1))
        # Geometry-only height shading; never fabricated object classes.
        base = np.array(_GEOMETRY) * (0.48 + 0.4 * lift)
        for face, shade in (([0, 1, 5, 4], 0.8), ([1, 2, 6, 5], 0.65),
                            ([4, 5, 6, 7], 1.0)):
            cv2.fillConvexPoly(image, box[face], tuple(map(int, base * shade)), cv2.LINE_AA)
        cv2.polylines(image, [box[[4, 5, 6, 7]]], True,
                      tuple(map(int, base * 1.15)), 1, cv2.LINE_AA)

    route = _teacher_points(teacher, frame)
    for a, b in zip(route[:-1], route[1:], strict=False):
        # A 100 m planned route must not shrink the 35 m local geometry display.
        # Clip to the configured volume; don't project hidden beyond-volume points.
        segment = _volume_segment(a, b, spec)
        if segment is not None:
            uv = project(segment)
            _clipped_line(image, *uv, (42, 23, 48), 7)
            _clipped_line(image, *uv, _TEACHER, 3)

    axes = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 2]])
    origin, *ends = np.rint(project(axes)).astype(int)
    for end, name in zip(ends, ("X front", "Y right", "Z up"), strict=True):
        cv2.arrowedLine(image, tuple(origin), tuple(end), (224, 229, 231), 1,
                       cv2.LINE_AA, tipLength=0.16)
        _label(image, name, tuple(end + [6, -4]), scale=0.4)
    _label(image, "VOXEL 3D", (24, 30), scale=0.65)
    _label(image, f"Frame {frame}  /  camera-local  /  {spec.resolution:g} m cells",
           (24, 54), scale=0.45)
    _label(image, "RGB estimated surfaces", (833, 29), color=_GEOMETRY)
    _label(image, "CARLA waypoint teacher" if len(route) else "CARLA waypoint unavailable",
           (833, 52), color=_TEACHER if len(route) else (150, 153, 157))
    # Source thumbnail is intentionally small and is drawn last, never occluded.
    height, width = source_bgr.shape[:2]
    ratio = min(248 / width, 140 / height)
    thumb = cv2.resize(source_bgr, (round(width * ratio), round(height * ratio)))
    image[88:88 + thumb.shape[0], 24:24 + thumb.shape[1]] = thumb
    _label(image, "Exact source RGB", (24, 88 + thumb.shape[0] + 19), scale=0.4)
    _label(image, "Estimated distance. Unknown space is not free space.", (24, 677), scale=0.47)
    _label(image, "Grid: camera height | Teacher: map reference, no occlusion test",
           (24, 700), scale=0.43)
    return image


def _volume_segment(a: np.ndarray, b: np.ndarray, spec: Any) -> np.ndarray | None:
    """Clip a segment to the displayed volume without changing its altitude."""
    minimum = (spec.x_min, spec.y_min, spec.z_min)
    maximum = (spec.x_max, spec.y_max, spec.z_max)
    direction = b - a
    lo, hi = 0.0, 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if not minimum[axis] <= a[axis] <= maximum[axis]:
                return None
        else:
            start = (minimum[axis] - a[axis]) / direction[axis]
            end = (maximum[axis] - a[axis]) / direction[axis]
            lo, hi = max(lo, min(start, end)), min(hi, max(start, end))
            if lo > hi:
                return None
    return np.array([a + direction * lo, a + direction * hi])
