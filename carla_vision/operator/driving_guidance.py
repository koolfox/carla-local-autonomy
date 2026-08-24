"""Projection helpers for the operator-only CARLA driving guide.

The guide is intentionally separate from perception.  Its world points come
from CARLA's privileged map/route state and are projected into the RGB camera
only for operator visualization and research evidence.  They are never an
input to a vision policy or an actuator.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np


def _camera_to_world_matrix(
    transform: Sequence[float],
) -> np.ndarray:
    if len(transform) != 6:
        raise ValueError("camera transform must contain x, y, z, pitch, yaw, roll")
    x, y, z, pitch, yaw, roll = (float(value) for value in transform)
    if not all(math.isfinite(value) for value in (x, y, z, pitch, yaw, roll)):
        raise ValueError("camera transform values must be finite")

    pitch_radians = math.radians(pitch)
    yaw_radians = math.radians(yaw)
    roll_radians = math.radians(roll)
    c_p, s_p = math.cos(pitch_radians), math.sin(pitch_radians)
    c_y, s_y = math.cos(yaw_radians), math.sin(yaw_radians)
    c_r, s_r = math.cos(roll_radians), math.sin(roll_radians)

    return np.array(
        [
            [
                c_p * c_y,
                c_y * s_p * s_r - s_y * c_r,
                -c_y * s_p * c_r - s_y * s_r,
                x,
            ],
            [
                c_p * s_y,
                s_y * s_p * s_r + c_y * c_r,
                -s_y * s_p * c_r + c_y * s_r,
                y,
            ],
            [s_p, -c_p * s_r, c_p * c_r, z],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _point3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return None
    try:
        point = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in point):
        return None
    return point  # type: ignore[return-value]


def _project_point(
    point: tuple[float, float, float],
    *,
    world_to_camera: np.ndarray,
    focal: float,
    width: int,
    height: int,
) -> tuple[list[float], float] | None:
    camera_unreal = world_to_camera @ np.array([*point, 1.0], dtype=np.float64)
    # CARLA/Unreal uses x-forward, y-right, z-up.  The pinhole camera expects
    # x-right, y-down, z-forward.
    depth = float(camera_unreal[0])
    if depth <= 0.35:
        return None
    camera_x = float(camera_unreal[1])
    camera_y = -float(camera_unreal[2])
    u = focal * camera_x / depth + width / 2.0
    v = focal * camera_y / depth + height / 2.0
    if not math.isfinite(u) or not math.isfinite(v):
        return None
    # Retain a bounded off-screen margin so Canvas can clip a path that enters
    # the image without receiving extreme coordinates.
    u = min(width * 3.0, max(-width * 2.0, u))
    v = min(height * 3.0, max(-height * 2.0, v))
    return [round(u, 2), round(v, 2)], depth


def project_driving_guidance(
    guidance: Mapping[str, Any] | None,
    *,
    camera_transform: Sequence[float],
    width: int,
    height: int,
    fov: float,
    camera_sequence: int,
) -> dict[str, Any]:
    """Project bounded CARLA world guidance into one exact RGB camera pose."""

    unavailable = {
        "available": False,
        "source": None,
        "label": "PATH GUIDE UNAVAILABLE",
        "privileged": True,
        "feeds_control": False,
        "camera_sequence": int(camera_sequence),
        "frame_size": [int(width), int(height)],
        "center": [],
        "left": [],
        "right": [],
        "steering_target": None,
    }
    if not isinstance(guidance, Mapping) or not bool(guidance.get("available")):
        if isinstance(guidance, Mapping) and guidance.get("reason"):
            unavailable["reason"] = str(guidance["reason"])
        return unavailable
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or int(width) <= 0
        or int(height) <= 0
    ):
        raise ValueError("camera width and height must be positive integers")
    field_of_view = float(fov)
    if not math.isfinite(field_of_view) or not 1.0 <= field_of_view < 179.0:
        raise ValueError("camera field of view must be in [1, 179) degrees")

    raw_points = guidance.get("points")
    if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
        return {**unavailable, "reason": "guidance points are missing"}

    world_to_camera = np.linalg.inv(_camera_to_world_matrix(camera_transform))
    focal = float(width) / (2.0 * math.tan(math.radians(field_of_view) / 2.0))
    projected: list[dict[str, Any]] = []
    for raw in raw_points[:96]:
        if not isinstance(raw, Mapping):
            continue
        center_world = _point3(raw.get("center"))
        left_world = _point3(raw.get("left"))
        right_world = _point3(raw.get("right"))
        if center_world is None or left_world is None or right_world is None:
            continue
        center_result = _project_point(
            center_world,
            world_to_camera=world_to_camera,
            focal=focal,
            width=int(width),
            height=int(height),
        )
        left_result = _project_point(
            left_world,
            world_to_camera=world_to_camera,
            focal=focal,
            width=int(width),
            height=int(height),
        )
        right_result = _project_point(
            right_world,
            world_to_camera=world_to_camera,
            focal=focal,
            width=int(width),
            height=int(height),
        )
        if center_result is None or left_result is None or right_result is None:
            continue
        try:
            distance_m = float(raw.get("distance_m", center_result[1]))
        except (TypeError, ValueError):
            distance_m = center_result[1]
        projected.append(
            {
                "center": center_result[0],
                "left": left_result[0],
                "right": right_result[0],
                "distance_m": round(max(0.0, distance_m), 2),
                "depth_m": round(center_result[1], 2),
            }
        )

    if len(projected) < 2:
        return {**unavailable, "reason": "guidance is outside the camera view"}

    steering_target = min(projected, key=lambda point: abs(point["distance_m"] - 12.0))[
        "center"
    ]
    return {
        "available": True,
        "source": str(guidance.get("source", "carla_map_lane_reference")),
        "label": str(guidance.get("label", "CARLA MAP REFERENCE")),
        "privileged": True,
        "feeds_control": False,
        "camera_sequence": int(camera_sequence),
        "frame_size": [int(width), int(height)],
        "lookahead_m": float(guidance.get("lookahead_m", 0.0) or 0.0),
        "center": [point["center"] for point in projected],
        "left": [point["left"] for point in projected],
        "right": [point["right"] for point in projected],
        "steering_target": steering_target,
        "point_count": len(projected),
    }


def _screen_polyline(value: Any) -> np.ndarray | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    points: list[tuple[int, int]] = []
    for raw in value:
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
            continue
        try:
            x, y = float(raw[0]), float(raw[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            points.append((round(x), round(y)))
    if len(points) < 2:
        return None
    return np.asarray(points, dtype=np.int32)


def render_driving_guidance_jpeg(
    jpeg: bytes,
    guidance: Mapping[str, Any] | None,
    *,
    quality: int = 92,
) -> bytes:
    """Draw guidance into one browser-only JPEG derived from that exact frame.

    The original cached JPEG and research recordings are never modified. When
    guidance is unavailable or does not match the decoded frame, the original
    bytes are returned without a decode/re-encode quality loss.
    """

    if not isinstance(jpeg, bytes) or not jpeg:
        raise ValueError("jpeg must be non-empty bytes")
    if not isinstance(guidance, Mapping) or not bool(guidance.get("available")):
        return jpeg
    encoded = np.frombuffer(jpeg, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        return jpeg
    height, width = image.shape[:2]
    frame_size = guidance.get("frame_size")
    if not isinstance(frame_size, Sequence) or isinstance(frame_size, (str, bytes)):
        return jpeg
    try:
        declared_size = [int(frame_size[0]), int(frame_size[1])]
    except (IndexError, TypeError, ValueError):
        return jpeg
    if declared_size != [width, height]:
        return jpeg

    left = _screen_polyline(guidance.get("left"))
    right = _screen_polyline(guidance.get("right"))
    center = _screen_polyline(guidance.get("center"))
    if left is None or right is None or center is None:
        return jpeg
    pair_count = min(len(left), len(right))
    if pair_count < 2:
        return jpeg

    corridor = np.concatenate((left[:pair_count], right[:pair_count][::-1]), axis=0)
    tint = image.copy()
    cv2.fillPoly(tint, [corridor], color=(255, 142, 45), lineType=cv2.LINE_AA)
    cv2.addWeighted(tint, 0.24, image, 0.76, 0.0, dst=image)

    line_color = (255, 191, 89)
    if guidance.get("source") == "carla_map_lane_reference":
        for index in range(0, len(center) - 1, 2):
            cv2.line(
                image,
                tuple(center[index]),
                tuple(center[index + 1]),
                line_color,
                5,
                cv2.LINE_AA,
            )
    else:
        cv2.polylines(image, [center], False, line_color, 5, cv2.LINE_AA)

    target = guidance.get("steering_target")
    if isinstance(target, Sequence) and not isinstance(target, (str, bytes)) and len(target) == 2:
        try:
            target_point = (round(float(target[0])), round(float(target[1])))
        except (TypeError, ValueError):
            target_point = None
        if target_point is not None and all(math.isfinite(float(value)) for value in target):
            cv2.circle(image, target_point, 11, line_color, 2, cv2.LINE_AA)
            cv2.circle(image, target_point, 5, (255, 255, 255), -1, cv2.LINE_AA)

    ok, rendered = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, int(quality)))],
    )
    return rendered.tobytes() if ok else jpeg


__all__ = ["project_driving_guidance", "render_driving_guidance_jpeg"]
