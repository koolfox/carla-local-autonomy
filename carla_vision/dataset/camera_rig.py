"""RGB rig configuration and calibrated frame contracts (no CARLA runtime import)."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from ..bridge import CarlaImageFrame
from ..scenarios.contracts import CameraRecipe, TransformRecipe

RIG_SCHEMA_VERSION = "1.0"
CAMERA_AXES = "x_forward_y_right_z_up"
OPTICAL_AXES = "x_right_y_down_z_forward"
_CAMERA_ID = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


def resolve_camera_rig(
    primary: CameraRecipe, *, preset: str = "front", config: Mapping[str, Any] | None = None
) -> dict[str, CameraRecipe]:
    """RGB views inherit scene timing/size; custom mounts/FOV are explicit."""
    if preset not in {"front", "front-three"}:
        raise ValueError("camera rig preset must be front or front-three")
    result = {"front": primary}
    if config is None:
        if preset == "front-three":
            for name, angle in (("front_left", -60.0), ("front_right", 60.0)):
                result[name] = replace(
                    primary, mount=replace(primary.mount, yaw=primary.mount.yaw + angle)
                )
        return result
    if preset != "front":
        raise ValueError("choose either a camera rig preset or a custom rig")
    if not isinstance(config, Mapping) or (
        set(config) - {"schema_version", "additional_views", "primary_view"}
        or not {"schema_version", "additional_views"} <= set(config)
    ):
        raise ValueError("rig requires schema_version, additional_views and optional primary_view")
    if config["schema_version"] != RIG_SCHEMA_VERSION:
        raise ValueError("unsupported camera rig schema_version")
    views = config["additional_views"]
    if not isinstance(views, list) or not 0 <= len(views) <= 7:
        raise ValueError("rig additional_views must contain 0 to 7 RGB cameras")
    if "primary_view" in config:
        view = config["primary_view"]
        if not isinstance(view, Mapping) or set(view) != {"mount", "fov_degrees"}:
            raise ValueError("primary_view requires mount and fov_degrees")
        result["front"] = CameraRecipe.from_mapping({**primary.as_dict(), **view})
    for view in views:
        if not isinstance(view, Mapping) or set(view) != {"id", "mount", "fov_degrees"}:
            raise ValueError("each additional view requires id, mount and fov_degrees")
        name = view["id"]
        if (
            not isinstance(name, str)
            or not _CAMERA_ID.fullmatch(name)
            or name in result
            or name == "front_teacher"
        ):
            raise ValueError(
                "camera IDs must be unique safe names; front/front_teacher are reserved"
            )
        mount = view["mount"]
        if not isinstance(mount, Mapping):
            raise ValueError("camera mount must be an object")
        # Reuse the same dimensional, timing, FOV and mount validation as scenes.
        result[name] = CameraRecipe.from_mapping(
            {
                **primary.as_dict(),
                "mount": TransformRecipe.from_mapping(mount, f"camera.{name}.mount").as_dict(),
                "fov_degrees": view["fov_degrees"],
            }
        )
    return result


def intrinsic_matrix(width: int, height: int, fov: float) -> list[list[float]]:
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    return [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]


def calibration(camera: CameraRecipe, camera_to_ego: Any) -> dict[str, Any]:
    """Store CARLA's actual rigid-mount matrix, not a guessed Euler composition."""
    result = {
        "schema_version": RIG_SCHEMA_VERSION,
        "projection": "pinhole",
        "width": camera.width,
        "height": camera.height,
        "fov_degrees": camera.fov_degrees,
        "intrinsics": intrinsic_matrix(camera.width, camera.height, camera.fov_degrees),
        "camera_to_ego": np.asarray(camera_to_ego, dtype=float).tolist(),
        "camera_and_ego_axes": CAMERA_AXES,
        "optical_axes": OPTICAL_AXES,
        "optical_from_camera": [[0, 1, 0], [0, 0, -1], [1, 0, 0]],
        "translation_unit": "metres",
        "sensor_tick_seconds": camera.sensor_tick_seconds,
    }
    validate_calibration(result)
    return result


def validate_calibration(raw: Mapping[str, Any]) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError("camera calibration must be an object")
    if (
        raw.get("schema_version") != RIG_SCHEMA_VERSION
        or raw.get("projection") != "pinhole"
        or raw.get("camera_and_ego_axes") != CAMERA_AXES
        or raw.get("optical_axes") != OPTICAL_AXES
        or raw.get("translation_unit") != "metres"
        or raw.get("optical_from_camera") != [[0, 1, 0], [0, 0, -1], [1, 0, 0]]
    ):
        raise ValueError("invalid RGB camera calibration conventions")
    for name in ("width", "height"):
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"camera {name} must be a positive integer")
    fov = float(raw["fov_degrees"])
    tick = float(raw["sensor_tick_seconds"])
    if not math.isfinite(fov) or not 0 < fov < 180 or not math.isfinite(tick) or tick <= 0:
        raise ValueError("invalid camera FOV or sensor period")
    matrix = np.asarray(raw["camera_to_ego"], dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("camera_to_ego must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
        or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5)
    ):
        raise ValueError("camera_to_ego must be a rigid transform")
    intrinsic = np.asarray(raw["intrinsics"], dtype=float)
    expected = intrinsic_matrix(raw["width"], raw["height"], fov)
    if intrinsic.shape != (3, 3) or not np.allclose(intrinsic, expected, atol=1e-6):
        raise ValueError("intrinsics disagree with image dimensions/FOV")


def validate_bundle(
    front: CarlaImageFrame,
    views: Mapping[str, CarlaImageFrame],
    calibrations: Mapping[str, Mapping[str, Any]],
) -> None:
    if not 1 <= len(views) <= 8 or views.get("front") is not front:
        raise ValueError("RGB bundle must include the exact primary front frame")
    if set(views) != set(calibrations):
        raise ValueError("every RGB view must have calibration")
    periods = set()
    for name, frame in views.items():
        if not isinstance(name, str) or not _CAMERA_ID.fullmatch(name) or name == "front_teacher":
            raise ValueError("invalid RGB camera ID")
        if frame.sensor_type != 0 or frame.frame != front.frame:
            raise ValueError(f"{name}: RGB views must share the exact CARLA frame")
        if not math.isfinite(frame.timestamp) or not math.isclose(
            frame.timestamp, front.timestamp, rel_tol=0, abs_tol=1e-6
        ):
            raise ValueError(f"{name}: RGB camera timestamps differ")
        entry = calibrations[name]
        validate_calibration(entry)
        periods.add(float(entry["sensor_tick_seconds"]))
        if (frame.width, frame.height) != (entry["width"], entry["height"]) or not math.isclose(
            frame.fov, entry["fov_degrees"], rel_tol=0, abs_tol=1e-4
        ):
            raise ValueError(f"{name}: frame dimensions/FOV disagree with calibration")
    if len(periods) != 1:
        raise ValueError("RGB cameras must share one sensor period")
