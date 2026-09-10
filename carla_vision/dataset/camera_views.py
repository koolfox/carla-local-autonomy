"""Read and verify the additive RGB-view records in an existing dataset release."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import cv2

from .camera_rig import validate_calibration


def view_path(root: Path, relative: str) -> Path:
    """Accept only regular files inside the dataset, including on Windows."""
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("invalid dataset artifact path")
    parts = relative.split("/")
    if PurePosixPath(relative).is_absolute() or any(p in {"", ".", ".."} for p in parts):
        raise ValueError("dataset artifact path must be safely relative")
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("dataset artifacts must not contain symlinks")
    current.resolve().relative_to(root.resolve())
    if not current.is_file():
        raise ValueError(f"missing dataset artifact: {relative}")
    return current


def verify_camera_views(
    root: Path,
    sample: Mapping[str, Any],
    metadata: Mapping[str, Any],
    camera_ids: Any,
    checksums: Mapping[str, str],
) -> list[str]:
    """Validate every view against this sample, its calibration and checksum index."""
    views = metadata.get("rgb_views")
    if camera_ids is None and views is None:
        return []  # Legacy single-camera format stays supported.
    if (
        not isinstance(camera_ids, list)
        or not 1 <= len(camera_ids) <= 8
        or any(not isinstance(name, str) for name in camera_ids)
        or len(set(camera_ids)) != len(camera_ids)
        or "front" not in camera_ids
        or not isinstance(views, Mapping)
        or set(views) != set(camera_ids)
    ):
        return ["RGB views do not match the declared camera rig"]
    errors = []
    image_paths: set[str] = set()
    periods: set[float] = set()
    for name, view in views.items():
        try:
            if not isinstance(view, Mapping):
                raise ValueError("RGB view must be an object")
            if type(view["carla_frame"]) is not int or view["carla_frame"] != sample["carla_frame"]:
                raise ValueError("RGB view frame differs from sample")
            timestamp = float(view["timestamp_seconds"])
            if not math.isfinite(timestamp) or not math.isclose(
                timestamp, sample["source_timestamp"], rel_tol=0, abs_tol=1e-6
            ):
                raise ValueError("RGB view timestamp differs from sample")
            spec = view["calibration"]
            validate_calibration(spec)
            periods.add(float(spec["sensor_tick_seconds"]))
            artifact = view["image"]
            relative = artifact["path"]
            path = view_path(root, relative)
            if relative in image_paths:
                raise ValueError("different cameras reference the same image")
            image_paths.add(relative)
            if name == "front" and artifact != sample["rgb"]:
                raise ValueError("front view differs from primary RGB artifact")
            if name != "front" and not relative.startswith(f"cameras/{name}/{sample['split']}/"):
                raise ValueError("RGB view path differs from its camera/split")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != artifact["sha256"] or checksums.get(relative) != digest:
                raise ValueError("RGB view checksum mismatch or not indexed")
            if path.stat().st_size != artifact["size_bytes"]:
                raise ValueError("RGB view size mismatch")
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (spec["height"], spec["width"]):
                raise ValueError("RGB image cannot be decoded or has wrong dimensions")
        except (KeyError, TypeError, ValueError, OSError, OverflowError) as error:
            errors.append(f"camera {name}: {error}")
    if len(periods) > 1:
        errors.append("RGB cameras do not share one sensor period")
    return errors
