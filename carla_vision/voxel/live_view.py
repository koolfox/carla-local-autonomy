"""Optional RGB observer with a separate, read-only teacher display annotation.

The predictor sees only RGB and camera calibration. An optional callback supplies
map waypoints for display/export *after* prediction, never for inference/control.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from .contracts import VoxelGridSpec
from .rendering import project_waypoint_teacher, render_voxel_overlay, render_voxel_view


@dataclass(frozen=True)
class VoxelViewFrame:
    sequence: int
    source_frame: int
    source_timestamp: float
    received_monotonic: float
    latency_ms: float
    image_bgr: np.ndarray
    jpeg: bytes
    voxels: Any
    overlay_bgr: np.ndarray
    overlay_jpeg: bytes
    source_bgr: np.ndarray
    waypoint_teacher: dict[str, Any] | None
    waypoint_status: str
    waypoint_error: str | None

    def record(self) -> dict[str, Any]:
        return {**self.voxels.metadata, "sequence": self.sequence, "source_frame": self.source_frame,
                "source_timestamp": self.source_timestamp, "latency_ms": self.latency_ms,
                "actuated": False, "input": "rgb_only",
                "waypoint_status": self.waypoint_status,
                "waypoint_error": self.waypoint_error,
                "waypoint_teacher": self.waypoint_teacher}


class VoxelViewWorker:
    """Latest-only, rate-limited observer; loading and failures never block Drive."""

    def __init__(self, *, device: str, predictor_factory: Callable[[], Any] | None = None,
                 max_fps: float = 2.0,
                 workspace: str | None = None,
                 waypoint_provider: Callable[[Any], dict[str, Any]] | None = None) -> None:
        if not np.isfinite(max_fps) or not 0 < max_fps <= 10:
            raise ValueError("voxel observer max_fps must be in (0, 10]")
        self._condition = threading.Condition()
        self._closed = False
        self._pending: Any = None
        self._latest: VoxelViewFrame | None = None
        self._status = "loading"
        self._error: str | None = None
        self._dropped = 0
        self._processed = 0
        self._period = 1.0 / max_fps
        self._device = device
        self._workspace = workspace
        self._factory = predictor_factory
        self._waypoint_provider = waypoint_provider
        self._thread = threading.Thread(target=self._run, name="rgb-voxel-view", daemon=True)
        self._thread.start()

    def submit(self, frame: Any) -> None:
        with self._condition:
            if self._closed or self._status == "failed":
                return
            if self._pending is not None:
                self._dropped += 1
            self._pending = frame
            self._condition.notify_all()

    def latest(self) -> VoxelViewFrame | None:
        with self._condition:
            return self._latest

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            result = self._latest
            return {"enabled": True, "status": self._status, "error": self._error,
                    "actuated": False, "input": "rgb_only", "processed": self._processed,
                    "dropped": self._dropped, "sequence": -1 if result is None else result.sequence,
                    "source_frame": None if result is None else result.source_frame,
                    "latency_ms": None if result is None else result.latency_ms,
                    "waypoint_status": result.waypoint_status if result else (
                        "pending" if self._waypoint_provider is not None else "unavailable"),
                    "waypoint_error": None if result is None else result.waypoint_error,
                    "waypoint_source": None if result is None or result.waypoint_teacher is None
                    else result.waypoint_teacher.get("source"),
                    "age_seconds": None if result is None else max(
                        0.0, time.monotonic() - result.received_monotonic)}

    def close(self) -> None:
        # ML/download calls cannot be interrupted safely. The observer publishes
        # nothing after close; its optional read-only provider has a short timeout.
        with self._condition:
            self._closed = True
            self._pending = None
            self._status = "stopped"
            self._condition.notify_all()
        self._thread.join(timeout=0.1)

    def _run(self) -> None:
        try:
            if self._factory is None:
                from .rgb_depth import RgbDepthVoxelPredictor
                predictor = RgbDepthVoxelPredictor(
                    device=self._device,
                    workspace=self._workspace,
                    spec=VoxelGridSpec(x_max=35, y_min=-17.5, y_max=17.5,
                                       z_min=-5, z_max=5, resolution=0.5),
                    pixel_stride=12, max_rays=2048,
                )
            else:
                predictor = self._factory()
            predictor.load()
            next_at = 0.0
            while True:
                with self._condition:
                    while not self._closed:
                        wait = next_at - time.monotonic()
                        if self._pending is not None and wait <= 0:
                            break
                        self._condition.wait(timeout=max(0.001, wait) if wait > 0 else None)
                    if self._closed:
                        return
                    frame, self._pending = self._pending, None
                started = time.monotonic()
                source = frame.bgr()
                voxels = predictor.predict(cv2.cvtColor(source, cv2.COLOR_BGR2RGB), frame.fov,
                                           frame=frame.frame, timestamp=frame.timestamp,
                                           sequence=frame.sequence)
                with self._condition:
                    if self._closed:
                        return
                # Teacher geometry is obtained only after RGB-only inference.
                # Missing/busy/older bridges omit the annotation, not the view.
                teacher = None
                waypoint_error = None
                waypoint_status = "unavailable"
                if self._waypoint_provider is not None:
                    try:
                        teacher = project_waypoint_teacher(
                            self._waypoint_provider(frame), transform=frame.transform,
                            width=source.shape[1], height=source.shape[0],
                            fov=frame.fov, source_frame=frame.frame,
                        )
                        waypoint_status = "available" if len(teacher["camera_points_xyz"]) > 1 else "empty"
                    except Exception as error:
                        waypoint_status = "error"
                        waypoint_error = f"{type(error).__name__}: {error}"
                canvas = render_voxel_view(source, voxels, frame=frame.frame, teacher=teacher)
                overlay = render_voxel_overlay(source, voxels, frame=frame.frame,
                                               fov=frame.fov, teacher=teacher)
                result = VoxelViewFrame(frame.sequence, frame.frame, frame.timestamp,
                                       frame.received_monotonic,
                                       (time.monotonic() - started) * 1000,
                                       canvas, _jpeg(canvas), voxels, overlay, _jpeg(overlay),
                                       source, teacher, waypoint_status, waypoint_error)
                with self._condition:
                    if self._closed:
                        return
                    self._latest = result
                    self._processed += 1
                    self._status = "running"
                next_at = started + self._period
        except Exception as error:
            with self._condition:
                if not self._closed:
                    self._status = "failed"
                    self._error = f"{type(error).__name__}: {error}"
                self._pending = None


def _jpeg(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("could not encode voxel view")
    return encoded.tobytes()
