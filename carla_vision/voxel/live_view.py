"""Optional RGB spatial observer. No CARLA client, control, or world-state access."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from .contracts import VoxelGridSpec


def render_voxel_view(source_bgr: np.ndarray, result: Any, *, frame: int) -> np.ndarray:
    """Show the exact source image and actual occupied cells in a camera-local view."""
    image = np.full((720, 1280, 3), (26, 29, 31), dtype=np.uint8)

    def label(text: str, xy: tuple[int, int], size: float = 0.55) -> None:
        cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, size,
                    (218, 223, 226), 1, cv2.LINE_AA)

    label("RGB VOXEL / OBSERVER ONLY", (24, 34), 0.7)
    label(f"Source frame {frame}  |  predicted depth, not ground truth", (24, 61))
    h, w = source_bgr.shape[:2]
    scale = min(420 / w, 270 / h)
    thumb = cv2.resize(source_bgr, (round(w * scale), round(h * scale)))
    image[92:92 + thumb.shape[0], 24:24 + thumb.shape[1]] = thumb
    label("Same RGB frame used by the model", (24, 385))
    for index, text in enumerate((
        "Origin: front camera", "X forward / Y right / Z up",
        "Distances are estimates, not calibrated.",
        "Unseen / occluded space stays unknown.",
        "Surfaces are not road or lane labels.",
        "No path or vehicle control is generated.",
    )):
        label(text, (24, 428 + 29 * index))

    spec = result.spec
    # Fixed metric projection: the scene cannot rescale misleadingly per frame.
    scale = min(16.0, 520.0 / (spec.x_max - spec.x_min))

    def project(points: np.ndarray) -> np.ndarray:
        x, y, z = np.asarray(points).T
        return np.column_stack((935 + (y - 0.28 * x) * scale,
                                613 - (0.76 * x + z) * scale)).astype(np.int32)

    for x in range(0, int(spec.x_max) + 1, 5):
        a, b = project(np.array([[x, spec.y_min, 0], [x, spec.y_max, 0]]))
        cv2.line(image, tuple(a), tuple(b), (50, 55, 58), 1, cv2.LINE_AA)
        label(f"{x}m", (int(b[0]) + 5, int(b[1])), 0.4)
    for y in range(int(spec.y_min), int(spec.y_max) + 1, 5):
        a, b = project(np.array([[0, y, 0], [spec.x_max, y, 0]]))
        cv2.line(image, tuple(a), tuple(b), (50, 55, 58), 1, cv2.LINE_AA)

    indices = np.argwhere(result.occupancy == 1)
    # Rendering all occupied cells is bounded by the modest fixed live grid.
    points = np.column_stack((spec.x_min + (indices[:, 2] + 0.5) * spec.resolution,
                              spec.y_min + (indices[:, 1] + 0.5) * spec.resolution,
                              spec.z_min + (indices[:, 0] + 0.5) * spec.resolution))
    points = points[np.argsort(-points[:, 0], kind="stable")]
    r = spec.resolution / 2
    corners = np.array([[-r, -r, -r], [-r, r, -r], [r, r, -r], [r, -r, -r],
                        [-r, -r, r], [-r, r, r], [r, r, r], [r, -r, r]])
    for point in points:
        box = project(point + corners)
        if box[:, 0].max() < 465 or box[:, 0].min() >= 1280:
            continue
        # Height shading is geometry only, never a fabricated semantic class.
        lift = int(np.clip((point[2] - spec.z_min) * 12, 0, 90))
        for face, color in (([0, 1, 5, 4], (100 + lift, 92 + lift, 64 + lift)),
                            ([1, 2, 6, 5], (80 + lift, 74 + lift, 52 + lift)),
                            ([4, 5, 6, 7], (135 + lift, 121 + lift, 80 + lift))):
            cv2.fillConvexPoly(image, box[face], color, cv2.LINE_AA)

    origin, forward, right, up = project(np.array([[0, 0, 0], [7, 0, 0],
                                                  [0, 7, 0], [0, 0, 4]]))
    for end, name in ((forward, "X front"), (right, "Y right"), (up, "Z up")):
        cv2.arrowedLine(image, tuple(origin), tuple(end), (233, 235, 237), 2,
                       cv2.LINE_AA, tipLength=0.12)
        label(name, (int(end[0]) + 8, int(end[1]) - 8), 0.45)
    label("Behind the front camera: unknown", (740, 676), 0.5)
    return image


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

    def record(self) -> dict[str, Any]:
        return {"sequence": self.sequence, "source_frame": self.source_frame,
                "source_timestamp": self.source_timestamp, "latency_ms": self.latency_ms,
                "actuated": False, "input": "rgb_only", **self.voxels.metadata}


class VoxelViewWorker:
    """Latest-only, rate-limited observer; loading and failures never block Drive."""

    def __init__(self, *, device: str, predictor_factory: Callable[[], Any] | None = None,
                 max_fps: float = 2.0) -> None:
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
        self._factory = predictor_factory
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
                    "age_seconds": None if result is None else max(
                        0.0, time.monotonic() - result.received_monotonic)}

    def close(self) -> None:
        # Native ML/download calls cannot be interrupted safely. The daemon owns
        # only its model, publishes nothing after close, and never touches CARLA/files.
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
                canvas = render_voxel_view(source, voxels, frame=frame.frame)
                ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not ok:
                    raise RuntimeError("could not encode voxel view")
                result = VoxelViewFrame(frame.sequence, frame.frame, frame.timestamp,
                                       frame.received_monotonic,
                                       (time.monotonic() - started) * 1000,
                                       canvas, encoded.tobytes(), voxels)
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
