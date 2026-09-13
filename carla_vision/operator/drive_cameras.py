"""Optional RGB rig recording for normal Drive; never owns vehicle control.

One bounded newest-frame stream and encoder thread per view. These are review
videos, not synchronized/lossless teacher datasets: the index records exactly
which simulator frames made it into each MP4.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from ..dataset.camera_rig import intrinsic_matrix, resolve_camera_rig
from ..scenarios.contracts import CameraRecipe, TransformRecipe
from ..video import BrowserVideoWriter
from .world_worker_client import WorldWorkerCameraStream


def recording_views(rig: dict, *, width: int, height: int, fps: float, fov: float) -> dict:
    primary = CameraRecipe(
        width, height, fov, 1 / fps, 2.2, True, TransformRecipe(1.5, 0, 1.7, 0, 0, 0)
    )
    recipes = resolve_camera_rig(primary, config=rig)
    for recipe in recipes.values():
        if any(abs(getattr(recipe.mount, axis)) > 100 for axis in ("x", "y", "z")):
            raise ValueError("recording camera positions must be within 100 metres of ego")
    return {
        name: {"mount": recipe.mount.as_dict(), "fov_degrees": recipe.fov_degrees}
        for name, recipe in recipes.items()
    }


class _ViewRecording:
    def __init__(
        self, client: Any, scene: Any, name: str, root: Path, width: int, height: int, fps: float
    ) -> None:
        self.name = name
        self.video = root / f"{name}.mp4"
        self.index = root / f"{name}.frames.jsonl"
        self.stop = threading.Event()
        self.error: str | None = None
        self.written = 0
        self.skipped = 0
        self.stream = WorldWorkerCameraStream(client, scene, timeout=5, view=name)
        try:
            # Encoder availability is checked before Drive starts moving.
            self.writer = BrowserVideoWriter(self.video, (width, height), fps)
        except BaseException:
            self.stream.close()
            raise
        self.thread = threading.Thread(target=self._run, name=f"drive-record-{name}", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        sequence = -1
        try:
            with self.index.open("x", encoding="utf-8") as index:
                while not self.stop.is_set():
                    frame = self.stream.wait_for_frame(sequence, timeout=10)
                    if self.stop.is_set():
                        break
                    self.writer.write(frame.bgr())
                    index.write(
                        json.dumps(
                            {
                                "video_frame": self.written,
                                "carla_frame": frame.frame,
                                "timestamp": frame.timestamp,
                                "sequence": frame.sequence,
                                "camera_world_transform": frame.transform,
                            }
                        )
                        + "\n"
                    )
                    if sequence >= 0:
                        self.skipped += max(0, frame.sequence - sequence - 1)
                    sequence = frame.sequence
                    self.written += 1
        except Exception as error:
            if not self.stop.is_set():
                self.error = f"{type(error).__name__}: {error}"
        finally:
            try:
                self.writer.release()
            except Exception as error:
                self.error = f"video finalization failed: {error}"

    def close(self) -> dict:
        self.stop.set()
        self.stream.close()
        self.thread.join(timeout=20)
        if self.thread.is_alive():
            self.writer.abort()
            self.thread.join(timeout=2)
            self.error = "recording encoder did not finish in time"
        if not self.written and self.error is None:
            self.error = "no camera frames recorded"
        return {
            "frames_written": self.written,
            "relay_frames_skipped": self.skipped,
            "error": self.error,
        }


class DriveCameraRecording:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.views: list[_ViewRecording] = []
        self.metadata: dict = {}

    def start(self, client: Any, scene: Any, config: Any) -> None:
        self.root.mkdir(parents=True, exist_ok=False)
        views = recording_views(
            config.recording_rig,
            width=config.width,
            height=config.height,
            fps=config.recording_rig_fps,
            fov=config.camera_fov,
        )
        response = client.start_recording_cameras(
            scene,
            views=views,
            width=config.width,
            height=config.height,
            fps=config.recording_rig_fps,
        )
        self.metadata = {
            "schema_version": "1.0",
            "kind": "drive_rgb_review_rig",
            "synchronized": False,
            "lossless": False,
            "model_input": False,
            "coordinate_frame": "ego_x_forward_y_right_z_up",
            "width": config.width,
            "height": config.height,
            "video_fps": config.recording_rig_fps,
            "views": views,
            "camera_to_ego": response.get("camera_to_ego", {}),
            "intrinsics": {
                name: intrinsic_matrix(config.width, config.height, view["fov_degrees"])
                for name, view in views.items()
            },
        }
        for name in views:
            self.views.append(
                _ViewRecording(
                    client,
                    scene,
                    name,
                    self.root,
                    config.width,
                    config.height,
                    config.recording_rig_fps,
                )
            )

    def close(self, tracker: Any) -> list[str]:
        self.request_stop()
        results = {view.name: view.close() for view in self.views}
        self.metadata["results"] = results
        errors = [
            f"camera {name}: {result['error']}"
            for name, result in results.items()
            if result["error"]
        ]
        manifest = self.root / "rig.json"
        manifest.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
        tracker.register_artifact(manifest, role="drive_camera_rig")
        for view in self.views:
            for path, role in (
                (view.video, "drive_camera_video"),
                (view.index, "drive_camera_frame_index"),
            ):
                if path.is_file():
                    tracker.register_artifact(path, role=role, metadata={"camera_id": view.name})
        return errors

    def request_stop(self) -> None:
        for view in self.views:
            view.stop.set()

    def check_health(self) -> None:
        for view in self.views:
            if view.error:
                raise RuntimeError(f"Recording camera {view.name}: {view.error}")
