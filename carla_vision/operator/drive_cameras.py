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

import cv2

from ..dataset.camera_rig import intrinsic_matrix, resolve_camera_rig
from ..scenarios.contracts import CameraRecipe, TransformRecipe
from ..video import BrowserVideoWriter
from .camera_perception import CameraPerception
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
        self, client: Any, scene: Any, name: str, root: Path, width: int, height: int, fps: float,
        on_frame: Any = None,
    ) -> None:
        self.name = name
        self.on_frame = on_frame
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
                    if self.on_frame:
                        self.on_frame(frame)
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
        self.perception_views: dict[str, CameraPerception] = {}
        self.perception_errors: dict[str, str] = {}

    def start(
        self, client: Any, scene: Any, config: Any, *, perception: Any = None,
        publish: Any = None, hud_factory: Any = None,
    ) -> None:
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
            "model_input": bool(getattr(config, "recording_perception", None)),
            "perception_views": dict(getattr(config, "recording_perception", None) or {}),
            "overlay_timing": "sampled inference frames at video_fps; use detections timestamps for elapsed time",
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
            mode = (getattr(config, "recording_perception", None) or {}).get(name)
            if mode and perception is not None:
                try:
                    self.perception_views[name] = CameraPerception(
                        perception, name, self.root, width=config.width, height=config.height,
                        fps=config.recording_rig_fps, publish=publish,
                        hud=hud_factory(mode) if hud_factory else {},
                    )
                except Exception as error:
                    self.perception_errors[name] = f"Overlay setup failed: {error}"

            def on_frame(frame: Any, name: str = name) -> None:
                if publish:
                    jpeg = getattr(frame, "jpeg", None)
                    if not isinstance(jpeg, bytes):
                        ok, encoded = cv2.imencode(".jpg", frame.bgr())
                        if not ok:
                            raise RuntimeError("Could not encode rig preview")
                        jpeg = encoded.tobytes()
                    publish(f"rig:{name}:raw", frame.sequence, jpeg)
                observer = self.perception_views.get(name)
                if observer:
                    observer.submit(frame)

            self.views.append(
                _ViewRecording(
                    client,
                    scene,
                    name,
                    self.root,
                    config.width,
                    config.height,
                    config.recording_rig_fps,
                    on_frame=on_frame,
                )
            )

    def close(self, tracker: Any) -> list[str]:
        self.request_stop()
        results = {view.name: view.close() for view in self.views}
        self.metadata["results"] = results
        self.metadata["perception_results"] = {
            name: observer.close() for name, observer in self.perception_views.items()
        }
        self.metadata["perception_setup_errors"] = dict(self.perception_errors)
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
        for name, observer in self.perception_views.items():
            for path, role in ((observer.video, "drive_camera_overlay_video"),
                               (observer.index, "drive_camera_detections")):
                if path.is_file():
                    tracker.register_artifact(path, role=role, metadata={"camera_id": name})
        return errors

    def request_stop(self) -> None:
        for view in self.views:
            view.stop.set()
        for observer in self.perception_views.values():
            observer.stop.set()

    def snapshot(self) -> dict:
        modes = self.metadata.get("perception_views", {})
        return {view.name: {
            "mode": modes.get(view.name, "off"), "raw_error": view.error,
            "perception_error": self.perception_errors.get(view.name) or (
                self.perception_views[view.name].error if view.name in self.perception_views else None),
        } for view in self.views}

    def check_health(self) -> None:
        for view in self.views:
            if view.error:
                raise RuntimeError(f"Recording camera {view.name}: {view.error}")
