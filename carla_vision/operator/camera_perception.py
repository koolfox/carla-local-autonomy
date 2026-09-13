"""Exact-frame, non-actuating perception evidence for one selected Drive camera."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

from ..display import OverlayRenderer
from ..perception import PerceptionWorker
from ..video import BrowserVideoWriter


class CameraPerception:
    """Consume a shared scheduler's results; never own or reload a model."""

    def __init__(
        self,
        worker: PerceptionWorker,
        name: str,
        root: Path,
        *,
        width: int,
        height: int,
        fps: float,
        publish: Any,
        hud: dict[str, str],
        show_rejection_status: bool = False,
    ) -> None:
        self.worker, self.name = worker, name
        self.camera_id = f"rig:{name}"
        self.publish, self.hud = publish, {**hud, "CAMERA": name}
        self.show_rejection_status = show_rejection_status
        self.video = root / f"{name}.overlay.mp4"
        self.index = root / f"{name}.detections.jsonl"
        self.error: str | None = None
        self.written = 0
        self.stop = threading.Event()
        self.writer = BrowserVideoWriter(self.video, (width, height), fps)
        self.thread = threading.Thread(target=self._run, name=f"camera-overlay-{name}", daemon=True)
        self.thread.start()

    def submit(self, frame: Any) -> None:
        if self.stop.is_set() or self.error:
            return
        try:
            self.worker.submit(frame, camera_id=self.camera_id)
        except Exception as error:
            self.error = str(error)
            self.stop.set()

    def _run(self) -> None:
        sequence = -1
        renderer = OverlayRenderer(stale_after_seconds=2, show_rejection_status=self.show_rejection_status)
        try:
            with self.index.open("x", encoding="utf-8") as stream:
                while not self.stop.is_set():
                    try:
                        result = self.worker.wait_for_result(
                            sequence, timeout=0.2, camera_id=self.camera_id
                        )
                    except TimeoutError:
                        continue
                    if self.stop.is_set():
                        break
                    overlay = renderer.render(result, hud=self.hud)
                    ok, jpeg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    if not ok:
                        raise RuntimeError("Could not encode camera overlay")
                    if self.publish:
                        self.publish(f"rig:{self.name}:overlay", result.sequence, jpeg.tobytes())
                    self.writer.write(overlay)
                    stream.write(
                        json.dumps(
                            {
                                "camera_id": self.name,
                                "video_frame": self.written,
                                "sequence": result.sequence,
                                "carla_frame": result.carla_frame,
                                "timestamp": result.source_timestamp,
                                "camera_world_transform": result.source_transform,
                                "fov": result.source_fov,
                                "detector": result.detector_name,
                                "inference_seconds": result.inference_seconds,
                                "model_output_actuated": False,
                                "detections": [asdict(item) for item in result.detections],
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    sequence = result.sequence
                    self.written += 1
        except Exception as error:
            if not self.stop.is_set():
                self.error = f"{type(error).__name__}: {error}"
        finally:
            try:
                self.writer.release()
            except Exception as error:
                self.error = f"overlay finalization failed: {error}"

    def close(self) -> dict[str, Any]:
        self.stop.set()
        self.thread.join(timeout=20)
        if self.thread.is_alive():
            self.writer.abort()
            self.thread.join(timeout=2)
            self.error = "overlay encoder did not finish in time"
        return {
            "frames_written": self.written,
            "error": self.error,
            "scheduler": asdict(self.worker.stats(camera_id=self.camera_id)),
        }
