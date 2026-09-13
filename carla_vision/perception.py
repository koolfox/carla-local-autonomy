from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .bridge import CarlaImageFrame
from .contracts import Detector, PerceptionResult


@dataclass(frozen=True)
class WorkerStats:
    submitted: int
    processed: int
    dropped_before_inference: int


class PerceptionWorker:
    """One model, fair camera turns, and at most one pending frame per camera."""

    def __init__(
        self,
        detector: Detector,
        *,
        camera_ids: tuple[str, ...] = ("front",),
        sign_cameras: frozenset[str] | None = None,
    ) -> None:
        if not camera_ids or len(camera_ids) > 9 or len(set(camera_ids)) != len(camera_ids):
            raise ValueError("Use 1-9 unique camera IDs (live front plus up to 8 rig views)")
        if sign_cameras is not None and not sign_cameras <= set(camera_ids):
            raise ValueError("Sign cameras must be registered perception cameras")
        self.detector = detector
        self._camera_ids = camera_ids
        self._sign_cameras = sign_cameras
        self._condition = threading.Condition()
        self._pending: dict[str, CarlaImageFrame] = {}
        self._latest: dict[str, PerceptionResult] = {}
        self._errors: dict[str, BaseException] = {}
        self._close_error: BaseException | None = None
        self._closed = False
        self._counts = {name: [0, 0, 0] for name in camera_ids}
        safe_name = detector.name.replace(" ", "-").replace(":", "-")
        self._thread = threading.Thread(
            target=self._run,
            name=f"perception-{safe_name}",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise TimeoutError(
                "Perception is still stopping; model closes after its active inference"
            )
        if self._close_error is not None:
            raise RuntimeError(f"detector close failed: {self._close_error}")

    def __enter__(self) -> "PerceptionWorker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def submit(self, frame: CarlaImageFrame, *, camera_id: str = "front") -> None:
        with self._condition:
            self._raise_if_failed(camera_id)
            if self._closed:
                raise RuntimeError("perception worker is closed")
            self._counts[camera_id][0] += 1
            if camera_id in self._pending:
                self._counts[camera_id][2] += 1
            # Replacing an existing dict value preserves its place in the queue.
            # A busy front camera cannot jump ahead of waiting side cameras.
            self._pending[camera_id] = frame
            self._condition.notify_all()

    def latest(self, *, camera_id: str = "front") -> PerceptionResult | None:
        with self._condition:
            self._raise_if_failed(camera_id)
            return self._latest.get(camera_id)

    def stats(self, *, camera_id: str | None = None) -> WorkerStats:
        with self._condition:
            counts = self._counts.values() if camera_id is None else [self._counts[camera_id]]
            totals = [sum(column) for column in zip(*counts, strict=True)]
            return WorkerStats(
                submitted=totals[0],
                processed=totals[1],
                dropped_before_inference=totals[2],
            )

    def wait_for_result(
        self,
        after_sequence: int = -1,
        timeout: float = 5.0,
        *,
        camera_id: str = "front",
    ) -> PerceptionResult:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                self._raise_if_failed(camera_id)
                result = self._latest.get(camera_id)
                if result is not None and result.sequence > after_sequence:
                    return result
                if self._closed:
                    raise EOFError("perception worker stopped")
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("timed out waiting for detector inference")
                self._condition.wait(remaining)

    def _raise_if_failed(self, camera_id: str) -> None:
        if camera_id not in self._counts:
            raise ValueError(f"Unknown perception camera {camera_id!r}")
        if camera_id in self._errors:
            raise RuntimeError(f"perception worker failed ({camera_id}): {self._errors[camera_id]}")

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._pending and not self._closed:
                        self._condition.wait()
                    if self._closed:
                        return
                    camera_id = next(iter(self._pending))
                    frame = self._pending.pop(camera_id)
                try:
                    source_bgr = frame.bgr()
                    inference_started = time.monotonic()
                    infer = self.detector.infer
                    name = self.detector.name
                    if self._sign_cameras is not None and camera_id not in self._sign_cameras:
                        infer = getattr(self.detector, "infer_without_signs", infer)
                        name = getattr(self.detector, "detection_only_name", name)
                    detections = infer(source_bgr)
                    result = PerceptionResult(
                        sequence=frame.sequence,
                        carla_frame=frame.frame,
                        source_timestamp=frame.timestamp,
                        source_received_monotonic=frame.received_monotonic,
                        completed_monotonic=time.monotonic(),
                        detections=detections,
                        source_bgr=source_bgr,
                        detector_name=name,
                        inference_started_monotonic=inference_started,
                        source_transform=frame.transform,
                        source_fov=frame.fov,
                    )
                except Exception as error:
                    with self._condition:
                        self._errors[camera_id] = error
                        self._pending.pop(camera_id, None)
                        self._condition.notify_all()
                    continue
                with self._condition:
                    self._latest[camera_id] = result
                    self._counts[camera_id][1] += 1
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._errors.update({name: exc for name in self._camera_ids})
                self._condition.notify_all()
        finally:
            try:
                self.detector.close()
            except BaseException as error:
                self._close_error = error
