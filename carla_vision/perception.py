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
    """Run any detector on the newest frame without building a stale backlog."""

    def __init__(self, detector: Detector) -> None:
        self.detector = detector
        self._condition = threading.Condition()
        self._pending: CarlaImageFrame | None = None
        self._latest: PerceptionResult | None = None
        self._error: BaseException | None = None
        self._closed = False
        self._submitted = 0
        self._processed = 0
        self._dropped = 0
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
        self.detector.close()

    def __enter__(self) -> "PerceptionWorker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def submit(self, frame: CarlaImageFrame) -> None:
        with self._condition:
            self._raise_if_failed()
            self._submitted += 1
            if self._pending is not None:
                self._dropped += 1
            self._pending = frame
            self._condition.notify_all()

    def latest(self) -> PerceptionResult | None:
        with self._condition:
            self._raise_if_failed()
            return self._latest

    def stats(self) -> WorkerStats:
        with self._condition:
            return WorkerStats(
                submitted=self._submitted,
                processed=self._processed,
                dropped_before_inference=self._dropped,
            )

    def wait_for_result(
        self,
        after_sequence: int = -1,
        timeout: float = 5.0,
    ) -> PerceptionResult:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                self._raise_if_failed()
                if self._latest is not None and self._latest.sequence > after_sequence:
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("timed out waiting for detector inference")
                self._condition.wait(remaining)

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"perception worker failed: {self._error}")

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while self._pending is None and not self._closed:
                        self._condition.wait()
                    if self._closed:
                        return
                    frame = self._pending
                    self._pending = None
                if frame is None:
                    continue
                source_bgr = frame.bgr()
                inference_started = time.monotonic()
                detections = self.detector.infer(source_bgr)
                result = PerceptionResult(
                    sequence=frame.sequence,
                    carla_frame=frame.frame,
                    source_timestamp=frame.timestamp,
                    source_received_monotonic=frame.received_monotonic,
                    completed_monotonic=time.monotonic(),
                    detections=detections,
                    source_bgr=source_bgr,
                    detector_name=self.detector.name,
                    inference_started_monotonic=inference_started,
                    source_transform=frame.transform,
                    source_fov=frame.fov,
                )
                with self._condition:
                    self._latest = result
                    self._processed += 1
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
