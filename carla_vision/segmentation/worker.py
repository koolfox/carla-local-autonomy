"""Newest-frame asynchronous worker for RGB road segmentation."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..contracts import PerceptionResult
from .contracts import RoadSegmenter, SegmentationMetadata, SegmentationResult


class _FrameSource(Protocol):
    sequence: int
    frame: int
    timestamp: float
    received_monotonic: float
    transform: tuple[float, float, float, float, float, float] | None
    fov: float | None

    def bgr(self) -> np.ndarray: ...


@dataclass(frozen=True)
class SegmentationFrameInput:
    """An exact RGB frame submitted without CARLA or detector dependencies."""

    sequence: int
    frame: int
    timestamp: float
    received_monotonic: float
    image_bgr: np.ndarray
    transform: tuple[float, float, float, float, float, float] | None = None
    fov: float | None = None
    perception: PerceptionResult | None = None

    @classmethod
    def from_perception(cls, result: PerceptionResult) -> "SegmentationFrameInput":
        return cls(
            sequence=result.sequence,
            frame=result.carla_frame,
            timestamp=result.source_timestamp,
            received_monotonic=result.source_received_monotonic,
            image_bgr=result.source_bgr,
            transform=result.source_transform,
            fov=result.source_fov,
            perception=result,
        )

    def bgr(self) -> np.ndarray:
        return self.image_bgr


@dataclass(frozen=True)
class SegmentationFrameResult:
    """A model result tied to the exact RGB frame used for inference."""

    sequence: int
    carla_frame: int
    source_timestamp: float
    source_received_monotonic: float
    inference_started_monotonic: float
    completed_monotonic: float
    source_bgr: np.ndarray
    segmentation: SegmentationResult
    segmenter_name: str
    source_transform: tuple[float, float, float, float, float, float] | None = None
    source_fov: float | None = None
    perception: PerceptionResult | None = None

    def __post_init__(self) -> None:
        if self.sequence < 0 or self.carla_frame < 0:
            raise ValueError("segmentation frame identifiers must be non-negative")
        if not math.isfinite(self.source_timestamp):
            raise ValueError("segmentation source timestamp must be finite")
        if not (
            self.source_received_monotonic
            <= self.inference_started_monotonic
            <= self.completed_monotonic
        ):
            raise ValueError("segmentation inference timing is inconsistent")
        if (
            not isinstance(self.source_bgr, np.ndarray)
            or self.source_bgr.dtype != np.uint8
            or self.source_bgr.ndim != 3
            or self.source_bgr.shape[2] != 3
        ):
            raise ValueError("segmentation source_bgr must be uint8 HxWx3")
        if self.segmentation.class_ids.shape != self.source_bgr.shape[:2]:
            raise ValueError("segmentation mask must match its exact RGB source frame")
        if not self.segmenter_name.strip() or self.segmenter_name != self.segmentation.model_name:
            raise ValueError("segmenter identity must match the segmentation result")
        if self.perception is not None and (
            self.perception.sequence != self.sequence
            or self.perception.carla_frame != self.carla_frame
        ):
            raise ValueError("carried detector result must match the segmentation frame")

    @property
    def inference_seconds(self) -> float:
        return self.completed_monotonic - self.source_received_monotonic

    @property
    def model_inference_seconds(self) -> float:
        return self.completed_monotonic - self.inference_started_monotonic


@dataclass(frozen=True)
class SegmentationWorkerStats:
    submitted: int
    processed: int
    dropped_before_inference: int
    failed_during_inference: int = 0


class SegmentationWorker:
    """Run a segmenter on only the newest pending frame, never a stale queue."""

    def __init__(
        self,
        segmenter: RoadSegmenter,
        *,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self.segmenter = segmenter
        self._on_error = on_error
        self._condition = threading.Condition()
        self._pending: _FrameSource | None = None
        self._latest: SegmentationFrameResult | None = None
        self._error: BaseException | None = None
        self._closed = False
        self._submitted = 0
        self._processed = 0
        self._dropped = 0
        self._failed = 0
        safe_name = segmenter.name.replace(" ", "-").replace(":", "-")
        self._thread = threading.Thread(
            target=self._run,
            name=f"segmentation-{safe_name}",
            daemon=True,
        )
        self._thread.start()

    def submit(self, frame: _FrameSource) -> None:
        with self._condition:
            self._raise_if_failed()
            if self._closed:
                raise RuntimeError("segmentation worker is closed")
            self._submitted += 1
            if self._pending is not None:
                self._dropped += 1
            self._pending = frame
            self._condition.notify_all()

    def latest(self) -> SegmentationFrameResult | None:
        with self._condition:
            self._raise_if_failed()
            return self._latest

    def stats(self) -> SegmentationWorkerStats:
        with self._condition:
            return SegmentationWorkerStats(
                submitted=self._submitted,
                processed=self._processed,
                dropped_before_inference=self._dropped,
                failed_during_inference=self._failed,
            )

    def wait_for_result(
        self,
        *,
        after_sequence: int = -1,
        timeout: float = 5.0,
    ) -> SegmentationFrameResult:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                self._raise_if_failed()
                if self._latest is not None and self._latest.sequence > after_sequence:
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("timed out waiting for road segmentation")
                self._condition.wait(remaining)

    def close(self) -> None:
        with self._condition:
            if not self._closed:
                self._closed = True
                if self._pending is not None:
                    # A newest-only frame that never reaches inference is still
                    # a real drop and must reconcile final lifecycle evidence.
                    self._pending = None
                    self._dropped += 1
                self._condition.notify_all()
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            raise TimeoutError(
                "segmentation inference did not stop within 10 seconds; "
                "the worker thread retains model ownership until it exits"
            )
        with self._condition:
            self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"segmentation worker failed: {self._error}")

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
                started = time.monotonic()
                segmentation = self.segmenter.infer(source_bgr)
                result = SegmentationFrameResult(
                    sequence=frame.sequence,
                    carla_frame=frame.frame,
                    source_timestamp=frame.timestamp,
                    source_received_monotonic=frame.received_monotonic,
                    inference_started_monotonic=started,
                    completed_monotonic=time.monotonic(),
                    source_bgr=source_bgr,
                    segmentation=segmentation,
                    segmenter_name=self.segmenter.name,
                    source_transform=getattr(frame, "transform", None),
                    source_fov=getattr(frame, "fov", None),
                    perception=getattr(frame, "perception", None),
                )
                with self._condition:
                    self._latest = result
                    self._processed += 1
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._failed += 1
            self._set_error(error)
        finally:
            try:
                self.segmenter.close()
            except BaseException as error:
                self._set_error(error)

    def _set_error(self, error: BaseException) -> None:
        callback: Callable[[BaseException], None] | None = None
        with self._condition:
            if self._error is None:
                self._error = error
                callback = self._on_error
            self._condition.notify_all()
        if callback is not None:
            try:
                callback(error)
            except Exception:
                # Error reporting must never hide or replace the model failure.
                pass


class AsyncSegmentationRuntime:
    """Load a segmenter off-thread, then expose its newest-frame worker."""

    def __init__(self, loader: Callable[[], RoadSegmenter]) -> None:
        self._loader = loader
        self._condition = threading.Condition()
        self._state = "initializing"
        self._error: str | None = None
        self._failure_phase: str | None = None
        self._worker: SegmentationWorker | None = None
        self._final_stats: SegmentationWorkerStats | None = None
        self._metadata: SegmentationMetadata | None = None
        self._close_requested = False
        self._thread = threading.Thread(
            target=self._initialize,
            name="road-segmentation-initialize",
            daemon=True,
        )
        self._thread.start()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "state": self._state,
                "error": self._error,
                "failure_phase": self._failure_phase,
                "name": None if self._metadata is None else self._metadata.name,
                "metadata": None if self._metadata is None else self._metadata.as_dict(),
            }

    def ready(self) -> bool:
        with self._condition:
            return self._state == "ready" and self._worker is not None

    def submit(self, frame: _FrameSource) -> bool:
        with self._condition:
            state, worker = self._state, self._worker
            error = self._error
        if state == "initializing":
            return False
        if state == "failed":
            raise RuntimeError(self._failure_message(error))
        if state != "ready" or worker is None:
            raise RuntimeError("road segmentation runtime is closed")
        worker.submit(frame)
        return True

    def latest(self) -> SegmentationFrameResult | None:
        with self._condition:
            state, worker = self._state, self._worker
            error = self._error
        if state == "initializing":
            return None
        if state == "failed":
            raise RuntimeError(self._failure_message(error))
        if state != "ready" or worker is None:
            return None
        return worker.latest()

    def stats(self) -> SegmentationWorkerStats | None:
        with self._condition:
            worker = self._worker
            final_stats = self._final_stats
        return worker.stats() if worker is not None else final_stats

    def close(self) -> None:
        with self._condition:
            if self._state == "closed":
                return
            self._close_requested = True
            worker = self._worker
            self._condition.notify_all()
        worker_error: BaseException | None = None
        try:
            if worker is not None:
                worker.close()
        except BaseException as error:
            # A timed-out worker retains sole ownership of its model and closes
            # it from the inference thread's finally block.  The runtime still
            # becomes terminal so no caller can submit more frames.
            worker_error = error
        finally:
            final_stats = None if worker is None else worker.stats()
            self._thread.join(timeout=60.0)
            initializer_alive = self._thread.is_alive()
            with self._condition:
                if worker_error is not None and self._error is None:
                    self._error = f"{type(worker_error).__name__}: {worker_error}"
                    self._failure_phase = "inference"
                self._final_stats = final_stats
                self._worker = None
                self._state = "closed"
                self._condition.notify_all()
        if initializer_alive:
            raise TimeoutError(
                "road segmentation initialization did not stop within 60 seconds; "
                "the initializer retains model ownership until it exits"
            )
        if worker_error is not None:
            raise worker_error

    def _initialize(self) -> None:
        segmenter: RoadSegmenter | None = None
        try:
            segmenter = self._loader()
            with self._condition:
                if self._close_requested:
                    close_immediately = True
                else:
                    self._metadata = segmenter.metadata
                    self._worker = SegmentationWorker(
                        segmenter,
                        on_error=self._worker_failed,
                    )
                    self._state = "ready"
                    close_immediately = False
                    self._condition.notify_all()
            if close_immediately:
                segmenter.close()
        except BaseException as error:
            with self._condition:
                if not self._close_requested:
                    self._state = "failed"
                    self._error = f"{type(error).__name__}: {error}"
                    self._failure_phase = "initialization"
                self._condition.notify_all()

    def _worker_failed(self, error: BaseException) -> None:
        with self._condition:
            if self._state != "closed":
                self._state = "failed"
                self._error = f"{type(error).__name__}: {error}"
                self._failure_phase = "inference"
            self._condition.notify_all()

    def _failure_message(self, error: str | None) -> str:
        phase = self._failure_phase or "runtime"
        return f"road segmentation {phase} failed: {error}"


__all__ = [
    "AsyncSegmentationRuntime",
    "SegmentationFrameInput",
    "SegmentationFrameResult",
    "SegmentationWorker",
    "SegmentationWorkerStats",
]
