from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class RecordingStats:
    submitted: int
    written: int
    dropped: int
    first_sequence: int | None
    last_sequence: int | None

    def as_dict(self) -> dict[str, int | None]:
        return {
            "submitted": self.submitted,
            "written": self.written,
            "dropped": self.dropped,
            "first_sequence": self.first_sequence,
            "last_sequence": self.last_sequence,
        }


@dataclass(frozen=True)
class _QueuedFrame:
    sequence: int
    image: np.ndarray


class AsyncVideoRecorder:
    """Bounded recorder that never makes the control loop wait for disk I/O."""

    def __init__(
        self,
        path: str | Path,
        *,
        frame_size: tuple[int, int],
        fps: float,
        latest_frame_path: str | Path | None = None,
        codec: str = "mp4v",
        max_queue: int = 32,
        cv2_backend: Any | None = None,
    ) -> None:
        if frame_size[0] <= 0 or frame_size[1] <= 0:
            raise ValueError("frame_size must contain positive width and height")
        if fps <= 0.0:
            raise ValueError("fps must be positive")
        if len(codec) != 4:
            raise ValueError("codec must contain exactly four characters")
        if max_queue <= 0:
            raise ValueError("max_queue must be positive")

        self.path = Path(path).resolve()
        self.latest_frame_path = (
            Path(latest_frame_path).resolve() if latest_frame_path is not None else None
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.latest_frame_path is not None:
            self.latest_frame_path.parent.mkdir(parents=True, exist_ok=True)

        self._cv2 = cv2 if cv2_backend is None else cv2_backend
        fourcc = self._cv2.VideoWriter_fourcc(*codec)
        self._writer = self._cv2.VideoWriter(
            str(self.path),
            fourcc,
            float(fps),
            tuple(int(value) for value in frame_size),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"could not open video writer for {self.path}")

        self._frame_size = tuple(int(value) for value in frame_size)
        self._max_queue = max_queue
        self._condition = threading.Condition()
        self._queue: deque[_QueuedFrame] = deque()
        self._closed = False
        self._error: BaseException | None = None
        self._submitted = 0
        self._written = 0
        self._dropped = 0
        self._first_sequence: int | None = None
        self._last_sequence: int | None = None
        self._last_submitted_sequence = -1
        self._latest_written_image: np.ndarray | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="artifact-video-recorder",
            daemon=True,
        )
        self._thread.start()

    def submit(self, sequence: int, image_bgr: np.ndarray) -> bool:
        """Queue one new sequence; duplicates are ignored and overflow drops oldest."""

        image = np.asarray(image_bgr)
        expected_width, expected_height = self._frame_size
        if image.dtype != np.uint8 or image.shape != (expected_height, expected_width, 3):
            raise ValueError(
                f"video frame must be uint8 with shape ({expected_height}, {expected_width}, 3)"
            )
        with self._condition:
            self._raise_if_failed()
            if self._closed:
                raise RuntimeError("video recorder is closed")
            if sequence <= self._last_submitted_sequence:
                return False
            self._last_submitted_sequence = sequence
            self._submitted += 1
            if len(self._queue) >= self._max_queue:
                self._queue.popleft()
                self._dropped += 1
            self._queue.append(_QueuedFrame(sequence=sequence, image=image.copy()))
            self._condition.notify_all()
        return True

    def stats(self) -> RecordingStats:
        with self._condition:
            self._raise_if_failed()
            return RecordingStats(
                submitted=self._submitted,
                written=self._written,
                dropped=self._dropped,
                first_sequence=self._first_sequence,
                last_sequence=self._last_sequence,
            )

    def close(self) -> RecordingStats:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=15.0)
        if self._thread.is_alive():
            raise TimeoutError("video recorder did not finish")
        self._raise_if_failed()
        if self.latest_frame_path is not None and self._latest_written_image is not None:
            if not self._cv2.imwrite(
                str(self.latest_frame_path),
                self._latest_written_image,
            ):
                raise RuntimeError(f"could not write latest frame to {self.latest_frame_path}")
        return self.stats()

    def __enter__(self) -> "AsyncVideoRecorder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"video recorder failed: {self._error}")

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._queue and not self._closed:
                        self._condition.wait()
                    if not self._queue and self._closed:
                        break
                    queued = self._queue.popleft()
                self._writer.write(queued.image)
                with self._condition:
                    self._written += 1
                    if self._first_sequence is None:
                        self._first_sequence = queued.sequence
                    self._last_sequence = queued.sequence
                    self._latest_written_image = queued.image
            self._writer.release()
        except BaseException as exc:
            try:
                self._writer.release()
            finally:
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()
