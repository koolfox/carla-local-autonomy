from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from carla_vision.recording import AsyncVideoRecorder, RecordingStats


class FakeVideoWriter:
    def __init__(
        self,
        *,
        opened: bool = True,
        write_gate: threading.Event | None = None,
        write_error: BaseException | None = None,
    ) -> None:
        self.opened = opened
        self.write_gate = write_gate
        self.write_error = write_error
        self.write_started = threading.Event()
        self.write_finished = threading.Event()
        self.frames: list[np.ndarray] = []
        self.release_count = 0

    def isOpened(self) -> bool:
        return self.opened

    def write(self, image: np.ndarray) -> None:
        self.write_started.set()
        try:
            if self.write_gate is not None and not self.write_gate.wait(timeout=2.0):
                raise TimeoutError("test did not release fake video writer")
            if self.write_error is not None:
                raise self.write_error
            self.frames.append(image.copy())
        finally:
            self.write_finished.set()

    def release(self) -> None:
        self.release_count += 1


class FakeCv2:
    def __init__(self, writer: FakeVideoWriter, *, image_write_succeeds: bool = True) -> None:
        self.writer = writer
        self.image_write_succeeds = image_write_succeeds
        self.fourcc_calls: list[tuple[str, ...]] = []
        self.video_writer_calls: list[tuple[str, int, float, tuple[int, int]]] = []
        self.image_write_calls: list[tuple[str, np.ndarray]] = []

    def VideoWriter_fourcc(self, *codec: str) -> int:
        self.fourcc_calls.append(codec)
        return 1234

    def VideoWriter(
        self,
        path: str,
        fourcc: int,
        fps: float,
        frame_size: tuple[int, int],
    ) -> FakeVideoWriter:
        self.video_writer_calls.append((path, fourcc, fps, frame_size))
        return self.writer

    def imwrite(self, path: str, image: np.ndarray) -> bool:
        self.image_write_calls.append((path, image.copy()))
        return self.image_write_succeeds


def frame(value: int) -> np.ndarray:
    return np.full((2, 3, 3), value, dtype=np.uint8)


class RecordingStatsTests(unittest.TestCase):
    def test_as_dict_is_stable_and_json_ready(self) -> None:
        stats = RecordingStats(
            submitted=5,
            written=4,
            dropped=1,
            first_sequence=10,
            last_sequence=14,
        )

        self.assertEqual(
            stats.as_dict(),
            {
                "submitted": 5,
                "written": 4,
                "dropped": 1,
                "first_sequence": 10,
                "last_sequence": 14,
            },
        )


class AsyncVideoRecorderTests(unittest.TestCase):
    def test_lifecycle_deduplicates_drops_oldest_and_writes_latest_snapshot(self) -> None:
        gate = threading.Event()
        writer = FakeVideoWriter(write_gate=gate)
        backend = FakeCv2(writer)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            recorder = AsyncVideoRecorder(
                root / "video" / "overlay.mp4",
                frame_size=(3, 2),
                fps=10.0,
                latest_frame_path=root / "images" / "latest.png",
                codec="TEST",
                max_queue=2,
                cv2_backend=backend,
            )

            first = frame(1)
            self.assertTrue(recorder.submit(1, first))
            first.fill(99)
            self.assertTrue(writer.write_started.wait(timeout=1.0))

            second = frame(2)
            third = frame(3)
            fourth = frame(4)
            self.assertTrue(recorder.submit(2, second))
            self.assertTrue(recorder.submit(3, third))
            third.fill(88)
            self.assertTrue(recorder.submit(4, fourth))
            fourth.fill(77)
            self.assertFalse(recorder.submit(4, frame(40)))
            self.assertFalse(recorder.submit(3, frame(30)))

            self.assertEqual(
                recorder.stats(),
                RecordingStats(
                    submitted=4,
                    written=0,
                    dropped=1,
                    first_sequence=None,
                    last_sequence=None,
                ),
            )

            gate.set()
            stats = recorder.close()

            self.assertEqual(
                stats,
                RecordingStats(
                    submitted=4,
                    written=3,
                    dropped=1,
                    first_sequence=1,
                    last_sequence=4,
                ),
            )
            self.assertEqual([int(image[0, 0, 0]) for image in writer.frames], [1, 3, 4])
            self.assertEqual(writer.release_count, 1)
            self.assertEqual(backend.fourcc_calls, [("T", "E", "S", "T")])
            self.assertEqual(
                backend.video_writer_calls,
                [
                    (
                        str((root / "video" / "overlay.mp4").resolve()),
                        1234,
                        10.0,
                        (3, 2),
                    )
                ],
            )
            self.assertEqual(len(backend.image_write_calls), 1)
            latest_path, latest_image = backend.image_write_calls[0]
            self.assertEqual(latest_path, str((root / "images" / "latest.png").resolve()))
            self.assertTrue(np.all(latest_image == 4))
            with self.assertRaisesRegex(RuntimeError, "closed"):
                recorder.submit(5, frame(5))

    def test_empty_context_manager_releases_writer_with_zero_stats(self) -> None:
        writer = FakeVideoWriter()
        backend = FakeCv2(writer)

        with tempfile.TemporaryDirectory() as temporary_directory:
            with AsyncVideoRecorder(
                Path(temporary_directory) / "empty.mp4",
                frame_size=(3, 2),
                fps=5.0,
                latest_frame_path=Path(temporary_directory) / "latest.png",
                cv2_backend=backend,
            ) as recorder:
                self.assertEqual(
                    recorder.stats(),
                    RecordingStats(0, 0, 0, None, None),
                )

        self.assertEqual(writer.release_count, 1)
        self.assertEqual(backend.image_write_calls, [])

    def test_background_writer_failure_is_reported_by_stats_and_close(self) -> None:
        writer = FakeVideoWriter(write_error=OSError("disk unavailable"))
        backend = FakeCv2(writer)

        with tempfile.TemporaryDirectory() as temporary_directory:
            recorder = AsyncVideoRecorder(
                Path(temporary_directory) / "failed.mp4",
                frame_size=(3, 2),
                fps=5.0,
                cv2_backend=backend,
            )
            self.assertTrue(recorder.submit(1, frame(1)))
            self.assertTrue(writer.write_finished.wait(timeout=1.0))

            with self.assertRaisesRegex(RuntimeError, "disk unavailable"):
                recorder.stats()
            with self.assertRaisesRegex(RuntimeError, "disk unavailable"):
                recorder.close()

        self.assertEqual(writer.release_count, 1)

    def test_invalid_configuration_and_frame_contracts_are_rejected(self) -> None:
        writer = FakeVideoWriter()
        backend = FakeCv2(writer)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "video.mp4"
            invalid_options = (
                {"frame_size": (0, 2), "fps": 5.0},
                {"frame_size": (3, 2), "fps": 0.0},
                {"frame_size": (3, 2), "fps": 5.0, "codec": "bad"},
                {"frame_size": (3, 2), "fps": 5.0, "max_queue": 0},
            )
            for options in invalid_options:
                with self.subTest(options=options):
                    with self.assertRaises(ValueError):
                        AsyncVideoRecorder(path, cv2_backend=backend, **options)

            recorder = AsyncVideoRecorder(
                path,
                frame_size=(3, 2),
                fps=5.0,
                cv2_backend=backend,
            )
            try:
                with self.assertRaisesRegex(ValueError, r"uint8.*\(2, 3, 3\)"):
                    recorder.submit(1, np.zeros((2, 3), dtype=np.uint8))
                with self.assertRaisesRegex(ValueError, "uint8"):
                    recorder.submit(1, np.zeros((2, 3, 3), dtype=np.float32))
            finally:
                recorder.close()


if __name__ == "__main__":
    unittest.main()
