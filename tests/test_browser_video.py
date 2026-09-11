"""Exercise actual saved codecs, not merely the .mp4 filename."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from carla_vision.recording import AsyncVideoRecorder
from carla_vision.video import BrowserVideoWriter, _check_encoder


class BrowserVideoTests(unittest.TestCase):
    def test_missing_encoder_is_actionable_without_silent_codec_fallback(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "carla_vision.video.shutil.which",
                return_value=None,
            ),
        ):
            target = Path(directory) / "test.mp4"
            with self.assertRaisesRegex(RuntimeError, "FFmpeg on the recording/WebUI"):
                BrowserVideoWriter(target, (64, 48), 10)
            self.assertFalse(target.exists())

    def test_build_without_h264_is_rejected(self):
        with patch("carla_vision.video.subprocess.run") as run:
            run.return_value.stdout = " V..... mpeg4 MPEG-4 part 2\n"
            with self.assertRaisesRegex(RuntimeError, "libx264"):
                _check_encoder("/test/ffmpeg-without-h264")

    def test_does_not_overwrite_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "saved.mp4"
            target.write_bytes(b"original")
            with self.assertRaises(FileExistsError):
                BrowserVideoWriter(target, (64, 48), 10)
            self.assertEqual(target.read_bytes(), b"original")

    def test_rejects_nonfinite_fps_and_invalid_dimensions(self):
        for size, fps in [((0, 8), 10), ((8, 8), float("nan")), ((8, 8), float("inf"))]:
            with self.subTest(size=size, fps=fps), self.assertRaises(ValueError):
                BrowserVideoWriter("unused.mp4", size, fps)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires FFmpeg")
    def test_new_recording_is_h264_yuv420p_faststart_and_decodable(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "overlay.mp4"
            # Odd dimensions exercise padding rather than losing the last row/column.
            recorder = AsyncVideoRecorder(target, frame_size=(65, 49), fps=12.5)
            for index in range(6):
                recorder.submit(index, np.full((49, 65, 3), index * 30, np.uint8))
            stats = recorder.close()
            self.assertEqual(stats.written, 6)
            self.assertEqual(stats.dropped, 0)
            info = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_streams",
                        "-of",
                        "json",
                        str(target),
                    ]
                )
            )["streams"][0]
            self.assertEqual(info["codec_name"], "h264")
            self.assertEqual(info["codec_tag_string"], "avc1")
            self.assertEqual(info["pix_fmt"], "yuv420p")
            self.assertEqual((info["width"], info["height"]), (66, 50))
            self.assertEqual(info["nb_frames"], "6")
            self.assertEqual(info["avg_frame_rate"], "25/2")
            data = target.read_bytes()
            self.assertLess(data.index(b"moov"), data.index(b"mdat"))
            decoded = subprocess.check_output(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(target),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "pipe:1",
                ]
            )
            self.assertEqual(len(decoded), 6 * 66 * 50 * 3)

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires FFmpeg")
    def test_encoder_failure_is_reported_and_process_is_reaped(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = BrowserVideoWriter(Path(directory) / "failed.mp4", (64, 48), 10)
            writer.abort()
            with self.assertRaisesRegex(RuntimeError, "H.264 recording failed"):
                writer.release()
            self.assertIsNotNone(writer._process.poll())
            self.assertTrue(writer._stderr.closed)
