"""Encode review video directly as browser-playable H.264 MP4.

Training images are not processed here. Callers must keep this disk/encoder I/O
off the live control loop (AsyncVideoRecorder does that for live recordings).
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=4)
def _check_encoder(executable: str) -> None:
    try:
        result = subprocess.run(
            [executable, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            "Could not check FFmpeg; restart the WebUI after fixing FFmpeg"
        ) from error
    if not any("libx264" in line.split() for line in result.stdout.splitlines()):
        raise RuntimeError("Recording requires an FFmpeg build with the libx264 encoder")


class BrowserVideoWriter:
    """Small write/release interface shared by live capture and offline replay.

    No MPEG-4 Part 2 fallback: a successful recording must already be playable.
    Odd dimensions are padded at the right/bottom, never cropped or stretched.
    """

    def __init__(self, path: str | Path, frame_size: tuple[int, int], fps: float) -> None:
        width, height = frame_size
        if (
            not all(isinstance(value, int) and value > 0 for value in frame_size)
            or not math.isfinite(fps)
            or fps <= 0
        ):
            raise ValueError("video requires positive integer dimensions and finite positive FPS")
        self.path = Path(path).absolute()
        if self.path.suffix.lower() != ".mp4":
            raise ValueError("browser recordings must use an .mp4 filename")
        if self.path.exists() or self.path.is_symlink():
            raise FileExistsError(f"recording already exists: {self.path}")
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError(
                "Recording requires FFmpeg on the recording/WebUI computer "
                "(macOS: brew install ffmpeg); restart the WebUI after installation"
            )
        _check_encoder(executable)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._shape = (height, width, 3)
        self._closed = False
        self._error: RuntimeError | None = None
        # A file avoids a filled stderr pipe blocking the encoder/control queue.
        self._stderr = tempfile.TemporaryFile()
        try:
            self._process = subprocess.Popen(
                [
                    executable,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-n",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "bgr24",
                    "-video_size",
                    f"{width}x{height}",
                    "-framerate",
                    str(float(fps)),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "2",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-f",
                    "mp4",
                    str(self.path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr,
            )
        except BaseException:
            self._stderr.close()
            raise

    def write(self, image: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("video writer is closed")
        if image.dtype != np.uint8 or image.shape != self._shape:
            raise ValueError(f"video frame must be uint8 with shape {self._shape}")
        try:
            self._process.stdin.write(image.tobytes())
        except (OSError, ValueError) as error:
            # release captures encoder diagnostics and reaps the process.
            self.release()
            raise RuntimeError("video encoder stopped accepting frames") from error

    def abort(self) -> None:
        """Unblock a stuck pipe write before the recorder joins its thread."""
        if self._process.poll() is None:
            self._process.kill()

    def release(self) -> None:
        if self._closed:
            if self._error:
                raise self._error
            return
        self._closed = True
        try:
            try:
                self._process.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                code = self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.abort()
                self._process.wait(timeout=5)
                raise RuntimeError("video encoder did not finish saving the MP4") from None
            if code != 0:
                self._stderr.seek(0, 2)
                self._stderr.seek(max(0, self._stderr.tell() - 2048))
                detail = self._stderr.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"H.264 recording failed (exit {code}): {detail}")
        except RuntimeError as error:
            self._error = error
            raise
        finally:
            self._stderr.close()
