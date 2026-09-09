"""On-demand browser copies; original research artifacts are never modified."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path


class RecordingPreviews:
    def __init__(self) -> None:
        self._directory = tempfile.TemporaryDirectory(prefix="carla-recording-preview-")
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="recording-preview")
        self._lock = threading.Lock()
        self._jobs: dict[str, Future[Path]] = {}

    def request(self, source: Path) -> dict[str, str]:
        stat = source.stat()
        key = hashlib.sha256(
            f"{source}:{stat.st_size}:{stat.st_mtime_ns}".encode()
        ).hexdigest()
        with self._lock:
            if key not in self._jobs:
                if any(not job.done() for job in self._jobs.values()):
                    return {"status": "busy"}
                executable = shutil.which("ffmpeg")
                if executable is None:
                    raise RuntimeError("Install FFmpeg on the WebUI computer to prepare browser copies")
                target = Path(self._directory.name) / f"{key}.mp4"
                self._jobs[key] = self._pool.submit(self._convert, executable, source, target)
            job = self._jobs[key]
            if not job.done():
                return {"status": "preparing", "id": key}
            try:
                job.result()
            except Exception:
                del self._jobs[key]
                raise
            return {"status": "ready", "id": key}

    @staticmethod
    def _convert(executable: str, source: Path, target: Path) -> Path:
        try:
            subprocess.run(
                [executable, "-nostdin", "-v", "error", "-n", "-i", str(source),
                 "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264",
                 "-threads", "2", "-preset", "fast", "-crf", "20",
                 "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-movflags", "+faststart", str(target)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, timeout=600, check=True,
            )
        except (subprocess.SubprocessError, OSError) as error:
            target.unlink(missing_ok=True)
            raise RuntimeError("Browser copy failed; the original recording is unchanged") from error
        return target

    def file(self, key: str) -> Path:
        with self._lock:
            job = self._jobs.get(key)
            if job is None or not job.done():
                raise FileNotFoundError("Browser copy is not ready")
            return job.result()

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._directory.cleanup()
