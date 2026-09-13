"""Stdlib-only native jobs served by the normal World Worker.

This is process isolation, not a security sandbox. There is deliberately no
network endpoint for uploading Python, choosing an executable, or installing pip
packages. Built-in tasks run directly from this checkout; optional local task
directories are for host-installed extensions, not normal operation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")
_ACTIVE = {"queued", "running", "cancelling"}
_MAX_LOG = 8 * 1024 * 1024
_MAX_RESULT = 1024 * 1024
_CANCEL_GRACE_SECONDS = 15.0


class ResearchJobError(RuntimeError):
    def __init__(self, message: str, *, status: int = 400, code: str = "research_job_error"):
        super().__init__(message)
        self.status, self.code = status, code


def identifier(value: Any) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ResearchJobError("IDs must contain only letters, numbers, underscores or hyphens")
    return value


def inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or any(c in relative for c in ("\\", ":", "\x00")):
        raise ResearchJobError("invalid relative task path")
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ResearchJobError("task paths must stay inside their directory")
    path = root
    for part in parts:
        path /= part
        if path.is_symlink():
            raise ResearchJobError("task paths must not contain symlinks")
    path.resolve().relative_to(root.resolve())
    return path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class ResearchJobs:
    def __init__(
        self, tasks: Path | None, root: Path, worker: Any, *, python: str = sys.executable
    ):
        self.tasks = Path(tasks).expanduser().resolve(strict=True) if tasks is not None else None
        self.source = Path(__file__).resolve().parent.parent
        self.root = Path(root).expanduser().resolve()
        if self.tasks is not None and not self.tasks.is_dir():
            raise ValueError("research tasks directory must exist")
        self.root.mkdir(parents=True, exist_ok=True)
        # Resolving a virtualenv's Python symlink selects the system interpreter
        # and silently loses its installed dependencies on macOS/Linux.
        interpreter = Path(python).expanduser().absolute()
        if not interpreter.is_file():
            raise ValueError("research interpreter must exist")
        self.python = str(interpreter)
        self.worker = worker
        self._lock = threading.RLock()
        self._active: dict[str, Any] | None = None
        self._thread: threading.Thread | None = None
        self._closed = False
        # Failed capture receipts are evidence, not permanent world locks.
        # A legacy recovery marker must not disable an otherwise healthy Worker.
        if (self.root / "recovery-required.json").exists():
            self.worker.research_reset_pending = True
        for path in self.root.glob("*/status.json"):
            if path.is_symlink() or path.parent.is_symlink():
                continue
            try:
                status = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(status, dict):
                    raise ValueError("invalid status object")
            except (ValueError, OSError):
                self.worker.research_reset_pending = True
                continue
            if not status.get("cleanup_confirmed", True):
                self.worker.research_reset_pending = True
            if status.get("status") in _ACTIVE:
                status.update(
                    status="interrupted", cleanup_confirmed=False,
                    task_error="World Worker restarted while capture was active; recording interrupted",
                )
                write_json(path, status)
                self.worker.research_reset_pending = True

    def _task(self, task_id: str, *, verify: bool = False) -> tuple[Path, dict[str, Any]]:
        identifier(task_id)
        if (
            task_id == "teacher_capture"
            and (self.source / "native/tasks/run.py").is_file()
            and (self.tasks is None or not (self.tasks / task_id / "task.json").is_file())
        ):
            # No copied source tree or generated manifest is needed for code
            # already shipped with the Worker. Record the exact source identity.
            files = {
                path.relative_to(self.source.parent).as_posix(): digest(path)
                for path in sorted(self.source.rglob("*.py"))
                if not path.is_symlink()
            }
            source_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
            return self.source.parent, {
                "schema_version": SCHEMA_VERSION,
                "id": task_id,
                "version": source_hash[:12],
                "manifest_sha256": source_hash,
                "entrypoint": "carla_vision/native/tasks/run.py",
                "world_access": "exclusive",
                "max_seconds": 900,
                "builtin": True,
            }
        if self.tasks is None:
            raise ResearchJobError("native task is not available on this Worker", status=404)
        directory = inside(self.tasks, identifier(task_id))
        path = inside(directory, "task.json")
        if not path.is_file():
            raise ResearchJobError("task is not installed on this host", status=404)
        if path.stat().st_size > 512 * 1024:
            raise ResearchJobError("task manifest is too large")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("id") != task_id
            or manifest.get("world_access") != "exclusive"
            or not isinstance(manifest.get("version"), str)
            or not manifest["version"].strip()
        ):
            raise ResearchJobError("unsupported task manifest")
        seconds = manifest.get("max_seconds")
        if type(seconds) is not int or not 1 <= seconds <= 3600:
            raise ResearchJobError("task max_seconds must be in [1, 3600]")
        files = manifest.get("files")
        if not isinstance(files, dict) or not 1 <= len(files) <= 4096:
            raise ResearchJobError("task requires a bounded file checksum inventory")
        entry = manifest.get("entrypoint")
        if not isinstance(entry, str) or not entry.endswith(".py") or entry not in files:
            raise ResearchJobError("task requires an indexed Python entrypoint")
        for relative, expected in files.items():
            candidate = inside(directory, relative)
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ResearchJobError("task file checksum is invalid")
            if verify and (not candidate.is_file() or digest(candidate) != expected):
                raise ResearchJobError(f"installed task checksum mismatch: {relative}")
        return directory, {**manifest, "manifest_sha256": digest(path)}

    def catalog(self) -> dict[str, Any]:
        tasks, invalid = [], []
        names = {"teacher_capture"} if (self.source / "native/tasks/run.py").is_file() else set()
        if self.tasks is not None:
            names.update(
                path.name
                for path in self.tasks.iterdir()
                if path.is_dir() and not path.is_symlink()
            )
        for name in sorted(names):
            try:
                _, item = self._task(name)
                tasks.append(
                    {
                        key: item[key]
                        for key in (
                            "id",
                            "version",
                            "manifest_sha256",
                            "world_access",
                            "max_seconds",
                        )
                    }
                )
            except (OSError, ValueError, ResearchJobError) as error:
                invalid.append({"id": name, "error": str(error)})
        return {
            "schema_version": SCHEMA_VERSION,
            "tasks": tasks,
            "invalid": invalid,
            "recovery_required": False,  # Retained for older clients; never a persistent lock.
        }

    def submit(self, raw: dict[str, Any]) -> dict[str, Any]:
        required = {"job_id", "task_id", "task_sha256", "parameters", "acknowledge_world_reload"}
        if set(raw) != required or raw["acknowledge_world_reload"] is not True:
            raise ResearchJobError(
                "submit requires task identity, parameters and world-reload acknowledgement"
            )
        job_id = identifier(raw["job_id"])
        if not isinstance(raw["parameters"], dict):
            raise ResearchJobError("task parameters must be a JSON object")
        request_hash = hashlib.sha256(
            json.dumps(raw, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        with self._lock:
            if self._closed:
                raise ResearchJobError("research runner is shutting down", status=503)
            directory = inside(self.root, job_id)
            if directory.exists():
                previous = self.get(job_id)
                if previous["request_sha256"] != request_hash:
                    raise ResearchJobError("job ID already belongs to another request", status=409)
                return previous  # Safe retry after an HTTP timeout; never rerun it.
            if self._active is not None:
                raise ResearchJobError("another native job is active", status=409)
            task_dir, task = self._task(raw["task_id"], verify=True)
            if task["manifest_sha256"] != raw["task_sha256"]:
                raise ResearchJobError(
                    "task changed; refresh the task catalog before submitting", status=409
                )
            endpoint = self.worker.reserve_research(job_id)
            try:
                directory.mkdir()
                (directory / "output").mkdir()
                request = {
                    "schema_version": SCHEMA_VERSION,
                    "job_id": job_id,
                    "parameters": raw["parameters"],
                    "endpoint": endpoint,
                    "task": {
                        "id": task["id"],
                        "version": task["version"],
                        "manifest_sha256": task["manifest_sha256"],
                    },
                    "cancel_file": str(directory / "cancel.requested"),
                }
                write_json(directory / "request.json", request)
                status = {
                    "schema_version": SCHEMA_VERSION,
                    "job_id": job_id,
                    "task_id": task["id"],
                    "task_version": task["version"],
                    "task_sha256": task["manifest_sha256"],
                    "request_sha256": request_hash,
                    "status": "queued",
                    "cleanup_confirmed": False,
                    "submitted_at": time.time(),
                    "task_error": None,
                    "result": None,
                }
                write_json(directory / "status.json", status)
                self._active = status
                self._thread = threading.Thread(
                    target=self._run,
                    args=(directory, task_dir, task),
                    name="native-research-job",
                    daemon=True,
                )
                self._thread.start()
                return dict(status)
            except BaseException:
                self._active = None
                self.worker.release_research(job_id, cleanup_confirmed=True)
                raise

    def _run(self, directory: Path, task_dir: Path, task: dict[str, Any]) -> None:
        process = None
        clean = True  # Nothing has been launched yet.
        final, error, result = "failed", None, None
        log_thread = None
        forced = False
        try:
            env = {
                key: value
                for key, value in os.environ.items()
                if key.upper()
                in {
                    "PATH",
                    "SYSTEMROOT",
                    "WINDIR",
                    "TEMP",
                    "TMP",
                    "HOME",
                    "USERPROFILE",
                    "APPDATA",
                    "LOCALAPPDATA",
                    "PYTHONPATH",
                    "LD_LIBRARY_PATH",
                    "DYLD_LIBRARY_PATH",
                    "LANG",
                    "LC_ALL",
                    "VIRTUAL_ENV",
                }
            }
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                [
                    self.python,
                    str(inside(task_dir, task["entrypoint"])),
                    "--request",
                    str(directory / "request.json"),
                    "--output",
                    str(directory / "output"),
                ],
                cwd=task_dir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            clean = False
            with self._lock:
                self._active.update(status="running", pid=process.pid, started_at=time.time())
                write_json(directory / "status.json", self._active)
            log_thread = threading.Thread(
                target=self._drain_log,
                args=(process.stdout, directory / "task.log"),
                daemon=True,
                name="native-research-log",
            )
            log_thread.start()
            deadline = time.monotonic() + task["max_seconds"]
            cancel_deadline = None
            timed_out = False
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    timed_out = True
                    (directory / "cancel.requested").touch(exist_ok=True)
                if (directory / "cancel.requested").exists():
                    if cancel_deadline is None:
                        cancel_deadline = time.monotonic() + _CANCEL_GRACE_SECONDS
                        with self._lock:
                            self._active["status"] = "cancelling"
                            write_json(directory / "status.json", self._active)
                    elif time.monotonic() >= cancel_deadline:
                        forced = True
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=3)
                        break
                time.sleep(0.1)
            receipt = inside(directory / "output", "result.json")
            if receipt.is_file() and receipt.stat().st_size <= _MAX_RESULT:
                result = json.loads(receipt.read_text(encoding="utf-8"))
                clean = isinstance(result, dict) and result.get("cleanup_confirmed") is True
            if forced:
                clean = False
            cancelled = (directory / "cancel.requested").exists()
            final = (
                "timed_out"
                if timed_out
                else "cancelled"
                if cancelled
                else (
                    "succeeded"
                    if process.returncode == 0 and clean and result.get("status") == "succeeded"
                    else "failed"
                )
            )
            if final == "failed":
                error = f"task exited with code {process.returncode}; inspect task.log and result"
        except BaseException as caught:
            error = f"{type(caught).__name__}: {caught}"
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=3)
                clean = False
        finally:
            if log_thread is not None:
                log_thread.join(timeout=2)
            with self._lock:
                self._active.update(
                    status=final,
                    task_error=error,
                    result=result,
                    cleanup_confirmed=clean,
                    finished_at=time.time(),
                )
                try:
                    write_json(directory / "status.json", self._active)
                except OSError:
                    # Retain uncertainty in the receipt without disabling Garage.
                    clean = False
                finally:
                    self.worker.release_research(directory.name, cleanup_confirmed=clean)
                    self._active = None

    @staticmethod
    def _drain_log(stream: Any, path: Path) -> None:
        with stream, path.open("wb") as target:
            retained = 0
            while chunk := stream.read1(4096):
                keep = chunk[: max(0, _MAX_LOG - retained)]
                target.write(keep)
                target.flush()
                retained += len(keep)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            path = inside(self.root, identifier(job_id) + "/status.json")
            if not path.is_file():
                raise ResearchJobError("native job not found", status=404)
            return json.loads(path.read_text(encoding="utf-8"))

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            status = self.get(job_id)
            if status["status"] in _ACTIVE:
                inside(self.root, job_id + "/cancel.requested").touch(exist_ok=True)
            return status

    def log(self, job_id: str) -> dict[str, Any]:
        self.get(job_id)
        path = inside(self.root, job_id + "/task.log")
        if not path.exists():
            return {"schema_version": SCHEMA_VERSION, "text": "", "truncated": False}
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 32768))
            return {
                "schema_version": SCHEMA_VERSION,
                "text": stream.read(32768).decode("utf-8", errors="replace"),
                "truncated": path.stat().st_size > 32768,
            }

    def output_file(self, job_id: str, relative: str) -> Path:
        status = self.get(job_id)
        if status["status"] in _ACTIVE:
            raise ResearchJobError("download is available after the job finishes", status=409)
        if relative not in {"artifacts.zip", "result.json"}:
            raise ResearchJobError(
                "only the task result and artifact archive may be downloaded", status=404
            )
        path = inside(self.root, job_id + "/output/" + relative)
        if not path.is_file():
            raise ResearchJobError("task output is not available", status=404)
        return path

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._active is not None:
                self.cancel(self._active["job_id"])
        if self._thread is not None:
            self._thread.join(timeout=21)
