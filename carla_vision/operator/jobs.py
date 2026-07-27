"""Small persistent subprocess manager for the local operator UI."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..artifacts import RunArtifactTracker, fingerprint_file
from .commands import CommandPlan
from .contracts import OperatorJobRequest


def _utc_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@dataclass
class _ManagedJob:
    job_id: str
    request: OperatorJobRequest
    plan: CommandPlan
    tracker: RunArtifactTracker
    created_at: str
    status: str = "queued"
    started_at: str | None = None
    finished_at: str | None = None
    returncode: int | None = None
    error: str | None = None
    stop_requested: bool = False
    tracker_finalized: bool = False
    process: subprocess.Popen[bytes] | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    def snapshot(self, *, for_artifact: bool = False) -> dict[str, Any]:
        status = self.status
        if (
            not for_artifact
            and not self.tracker_finalized
            and status in {"success", "failed", "stopped"}
        ):
            status = "finalizing"
        return {
            "job_id": self.job_id,
            "kind": self.request.kind,
            "title": self.plan.title,
            "status": status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "error": self.error,
            "stop_requested": self.stop_requested,
            "motion_authorized": self.plan.motion_authorized,
            "destructive": self.plan.destructive,
            "note": self.plan.note,
            "session_path": str(self.tracker.run_dir),
            "expected_output": (
                str(self.plan.expected_output) if self.plan.expected_output is not None else None
            ),
            "expected_output_exists": (
                self.plan.expected_output.exists()
                if self.plan.expected_output is not None
                else False
            ),
        }


class JobManager:
    """Runs allow-listed command plans and retains a verifiable operator session."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        sessions_root: str | Path = "operator_sessions",
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve(strict=True)
        sessions = Path(sessions_root).expanduser()
        if not sessions.is_absolute():
            sessions = self.workspace / sessions
        self.sessions_root = sessions.resolve()
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, _ManagedJob] = {}

    def submit(
        self,
        request: OperatorJobRequest,
        plan: CommandPlan,
    ) -> dict[str, Any]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz")
        job_kind = request.kind.replace("_", "-")
        job_id = f"op-{timestamp}-{job_kind}-{uuid.uuid4().hex[:8]}"
        tracker = RunArtifactTracker(
            self.sessions_root,
            run_id=job_id,
            cli_args=(),
            config={
                "schema_version": "1.0",
                "object_type": "operator_session",
                "request": request.as_dict(),
                "command_plan": plan.as_dict(),
            },
            repository_root=self.workspace,
        )
        job = _ManagedJob(
            job_id=job_id,
            request=request,
            plan=plan,
            tracker=tracker,
            created_at=_utc_text(),
        )
        thread = threading.Thread(
            target=self._run,
            args=(job,),
            name=f"operator-{job_id}",
            daemon=True,
        )
        job.thread = thread
        with self._lock:
            self._jobs[job_id] = job
        thread.start()
        return job.snapshot()

    def _write_status(self, job: _ManagedJob, path: Path) -> None:
        _atomic_json(path, job.snapshot(for_artifact=True))

    def _stop_process(self, process: subprocess.Popen[bytes], *, force: bool = False) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(
                    process.pid,
                    signal.SIGKILL if force else signal.SIGTERM,
                )
            elif force:
                process.kill()
            else:
                process.terminate()
        except ProcessLookupError:
            return

    def _run(self, job: _ManagedJob) -> None:
        try:
            self._run_tracked(job)
        except BaseException:
            # The tracker and status artifact already retain the redacted
            # failure. Do not leak a background-thread traceback into the UI
            # server's terminal.
            return
        finally:
            with self._lock:
                job.tracker_finalized = True

    def _run_tracked(self, job: _ManagedJob) -> None:
        tracker = job.tracker
        request_path = tracker.artifact_path("request.json")
        command_path = tracker.artifact_path("command.json")
        status_path = tracker.artifact_path("status.json")
        stdout_path = tracker.artifact_path("logs/stdout.log")
        stderr_path = tracker.artifact_path("logs/stderr.log")
        failure: BaseException | None = None
        with tracker:
            _atomic_json(request_path, job.request.as_dict())
            _atomic_json(command_path, job.plan.as_dict())
            job.status = "starting"
            job.started_at = _utc_text()
            self._write_status(job, status_path)
            try:
                with (
                    stdout_path.open("wb") as stdout_stream,
                    stderr_path.open("wb") as stderr_stream,
                ):
                    process = subprocess.Popen(
                        job.plan.command,
                        cwd=self.workspace,
                        stdout=stdout_stream,
                        stderr=stderr_stream,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    with self._lock:
                        job.process = process
                        job.status = "running"
                    self._write_status(job, status_path)
                    while process.poll() is None:
                        if job.stop_event.wait(0.2):
                            with self._lock:
                                job.stop_requested = True
                                job.status = "stopping"
                            self._write_status(job, status_path)
                            self._stop_process(process)
                            try:
                                process.wait(timeout=5.0)
                            except subprocess.TimeoutExpired:
                                self._stop_process(process, force=True)
                            break
                    job.returncode = process.wait()
                if job.stop_requested:
                    job.status = "stopped"
                    failure = RuntimeError("operator stop requested")
                elif job.returncode == 0:
                    job.status = "success"
                else:
                    job.status = "failed"
                    failure = RuntimeError(
                        f"operator child process exited with code {job.returncode}"
                    )
            except BaseException as error:
                job.status = "failed"
                job.error = f"{type(error).__qualname__}: {error}"
                failure = error
            finally:
                job.finished_at = _utc_text()
                if failure is not None and job.error is None:
                    job.error = f"{type(failure).__qualname__}: {failure}"
                self._write_status(job, status_path)
                for path, role in (
                    (request_path, "operator_job_request"),
                    (command_path, "operator_command_plan"),
                    (status_path, "operator_job_status"),
                    (stdout_path, "operator_stdout_log"),
                    (stderr_path, "operator_stderr_log"),
                ):
                    if path.is_file():
                        tracker.register_artifact(
                            path,
                            role=role,
                            metadata={
                                "job_id": job.job_id,
                                "job_kind": job.request.kind,
                            },
                        )
                output = job.plan.expected_output
                if output is not None and (output / "manifest.json").is_file():
                    tracker.add_input_reference(
                        {
                            "kind": "operator_child_output",
                            "job_kind": job.request.kind,
                            "run_id": output.name,
                            **fingerprint_file(output / "manifest.json"),
                        }
                    )
            if failure is not None:
                raise failure

    def stop(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"unknown active job {job_id!r}")
            if job.status not in {"queued", "starting", "running", "stopping"}:
                raise RuntimeError(f"job {job_id} is already terminal")
            job.stop_requested = True
            job.stop_event.set()
            return job.snapshot()

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return job.snapshot()
        return self._historical_snapshot(job_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            active = {job_id: job.snapshot() for job_id, job in self._jobs.items()}
        for directory in sorted(self.sessions_root.iterdir(), reverse=True):
            if not directory.is_dir() or directory.name in active:
                continue
            try:
                active[directory.name] = self._historical_snapshot(directory.name)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return sorted(
            active.values(),
            key=lambda row: str(row.get("created_at") or ""),
            reverse=True,
        )

    def _historical_snapshot(self, job_id: str) -> dict[str, Any]:
        directory = (self.sessions_root / job_id).resolve(strict=True)
        try:
            directory.relative_to(self.sessions_root)
        except ValueError as error:
            raise KeyError(f"invalid operator job ID {job_id!r}") from error
        status_path = directory / "status.json"
        if status_path.is_file():
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        return {
            "job_id": job_id,
            "kind": manifest.get("invocation", {})
            .get("config", {})
            .get("request", {})
            .get("kind", "unknown"),
            "title": job_id,
            "status": manifest.get("status", "unknown"),
            "created_at": manifest.get("timestamps", {}).get("created_at"),
            "started_at": manifest.get("timestamps", {}).get("started_at"),
            "finished_at": manifest.get("timestamps", {}).get("finished_at"),
            "returncode": None,
            "error": None,
            "stop_requested": False,
            "motion_authorized": False,
            "destructive": False,
            "note": "Historical operator session",
            "session_path": str(directory),
            "expected_output": None,
            "expected_output_exists": False,
        }

    def log_tail(
        self,
        job_id: str,
        *,
        stream: str = "stdout",
        max_bytes: int = 64 * 1024,
    ) -> str:
        if stream not in {"stdout", "stderr"}:
            raise ValueError("stream must be stdout or stderr")
        snapshot = self.get(job_id)
        path = Path(str(snapshot["session_path"])) / "logs" / f"{stream}.log"
        if not path.is_file():
            return ""
        with path.open("rb") as handle:
            size = path.stat().st_size
            handle.seek(max(0, size - max_bytes))
            payload = handle.read(max_bytes)
        return payload.decode("utf-8", errors="replace")

    def shutdown(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.status in {"queued", "starting", "running", "stopping"}:
                job.stop_requested = True
                job.stop_event.set()
        deadline = time.monotonic() + 8.0
        for job in jobs:
            thread = job.thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))


__all__ = ["JobManager"]
