"""Garage capture workflow over its existing World Worker connection.

The native Worker owns collection. This owner only coordinates preview handoff,
observes the job, and imports its verified output for the existing recording UI.
"""

from __future__ import annotations

import json
import shutil
import stat
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from ..dataset.camera_rig import resolve_camera_rig
from .configuration import build_situation_request
from .jobs import _atomic_json
from .situations import SituationSpec, build_scenario_suite
from .world_worker_client import WorldWorkerError


def unpack_capture(archive: Path, destination: Path, *, max_bytes: int = 8 * 1024**3) -> None:
    """Extract only bounded regular artifact files into a new private directory."""
    with zipfile.ZipFile(archive) as source:
        entries = source.infolist()
        if len(entries) > 200_000 or sum(item.file_size for item in entries) > max_bytes:
            raise ValueError("capture archive exceeds extraction limits")
        seen = set()
        for item in entries:
            path = PurePosixPath(item.filename)
            kind = stat.S_IFMT(item.external_attr >> 16)
            if (
                not path.parts
                or path.is_absolute()
                or any(part in {"", ".", ".."} for part in item.filename.rstrip("/").split("/"))
                or path.parts[0] not in {"inputs", "runs", "datasets"}
                or any(c in item.filename for c in ("\\", ":", "\x00"))
                or kind not in {0, stat.S_IFREG, stat.S_IFDIR}
                or item.filename.casefold() in seen
            ):
                raise ValueError("unsafe capture archive entry")
            seen.add(item.filename.casefold())
        destination.mkdir(exist_ok=False)
        source.extractall(destination)


class GarageCapture:
    def __init__(self, application: Any, world_lock: threading.RLock):
        self.application, self.world_lock = application, world_lock
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self.path = application.jobs.sessions_root / "native_capture_state.json"
        self._state = {
            "phase": "idle",
            "active": False,
            "holds_world": False,
            "job_id": None,
            "error": None,
            "log": "",
        }
        if self.path.is_file():
            try:
                self._state.update(json.loads(self.path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                self._state.update(
                    phase="failed",
                    active=False,
                    holds_world=False,
                    error="Previous capture state is unreadable; Garage remains available",
                )
        if not self._state["active"]:
            self._state["holds_world"] = False

    def resume(self) -> None:
        if self._state["active"] and self.application.world_worker is not None:
            self._launch(None)

    def state(self) -> dict:
        with self._lock:
            return {
                "schema_version": "1.0",
                **self._state,
                "available": self.application.world_worker is not None,
            }

    def _update(self, **values) -> None:
        with self._lock:
            self._state.update(values)
            _atomic_json(self.path, self._state)

    def require_world_available(self) -> None:
        if self.state()["holds_world"]:
            raise RuntimeError(
                "Teacher capture owns CARLA; finish or cancel capture before driving"
            )

    def start(self, raw: dict) -> dict:
        self.require_world_available()
        if set(raw) != {"session", "situation", "camera_rig", "acknowledge"}:
            raise ValueError("capture requires session, situation, camera_rig and acknowledgement")
        if raw["acknowledge"] is not True:
            raise ValueError(
                "capture reloads the scene and moves the teacher vehicle; acknowledgement required"
            )
        if self.application.world_worker is None:
            raise ValueError("connect the normal World Worker before capturing")
        spec = SituationSpec.from_mapping(
            build_situation_request(
                raw["session"],
                raw["situation"],
                current_map=self.application.preview.state().get("map"),
            )
        )
        if spec.repetitions > 32:
            raise ValueError("one native capture supports at most 32 episodes")
        suite = build_scenario_suite(spec)
        rig = raw["camera_rig"]
        if isinstance(rig, str):
            resolve_camera_rig(suite.recipes[0].camera, preset=rig)
            rig_parameter = {"camera_rig": rig}
        elif isinstance(rig, dict):
            resolve_camera_rig(suite.recipes[0].camera, config=rig)
            rig_parameter = {"camera_rig_config": json.loads(json.dumps(rig, allow_nan=False))}
        else:
            raise ValueError("camera_rig must be a preset name or a camera rig object")
        parameters = {
            "scenario_suite": suite.as_dict(),
            "split_plan": {
                "schema_version": "1.0",
                "plan_id": "garage-capture",
                "test_map_families": [],
                "test_weather_ids": [],
                "validation_map_families": [],
                "validation_weather_ids": [],
            },
            **rig_parameter,
            "max_episodes": spec.repetitions,
            "behavior": "cautious",
            "target_speed_kmh": 20,
        }
        if not self.world_lock.acquire(blocking=False):
            raise RuntimeError("Garage is updating; wait for it to finish before capture")
        try:
            if self.state()["active"]:
                raise RuntimeError("a capture is already active")
            if self.application.drive.state()["status"] in {"starting", "running", "stopping"}:
                raise RuntimeError("Stop & Save the active drive before teacher capture")
            self._update(
                phase="preparing",
                active=True,
                holds_world=True,
                job_id=f"capture-{uuid.uuid4().hex[:16]}",
                error=None,
                log="",
                results=[],
                cancel_requested=False,
            )
            self._launch(parameters)
        finally:
            self.world_lock.release()
        return self.state()

    def _launch(self, parameters: dict | None) -> None:
        self._thread = threading.Thread(
            target=self._run, args=(parameters,), daemon=True, name="garage-native-capture"
        )
        self._thread.start()

    def _run(self, parameters: dict | None) -> None:
        job_id = self.state()["job_id"]
        try:
            client = self.application.world_worker.research()
            if parameters is not None:
                catalog = client.tasks()
                task = next(
                    (item for item in catalog["tasks"] if item["id"] == "teacher_capture"), None
                )
                if task is None:
                    raise ValueError(
                        "update the Windows checkout and restart its usual Worker command"
                    )
                if self.state().get("cancel_requested") or self._stop.is_set():
                    self._update(phase="cancelled", active=False, holds_world=False)
                    return
                with self.world_lock:
                    self.application.preview.cancel_preparation()
                    self.application.preview.stop_for_drive()
                if self.state().get("cancel_requested") or self._stop.is_set():
                    self._update(phase="cancelled", active=False, holds_world=False)
                    return
                try:
                    client.submit(
                        task_id="teacher_capture",
                        job_id=job_id,
                        parameters=parameters,
                        task_sha256=task["manifest_sha256"],
                        acknowledge_world_reload=True,
                    )
                except WorldWorkerError as error:
                    if error.status is not None and 400 <= error.status < 500:
                        raise
                    self._update(phase="reconnecting", error=str(error))
            while not self._stop.is_set():
                try:
                    remote = client.status(job_id)
                    if remote["status"] not in {"queued", "running", "cancelling"}:
                        break
                    if self.state().get("cancel_requested") and remote["status"] != "cancelling":
                        client.cancel(job_id)
                    self._update(phase=remote["status"], error=None, log=client.log(job_id)["text"])
                except WorldWorkerError as error:
                    if error.status == 404:
                        raise ValueError("capture was not accepted; it is safe to retry") from error
                    self._update(phase="reconnecting", error=str(error))
                self._stop.wait(1)
            else:
                return  # Retain state so a WebUI restart can resume observation.
            self._update(holds_world=False)
            if not remote.get("cleanup_confirmed"):
                raise RuntimeError(
                    f"{(remote.get('result') or {}).get('error') or remote.get('task_error') or remote['status']}. "
                    "Capture did not confirm cleanup; no dataset was imported. Garage remains available."
                )
            if remote["status"] == "cancelled":
                self._update(phase="cancelled", active=False, error=None)
                return
            if remote["status"] != "succeeded":
                raise RuntimeError(
                    (remote.get("result") or {}).get("error")
                    or remote.get("task_error")
                    or remote["status"]
                )
            self._update(phase="receiving", error=None)
            results = self._receive(client, job_id)
            self._update(phase="saved", active=False, error=None, results=results)
        except Exception as error:
            self._update(
                phase="failed",
                active=False,
                error=str(error),
                holds_world=False,
            )

    def _receive(self, client, job_id: str) -> list[str]:
        root = self.application.workspace
        folder = self.application.jobs.sessions_root / "native" / job_id
        folder.mkdir(parents=True, exist_ok=True)
        archive = folder / "artifacts.zip"
        # Downloads are restartable without overwriting a previously received file.
        if archive.exists():
            from ..native.research_jobs import digest

            if digest(archive) != client.status(job_id)["result"]["archive"]["sha256"]:
                raise ValueError("retained capture archive checksum mismatch")
        else:
            client.fetch(job_id, archive)
        from ..dataset.episode_replay import replay_teacher_episode
        from ..native.teacher_verify import verify_teacher_dataset

        with tempfile.TemporaryDirectory(prefix="native-import-", dir=folder) as temporary:
            extracted = Path(temporary) / "artifacts"
            unpack_capture(archive, extracted)
            source = extracted / "datasets" / job_id
            report = verify_teacher_dataset(source)
            if report["status"] != "passed":
                raise ValueError(
                    "received dataset failed verification: " + "; ".join(report["errors"])
                )
            target = root / "datasets" / job_id
            if target.is_symlink() or target.parent.is_symlink():
                raise ValueError("dataset destination must not be a symbolic link")
            if not target.exists():
                shutil.copytree(source, target)
            else:
                # Restart after a completed import: verify and compare release identity.
                if (target / "checksums.sha256").read_bytes() != (
                    source / "checksums.sha256"
                ).read_bytes():
                    raise FileExistsError("a different dataset already has this capture ID")
                if verify_teacher_dataset(target)["status"] != "passed":
                    raise ValueError("previously imported dataset is incomplete")
        self._update(phase="preparing replay")
        dataset = json.loads((target / "dataset.json").read_text())
        episodes = sorted({row["episode_id"] for row in dataset["samples"]})
        outputs = []
        for index, episode in enumerate(episodes):
            run_id = f"{job_id}-{index + 1}"
            run = root / "runs" / run_id
            if not run.exists():
                replay_teacher_episode(
                    target, episode_id=episode, runs_root=root / "runs", run_id=run_id
                )
            elif (
                run.is_symlink()
                or json.loads((run / "manifest.json").read_text())["status"] != "success"
            ):
                raise ValueError("previous replay is incomplete; the verified dataset is retained")
            outputs.append(str(run.relative_to(root)))
        return outputs

    def cancel(self) -> dict:
        state = self.state()
        if state["active"] and state["holds_world"]:
            self._update(cancel_requested=True)
        return self.state()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
