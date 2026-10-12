"""Start existing product processes; never take ownership of an external server."""

from __future__ import annotations

import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import webbrowser
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

import psutil

from .config import (
    CARLA_VERSION,
    LauncherConfig,
    LauncherError,
    atomic_json,
    find_carla,
    native_platform,
    operation_lock,
    read_json,
    worker_token,
)
from .downloads import Report, model_status


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
        return False


def _process(entry: dict) -> psutil.Process | None:
    """A PID alone is insufficient: reject reused PIDs and malformed receipts."""
    if type(entry.get("pid")) is not int or type(entry.get("create_time")) not in {int, float}:
        return None
    try:
        process = psutil.Process(entry["pid"])
        if abs(process.create_time() - entry["create_time"]) > 0.01:
            return None
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return None
        return process
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise LauncherError("Local service requests must not redirect.")


def _local_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirects())


def http_json(
    url: str, *, token: str | None = None, body: dict | None = None,
    operator_token: str | None = None, timeout: float = 2,
) -> dict:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    if operator_token:
        headers["X-Operator-Token"] = operator_token
    data = None if body is None else json.dumps(body).encode()
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    # Loopback requests must not be sent through an ambient HTTP proxy.
    with _local_opener().open(request, timeout=timeout) as response:
        payload = json.loads(response.read(1024 * 1024))
    if not isinstance(payload, dict):
        raise LauncherError("Local service returned an invalid JSON response.")
    return payload


def web_ready(config: LauncherConfig) -> None:
    http_json(config.web_url + "api/bootstrap")
    with _local_opener().open(config.web_url, timeout=2) as response:
        if response.headers.get_content_type() != "text/html" or not response.read(4096):
            raise LauncherError("The web console entrypoint is unavailable; repair the installation.")


class LocalStack:
    def __init__(self, root: Path, report: Report = print) -> None:
        self.root = root.resolve()
        self.directory = self.root / ".launcher"
        self.state_path = self.directory / "services.json"
        self.report = report
        self._receipt_lock = threading.RLock()

    @property
    def config(self) -> LauncherConfig:
        return LauncherConfig.load(self.root)

    def state(self) -> dict:
        state = read_json(self.state_path, {"schema_version": 1, "services": {}})
        if not isinstance(state, dict) or not isinstance(state.get("services"), dict):
            raise LauncherError("Launcher service receipt is malformed; inspect its private state file.")
        return state

    def active(self) -> bool:
        return any(
            _process(entry) or any(_process(child) for child in entry.get("descendants", []))
            for entry in self.state()["services"].values()
        )

    def configure(self, config: LauncherConfig) -> None:
        with operation_lock(self.root):
            if self.active():
                raise LauncherError("Stop launcher-owned services before changing configuration.")
            config.save(self.root)

    def _server_facts(self, config: LauncherConfig) -> dict:
        # The official client supplies genuine read-only version/world/map facts.
        import carla

        client = carla.Client("127.0.0.1", config.carla_port)
        client.set_timeout(1.0)
        server_version = str(client.get_server_version())
        client_version = str(client.get_client_version())
        if server_version != CARLA_VERSION or client_version != CARLA_VERSION:
            raise LauncherError(
                f"CARLA/PythonAPI must both be {CARLA_VERSION}; "
                f"server={server_version}, client={client_version}."
            )
        return {"version": server_version, "map": str(client.get_world().get_map().name)}

    def snapshot(self) -> dict:
        config = self.config
        receipt = self.state()
        rows: dict[str, dict] = {}
        server = {"status": "stopped", "detail": f"127.0.0.1:{config.carla_port}"}
        if port_open(config.carla_port):
            try:
                facts = self._server_facts(config)
                entry = receipt["services"].get("carla", {})
                server = {
                    "status": "ready", "detail": f"{facts['version']} · {facts['map']}",
                    "owned": _process(entry) is not None or any(
                        _process(child) for child in entry.get("descendants", [])
                    ),
                }
            except (ImportError, RuntimeError, OSError) as error:
                server = {"status": "unavailable", "detail": str(error)}
        rows["carla"] = server
        token = worker_token(self.root)
        for role in ("worker", "web"):
            entry = receipt["services"].get(role, {})
            alive = _process(entry) is not None
            row = {"status": "running" if alive else "stopped", "owned": alive}
            if alive:
                try:
                    if role == "worker":
                        payload = http_json(config.worker_url + "/v1/health", token=token)
                        row["status"] = "ready" if payload.get("ready") is True else "unavailable"
                        row["detail"] = str(payload.get("status", "unknown"))
                    else:
                        web_ready(config)
                        row["status"], row["detail"] = "ready", config.web_url
                except (OSError, ValueError, LauncherError) as error:
                    row["detail"] = str(error)
                    row["status"] = "unavailable"
            else:
                row["detail"] = "Not started by this launcher"
            rows[role] = row
        try:
            models = model_status(self.root)
        except (OSError, ValueError, LauncherError) as error:
            models = {"ready": False, "error": str(error), "present": 0, "total": 0}
        return {
            "services": rows, "models": models, "config": asdict(config),
            "web_url": config.web_url, "owned_services_active": self.active(),
            "ready": all(row["status"] == "ready" for row in rows.values()),
        }

    def read_logs(self, offsets: dict[str, int]) -> tuple[list[dict], dict[str, int]]:
        """Read only new, bounded diagnostic output, including the last failed run."""
        receipt = self.state()
        entries = receipt["services"] or receipt.get("last_run", {}).get("services", {})
        updated, chunks = dict(offsets), []
        token = worker_token(self.root)
        for role, entry in entries.items():
            path = Path(entry.get("log_path", ""))
            if (
                not path.resolve().is_relative_to((self.directory / "logs").resolve())
                or path.is_symlink() or not path.is_file()
            ):
                continue
            size = path.stat().st_size
            previous = offsets.get(str(path), 0)
            start = max(0, size - 16_384, previous if previous <= size else 0)
            if start == size:
                continue
            with path.open("rb") as stream:
                stream.seek(start)
                data = stream.read(16_384)
                updated[str(path)] = stream.tell()
            text = data.decode("utf-8", errors="replace")
            if token:
                text = text.replace(token, "[private credential]")
            chunks.append({"role": role, "path": str(path), "text": text})
        return chunks, updated

    def _spawn(self, role: str, command: list[str], env: dict, state: dict) -> dict:
        run = state["run_id"]
        logs = self.directory / "logs" / run
        logs.mkdir(parents=True, exist_ok=True)
        log_path = logs / (role + ".log")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" and role != "carla" else 0
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(
                command, cwd=self.root if role != "carla" else state["carla_cwd"],
                env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                creationflags=flags, start_new_session=os.name != "nt",
            )
        try:
            created = psutil.Process(process.pid).create_time()
        except psutil.NoSuchProcess as error:
            raise LauncherError(f"{role} exited immediately; inspect {log_path}.") from error
        entry = {
            "pid": process.pid, "create_time": created, "role": role,
            "log_path": str(log_path), "control_file": str(self.directory / "control" / (role + "-" + run)),
            "descendants": [],
        }
        with self._receipt_lock:
            state["services"][role] = entry
            atomic_json(self.state_path, state)
        return entry

    def _remember_children(self, entry: dict, state: dict) -> None:
        process = _process(entry)
        if process is None:
            return
        try:
            with self._receipt_lock:
                known = {child["pid"]: child for child in entry.get("descendants", [])}
                for child in process.children(recursive=True):
                    known[child.pid] = {"pid": child.pid, "create_time": child.create_time()}
                entry["descendants"] = list(known.values())
                atomic_json(self.state_path, state)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return

    def _environment(self, config: LauncherConfig, token: str) -> dict[str, str]:
        environment = dict(os.environ)
        environment.update({
            "CARLA_WORLD_WORKER_TOKEN": token, "CARLA_HOST": "127.0.0.1",
            "CARLA_PORT": str(config.carla_port), "CARLA_WORKSPACE": str(self.root),
            "CARLA_WORLD_WORKER_URL": config.worker_url,
            "CARLA_WORLD_WORKER_PORT": str(config.worker_port),
            "CARLA_OPERATOR_BIND": "127.0.0.1", "CARLA_OPERATOR_PORT": str(config.web_port),
            "CARLA_ENABLE_EXPERIMENTAL": str(config.enable_experimental).lower(),
            "PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1",
            "PATH": str(self.directory / "bin") + os.pathsep + os.environ.get("PATH", ""),
        })
        return environment

    def start(self, *, open_browser: bool = True) -> dict:
        native_platform()
        if sys.version_info[:2] != (3, 12):
            raise LauncherError("Use the setup-installed Python 3.12 environment to start the local stack.")
        with operation_lock(self.root):
            config = self.config
            if self.active():
                snapshot = self.snapshot()
                if snapshot["ready"]:
                    if open_browser:
                        webbrowser.open(config.web_url)
                    return snapshot
                raise LauncherError("Some owned services are already running; stop them before retrying.")
            try:
                if importlib.metadata.version("carla") != CARLA_VERSION:
                    raise LauncherError("Install/repair the native-host profile before Start.")
            except importlib.metadata.PackageNotFoundError as error:
                raise LauncherError("Choose Install / Repair first; CARLA PythonAPI is missing.") from error
            if not model_status(self.root)["ready"]:
                raise LauncherError("Choose Install / Repair first; some model weights are not hydrated.")
            from ..operator.product_console import console_available

            if not console_available():
                raise LauncherError("The compiled web console is missing; repair the source installation.")
            for role, port in (("Worker", config.worker_port), ("web app", config.web_port)):
                if port_open(port):
                    raise LauncherError(f"{role} port {port} is occupied; select a free port in Setup.")
            token = worker_token(self.root, create=True)
            assert token is not None
            state: dict[str, Any] = {
                "schema_version": 1, "run_id": uuid.uuid4().hex, "services": {},
                "config": asdict(config), "phase": "starting",
            }
            env = self._environment(config, token)
            try:
                if port_open(config.carla_port):
                    self._server_facts(config)
                    self.report("Using the existing CARLA server; it remains externally owned.")
                else:
                    executable = find_carla(self.root, config.carla_root)
                    if executable is None:
                        raise LauncherError("Select the CARLA folder in Setup, or enable its official download.")
                    state["carla_cwd"] = str(executable.parent)
                    command = [str(executable), f"-carla-rpc-port={config.carla_port}",
                               f"-quality-level={config.quality}"]
                    if os.name != "nt":
                        command.insert(0, "/bin/bash")
                    if config.offscreen:
                        command.append("-RenderOffScreen")
                    carla_entry = self._spawn("carla", command, env, state)
                    tracking = threading.Event()

                    def track_bootstrap_children() -> None:
                        while not tracking.wait(0.025) and _process(carla_entry):
                            self._remember_children(carla_entry, state)

                    tracker = threading.Thread(target=track_bootstrap_children, daemon=True)
                    tracker.start()
                    self.report("Starting CARLA; waiting for its real version/world API.")
                    deadline = time.monotonic() + config.startup_timeout_seconds
                    try:
                        while True:
                            self._remember_children(carla_entry, state)
                            try:
                                self._server_facts(config)
                                break
                            except LauncherError:
                                raise
                            except (ImportError, RuntimeError, OSError) as error:
                                if time.monotonic() >= deadline:
                                    raise LauncherError(
                                        "CARLA readiness timed out; inspect the CARLA log in Logs."
                                    ) from error
                                if _process(carla_entry) is None and not any(
                                    _process(child) for child in carla_entry["descendants"]
                                ):
                                    raise LauncherError(
                                        "CARLA exited before readiness; inspect its log."
                                    ) from error
                                time.sleep(0.5)
                    finally:
                        tracking.set()
                        tracker.join(timeout=1)
                self.report("CARLA 0.9.16 responded; starting the authenticated loopback Worker.")
                self.directory.joinpath("control").mkdir(exist_ok=True)
                for role in ("worker", "web"):
                    control = self.directory / "control" / (role + "-" + state["run_id"])
                    if role == "worker":
                        arguments = [
                            "--internal-worker", "--bind", "127.0.0.1", "--port", str(config.worker_port),
                            "--carla-host", "127.0.0.1", "--carla-port", str(config.carla_port),
                            "--traffic-manager-port", str(config.traffic_manager_port),
                            "--expected-carla-version", CARLA_VERSION,
                        ]
                    else:
                        arguments = [
                            "--bind", "127.0.0.1", "--port", str(config.web_port),
                            "--workspace", str(self.root), "--carla-host", "127.0.0.1",
                            "--carla-port", str(config.carla_port), "--world-worker-url", config.worker_url,
                            "--world-worker-port", str(config.worker_port),
                        ]
                        if not config.enable_experimental:
                            arguments.append("--no-enable-experimental")
                    command = [sys.executable, "-m", "carla_vision.launcher.child",
                               "--role", role, "--control-file", str(control), "--", *arguments]
                    entry = self._spawn(role, command, env, state)
                    deadline = time.monotonic() + 45
                    while True:
                        if _process(entry) is None:
                            raise LauncherError(f"{role} exited; inspect its log in Logs.")
                        try:
                            if role == "worker":
                                payload = http_json(config.worker_url + "/v1/health", token=token)
                                if payload.get("ready") is not True:
                                    raise LauncherError("Worker is not ready for this CARLA server.")
                                if payload.get("carla", {}).get("server_version") != CARLA_VERSION:
                                    raise LauncherError("Worker reported a different CARLA version.")
                            else:
                                web_ready(config)
                            break
                        except (OSError, ValueError, LauncherError) as error:
                            if time.monotonic() >= deadline:
                                raise LauncherError(f"{role} readiness failed: {error}") from error
                            time.sleep(0.4)
                    self.report(f"{role.capitalize()} is ready.")
                state["phase"] = "ready"
                atomic_json(self.state_path, state)
                if open_browser:
                    webbrowser.open(config.web_url)
                return self.snapshot()
            except BaseException:
                self._stop_processes(state)
                for chunk in self.read_logs({})[0]:
                    self.report(f"{chunk['role'].capitalize()} log: {chunk['path']}\n{chunk['text']}")
                raise

    def _stop_processes(self, state: dict) -> None:
        for role in ("web", "worker", "carla"):
            entry = state["services"].get(role)
            if not entry:
                continue
            self._remember_children(entry, state)
            children = [_process(child) for child in entry.get("descendants", [])]
            process = _process(entry)
            if process is not None:
                if role in {"web", "worker"}:
                    control = Path(entry["control_file"])
                    control.resolve().relative_to(self.directory.resolve() / "control")
                    control.parent.mkdir(exist_ok=True)
                    control.touch()
                else:
                    process.terminate()
                try:
                    process.wait(timeout=20)
                except psutil.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for child in reversed(children):
                if child is None:
                    continue
                try:
                    if child.is_running():
                        child.terminate()
                        try:
                            child.wait(timeout=5)
                        except psutil.TimeoutExpired:
                            child.kill()
                except psutil.NoSuchProcess:
                    pass
        state["last_run"] = {"run_id": state.get("run_id"), "services": state["services"]}
        state["phase"] = "stopped"
        state["services"] = {}
        atomic_json(self.state_path, state)

    def _worker_health(self, config: LauncherConfig) -> dict | None:
        try:
            return http_json(config.worker_url + "/v1/health", token=worker_token(self.root))
        except urllib.error.HTTPError:
            # Authentication and server errors cannot prove an absent Worker.
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            return None

    @staticmethod
    def _worker_idle(health: dict) -> bool:
        return (
            "active_scene" in health and health["active_scene"] is None
            and health.get("research_job_id") is None
            and health.get("busy", False) is False
            and health.get("status") not in {"busy", "preparing", "research_running"}
            and (health.get("preparation") or {}).get("status") != "preparing"
        )

    @staticmethod
    def _owners_finished(drive: dict, preview: dict, capture: dict) -> bool:
        errors = drive.get("cleanup_errors") or []
        if errors or drive.get("cleanup_error_count") or (
            drive.get("world_worker", {}).get("cleanup_guard_passed") is False
        ):
            raise LauncherError("Drive cleanup failed: " + "; ".join(map(str, errors)))
        if preview.get("cleanup_errors") or preview.get("error"):
            raise LauncherError(
                "Garage preview cleanup is unconfirmed: "
                + str(preview.get("error") or preview["cleanup_errors"])
            )
        return (
            drive.get("status") in {"idle", "success", "failed"}
            and drive.get("recording") is False
            and preview.get("active") is False
            and capture.get("active") is False
            and capture.get("holds_world") is False
        )

    def _owner_states(self, config: LauncherConfig) -> tuple[dict, dict, dict]:
        return (
            http_json(config.web_url + "api/drive/state"),
            http_json(config.web_url + "api/garage/preview/state"),
            http_json(config.web_url + "api/garage/capture"),
        )

    def _cleanup_scene(self, config: LauncherConfig) -> None:
        """Wait for the actual owners to finish recorders, imports and scene cleanup."""
        bootstrap = http_json(config.web_url + "api/bootstrap")
        operator_token = bootstrap.get("token")
        if not isinstance(operator_token, str):
            raise LauncherError("Web app did not provide its local control token.")
        drive, preview, capture = self._owner_states(config)
        health = self._worker_health(config)
        capture_was_active = capture.get("active") is True
        requests = []
        if drive.get("status") in {"starting", "running", "stopping"}:
            if not drive.get("session_id"):
                raise LauncherError("Active Drive has no session identity; inspect the web app.")
            requests.append(("api/drive/stop", {"session_id": drive["session_id"]}))
        if capture_was_active:
            requests.append(("api/garage/capture/cancel", {}))
        if preview.get("active") or health is not None:
            requests.append(("api/garage/preview/stop", {}))
        for route, body in requests:
            # Parse every owner response; 403/404/409 are failures, never a clean stop.
            result = http_json(
                config.web_url + route, body=body, operator_token=operator_token, timeout=35,
            )
            if result.get("cleanup_errors"):
                raise LauncherError(f"{route} reported cleanup errors: {result['cleanup_errors']}")
        deadline = time.monotonic() + 120
        while True:
            drive, preview, capture = self._owner_states(config)
            finished = self._owners_finished(drive, preview, capture)
            health = self._worker_health(config)
            if finished:
                if health is None:
                    if capture.get("phase") not in {"idle", "saved", "cancelled"}:
                        raise LauncherError("Worker is unavailable and capture cleanup is unconfirmed.")
                    self.report("All web owners are idle; stopping the remaining owned processes.")
                    return
                if self._worker_idle(health):
                    if capture.get("job_id") and (
                        capture_was_active or capture.get("phase") == "failed"
                    ):
                        try:
                            job = http_json(
                                config.worker_url + "/v1/research/jobs/"
                                + quote(str(capture["job_id"]), safe=""),
                                token=worker_token(self.root),
                            )
                        except urllib.error.HTTPError as error:
                            if error.code != 404:
                                raise
                            # The native owner proves the job was never accepted and is idle.
                        else:
                            if job.get("cleanup_confirmed") is not True:
                                raise LauncherError("Native capture did not confirm actor cleanup.")
                    elif capture.get("phase") == "failed":
                        raise LauncherError("Capture state is unreadable; inspect its retained state.")
                    return
            if time.monotonic() >= deadline:
                raise LauncherError(
                    "Drive recording, preview or capture is still active; "
                    "services were preserved. Wait for Stop & Save in the web app, then retry."
                )
            time.sleep(0.5)

    def stop(self) -> dict:
        with operation_lock(self.root):
            config = self.config
            state = self.state()
            if _process(state["services"].get("web", {})) is not None:
                self.report("Stopping Drive and Garage through their existing owners.")
                try:
                    self._cleanup_scene(config)
                except (OSError, ValueError, LauncherError) as error:
                    raise LauncherError(
                        "Cannot confirm scene/recording cleanup. "
                        f"Owned services were kept running. {error}"
                    ) from error
            elif self.active():
                health = self._worker_health(config)
                if health is None or not self._worker_idle(health):
                    raise LauncherError(
                        "Web app is unavailable and the Worker cannot confirm an idle scene. "
                        "Services were preserved; inspect their logs and recover the scene owner."
                    )
            self._stop_processes(state)
            self.report("Launcher-owned services stopped. External CARLA was preserved.")
            return self.snapshot()
