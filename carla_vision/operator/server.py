"""Local-only HTTP operator panel backed by the existing research CLIs."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
import threading
import webbrowser
from collections.abc import Mapping, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..discovery import discover_carla_servers
from .artifacts import _VIDEO_SUFFIXES as _VIDEO_SUFFIXES
from .artifacts import ArtifactStore
from .artifacts import _path_parts as _path_parts
from .artifacts import _safe_mime_type as _safe_mime_type
from .artifacts import _walk_has_symlink as _walk_has_symlink
from .catalog import build_catalog
from .commands import build_command_plan
from .configuration import (
    CONFIGURATION_SCHEMA_VERSION,
    build_configuration_evidence,
    build_situation_request,
)
from .contracts import OperatorJobRequest
from .drive import DriveSessionManager
from .jobs import JobManager
from .recording_preview import RecordingPreviews
from .situations import PROP_PRESETS, WEATHER_PRESETS, SituationSpec, save_situation_suite
from .world_worker_client import WorldWorkerClient

STATIC_ROOT = Path(__file__).resolve().parent / "static"
_LOCAL_BINDS = frozenset({"127.0.0.1", "::1", "localhost"})
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_APPLICATION_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; "
    "img-src 'self' data:; media-src 'self'; object-src 'none'; "
    "base-uri 'none'; frame-ancestors 'none'"
)
_ARTIFACT_CSP = (
    "default-src 'none'; script-src 'none'; style-src 'unsafe-inline'; "
    "img-src data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; sandbox"
)


class OperatorApplication:
    def __init__(
        self,
        *,
        workspace: str | Path,
        sessions_root: str | Path,
        carla_host: str,
        carla_port: int,
        world_worker: WorldWorkerClient | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve(strict=True)
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.world_worker = world_worker
        self.token = secrets.token_urlsafe(24)
        self._discovery_lock = threading.Lock()
        self.artifacts = ArtifactStore(self.workspace)
        self.recording_previews = RecordingPreviews()
        self.jobs = JobManager(
            workspace=self.workspace,
            sessions_root=sessions_root,
        )
        self.drive = DriveSessionManager(
            workspace=self.workspace,
            carla_host=self.carla_host,
            carla_port=self.carla_port,
            world_worker=world_worker,
        )

    def bootstrap(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "token": self.token,
            "catalog": build_catalog(
                self.workspace,
                carla_host=self.carla_host,
                carla_port=self.carla_port,
            ),
            "weather_presets": list(WEATHER_PRESETS),
            "prop_presets": list(PROP_PRESETS),
            "jobs": self.jobs.list(),
            "drive": self.drive.state(),
        }

    def close(self) -> None:
        self.drive.shutdown()
        self.recording_previews.close()

    def discover_carla(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise TypeError("CARLA discovery request must be an object")
        if raw:
            raise ValueError("CARLA discovery request does not accept parameters")
        with self._discovery_lock:
            result = discover_carla_servers(port=self.carla_port)
        result["configured_host"] = self.carla_host
        result["configured_match"] = any(
            server["host"] == self.carla_host for server in result["servers"]
        )
        return result

    def save_situation(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise TypeError("situation request must be an object")
        canonical = frozenset(str(key) for key in raw) == {
            "schema_version",
            "session",
            "situation",
        }
        if canonical:
            if str(raw["schema_version"]) != CONFIGURATION_SCHEMA_VERSION:
                raise ValueError(
                    "situation schema_version must be "
                    f"{CONFIGURATION_SCHEMA_VERSION!r}"
                )
            session_raw = raw["session"]
            scene_raw = session_raw.get("scene") if isinstance(session_raw, Mapping) else None
            current_map = None
            if isinstance(scene_raw, Mapping) and str(scene_raw.get("mapName")) == "current":
                catalog = self.drive.catalog()
                current_map = catalog.get("map") if isinstance(catalog, Mapping) else None
            resolved = build_situation_request(
                raw["session"],
                raw["situation"],
                current_map=str(current_map) if current_map else None,
            )
        else:
            resolved = dict(raw)
        spec = SituationSpec.from_mapping(resolved)
        output = save_situation_suite(spec, workspace=self.workspace)
        suite = json.loads(output.read_text(encoding="utf-8"))
        response = {
            "status": "saved",
            "path": output.relative_to(self.workspace).as_posix(),
            "suite_id": suite["suite_id"],
            "recipe_id": suite["recipes"][0]["recipe_id"],
        }
        if canonical:
            response["configuration"] = build_configuration_evidence(
                requested=raw,
                resolved=spec.as_dict(),
                applied={
                    "status": "saved_offline",
                    "carla_mutated": False,
                    "path": response["path"],
                },
            )
        return response

    def start_job(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise TypeError("job request must be an object")
        request = OperatorJobRequest.from_mapping(raw)
        plan = build_command_plan(
            request,
            workspace=self.workspace,
            executable=sys.executable,
        )
        return self.jobs.submit(request, plan)

    def inspect_research_object(self, raw: str) -> dict[str, Any]:
        return self.artifacts.inspect_research_object(raw)

    def artifact_path(self, object_raw: str, artifact_raw: str) -> tuple[Path, bool]:
        return self.artifacts.artifact_path(object_raw, artifact_raw)


class OperatorHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        application: OperatorApplication,
    ) -> None:
        self.application = application
        super().__init__(server_address, OperatorRequestHandler)

    def server_close(self) -> None:
        self.application.close()
        super().server_close()


class OperatorRequestHandler(BaseHTTPRequestHandler):
    def handle(self) -> None:
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    server: OperatorHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def _headers(
        self,
        status: HTTPStatus,
        *,
        content_type: str,
        length: int,
        cache: str = "no-store",
        content_disposition: str | None = None,
        content_security_policy: str = _APPLICATION_CSP,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        if content_disposition is not None:
            self.send_header("Content-Disposition", content_disposition)
        for name, value in (extra_headers or {}).items():
            self.send_header(str(name), str(value))
        self.send_header("Content-Security-Policy", content_security_policy)
        self.end_headers()

    def _json(self, status: HTTPStatus, payload: Any) -> None:
        body = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        self._headers(
            status,
            content_type="application/json; charset=utf-8",
            length=len(body),
        )
        self.wfile.write(body)

    def _bytes(
        self,
        status: HTTPStatus,
        payload: bytes,
        *,
        content_type: str,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._headers(
            status,
            content_type=content_type,
            length=len(payload),
            extra_headers=extra_headers,
        )
        self.wfile.write(payload)

    def _file(
        self,
        path: Path,
        *,
        cache: str = "no-store",
        attachment: bool = False,
        untrusted_artifact: bool = False,
    ) -> None:
        size = path.stat().st_size
        mime = _safe_mime_type(path)
        start, end = 0, size - 1
        status = HTTPStatus.OK
        extra_headers = {"Accept-Ranges": "bytes"}
        requested = self.headers.get("Range")
        # If-Range cannot be validated without validators: send the full file.
        if requested and not self.headers.get("If-Range"):
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip())
            # Unsupported/multiple ranges are ignored, as allowed by HTTP.
            if match and any(match.groups()):
                first, last = match.groups()
                if first:
                    start = int(first)
                    end = min(int(last), size - 1) if last else size - 1
                else:
                    start = max(0, size - int(last))
                if start > end or start >= size:
                    self._headers(
                        HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                        content_type=mime, length=0, cache=cache,
                        extra_headers={"Content-Range": f"bytes */{size}"},
                    )
                    return
                status = HTTPStatus.PARTIAL_CONTENT
                extra_headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        safe_name = "".join(
            character
            if character.isascii() and (character.isalnum() or character in "._-")
            else "_"
            for character in path.name
        )
        self._headers(
            status,
            content_type=mime,
            length=end - start + 1,
            cache=cache,
            content_disposition=(f'attachment; filename="{safe_name}"' if attachment else None),
            content_security_policy=(_ARTIFACT_CSP if untrusted_artifact else _APPLICATION_CSP),
            extra_headers=extra_headers,
        )
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining and (chunk := stream.read(min(1024 * 1024, remaining))):
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _body(self) -> Any:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("request body is required")
        length = int(raw_length)
        if not 0 < length <= 1024 * 1024:
            raise ValueError("request body must be between 1 byte and 1 MiB")
        payload = self.rfile.read(length)
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("request body must be valid UTF-8 JSON") from error

    def _authorized(self) -> bool:
        return secrets.compare_digest(
            self.headers.get("X-Operator-Token", ""),
            self.server.application.token,
        )

    def _error(self, error: BaseException) -> None:
        if isinstance(error, (BrokenPipeError, ConnectionResetError)):
            self.close_connection = True
            return
        if isinstance(error, (KeyError, FileNotFoundError)):
            status = HTTPStatus.NOT_FOUND
        elif isinstance(error, (FileExistsError, PermissionError, RuntimeError)):
            status = HTTPStatus.CONFLICT
        elif isinstance(error, (TypeError, ValueError)):
            status = HTTPStatus.BAD_REQUEST
        else:
            status = HTTPStatus.INTERNAL_SERVER_ERROR
        self._json(
            status,
            {
                "error": {
                    "type": type(error).__qualname__,
                    "message": str(error),
                }
            },
        )

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/":
                self._file(STATIC_ROOT / "index.html", cache="no-cache")
                return
            if path.startswith("/static/"):
                name = path.removeprefix("/static/")
                if "/" in name or name not in {"app.css", "app.js"}:
                    raise FileNotFoundError("static asset not found")
                self._file(STATIC_ROOT / name, cache="no-cache")
                return
            if path == "/api/bootstrap":
                self._json(HTTPStatus.OK, self.server.application.bootstrap())
                return
            if path == "/api/jobs":
                self._json(HTTPStatus.OK, {"jobs": self.server.application.jobs.list()})
                return
            if path == "/api/drive/catalog":
                self._json(HTTPStatus.OK, self.server.application.drive.catalog())
                return
            if path == "/api/drive/state":
                self._json(HTTPStatus.OK, self.server.application.drive.state())
                return
            if path == "/api/drive/frame.jpg":
                view = parse_qs(parsed.query).get("view", ["raw"])[0]
                sequence, payload = self.server.application.drive.frame(view)
                self._bytes(
                    HTTPStatus.OK,
                    payload,
                    content_type="image/jpeg",
                    extra_headers={"X-Drive-Frame-Sequence": str(sequence)},
                )
                return
            if path == "/api/drive/stream.mjpg":
                view = parse_qs(parsed.query).get("view", ["raw"])[0]
                self._drive_stream(view)
                return
            if path == "/api/evidence":
                raw = parse_qs(parsed.query).get("path", [""])[0]
                self._json(
                    HTTPStatus.OK,
                    self.server.application.inspect_research_object(raw),
                )
                return
            if path == "/api/recording-preview":
                key = parse_qs(parsed.query).get("id", [""])[0]
                self._file(self.server.application.recording_previews.file(key), untrusted_artifact=True)
                return
            if path.startswith("/api/jobs/") and path.endswith("/log"):
                job_id = path.removeprefix("/api/jobs/").removesuffix("/log").rstrip("/")
                stream = parse_qs(parsed.query).get("stream", ["stdout"])[0]
                self._json(
                    HTTPStatus.OK,
                    {
                        "job_id": job_id,
                        "stream": stream,
                        "text": self.server.application.jobs.log_tail(
                            job_id,
                            stream=stream,
                        ),
                    },
                )
                return
            if path.startswith("/api/jobs/"):
                job_id = path.removeprefix("/api/jobs/")
                self._json(HTTPStatus.OK, self.server.application.jobs.get(job_id))
                return
            if path == "/api/artifact":
                query = parse_qs(parsed.query)
                object_raw = query.get("object", [""])[0]
                artifact_raw = query.get("path", [""])[0]
                artifact_path, attachment = self.server.application.artifact_path(
                    object_raw,
                    artifact_raw,
                )
                self._file(
                    artifact_path,
                    attachment=attachment,
                    untrusted_artifact=True,
                )
                return
            raise FileNotFoundError("route not found")
        except BaseException as error:
            self._error(error)

    def _drive_stream(self, view: str) -> None:
        """Relay cached newest frames over one persistent browser response."""

        if view not in {"raw", "overlay", "voxel", "voxel_overlay"}:
            raise ValueError(
                "drive frame view must be raw, overlay, voxel or voxel_overlay"
            )
        sequence, payload = self.server.application.drive.wait_for_frame(
            view,
            -1,
            timeout=10.0,
        )
        boundary = "carla-drive"
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            f"multipart/x-mixed-replace; boundary={boundary}",
        )
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", _APPLICATION_CSP)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            while True:
                header = (
                    f"--{boundary}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(payload)}\r\n"
                    f"X-Drive-Frame-Sequence: {sequence}\r\n\r\n"
                ).encode("ascii")
                self.wfile.write(header)
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                try:
                    sequence, payload = self.server.application.drive.wait_for_frame(
                        view,
                        sequence,
                        timeout=5.0,
                    )
                except TimeoutError:
                    continue
        except (BrokenPipeError, ConnectionResetError, EOFError, OSError):
            return

    def do_POST(self) -> None:
        try:
            if not self._authorized():
                self._json(
                    HTTPStatus.FORBIDDEN,
                    {"error": {"type": "PermissionError", "message": "invalid UI token"}},
                )
                return
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/api/situations":
                self._json(
                    HTTPStatus.CREATED,
                    self.server.application.save_situation(self._body()),
                )
                return
            if path == "/api/recording-preview":
                body = self._body()
                source, _ = self.server.application.artifact_path(body["object"], body["path"])
                if source.suffix.lower() not in _VIDEO_SUFFIXES:
                    raise ValueError("Select a registered video recording")
                self._json(HTTPStatus.OK, self.server.application.recording_previews.request(source))
                return
            if path == "/api/discovery/carla":
                self._json(
                    HTTPStatus.OK,
                    self.server.application.discover_carla(self._body()),
                )
                return
            if path == "/api/jobs":
                self._json(
                    HTTPStatus.ACCEPTED,
                    self.server.application.start_job(self._body()),
                )
                return
            if path == "/api/drive/start":
                body = self._body()
                if not isinstance(body, Mapping):
                    raise TypeError("drive start request must be an object")
                self._json(
                    HTTPStatus.ACCEPTED,
                    self.server.application.drive.start(body),
                )
                return
            if path == "/api/drive/control":
                body = self._body()
                if not isinstance(body, Mapping):
                    raise TypeError("drive control request must be an object")
                self._json(
                    HTTPStatus.OK,
                    self.server.application.drive.control(body),
                )
                return
            if path == "/api/drive/weather":
                body = self._body()
                if not isinstance(body, Mapping):
                    raise TypeError("drive weather request must be an object")
                self._json(
                    HTTPStatus.ACCEPTED,
                    self.server.application.drive.weather(body),
                )
                return
            if path == "/api/drive/mode":
                body = self._body()
                if not isinstance(body, Mapping):
                    raise TypeError("drive mode request must be an object")
                self._json(
                    HTTPStatus.ACCEPTED,
                    self.server.application.drive.mode(body),
                )
                return
            if path == "/api/drive/mark":
                body = self._body()
                if not isinstance(body, Mapping):
                    raise TypeError("drive human-marker request must be an object")
                self._json(
                    HTTPStatus.CREATED,
                    self.server.application.drive.mark(body),
                )
                return
            if path == "/api/drive/emergency-stop":
                body = self._body()
                if not isinstance(body, Mapping):
                    raise TypeError("drive emergency-stop request must be an object")
                self._json(
                    HTTPStatus.ACCEPTED,
                    self.server.application.drive.emergency_stop(body),
                )
                return
            if path == "/api/drive/stop":
                body = self._body()
                if not isinstance(body, Mapping):
                    raise TypeError("drive stop request must be an object")
                self._json(
                    HTTPStatus.ACCEPTED,
                    self.server.application.drive.stop(body),
                )
                return
            if path.startswith("/api/jobs/") and path.endswith("/stop"):
                job_id = path.removeprefix("/api/jobs/").removesuffix("/stop").rstrip("/")
                self._json(
                    HTTPStatus.ACCEPTED,
                    self.server.application.jobs.stop(job_id),
                )
                return
            raise FileNotFoundError("route not found")
        except BaseException as error:
            self._error(error)


def create_server(
    *,
    workspace: str | Path,
    bind: str = "127.0.0.1",
    port: int = 8765,
    sessions_root: str | Path = "operator_sessions",
    carla_host: str = "172.20.10.7",
    carla_port: int = 2000,
    world_worker_url: str | None = None,
    world_worker_token: str | None = None,
) -> OperatorHTTPServer:
    if bind not in _LOCAL_BINDS:
        raise ValueError("the operator UI is local-only; bind to 127.0.0.1, ::1, or localhost")
    if not 0 <= port <= 65535:
        raise ValueError("port must be in [0, 65535]")
    if (world_worker_url is None) != (world_worker_token is None):
        raise ValueError("World Worker URL and bearer token must be configured together")
    world_worker = (
        None
        if world_worker_url is None or world_worker_token is None
        else WorldWorkerClient(world_worker_url, world_worker_token)
    )
    application = OperatorApplication(
        workspace=workspace,
        sessions_root=sessions_root,
        carla_host=carla_host,
        carla_port=carla_port,
        world_worker=world_worker,
    )
    return OperatorHTTPServer((bind, port), application)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local CARLA Vision operator panel for situation planning, "
            "recording, replay, training, analysis, and verification"
        )
    )
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--sessions-root", default="operator_sessions")
    parser.add_argument(
        "--carla-host",
        default="auto",
        help="CARLA server address, or 'auto' to discover one on the directly connected LAN",
    )
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--world-worker-url")
    parser.add_argument(
        "--world-worker-port",
        type=int,
        default=8766,
        help="World Worker port used when --world-worker-url=auto",
    )
    parser.add_argument(
        "--world-worker-token-env",
        default="CARLA_WORLD_WORKER_TOKEN",
        help="environment variable containing the World Worker bearer token",
    )
    parser.add_argument(
        "--enable-experimental",
        action="store_true",
        help="enable opt-in research autonomy and experimental Garage jobs",
    )
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args(argv)


def resolve_world_worker_token(args: argparse.Namespace) -> str | None:
    """Resolve the configured worker credential without exposing it to the browser."""

    if args.world_worker_url is None:
        return None
    if not _ENVIRONMENT_NAME.fullmatch(args.world_worker_token_env):
        raise ValueError("World Worker token environment variable name is invalid")
    token = os.environ.get(args.world_worker_token_env)
    if not token:
        raise RuntimeError(
            f"World Worker token environment variable {args.world_worker_token_env!r} is empty"
        )
    return token


def resolve_carla_host(host: str, port: int) -> str:
    """Resolve an explicit host or require one unambiguous discovered server."""

    value = str(host).strip()
    if value.lower() != "auto":
        if not value or any(character.isspace() for character in value):
            raise ValueError("CARLA host must be a non-empty address or 'auto'")
        return value
    result = discover_carla_servers(port=int(port))
    servers = result["servers"]
    if not servers:
        scopes = ", ".join(scope["network"] for scope in result["scopes"]) or "none"
        raise RuntimeError(f"no CARLA server was discovered on local scopes: {scopes}")
    if len(servers) != 1:
        hosts = ", ".join(server["host"] for server in servers)
        raise RuntimeError(
            f"multiple CARLA servers were discovered; choose one explicitly: {hosts}"
        )
    return str(servers[0]["host"])


def resolve_world_worker_url(value: str | None, carla_host: str, port: int) -> str | None:
    """Resolve ``auto`` to the Worker colocated with the discovered CARLA host."""

    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        raise ValueError("World Worker URL must not be empty")
    if raw.lower() != "auto":
        return raw
    if isinstance(port, bool) or not 1 <= int(port) <= 65535:
        raise ValueError("World Worker port must be in [1, 65535]")
    return f"http://{carla_host}:{int(port)}"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    carla_host = resolve_carla_host(args.carla_host, args.carla_port)
    world_worker_url = resolve_world_worker_url(
        args.world_worker_url,
        carla_host,
        args.world_worker_port,
    )
    world_worker_token = resolve_world_worker_token(args)
    server = create_server(
        workspace=args.workspace,
        bind=args.bind,
        port=args.port,
        sessions_root=args.sessions_root,
        carla_host=carla_host,
        carla_port=args.carla_port,
        world_worker_url=world_worker_url,
        world_worker_token=world_worker_token,
    )
    address, port = server.server_address[:2]
    display_host = "127.0.0.1" if address in {"0.0.0.0", "::"} else address
    url = f"http://{display_host}:{port}/"
    print(
        json.dumps(
            {
                "status": "ready",
                "url": url,
                "workspace": str(Path(args.workspace).expanduser().resolve()),
                "local_only": True,
                "world_worker_configured": args.world_worker_url is not None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        server.application.jobs.shutdown()
    return 0


__all__ = [
    "OperatorApplication",
    "OperatorHTTPServer",
    "OperatorRequestHandler",
    "create_server",
    "main",
    "parse_args",
    "resolve_carla_host",
    "resolve_world_worker_token",
]
