"""Local-only HTTP operator panel backed by the existing research CLIs."""

from __future__ import annotations

import argparse
import json
import mimetypes
import secrets
import sys
import webbrowser
from collections.abc import Mapping, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .catalog import build_catalog
from .commands import build_command_plan
from .contracts import OperatorJobRequest
from .jobs import JobManager
from .situations import PROP_PRESETS, WEATHER_PRESETS, SituationSpec, save_situation_suite

STATIC_ROOT = Path(__file__).resolve().parent / "static"
_LOCAL_BINDS = frozenset({"127.0.0.1", "::1", "localhost"})
_ARTIFACT_SUFFIXES = frozenset(
    {
        ".csv",
        ".jpeg",
        ".jpg",
        ".json",
        ".jsonl",
        ".log",
        ".md",
        ".mp4",
        ".png",
        ".svg",
        ".txt",
    }
)


class OperatorApplication:
    def __init__(
        self,
        *,
        workspace: str | Path,
        sessions_root: str | Path,
        carla_host: str,
        carla_port: int,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve(strict=True)
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.token = secrets.token_urlsafe(24)
        self.jobs = JobManager(
            workspace=self.workspace,
            sessions_root=sessions_root,
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
        }

    def save_situation(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise TypeError("situation request must be an object")
        spec = SituationSpec.from_mapping(raw)
        output = save_situation_suite(spec, workspace=self.workspace)
        suite = json.loads(output.read_text(encoding="utf-8"))
        return {
            "status": "saved",
            "path": output.relative_to(self.workspace).as_posix(),
            "suite_id": suite["suite_id"],
            "recipe_id": suite["recipes"][0]["recipe_id"],
        }

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

    def artifact_path(self, raw: str) -> Path:
        candidate = Path(unquote(raw)).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        resolved = candidate.resolve(strict=True)
        allowed = False
        for root_name in (
            "bundles",
            "datasets",
            "models",
            "operator_sessions",
            "reports",
            "runs",
        ):
            root = (self.workspace / root_name).resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                continue
            allowed = True
            break
        if not allowed or not resolved.is_file():
            raise ValueError("artifact path is outside the allow-listed project roots")
        if resolved.suffix.lower() not in _ARTIFACT_SUFFIXES:
            raise ValueError("artifact type is not available through the operator UI")
        if resolved.stat().st_size > 100 * 1024 * 1024:
            raise ValueError("artifact is too large for the operator preview endpoint")
        return resolved


class OperatorHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        application: OperatorApplication,
    ) -> None:
        self.application = application
        super().__init__(server_address, OperatorRequestHandler)


class OperatorRequestHandler(BaseHTTPRequestHandler):
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
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; media-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'",
        )
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

    def _file(self, path: Path, *, cache: str = "no-store") -> None:
        body = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._headers(
            HTTPStatus.OK,
            content_type=mime,
            length=len(body),
            cache=cache,
        )
        self.wfile.write(body)

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
        if isinstance(error, KeyError):
            status = HTTPStatus.NOT_FOUND
        elif isinstance(error, (FileExistsError, PermissionError, RuntimeError)):
            status = HTTPStatus.CONFLICT
        elif isinstance(error, (TypeError, ValueError, FileNotFoundError)):
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
                raw = parse_qs(parsed.query).get("path", [""])[0]
                self._file(self.server.application.artifact_path(raw))
                return
            raise FileNotFoundError("route not found")
        except BaseException as error:
            self._error(error)

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
            if path == "/api/jobs":
                self._json(
                    HTTPStatus.ACCEPTED,
                    self.server.application.start_job(self._body()),
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
) -> OperatorHTTPServer:
    if bind not in _LOCAL_BINDS:
        raise ValueError("the operator UI is local-only; bind to 127.0.0.1, ::1, or localhost")
    if not 0 <= port <= 65535:
        raise ValueError("port must be in [0, 65535]")
    application = OperatorApplication(
        workspace=workspace,
        sessions_root=sessions_root,
        carla_host=carla_host,
        carla_port=carla_port,
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
    parser.add_argument("--carla-host", default="172.20.10.7")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    server = create_server(
        workspace=args.workspace,
        bind=args.bind,
        port=args.port,
        sessions_root=args.sessions_root,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
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
]
