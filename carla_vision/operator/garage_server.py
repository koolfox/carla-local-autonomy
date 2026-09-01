"""Garage APIs and lifecycle integration for the local operator server.

The base operator application owns the single canonical browser shell and static
assets. This wrapper adds the Garage drive manager, preview, canonical unified
session adapter, and allow-listed research-job endpoint without changing the
underlying Drive execution contracts during migration.
"""

from __future__ import annotations

import ipaddress
import json
import threading
import webbrowser
from collections.abc import Mapping
from http import HTTPStatus
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

from . import server as base
from .configuration import (
    CONFIGURATION_SCHEMA_VERSION,
    build_configuration_evidence,
    build_garage_preview_request,
    build_legacy_drive_request,
)
from .garage_drive import GarageDriveSessionManager
from .garage_preview import GaragePreviewConfig, GaragePreviewManager
from .garage_research import GarageResearchRequest, build_garage_research_plan

_LIVE_RESEARCH_KINDS = frozenset(
    {"voxel_capture", "voxel_flow_capture", "voxel_shadow", "closed_loop_evaluate"}
)
_LAN_IPV4_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16")
)
_LAN_IPV6_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in ("fc00::/7", "fe80::/10")
)
_WILDCARD_BINDS = frozenset({"0.0.0.0", "::"})


def _operator_bind_scope(bind: str) -> str:
    """Classify an explicit Operator bind as loopback, LAN, or unsupported."""

    value = str(bind).strip()
    if value in base._LOCAL_BINDS:
        return "loopback"
    if value in _WILDCARD_BINDS:
        return "lan"
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return "unsupported"
    networks = _LAN_IPV4_NETWORKS if address.version == 4 else _LAN_IPV6_NETWORKS
    return "lan" if any(address in network for network in networks) else "unsupported"


def _validate_research_runtime(
    request: GarageResearchRequest, drive_state: Mapping[str, Any]
) -> None:
    status = str(drive_state.get("status", "idle"))
    active = status in {"starting", "running", "stopping"}
    if request.kind == "teacher_capture" and active:
        raise RuntimeError(
            "end the active Garage drive before starting destructive teacher capture"
        )
    dry_run = request.parameters.get("dry_run") is True
    if request.kind in _LIVE_RESEARCH_KINDS and not dry_run and status != "running":
        raise RuntimeError(f"{request.kind} requires a running Garage ego or dry_run=true")


def _canonical_session_request(raw: Any, application: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError("unified session request must be an object")
    keys = frozenset(str(key) for key in raw)
    if keys != {"schema_version", "session"}:
        raise ValueError("unified session request must contain only schema_version and session")
    if str(raw["schema_version"]) != CONFIGURATION_SCHEMA_VERSION:
        raise ValueError(
            f"unified session schema_version must be {CONFIGURATION_SCHEMA_VERSION!r}"
        )

    worker_configured = application.world_worker is not None
    return build_legacy_drive_request(
        raw["session"],
        carla_host=application.carla_host,
        carla_port=application.carla_port,
        # Schema adaptation depends on the configured execution topology, not
        # a racy health probe. The actual scene operation below remains the
        # authoritative liveness check.
        worker_connected=worker_configured,
        capabilities={"autopilot": worker_configured},
    )


def _canonical_preview_request(raw: Any) -> tuple[Mapping[str, Any] | None, dict[str, Any]]:
    if not isinstance(raw, Mapping):
        raise TypeError("Garage preview request must be an object")
    keys = frozenset(str(key) for key in raw)
    if keys != {"schema_version", "session"}:
        return None, dict(raw)
    if str(raw["schema_version"]) != CONFIGURATION_SCHEMA_VERSION:
        raise ValueError(
            f"Garage preview schema_version must be {CONFIGURATION_SCHEMA_VERSION!r}"
        )
    session = raw["session"]
    if not isinstance(session, Mapping):
        raise TypeError("Garage preview session must be an object")
    return session, build_garage_preview_request(session)


def _with_configuration_evidence(
    result: Mapping[str, Any],
    *,
    requested: Any,
    resolved: Any,
    applied: Any | None = None,
) -> dict[str, Any]:
    response = dict(result)
    response["configuration"] = build_configuration_evidence(
        requested=requested,
        resolved=resolved,
        applied=result if applied is None else applied,
    )
    return response


class GarageOperatorRequestHandler(base.OperatorRequestHandler):
    """Serve Garage APIs and delegate the canonical shell to the base handler."""

    def do_GET(self) -> None:
        try:
            path = urlparse(self.path).path
            if path == "/api/garage/preview/state":
                self._json(HTTPStatus.OK, self.server.application.preview.state())
                return
            if path == "/api/garage/preview/frame.jpg":
                sequence, payload = self.server.application.preview.frame()
                self._bytes(
                    HTTPStatus.OK,
                    payload,
                    content_type="image/jpeg",
                    extra_headers={"X-Garage-Preview-Frame-Sequence": str(sequence)},
                )
                return
            if path == "/api/garage/preview/stream.mjpg":
                self._preview_stream()
                return
        except BaseException as error:
            self._error(error)
            return
        super().do_GET()

    def _preview_stream(self) -> None:
        """Serve new preview frames continuously until the browser disconnects."""

        boundary = "carla-garage-preview"
        session = self.server.application.preview.subscribe()
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            f"multipart/x-mixed-replace; boundary={boundary}",
        )
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        sequence = -1
        try:
            while True:
                try:
                    sequence, payload = session.wait_for_frame(sequence, timeout=5.0)
                except TimeoutError:
                    continue
                header = (
                    f"--{boundary}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(payload)}\r\n"
                    f"X-Garage-Preview-Frame-Sequence: {sequence}\r\n\r\n"
                ).encode("ascii")
                self.wfile.write(header)
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, EOFError, OSError):
            return

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        preview_routes = {
            "/api/garage/preview/configure",
            "/api/garage/preview/orbit",
            "/api/garage/preview/stop",
        }
        handled_routes = {"/api/garage/jobs", "/api/session/start", *preview_routes}
        if path not in handled_routes:
            super().do_POST()
            return
        try:
            if not self._authorized():
                self._json(
                    HTTPStatus.FORBIDDEN,
                    {"error": {"type": "PermissionError", "message": "invalid UI token"}},
                )
                return
            body = self._body()
            if path == "/api/session/start":
                request = _canonical_session_request(body, self.server.application)
                result = self.server.application.drive.start(request)
                self._json(
                    HTTPStatus.ACCEPTED,
                    _with_configuration_evidence(
                        result,
                        requested=body["session"],
                        resolved=request,
                    ),
                )
                return
            if path in preview_routes:
                if not isinstance(body, Mapping):
                    raise TypeError("Garage preview request must be an object")
                if path.endswith("/configure"):
                    requested, preview_request = _canonical_preview_request(body)
                    result = self.server.application.preview.configure(preview_request)
                    if requested is not None:
                        result = _with_configuration_evidence(
                            result,
                            requested=requested,
                            resolved=GaragePreviewConfig.from_mapping(
                                preview_request
                            ).as_dict(),
                        )
                    status = HTTPStatus.CREATED
                elif path.endswith("/orbit"):
                    result = self.server.application.preview.orbit(body)
                    status = HTTPStatus.OK
                else:
                    result = self.server.application.preview.stop(body)
                    status = HTTPStatus.OK
                self._json(status, result)
                return
            if not isinstance(body, Mapping):
                raise TypeError("Garage research request must be an object")
            if not self.server.application.experimental_enabled:
                raise PermissionError(
                    "Garage research jobs are experimental and disabled; "
                    "restart the Garage server with --enable-experimental to opt in"
                )
            request = GarageResearchRequest.from_mapping(body)
            application = self.server.application
            _validate_research_runtime(request, application.drive.state())
            plan = build_garage_research_plan(
                request,
                workspace=application.workspace,
                carla_host=application.carla_host,
                carla_port=application.carla_port,
            )
            self._json(
                HTTPStatus.ACCEPTED,
                application.jobs.submit(request, plan),
            )
        except BaseException as error:
            self._error(error)


class GarageOperatorDriveManager(GarageDriveSessionManager):
    """Garage manager with workspace checkpoint discovery for the browser selector."""

    def attach_preview(
        self,
        preview: GaragePreviewManager,
        world_mode_lock: threading.RLock,
    ) -> None:
        self._garage_preview = preview
        self._garage_world_mode_lock = world_mode_lock

    def start(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        preview = getattr(self, "_garage_preview", None)
        world_mode_lock = getattr(self, "_garage_world_mode_lock", None)
        if preview is None or world_mode_lock is None:
            return super().start(raw)
        with world_mode_lock:
            preview.stop_for_drive()
            return super().start(raw)

    def catalog(self) -> dict[str, Any]:
        payload = super().catalog()
        if not self.experimental_enabled:
            return payload
        checkpoints: list[str] = []
        ignored_roots = {".git", ".venv", "__pycache__", "node_modules"}
        for suffix in ("*.pt", "*.pth", "*.ckpt"):
            for path in self.workspace.rglob(suffix):
                try:
                    relative = path.relative_to(self.workspace)
                except ValueError:
                    continue
                if any(part in ignored_roots for part in relative.parts):
                    continue
                if path.is_symlink() or not path.is_file():
                    continue
                checkpoints.append(relative.as_posix())
        payload["policy_checkpoints"] = sorted(set(checkpoints), key=str.casefold)
        return payload


class GarageOperatorApplication(base.OperatorApplication):
    """Base operator state plus the mutually-exclusive real Garage preview."""

    preview: GaragePreviewManager
    experimental_enabled: bool

    def close(self) -> None:
        preview = getattr(self, "preview", None)
        if preview is not None:
            preview.shutdown()
        super().close()


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
    enable_experimental: bool = False,
) -> base.OperatorHTTPServer:
    """Create the normal operator server and add Garage-only extensions."""

    bind_scope = _operator_bind_scope(bind)
    if bind_scope == "unsupported":
        raise ValueError(
            "the operator UI may bind only to loopback, wildcard, or a private/link-local LAN IP"
        )
    if not 0 <= int(port) <= 65535:
        raise ValueError("port must be in [0, 65535]")
    if (world_worker_url is None) != (world_worker_token is None):
        raise ValueError("World Worker URL and bearer token must be configured together")
    world_worker = (
        None
        if world_worker_url is None or world_worker_token is None
        else base.WorldWorkerClient(str(world_worker_url), str(world_worker_token))
    )
    application = GarageOperatorApplication(
        workspace=workspace,
        sessions_root=sessions_root,
        carla_host=str(carla_host),
        carla_port=int(carla_port),
        world_worker=world_worker,
    )
    application.experimental_enabled = bool(enable_experimental)
    drive = GarageOperatorDriveManager(
        workspace=application.workspace,
        carla_host=application.carla_host,
        carla_port=application.carla_port,
        world_worker=application.world_worker,
        experimental_enabled=application.experimental_enabled,
    )
    application.drive = drive
    world_mode_lock = threading.RLock()
    application.preview = GaragePreviewManager(
        carla_host=application.carla_host,
        carla_port=application.carla_port,
        world_worker=application.world_worker,
        drive_state=drive.state,
        world_mode_lock=world_mode_lock,
    )
    drive.attach_preview(application.preview, world_mode_lock)
    server = base.OperatorHTTPServer((str(bind), int(port)), application)
    server.RequestHandlerClass = GarageOperatorRequestHandler
    return server


def main(argv: Sequence[str] | None = None) -> int:
    args = base.parse_args(argv)
    carla_host = base.resolve_carla_host(args.carla_host, args.carla_port)
    world_worker_url = base.resolve_world_worker_url(
        args.world_worker_url,
        carla_host,
        args.world_worker_port,
    )
    world_worker_token = base.resolve_world_worker_token(args)
    server = create_server(
        workspace=args.workspace,
        bind=args.bind,
        port=args.port,
        sessions_root=args.sessions_root,
        carla_host=carla_host,
        carla_port=args.carla_port,
        world_worker_url=world_worker_url,
        world_worker_token=world_worker_token,
        enable_experimental=args.enable_experimental,
    )
    address, port = server.server_address[:2]
    bind_scope = _operator_bind_scope(args.bind)
    display_host = "127.0.0.1" if address in _WILDCARD_BINDS else address
    url = f"http://{display_host}:{port}/"
    print(
        json.dumps(
            {
                "status": "ready",
                "url": url,
                "bind": str(address),
                "port": int(port),
                "workspace": str(Path(args.workspace).expanduser().resolve()),
                "local_only": bind_scope == "loopback",
                "lan_enabled": bind_scope == "lan",
                "world_worker_configured": world_worker_url is not None,
                "garage_research_bridge": args.enable_experimental,
                "garage_drive_modes": True,
                "garage_research_jobs": args.enable_experimental,
                "experimental_enabled": args.enable_experimental,
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
    "GarageOperatorDriveManager",
    "GarageOperatorRequestHandler",
    "_operator_bind_scope",
    "create_server",
    "main",
]
