"""Additive Garage integration for the existing local operator server.

The base operator application, HTTP routes, browser Drive engine, research jobs,
and artifact APIs remain intact.  This wrapper swaps in the additive Garage
drive manager and injects same-origin JavaScript/CSS that expose the extra
controls without rewriting the existing static application.
"""

from __future__ import annotations

import json
import webbrowser
from http import HTTPStatus
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

from . import server as base
from .garage_drive import GarageDriveSessionManager

GARAGE_STATIC_ROOT = Path(__file__).resolve().parent / "garage_static"
_GARAGE_SCRIPT = "/static/garage-integration.js"
_GARAGE_STYLE = "/static/garage-integration.css"


def _injected_index() -> bytes:
    source = (base.STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    if _GARAGE_SCRIPT in source or _GARAGE_STYLE in source:
        raise RuntimeError("garage integration assets are already present in the base index")
    if "</head>" not in source or "</body>" not in source:
        raise RuntimeError("operator index is missing expected head/body boundaries")
    source = source.replace(
        "</head>",
        f'    <link rel="stylesheet" href="{_GARAGE_STYLE}">\n  </head>',
        1,
    )
    source = source.replace(
        "</body>",
        f'    <script src="{_GARAGE_SCRIPT}"></script>\n  </body>',
        1,
    )
    return source.encode("utf-8")


class GarageOperatorRequestHandler(base.OperatorRequestHandler):
    """Serve the base operator app plus the additive Garage assets."""

    def do_GET(self) -> None:
        try:
            path = urlparse(self.path).path
            if path == "/":
                self._bytes(
                    HTTPStatus.OK,
                    _injected_index(),
                    content_type="text/html; charset=utf-8",
                )
                return
            if path == _GARAGE_SCRIPT:
                self._file(GARAGE_STATIC_ROOT / "garage-integration.js", cache="no-cache")
                return
            if path == _GARAGE_STYLE:
                self._file(GARAGE_STATIC_ROOT / "garage-integration.css", cache="no-cache")
                return
        except BaseException as error:
            self._error(error)
            return
        super().do_GET()


class GarageOperatorDriveManager(GarageDriveSessionManager):
    """Garage manager with workspace checkpoint discovery for the browser selector."""

    def catalog(self) -> dict[str, Any]:
        payload = super().catalog()
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


def create_server(**kwargs: Any) -> base.OperatorHTTPServer:
    """Create the normal operator server and add Garage-only extensions."""

    server = base.create_server(**kwargs)
    server.application.drive = GarageOperatorDriveManager(
        workspace=server.application.workspace,
        carla_host=server.application.carla_host,
        carla_port=server.application.carla_port,
    )
    server.RequestHandlerClass = GarageOperatorRequestHandler
    return server


def main(argv: Sequence[str] | None = None) -> int:
    args = base.parse_args(argv)
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
                "garage_research_bridge": True,
                "garage_drive_modes": True,
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
    "GARAGE_STATIC_ROOT",
    "GarageOperatorDriveManager",
    "GarageOperatorRequestHandler",
    "create_server",
    "main",
]
