"""Additive Garage-to-Research UI bridge for the existing operator server.

The underlying OperatorApplication, DriveSessionManager, HTTP API, and static
application remain unchanged. This wrapper only injects one same-origin script
and stylesheet into the existing index page and serves those two extra assets.
"""

from __future__ import annotations

import json
import webbrowser
from http import HTTPStatus
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

from . import server as base

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
    """Serve the unchanged operator app plus the additive Garage bridge assets."""

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


def create_server(**kwargs: Any) -> base.OperatorHTTPServer:
    """Create the normal operator server and replace only its request handler."""

    server = base.create_server(**kwargs)
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
    "GarageOperatorRequestHandler",
    "create_server",
    "main",
]
