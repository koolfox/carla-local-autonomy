"""Serve the packaged Svelte Garage over the model-aware Operator API handler."""

from __future__ import annotations

import base64
import hashlib
import os
import re
from http import HTTPStatus
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

from . import garage_server
from . import server as base
from .local_entrypoint import LocalConfigGarageRequestHandler, prepare_launch, render_operator_index

CONSOLE_ROOT = Path(__file__).resolve().parent / "console_static"
_CONSOLE_ASSET_SUFFIXES = frozenset(
    {
        ".css",
        ".gif",
        ".ico",
        ".jpeg",
        ".jpg",
        ".js",
        ".json",
        ".png",
        ".svg",
        ".webp",
        ".woff",
        ".woff2",
    }
)
_INLINE_SCRIPT = re.compile(
    r"<script(?:\s[^>]*)?>(?P<body>.*?)</script>", re.IGNORECASE | re.DOTALL
)
# SvelteKit's accessibility announcer uses this one generated style attribute.
# Keep the CSP strict while authorizing only that exact attribute value.
_SVELTE_ANNOUNCER_STYLE_HASH = "'sha256-S8qMpvofolR8Mpjy4kQvEm7m1q8clzU4dfDH0AmvZjo='"


def console_available(root: Path = CONSOLE_ROOT) -> bool:
    """Return whether a regular packaged Svelte entrypoint is available."""

    index = root / "index.html"
    return root.is_dir() and not root.is_symlink() and index.is_file() and not index.is_symlink()


def console_content_security_policy(index_html: str) -> str:
    """Allow only the exact generated inline bootstrap scripts by SHA-256 hash."""

    hashes: list[str] = []
    for match in _INLINE_SCRIPT.finditer(index_html):
        body = match.group("body")
        if not body.strip():
            continue
        digest = hashlib.sha256(body.encode("utf-8")).digest()
        token = base64.b64encode(digest).decode("ascii")
        hashes.append(f"'sha256-{token}'")
    script_sources = " ".join(["'self'", *hashes])
    return (
        "default-src 'self'; "
        f"script-src {script_sources}; "
        "style-src 'self'; "
        f"style-src-attr 'unsafe-hashes' {_SVELTE_ANNOUNCER_STYLE_HASH}; "
        "img-src 'self' data:; media-src 'self'; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    )


def resolve_console_asset(path: str, root: Path = CONSOLE_ROOT) -> Path:
    """Resolve one generated ``/_app`` asset without allowing traversal or symlinks."""

    if not path.startswith("/_app/"):
        raise FileNotFoundError("console asset route not found")
    parts = base._path_parts(path.removeprefix("/"), "console asset path")
    if not parts or parts[0] != "_app":
        raise FileNotFoundError("console asset route not found")
    if not console_available(root):
        raise FileNotFoundError("compiled Garage console is unavailable")
    if base._walk_has_symlink(root, parts):
        raise ValueError("console asset path must not contain symlinks")
    candidate = root.joinpath(*parts)
    if not candidate.is_file():
        raise FileNotFoundError("console asset not found")
    resolved_root = root.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    resolved.relative_to(resolved_root)
    if resolved.suffix.lower() not in _CONSOLE_ASSET_SUFFIXES:
        raise FileNotFoundError("console asset type is not served")
    return resolved


class ProductConsoleRequestHandler(LocalConfigGarageRequestHandler):
    """Add packaged Svelte serving while retaining all model-aware API routes."""

    console_root = CONSOLE_ROOT

    def _legacy_index(self) -> None:
        html = (base.STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        html = render_operator_index(
            html,
            detector_enabled=self.detector_enabled_default,
        )
        self._bytes(
            HTTPStatus.OK,
            html.encode("utf-8"),
            content_type="text/html; charset=utf-8",
        )

    def _console_index(self) -> None:
        index = self.console_root / "index.html"
        html = index.read_text(encoding="utf-8")
        payload = html.encode("utf-8")
        self._headers(
            HTTPStatus.OK,
            content_type="text/html; charset=utf-8",
            length=len(payload),
            cache="no-cache",
            content_security_policy=console_content_security_policy(html),
        )
        self.wfile.write(payload)

    def _missing_console_index(self) -> None:
        payload = (
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>CARLA console unavailable</title></head><body><main>"
            "<h1>Garage console build is missing</h1>"
            "<p>Install a release package or run npm ci and npm run build in web.</p>"
            "<p><a href=\"/legacy/\">Open the explicit legacy rollback UI</a></p>"
            "</main></body></html>"
        ).encode("utf-8")
        self._bytes(
            HTTPStatus.SERVICE_UNAVAILABLE,
            payload,
            content_type="text/html; charset=utf-8",
        )

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/legacy", "/legacy/"}:
            try:
                self._legacy_index()
            except BaseException as error:
                self._error(error)
            return
        if path == "/":
            try:
                if console_available(self.console_root):
                    self._console_index()
                else:
                    self._missing_console_index()
            except BaseException as error:
                self._error(error)
            return
        if path.startswith("/_app/"):
            try:
                asset = resolve_console_asset(path, self.console_root)
                cache = (
                    "public, max-age=31536000, immutable"
                    if "/immutable/" in path
                    else "no-cache"
                )
                self._file(asset, cache=cache)
            except BaseException as error:
                self._error(error)
            return
        super().do_GET()


def main(argv: Sequence[str] | None = None) -> int:
    plan = prepare_launch(argv)
    ProductConsoleRequestHandler.detector_enabled_default = plan.detector_enabled

    previous_env = {name: os.environ.get(name) for name in plan.env_updates}
    for name, value in plan.env_updates.items():
        os.environ[name] = value

    original_handler = garage_server.GarageOperatorRequestHandler
    garage_server.GarageOperatorRequestHandler = ProductConsoleRequestHandler
    try:
        return garage_server.main(plan.argv)
    finally:
        garage_server.GarageOperatorRequestHandler = original_handler
        for name, previous in previous_env.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


__all__ = [
    "CONSOLE_ROOT",
    "ProductConsoleRequestHandler",
    "console_available",
    "console_content_security_policy",
    "main",
    "resolve_console_asset",
]
