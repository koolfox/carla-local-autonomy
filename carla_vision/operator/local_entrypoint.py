"""Strict project-root ``.env.local`` support for the Operator/Garage entry point.

Precedence is deterministic: explicit CLI > process environment > .env.local >
existing built-in defaults. The actual HTTP/server lifecycle remains owned by
``garage_server.main``; this module resolves local machine configuration,
exposes the canonical secret-free product contract, and serves the packaged
Svelte Garage when its static build is present. The legacy shell remains
available at ``/legacy/`` and is also the fail-safe root fallback when the
compiled console is absent.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlparse

from . import garage_server
from . import server as base
from .configuration import build_configuration_contract

_ENV_FILE = ".env.local"
CONSOLE_ROOT = Path(__file__).resolve().parent / "console_static"
_CONSOLE_ASSET_SUFFIXES = frozenset(
    {".css", ".gif", ".ico", ".jpeg", ".jpg", ".js", ".json", ".png", ".svg", ".webp", ".woff", ".woff2"}
)
_INLINE_SCRIPT = re.compile(r"<script(?:\s[^>]*)?>(?P<body>.*?)</script>", re.IGNORECASE | re.DOTALL)
_SUPPORTED_KEYS = frozenset(
    {
        "CARLA_HOST",
        "CARLA_PORT",
        "CARLA_OPERATOR_BIND",
        "CARLA_OPERATOR_PORT",
        "CARLA_WORKSPACE",
        "CARLA_OPERATOR_SESSIONS_ROOT",
        "CARLA_WORLD_WORKER_URL",
        "CARLA_WORLD_WORKER_PORT",
        "CARLA_WORLD_WORKER_TOKEN",
        "CARLA_ENABLE_EXPERIMENTAL",
        "detector.enabled",
    }
)
_RUNTIME_ONLY_KEYS = frozenset({"overlay_frame_sequence"})
_BOOL_TRUE = frozenset({"1", "true", "yes", "on"})
_BOOL_FALSE = frozenset({"0", "false", "no", "off"})
_DETECTOR_INPUT = re.compile(
    r'(<input\s+id="drive-detector-enabled"\s+type="checkbox")(?P<attrs>[^>]*)>',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LocalLaunchPlan:
    argv: tuple[str, ...]
    detector_enabled: bool
    env_updates: Mapping[str, str]
    env_file: Path | None


def _parse_value(raw: str, *, line_number: int) -> str:
    value = raw.strip()
    if not value:
        return ""
    if value[0] == "'":
        if len(value) < 2 or value[-1] != "'":
            raise ValueError(f".env.local line {line_number}: unterminated single-quoted value")
        return value[1:-1]
    if value[0] == '"':
        if len(value) < 2 or value[-1] != '"':
            raise ValueError(f".env.local line {line_number}: unterminated double-quoted value")
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(
                f".env.local line {line_number}: invalid double-quoted value"
            ) from error
        if not isinstance(decoded, str):
            raise ValueError(f".env.local line {line_number}: value must decode to a string")
        return decoded
    return value


def load_local_env(path: Path) -> dict[str, str]:
    """Load the project-local dotenv file without shell expansion."""

    if not path.exists():
        return {}
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path} must be a regular non-symlink file")
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            raise ValueError(f".env.local line {line_number}: expected KEY=VALUE")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if key in _RUNTIME_ONLY_KEYS:
            continue
        if key not in _SUPPORTED_KEYS:
            raise ValueError(f"unsupported .env.local key: {key!r}")
        if key in values:
            raise ValueError(f"duplicate .env.local key: {key!r}")
        values[key] = _parse_value(raw_value, line_number=line_number)
    return values


def _bool_value(raw: str, name: str) -> bool:
    value = str(raw).strip().lower()
    if value in _BOOL_TRUE:
        return True
    if value in _BOOL_FALSE:
        return False
    raise ValueError(f"{name} must be one of true/false, 1/0, yes/no, or on/off")


def _port_value(raw: str, name: str, *, allow_zero: bool = False) -> int:
    try:
        value = int(str(raw).strip())
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    minimum = 0 if allow_zero else 1
    if not minimum <= value <= 65535:
        raise ValueError(f"{name} must be in [{minimum}, 65535]")
    return value


def _nonempty(raw: str, name: str) -> str:
    value = str(raw).strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _worker_url(raw: str) -> str:
    value = _nonempty(raw, "CARLA_WORLD_WORKER_URL")
    if value.lower() == "auto":
        return "auto"
    if value.startswith("[") or "](" in value:
        raise ValueError("CARLA_WORLD_WORKER_URL must be a raw URL, not Markdown")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("CARLA_WORLD_WORKER_URL must be an http:// or https:// URL")
    if parsed.username or parsed.password:
        raise ValueError("CARLA_WORLD_WORKER_URL must not contain credentials")
    return value


def _effective(
    file_values: Mapping[str, str],
    environ: Mapping[str, str],
    file_key: str,
    env_key: str | None = None,
) -> str | None:
    process_key = env_key or file_key
    if process_key in environ:
        return str(environ[process_key])
    return file_values.get(file_key)


def prepare_launch(
    argv: Sequence[str] | None = None,
    *,
    cwd: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> LocalLaunchPlan:
    """Resolve local configuration into argv/env updates for the existing server."""

    root = Path.cwd() if cwd is None else Path(cwd)
    env_path = root / _ENV_FILE
    file_values = load_local_env(env_path)
    process_env = os.environ if environ is None else environ
    prefix: list[str] = []

    option_specs = (
        ("CARLA_HOST", "--carla-host", lambda value: _nonempty(value, "CARLA_HOST")),
        ("CARLA_PORT", "--carla-port", lambda value: _port_value(value, "CARLA_PORT")),
        (
            "CARLA_OPERATOR_BIND",
            "--bind",
            lambda value: _nonempty(value, "CARLA_OPERATOR_BIND"),
        ),
        (
            "CARLA_OPERATOR_PORT",
            "--port",
            lambda value: _port_value(value, "CARLA_OPERATOR_PORT", allow_zero=True),
        ),
        ("CARLA_WORKSPACE", "--workspace", lambda value: _nonempty(value, "CARLA_WORKSPACE")),
        (
            "CARLA_OPERATOR_SESSIONS_ROOT",
            "--sessions-root",
            lambda value: _nonempty(value, "CARLA_OPERATOR_SESSIONS_ROOT"),
        ),
        ("CARLA_WORLD_WORKER_URL", "--world-worker-url", _worker_url),
        (
            "CARLA_WORLD_WORKER_PORT",
            "--world-worker-port",
            lambda value: _port_value(value, "CARLA_WORLD_WORKER_PORT"),
        ),
    )
    for key, option, validator in option_specs:
        raw = _effective(file_values, process_env, key)
        if raw is not None:
            prefix.extend((option, str(validator(raw))))

    raw_detector = _effective(
        file_values,
        process_env,
        "detector.enabled",
        "CARLA_DETECTOR_ENABLED",
    )
    detector_enabled = True if raw_detector is None else _bool_value(raw_detector, "detector.enabled")

    raw_experimental = _effective(file_values, process_env, "CARLA_ENABLE_EXPERIMENTAL")
    experimental_enabled = (
        False
        if raw_experimental is None
        else _bool_value(raw_experimental, "CARLA_ENABLE_EXPERIMENTAL")
    )
    user_argv = list(sys.argv[1:] if argv is None else argv)
    explicit_disable_experimental = "--no-enable-experimental" in user_argv
    user_argv = [item for item in user_argv if item != "--no-enable-experimental"]
    if experimental_enabled and not explicit_disable_experimental:
        prefix.append("--enable-experimental")

    env_updates: dict[str, str] = {}
    token = _effective(file_values, process_env, "CARLA_WORLD_WORKER_TOKEN")
    if token is not None:
        token = _nonempty(token, "CARLA_WORLD_WORKER_TOKEN")
        if "CARLA_WORLD_WORKER_TOKEN" not in process_env:
            env_updates["CARLA_WORLD_WORKER_TOKEN"] = token

    return LocalLaunchPlan(
        argv=tuple(prefix + user_argv),
        detector_enabled=detector_enabled,
        env_updates=env_updates,
        env_file=env_path if env_path.exists() else None,
    )


def render_operator_index(html: str, *, detector_enabled: bool) -> str:
    """Apply browser-only defaults to the legacy shell only."""

    match = _DETECTOR_INPUT.search(html)
    if match is None:
        raise RuntimeError("operator index is missing the drive detector checkbox")
    attrs = re.sub(
        r"\s+checked(?:=(?:\"[^\"]*\"|'[^']*'|[^\s>]+))?",
        "",
        match.group("attrs"),
        flags=re.IGNORECASE,
    )
    checked = " checked" if detector_enabled else ""
    replacement = f"{match.group(1)}{attrs}{checked}>"
    return html[: match.start()] + replacement + html[match.end() :]


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
        "style-src 'self'; img-src 'self' data:; media-src 'self'; "
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


class LocalConfigGarageRequestHandler(garage_server.GarageOperatorRequestHandler):
    """Serve the canonical Garage, APIs, and explicit legacy rollback shell."""

    detector_enabled_default = True
    console_root = CONSOLE_ROOT

    def _configuration_contract(self) -> dict[str, object]:
        return build_configuration_contract(
            self.server.application,
            detector_enabled=self.detector_enabled_default,
        )

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

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/configuration":
            try:
                self._json(HTTPStatus.OK, self._configuration_contract())
            except BaseException as error:
                self._error(error)
            return
        if path == "/api/bootstrap":
            try:
                payload = self.server.application.bootstrap()
                payload["configuration"] = self._configuration_contract()
                self._json(HTTPStatus.OK, payload)
            except BaseException as error:
                self._error(error)
            return
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
                    self._legacy_index()
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
    LocalConfigGarageRequestHandler.detector_enabled_default = plan.detector_enabled

    previous_env = {name: os.environ.get(name) for name in plan.env_updates}
    for name, value in plan.env_updates.items():
        os.environ[name] = value

    original_handler = garage_server.GarageOperatorRequestHandler
    garage_server.GarageOperatorRequestHandler = LocalConfigGarageRequestHandler
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
    "LocalConfigGarageRequestHandler",
    "LocalLaunchPlan",
    "console_available",
    "console_content_security_policy",
    "load_local_env",
    "main",
    "prepare_launch",
    "render_operator_index",
    "resolve_console_asset",
]
