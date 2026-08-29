"""Strict project-root ``.env.local`` support for the Operator/Garage entry point.

Precedence is deterministic: explicit CLI > process environment > .env.local >
existing built-in defaults. The actual HTTP/server lifecycle remains owned by
``garage_server.main``; this module resolves local machine configuration and
exposes the canonical, secret-free product configuration contract.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlparse

from ..model_registry import discover_model_packages
from . import garage_server
from . import server as base
from .configuration import build_configuration_contract
from .external_model_drive import start_registered_model_session

_ENV_FILE = ".env.local"
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
    """Apply browser-only defaults without turning runtime telemetry into config."""

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


class LocalConfigGarageRequestHandler(garage_server.GarageOperatorRequestHandler):
    """Serve the Garage shell and canonical local configuration contract."""

    detector_enabled_default = True

    def _configuration_contract(self) -> dict[str, object]:
        return build_configuration_contract(
            self.server.application,
            detector_enabled=self.detector_enabled_default,
        )

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/configuration":
            try:
                self._json(HTTPStatus.OK, self._configuration_contract())
            except BaseException as error:
                self._error(error)
            return
        if path == "/api/models":
            try:
                self._json(
                    HTTPStatus.OK,
                    discover_model_packages(self.server.application.workspace),
                )
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
        if path == "/":
            try:
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
            except BaseException as error:
                self._error(error)
            return
        super().do_GET()

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/session/start":
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
            request = garage_server._canonical_session_request(body, self.server.application)
            if str(request.get("control_mode", "manual")) == "model":
                result = start_registered_model_session(self.server.application.drive, request)
            else:
                result = self.server.application.drive.start(request)
            self._json(HTTPStatus.ACCEPTED, result)
        except BaseException as error:
            self._error(error)


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
    "LocalConfigGarageRequestHandler",
    "LocalLaunchPlan",
    "load_local_env",
    "main",
    "prepare_launch",
    "render_operator_index",
]
