"""Local operator entry point with strict project-root ``.env.local`` support.

Precedence is deterministic: explicit CLI > process environment > .env.local >
existing built-in defaults. Secrets are only exported to the child process
environment when required by the existing World Worker token contract; they are
never added to argv or browser-visible payloads.
"""

from __future__ import annotations

import json
import os
import re
import sys
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlparse

from . import garage_server
from . import server as base

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
    """Load the strict project-local dotenv file without shell expansion."""

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
            raise ValueError(
                f"{key} is runtime state and cannot be configured in .env.local"
            )
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
        ("CARLA_HOST", "CARLA_HOST", "--carla-host", lambda value: _nonempty(value, "CARLA_HOST")),
        ("CARLA_PORT", "CARLA_PORT", "--carla-port", lambda value: _port_value(value, "CARLA_PORT")),
        (
            "CARLA_OPERATOR_BIND",
            "CARLA_OPERATOR_BIND",
            "--bind",
            lambda value: _nonempty(value, "CARLA_OPERATOR_BIND"),
        ),
        (
            "CARLA_OPERATOR_PORT",
            "CARLA_OPERATOR_PORT",
            "--port",
            lambda value: _port_value(value, "CARLA_OPERATOR_PORT", allow_zero=True),
        ),
        (
            "CARLA_WORKSPACE",
            "CARLA_WORKSPACE",
            "--workspace",
            lambda value: _nonempty(value, "CARLA_WORKSPACE"),
        ),
        (
            "CARLA_OPERATOR_SESSIONS_ROOT",
            "CARLA_OPERATOR_SESSIONS_ROOT",
            "--sessions-root",
            lambda value: _nonempty(value, "CARLA_OPERATOR_SESSIONS_ROOT"),
        ),
        (
            "CARLA_WORLD_WORKER_URL",
            "CARLA_WORLD_WORKER_URL",
            "--world-worker-url",
            _worker_url,
        ),
        (
            "CARLA_WORLD_WORKER_PORT",
            "CARLA_WORLD_WORKER_PORT",
            "--world-worker-port",
            lambda value: _port_value(value, "CARLA_WORLD_WORKER_PORT"),
        ),
    )
    for file_key, env_key, option, validator in option_specs:
        raw = _effective(file_values, process_env, file_key, env_key)
        if raw is None:
            continue
        value = validator(raw)
        prefix.extend((option, str(value)))

    raw_detector = _effective(
        file_values,
        process_env,
        "detector.enabled",
        "CARLA_DETECTOR_ENABLED",
    )
    detector_enabled = True if raw_detector is None else _bool_value(raw_detector, "detector.enabled")

    raw_experimental = _effective(
        file_values,
        process_env,
        "CARLA_ENABLE_EXPERIMENTAL",
        "CARLA_ENABLE_EXPERIMENTAL",
    )
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
    token = _effective(
        file_values,
        process_env,
        "CARLA_WORLD_WORKER_TOKEN",
        "CARLA_WORLD_WORKER_TOKEN",
    )
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
    attrs = re.sub(r"\s+checked(?:=(?:\"[^\"]*\"|'[^']*'|[^\s>]+))?", "", match.group("attrs"), flags=re.IGNORECASE)
    checked = " checked" if detector_enabled else ""
    replacement = f"{match.group(1)}{attrs}{checked}>"
    return html[: match.start()] + replacement + html[match.end() :]


class LocalConfigGarageRequestHandler(garage_server.GarageOperatorRequestHandler):
    """Serve the normal Garage shell with local browser defaults applied."""

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/":
            try:
                html = (base.STATIC_ROOT / "index.html").read_text(encoding="utf-8")
                html = render_operator_index(
                    html,
                    detector_enabled=bool(
                        getattr(self.server.application, "local_detector_enabled", True)
                    ),
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


def main(argv: Sequence[str] | None = None) -> int:
    plan = prepare_launch(argv)
    for name, value in plan.env_updates.items():
        os.environ[name] = value

    args = base.parse_args(plan.argv)
    carla_host = base.resolve_carla_host(args.carla_host, args.carla_port)
    world_worker_url = base.resolve_world_worker_url(
        args.world_worker_url,
        carla_host,
        args.world_worker_port,
    )
    world_worker_token = base.resolve_world_worker_token(args)
    server = garage_server.create_server(
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
    server.RequestHandlerClass = LocalConfigGarageRequestHandler
    server.application.local_detector_enabled = plan.detector_enabled

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
                "local_env_loaded": plan.env_file is not None,
                "local_env_path": str(plan.env_file.resolve()) if plan.env_file else None,
                "world_worker_configured": world_worker_url is not None,
                "garage_research_bridge": args.enable_experimental,
                "garage_drive_modes": True,
                "garage_research_jobs": args.enable_experimental,
                "experimental_enabled": args.enable_experimental,
                "detector_enabled_default": plan.detector_enabled,
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
    "LocalConfigGarageRequestHandler",
    "LocalLaunchPlan",
    "load_local_env",
    "main",
    "prepare_launch",
    "render_operator_index",
]
