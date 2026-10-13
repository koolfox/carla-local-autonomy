"""Inspect and test a running console from another computer.

``carla-remote`` talks to the web app's existing API: a local URL when the web
app runs here against a LAN Worker, or a Cloudflare Tunnel hostname protected
by Cloudflare Access. Access service-token credentials come from the
environment or a private env file and are sent only as request headers; the
Operator request token is fetched per call and never printed.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .launcher.config import LauncherError
from .launcher.live_check import _request, describe_live_check, live_check_state, start_live_check

DEFAULT_URL = "http://127.0.0.1:8765/"
DEFAULT_ENV_FILE = Path.home() / ".config" / "carla-vision" / "remote.env"
_ENV_KEYS = ("CARLA_REMOTE_URL", "CF_ACCESS_CLIENT_ID", "CF_ACCESS_CLIENT_SECRET")


def load_env_file(path: Path) -> dict[str, str]:
    """Read KEY=VALUE lines from a private file; refuse one other users can read."""

    if not path.is_file():
        return {}
    if os.name != "nt" and path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise LauncherError(f"{path} must be private: run chmod 600 {path}")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in _ENV_KEYS:
            values[key] = value.strip().strip("'\"")
    return values


def resolve_target(
    url: str | None,
    env_file: Path,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    environ = os.environ if environ is None else environ
    settings = {**load_env_file(env_file), **{key: environ[key] for key in _ENV_KEYS if key in environ}}
    target = (url or settings.get("CARLA_REMOTE_URL") or DEFAULT_URL).rstrip("/") + "/"
    client_id = settings.get("CF_ACCESS_CLIENT_ID", "")
    secret = settings.get("CF_ACCESS_CLIENT_SECRET", "")
    if bool(client_id) != bool(secret):
        raise LauncherError("Set both CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET, or neither.")
    if target.startswith("http://") and client_id:
        raise LauncherError("Cloudflare Access credentials are only sent to an https:// URL.")
    if target.startswith("https://") and not client_id:
        raise LauncherError(
            "A remote https:// console needs a Cloudflare Access service token "
            f"(CF_ACCESS_CLIENT_ID / CF_ACCESS_CLIENT_SECRET in {env_file})."
        )
    headers = (
        {"CF-Access-Client-Id": client_id, "CF-Access-Client-Secret": secret} if client_id else {}
    )
    return target, headers


def _get(base: str, path: str, headers: Mapping[str, str]) -> dict[str, Any]:
    return _request(base + path.lstrip("/"), headers=headers)


def status(base: str, headers: Mapping[str, str]) -> list[str]:
    catalog = _get(base, "api/drive/catalog", headers)
    drive = _get(base, "api/drive/state", headers)
    preview = _get(base, "api/garage/preview/state", headers)
    check = live_check_state(base, headers)
    worker = catalog.get("world_worker") or {}
    return [
        f"Console     {base}",
        "CARLA       "
        + (
            f"online · {catalog.get('server_version') or 'version unknown'} · "
            f"{str(catalog.get('map') or 'map unknown').split('/')[-1]}"
            if catalog.get("connected")
            else "offline"
        ),
        "Worker      " + ("connected" if worker.get("connected") else "not connected"),
        f"Garage      {preview.get('status', 'unknown')}",
        f"Drive       {drive.get('status', 'unknown')}"
        + (f" · run {drive['run_id']}" if drive.get("run_id") else ""),
        f"Live check  {describe_live_check(check)}",
    ]


def snapshot(base: str, headers: Mapping[str, str], view: str, output: Path) -> Path:
    path = "api/drive/frame.jpg?view=raw" if view == "drive" else "api/garage/preview/frame.jpg"
    request = urllib.request.Request(base + path, headers=dict(headers))
    try:
        with urllib.request.urlopen(request, timeout=10.0) as response:
            if response.headers.get("Content-Type", "").split(";")[0] != "image/jpeg":
                raise LauncherError(
                    "Snapshot: the server did not return a camera frame; "
                    "check the Access token and that the Garage or Drive camera is live."
                )
            data = response.read()
    except urllib.error.HTTPError as error:
        raise LauncherError(f"Snapshot: no {view} frame is available (HTTP {error.code}).") from error
    except OSError as error:
        raise LauncherError(f"Snapshot: the console is not reachable ({error}).") from error
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    return output


def run_live_check(base: str, headers: Mapping[str, str], poll: float = 3.0) -> tuple[bool, str]:
    state = start_live_check(base, headers)
    while state.get("active"):
        print(describe_live_check(state), flush=True)
        time.sleep(poll)
        state = live_check_state(base, headers)
    return state.get("phase") == "passed", describe_live_check(state)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="carla-remote",
        description="Inspect and test a running CARLA Vision console on this computer, "
        "the LAN, or behind Cloudflare Access.",
    )
    parser.add_argument("--url", help=f"console URL (default: CARLA_REMOTE_URL or {DEFAULT_URL})")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help=f"private file with CARLA_REMOTE_URL and Access credentials (default: {DEFAULT_ENV_FILE})",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="show CARLA, Worker, Garage, Drive and live-check state")
    shot = commands.add_parser("snapshot", help="save the current Garage or Drive camera frame")
    shot.add_argument("--view", choices=("garage", "drive"), default="garage")
    shot.add_argument("--out", type=Path, default=Path("remote-snapshot.jpg"))
    commands.add_parser(
        "live-check", help="run the live CARLA check (reloads the map, drives a test car)"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        base, headers = resolve_target(args.url, args.env_file)
        if args.command == "status":
            print("\n".join(status(base, headers)))
            return 0
        if args.command == "snapshot":
            print(f"Saved {snapshot(base, headers, args.view, args.out)}")
            return 0
        passed, message = run_live_check(base, headers)
        print(message)
        return 0 if passed else 1
    except LauncherError as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
