"""Run the web app's live CARLA check from the launcher.

The Operator owns the check (``/api/garage/live-check``); the launcher only
starts it and reads its state, so browser and terminal show the same result.
The Operator listens on loopback, where ``/api/bootstrap`` hands out its
request token to local clients.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .config import LauncherError

_TIMEOUT = 5.0


def _request(url: str, *, token: str | None = None, body: Any = None) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    if token is not None:
        headers["X-Operator-Token"] = token
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            message = json.loads(error.read().decode("utf-8"))["error"]["message"]
        except (ValueError, KeyError, TypeError):
            message = f"HTTP {error.code}"
        raise LauncherError(f"Live check: {message}") from error
    except (OSError, ValueError) as error:
        raise LauncherError(f"Live check: the web app is not reachable ({error}).") from error


def live_check_state(web_url: str) -> dict[str, Any]:
    return _request(f"{web_url.rstrip('/')}/api/garage/live-check")


def start_live_check(web_url: str) -> dict[str, Any]:
    base = web_url.rstrip("/")
    token = _request(f"{base}/api/bootstrap").get("token")
    if not isinstance(token, str) or not token:
        raise LauncherError("Live check: the web app did not provide a request token.")
    return _request(f"{base}/api/garage/live-check", token=token, body={"acknowledge": True})


def describe_live_check(state: dict[str, Any]) -> str:
    phase = str(state.get("phase", "idle"))
    if state.get("active"):
        return f"Live check running ({phase}). The Garage preview resumes when it finishes."
    summary = state.get("summary") or {}
    if phase in {"passed", "failed"} and summary:
        text = (
            f"Live check {phase}: {summary.get('passed_checks', 0)} of "
            f"{summary.get('total_checks', 0)} checks passed."
        )
        failed = summary.get("failed_checks") or []
        if failed:
            text += " Failed: " + ", ".join(str(item) for item in failed[:5]) + "."
        if state.get("report"):
            text += f" Report: {state['report']}"
        return text
    if phase == "failed":
        return f"Live check failed: {state.get('error') or 'see the web app System panel.'}"
    return "Live check has not run yet."
