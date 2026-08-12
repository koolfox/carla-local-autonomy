"""Small authenticated HTTP client for the native CARLA World Worker.

The worker is intentionally an allow-listed service rather than a remote
Python or shell endpoint.  This client keeps the operator UI and model runtime
on the Mac while delegating world ownership to the version-matched official
PythonAPI process beside the simulator.
"""

from __future__ import annotations

import json
import math
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_SCENE_PREPARE_KEYS = frozenset(
    {
        "map_name",
        "weather_preset",
        "vehicle_blueprint",
        "color",
        "seed",
        "traffic_count",
        "walker_count",
        "prop_preset",
        "route_mode",
        "initial_control_mode",
    }
)
_CONTROL_KEYS = frozenset(
    {
        "sequence",
        "throttle",
        "steer",
        "brake",
        "hand_brake",
        "reverse",
    }
)
_CONTROL_MODES = frozenset({"manual", "autopilot"})


class WorldWorkerError(RuntimeError):
    """A transport, protocol, authentication, or worker-side failure."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class _RejectRedirects(HTTPRedirectHandler):
    """Never forward the bearer credential to a redirect target."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


@dataclass(frozen=True)
class WorldWorkerScene:
    """Validated scene lease returned by the World Worker."""

    scene_id: str
    lease_token: str
    status: str
    episode_id: int | None
    ego_actor_id: int | None
    map_name: str | None
    spawn_index: int | None
    route_mode: str | None
    route: Mapping[str, Any]
    destination: Any
    control_mode: str | None
    traffic_count: int | None
    walker_count: int | None
    prop_actor_ids: tuple[int, ...]
    lease_expires_in_seconds: float | None
    cleanup_guard_passed: bool | None
    cleanup_errors: tuple[str, ...]
    capabilities: Mapping[str, Any]

    @classmethod
    def from_response(cls, payload: Mapping[str, Any]) -> "WorldWorkerScene":
        raw = payload.get("scene")
        if not isinstance(raw, Mapping):
            raise WorldWorkerError("World Worker response is missing a scene object")

        scene_id = _required_text(raw.get("scene_id"), "scene.scene_id")
        lease_token = _required_text(raw.get("lease_token"), "scene.lease_token")
        status = _required_text(raw.get("status"), "scene.status")
        episode_id = _optional_integer(raw.get("episode_id"), "scene.episode_id", minimum=0)
        ego_actor_id = _optional_integer(
            raw.get("ego_actor_id"),
            "scene.ego_actor_id",
            minimum=1,
        )
        spawn_index = _optional_integer(raw.get("spawn_index"), "scene.spawn_index", minimum=0)
        traffic_count = _optional_integer(
            raw.get("traffic_count"),
            "scene.traffic_count",
            minimum=0,
        )
        walker_count = _optional_integer(
            raw.get("walker_count"),
            "scene.walker_count",
            minimum=0,
        )
        map_name = _optional_text(raw.get("map_name"), "scene.map_name")
        route_mode = _optional_text(raw.get("route_mode"), "scene.route_mode")
        route = raw.get("route", {})
        if not isinstance(route, Mapping):
            raise WorldWorkerError("scene.route must be an object")
        control_mode = _optional_text(raw.get("control_mode"), "scene.control_mode")
        if control_mode is not None and control_mode not in _CONTROL_MODES:
            raise WorldWorkerError(f"scene.control_mode must be one of {sorted(_CONTROL_MODES)!r}")

        raw_props = raw.get("prop_actor_ids", [])
        if not isinstance(raw_props, list):
            raise WorldWorkerError("scene.prop_actor_ids must be a list")
        prop_actor_ids = tuple(
            _required_integer(value, "scene.prop_actor_ids[]", minimum=1) for value in raw_props
        )

        lease_seconds = raw.get("lease_expires_in_seconds")
        if lease_seconds is not None:
            if (
                isinstance(lease_seconds, bool)
                or not isinstance(lease_seconds, (int, float))
                or not math.isfinite(float(lease_seconds))
                or float(lease_seconds) < 0.0
            ):
                raise WorldWorkerError(
                    "scene.lease_expires_in_seconds must be a non-negative finite number"
                )
            lease_seconds = float(lease_seconds)

        capabilities = raw.get("capabilities", {})
        if not isinstance(capabilities, Mapping):
            raise WorldWorkerError("scene.capabilities must be an object")
        cleanup_guard_passed = raw.get("cleanup_guard_passed")
        if cleanup_guard_passed is not None and not isinstance(cleanup_guard_passed, bool):
            raise WorldWorkerError("scene.cleanup_guard_passed must be boolean or null")
        raw_cleanup_errors = raw.get("cleanup_errors", [])
        if not isinstance(raw_cleanup_errors, list) or any(
            not isinstance(value, str) or len(value) > 2048 for value in raw_cleanup_errors
        ):
            raise WorldWorkerError("scene.cleanup_errors must be a list of bounded strings")

        return cls(
            scene_id=scene_id,
            lease_token=lease_token,
            status=status,
            episode_id=episode_id,
            ego_actor_id=ego_actor_id,
            map_name=map_name,
            spawn_index=spawn_index,
            route_mode=route_mode,
            route=dict(route),
            destination=raw.get("destination"),
            control_mode=control_mode,
            traffic_count=traffic_count,
            walker_count=walker_count,
            prop_actor_ids=prop_actor_ids,
            lease_expires_in_seconds=lease_seconds,
            cleanup_guard_passed=cleanup_guard_passed,
            cleanup_errors=tuple(raw_cleanup_errors),
            capabilities=dict(capabilities),
        )


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorldWorkerError(f"{name} must be a non-empty string")
    result = value.strip()
    if len(result) > 512 or any(character in result for character in "\x00\r\n"):
        raise WorldWorkerError(f"{name} is invalid")
    return result


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, name)


def _required_integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise WorldWorkerError(f"{name} must be an integer >= {minimum}")
    return value


def _optional_integer(value: Any, name: str, *, minimum: int) -> int | None:
    if value is None:
        return None
    return _required_integer(value, name, minimum=minimum)


class WorldWorkerClient:
    """Authenticated, JSON-only client for one native World Worker."""

    def __init__(self, base_url: str, bearer_token: str, *, timeout: float = 5.0) -> None:
        parsed = urlsplit(str(base_url).strip())
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("World Worker URL must use http or https")
        if not parsed.hostname or parsed.username is not None or parsed.password is not None:
            raise ValueError("World Worker URL must contain a host and no credentials")
        if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            raise ValueError("World Worker URL must not contain a path, query, or fragment")
        try:
            port = parsed.port
        except ValueError as error:
            raise ValueError("World Worker URL contains an invalid port") from error
        if port is not None and not 1 <= port <= 65535:
            raise ValueError("World Worker URL port must be in [1, 65535]")

        token = str(bearer_token)
        if not token or token != token.strip() or any(character in token for character in "\r\n"):
            raise ValueError("World Worker bearer token is missing or invalid")
        if not math.isfinite(float(timeout)) or not 0.1 <= float(timeout) <= 60.0:
            raise ValueError("World Worker timeout must be in [0.1, 60] seconds")

        netloc = parsed.hostname
        if ":" in netloc and not netloc.startswith("["):
            netloc = f"[{netloc}]"
        if port is not None:
            netloc = f"{netloc}:{port}"
        self.base_url = urlunsplit((parsed.scheme, netloc, "", "", ""))
        self._bearer_token = token
        self.timeout = float(timeout)
        # Connect directly to the explicitly configured LAN worker. Environment
        # proxy settings must never receive the bearer credential.
        self._opener = build_opener(ProxyHandler({}), _RejectRedirects())

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/v1/health")

    def catalog(self) -> dict[str, Any]:
        return self._request("GET", "/v1/catalog")

    def current_scene(self) -> dict[str, Any]:
        return self._request("GET", "/v1/scenes/current")

    def prepare_scene(self, payload: Mapping[str, Any]) -> WorldWorkerScene:
        keys = frozenset(str(key) for key in payload)
        if keys != _SCENE_PREPARE_KEYS:
            missing = sorted(_SCENE_PREPARE_KEYS - keys)
            unknown = sorted(keys - _SCENE_PREPARE_KEYS)
            detail = []
            if missing:
                detail.append(f"missing {', '.join(missing)}")
            if unknown:
                detail.append(f"unknown {', '.join(unknown)}")
            raise ValueError(f"World Worker scene payload has {'; '.join(detail)}")
        return WorldWorkerScene.from_response(
            # Native map reloads can legitimately outlive ordinary LAN control
            # calls. Keep the longer budget local to this destructive request;
            # heartbeat, control and stop retain the normal short timeout.
            self._request(
                "POST",
                "/v1/scenes/prepare",
                dict(payload),
                timeout=max(self.timeout, 120.0),
            )
        )

    def start_scene(self, scene: WorldWorkerScene) -> WorldWorkerScene:
        return self._scene_request(scene, "start", {"lease_token": scene.lease_token})

    def heartbeat(self, scene: WorldWorkerScene) -> WorldWorkerScene:
        return self._scene_request(scene, "heartbeat", {"lease_token": scene.lease_token})

    def control(
        self,
        scene: WorldWorkerScene,
        control: Mapping[str, Any],
    ) -> WorldWorkerScene:
        keys = frozenset(str(key) for key in control)
        if keys != _CONTROL_KEYS:
            raise ValueError("World Worker control payload has invalid fields")
        return self._scene_request(
            scene,
            "control",
            {"lease_token": scene.lease_token, **dict(control)},
        )

    def mode(self, scene: WorldWorkerScene, control_mode: str) -> WorldWorkerScene:
        if control_mode not in _CONTROL_MODES:
            raise ValueError("control mode must be manual or autopilot")
        return self._scene_request(
            scene,
            "mode",
            {"lease_token": scene.lease_token, "control_mode": control_mode},
        )

    def weather(self, scene: WorldWorkerScene, weather_preset: str) -> WorldWorkerScene:
        preset = str(weather_preset).strip()
        if not preset:
            raise ValueError("weather preset is required")
        return self._scene_request(
            scene,
            "weather",
            {"lease_token": scene.lease_token, "weather_preset": preset},
        )

    def stop_scene(self, scene: WorldWorkerScene) -> WorldWorkerScene:
        return self._scene_request(scene, "stop", {"lease_token": scene.lease_token})

    def _scene_request(
        self,
        scene: WorldWorkerScene,
        action: str,
        payload: Mapping[str, Any],
    ) -> WorldWorkerScene:
        scene_id = quote(scene.scene_id, safe="")
        response = self._request(
            "POST",
            f"/v1/scenes/{scene_id}/{action}",
            dict(payload),
        )
        updated = WorldWorkerScene.from_response(response)
        if updated.scene_id != scene.scene_id:
            raise WorldWorkerError("World Worker response changed the active scene_id")
        return updated

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        body = None
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._bearer_token}",
        }
        if payload is not None:
            body = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with self._opener.open(
                request,
                timeout=self.timeout if timeout is None else float(timeout),
            ) as response:
                status = int(response.status)
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except HTTPError as error:
            raw = error.read(_MAX_RESPONSE_BYTES + 1)
            if 300 <= error.code < 400:
                raise WorldWorkerError(
                    "World Worker redirects are not allowed",
                    status=error.code,
                    code="world_worker_redirect",
                ) from error
            parsed = self._decode(raw, status=error.code, allow_error=True)
            error_body = parsed.get("error")
            if isinstance(error_body, Mapping):
                code = str(error_body.get("code", "world_worker_error"))
                message = str(error_body.get("message", error.reason))
            else:
                code = "world_worker_http_error"
                message = str(error.reason)
            raise WorldWorkerError(message, status=error.code, code=code) from error
        except (URLError, TimeoutError, socket.timeout, OSError) as error:
            reason = getattr(error, "reason", error)
            raise WorldWorkerError(f"World Worker is unreachable: {reason}") from error

        if len(raw) > _MAX_RESPONSE_BYTES:
            raise WorldWorkerError("World Worker response exceeds the size limit", status=status)
        if not 200 <= status < 300:
            raise WorldWorkerError(f"World Worker returned HTTP {status}", status=status)
        return self._decode(raw, status=status, allow_error=False)

    @staticmethod
    def _decode(raw: bytes, *, status: int, allow_error: bool) -> dict[str, Any]:
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise WorldWorkerError("World Worker response exceeds the size limit", status=status)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise WorldWorkerError(
                "World Worker returned invalid UTF-8 JSON",
                status=status,
            ) from error
        if not isinstance(payload, dict):
            raise WorldWorkerError("World Worker response must be a JSON object", status=status)
        if not allow_error and "error" in payload:
            error_body = payload.get("error")
            message = (
                str(error_body.get("message", "World Worker rejected the request"))
                if isinstance(error_body, Mapping)
                else "World Worker rejected the request"
            )
            code = (
                str(error_body.get("code", "world_worker_error"))
                if isinstance(error_body, Mapping)
                else "world_worker_error"
            )
            raise WorldWorkerError(message, status=status, code=code)
        return payload


__all__ = ["WorldWorkerClient", "WorldWorkerError", "WorldWorkerScene"]
