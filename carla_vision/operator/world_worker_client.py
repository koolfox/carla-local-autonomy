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
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_SCENE_PREPARE_REQUIRED_KEYS = frozenset(
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
_SCENE_PREPARE_OPTIONAL_KEYS = frozenset(
    {
        "pedestrian_crossing_factor",
        "speed_difference_percent",
        "following_distance_metres",
    }
)
_SCENE_PREPARE_KEYS = _SCENE_PREPARE_REQUIRED_KEYS | _SCENE_PREPARE_OPTIONAL_KEYS
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


@dataclass(frozen=True)
class WorldWorkerCameraFrame:
    sequence: int
    frame: int
    timestamp: float
    received_monotonic: float
    width: int
    height: int
    fov: float
    transform: tuple[float, float, float, float, float, float]
    jpeg: bytes

    def bgr(self) -> Any:
        import cv2
        import numpy as np

        decoded = cv2.imdecode(np.frombuffer(self.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise WorldWorkerError("World Worker camera returned an invalid JPEG")
        return decoded


class WorldWorkerCameraStream:
    """Newest-frame reader for the worker-side persistent MJPEG relay.

    New workers expose one authenticated multipart response, avoiding an HTTP
    request/response round trip for every camera frame.  Older workers retain
    the long-poll endpoint, which remains an automatic compatibility fallback.
    """

    def __init__(
        self,
        client: "WorldWorkerClient",
        scene: "WorldWorkerScene",
        *,
        timeout: float = 5.0,
    ) -> None:
        self.client = client
        self.scene = scene
        self.timeout = float(timeout)
        self._condition = threading.Condition()
        self._latest: WorldWorkerCameraFrame | None = None
        self._error: BaseException | None = None
        self._closed = False
        self.transport = (
            "worker_mjpeg"
            if bool(scene.capabilities.get("persistent_mjpeg_camera_relay"))
            else "worker_jpeg_long_poll"
        )
        self._thread = threading.Thread(
            target=self._run,
            name=f"world-worker-camera-{scene.scene_id}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        sequence = -1
        try:
            if self.transport == "worker_mjpeg":
                failures = 0
                while not self._closed:
                    try:
                        for frame in self.client.camera_frames(
                            self.scene,
                            timeout=self.timeout,
                        ):
                            if self._closed:
                                return
                            failures = 0
                            if frame.sequence <= sequence:
                                continue
                            sequence = frame.sequence
                            self._publish(frame)
                        if self._closed:
                            return
                        raise EOFError("persistent camera stream ended")
                    except WorldWorkerError as error:
                        if error.status in {
                            HTTPStatus.NOT_FOUND,
                            HTTPStatus.METHOD_NOT_ALLOWED,
                            HTTPStatus.NOT_IMPLEMENTED,
                        }:
                            # Only a Worker that explicitly lacks the endpoint
                            # may use the compatibility long-poll lane. A CARLA
                            # stall or network timeout must reconnect MJPEG.
                            self.transport = "worker_jpeg_long_poll_fallback"
                            break
                        if (
                            error.status is not None
                            and error.status < HTTPStatus.INTERNAL_SERVER_ERROR
                            and error.status
                            not in {HTTPStatus.REQUEST_TIMEOUT, HTTPStatus.TOO_MANY_REQUESTS}
                        ):
                            raise
                        failures += 1
                        if failures >= 5:
                            raise
                    except (EOFError, TimeoutError, OSError):
                        failures += 1
                        if failures >= 5:
                            raise

                    retry_seconds = min(2.0, 0.25 * (2 ** (failures - 1)))
                    with self._condition:
                        if self._closed:
                            return
                        self._condition.wait(retry_seconds)
            while not self._closed:
                frame = self.client.camera_frame(
                    self.scene,
                    after_sequence=sequence,
                    timeout=self.timeout,
                )
                if frame.sequence <= sequence:
                    time.sleep(0.01)
                    continue
                sequence = frame.sequence
                self._publish(frame)
        except BaseException as error:
            if not self._closed:
                with self._condition:
                    self._error = error
                    self._condition.notify_all()

    def _publish(self, frame: WorldWorkerCameraFrame) -> None:
        with self._condition:
            self._latest = frame
            self._condition.notify_all()

    def latest(self) -> WorldWorkerCameraFrame | None:
        with self._condition:
            if self._error is not None:
                raise WorldWorkerError(f"compressed camera stream failed: {self._error}")
            return self._latest

    def wait_for_frame(
        self,
        after_sequence: int = -1,
        timeout: float = 5.0,
    ) -> WorldWorkerCameraFrame:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if self._error is not None:
                    raise WorldWorkerError(f"compressed camera stream failed: {self._error}")
                if self._latest is not None and self._latest.sequence > after_sequence:
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("timed out waiting for a compressed camera frame")
                self._condition.wait(remaining)

    def close(self) -> None:
        self._closed = True
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=self.timeout + 1.0)

    def __enter__(self) -> "WorldWorkerCameraStream":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


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
    traffic_count_requested: int | None = None
    walker_count_requested: int | None = None
    pedestrian_crossing_factor: float | None = None
    speed_difference_percent: float | None = None
    following_distance_metres: float | None = None

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
        traffic_count_requested = _optional_integer(
            raw.get("traffic_count_requested"),
            "scene.traffic_count_requested",
            minimum=0,
        )
        walker_count_requested = _optional_integer(
            raw.get("walker_count_requested"),
            "scene.walker_count_requested",
            minimum=0,
        )
        pedestrian_crossing_factor = _optional_number(
            raw.get("pedestrian_crossing_factor"),
            "scene.pedestrian_crossing_factor",
            minimum=0.0,
            maximum=1.0,
        )
        speed_difference_percent = _optional_number(
            raw.get("speed_difference_percent"),
            "scene.speed_difference_percent",
            minimum=-100.0,
            maximum=100.0,
        )
        following_distance_metres = _optional_number(
            raw.get("following_distance_metres"),
            "scene.following_distance_metres",
            minimum=0.1,
            maximum=20.0,
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
            traffic_count_requested=traffic_count_requested,
            walker_count_requested=walker_count_requested,
            pedestrian_crossing_factor=pedestrian_crossing_factor,
            speed_difference_percent=speed_difference_percent,
            following_distance_metres=following_distance_metres,
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


def _optional_number(
    value: Any,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorldWorkerError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise WorldWorkerError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return result


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
        if not _SCENE_PREPARE_REQUIRED_KEYS <= keys or not keys <= _SCENE_PREPARE_KEYS:
            missing = sorted(_SCENE_PREPARE_REQUIRED_KEYS - keys)
            unknown = sorted(keys - _SCENE_PREPARE_KEYS)
            detail = []
            if missing:
                detail.append(f"missing {', '.join(missing)}")
            if unknown:
                detail.append(f"unknown {', '.join(unknown)}")
            raise ValueError(f"World Worker scene payload has {'; '.join(detail)}")
        if keys & _SCENE_PREPARE_OPTIONAL_KEYS:
            catalog = self.catalog()
            capabilities = catalog.get("capabilities", {})
            if not isinstance(capabilities, Mapping) or not bool(
                capabilities.get("world_dynamics_controls")
            ):
                raise WorldWorkerError(
                    "the Windows World Worker is outdated and cannot apply pedestrian "
                    "crossing, traffic speed, or following distance; pull main and restart "
                    "the Worker"
                )
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

    def start_camera(
        self,
        scene: WorldWorkerScene,
        *,
        mode: str,
        width: int,
        height: int,
        fps: float,
        fov: float,
        yaw: float = 325.0,
        pitch: float = -10.0,
        distance: float = 6.5,
    ) -> dict[str, Any]:
        if mode not in {"garage", "drive"}:
            raise ValueError("camera mode must be garage or drive")
        scene_id = quote(scene.scene_id, safe="")
        return self._request(
            "POST",
            f"/v1/scenes/{scene_id}/camera",
            {
                "lease_token": scene.lease_token,
                "mode": mode,
                "width": int(width),
                "height": int(height),
                "fps": float(fps),
                "fov": float(fov),
                "yaw": float(yaw),
                "pitch": float(pitch),
                "distance": float(distance),
            },
        )

    def orbit_camera(
        self,
        scene: WorldWorkerScene,
        *,
        yaw: float,
        pitch: float,
        distance: float,
        preset: str = "orbit",
    ) -> dict[str, Any]:
        choices = {"orbit", "front", "rear", "top", "cockpit"}
        if not isinstance(preset, str) or preset not in choices:
            raise ValueError("camera preset must be orbit, front, rear, top, or cockpit")
        supports_presets = bool(scene.capabilities.get("garage_camera_presets"))
        if preset != "orbit" and not supports_presets:
            raise WorldWorkerError(
                "the Windows World Worker is outdated and does not support Garage camera "
                "presets; pull main and restart the Worker"
            )
        scene_id = quote(scene.scene_id, safe="")
        payload: dict[str, Any] = {
            "lease_token": scene.lease_token,
            "yaw": float(yaw),
            "pitch": float(pitch),
            "distance": float(distance),
        }
        if supports_presets:
            payload["preset"] = preset
        return self._request(
            "POST",
            f"/v1/scenes/{scene_id}/camera_orbit",
            payload,
        )

    def camera_frame(
        self,
        scene: WorldWorkerScene,
        *,
        after_sequence: int,
        timeout: float,
    ) -> WorldWorkerCameraFrame:
        scene_id = quote(scene.scene_id, safe="")
        request = Request(
            f"{self.base_url}/v1/scenes/{scene_id}/camera/frame.jpg",
            headers={
                "Accept": "image/jpeg",
                "Authorization": f"Bearer {self._bearer_token}",
                "X-Scene-Lease": scene.lease_token,
                "X-Camera-After": str(int(after_sequence)),
                "X-Camera-Timeout": str(float(timeout)),
            },
            method="GET",
        )
        try:
            with self._opener.open(request, timeout=float(timeout) + 2.0) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
                headers = response.headers
                status = int(response.status)
        except HTTPError as error:
            body = error.read(_MAX_RESPONSE_BYTES + 1)
            parsed = self._decode(body, status=error.code, allow_error=True)
            error_body = parsed.get("error")
            message = (
                str(error_body.get("message", error.reason))
                if isinstance(error_body, Mapping)
                else str(error.reason)
            )
            raise WorldWorkerError(message, status=error.code) from error
        except (URLError, TimeoutError, socket.timeout, OSError) as error:
            reason = getattr(error, "reason", error)
            raise WorldWorkerError(f"World Worker camera is unreachable: {reason}") from error
        if status != HTTPStatus.OK or len(raw) > _MAX_RESPONSE_BYTES:
            raise WorldWorkerError("World Worker camera response is invalid", status=status)
        return self._camera_frame_from_headers(raw, headers, status=status)

    def camera_frames(
        self,
        scene: WorldWorkerScene,
        *,
        timeout: float,
    ) -> Iterator[WorldWorkerCameraFrame]:
        """Yield authenticated newest-only frames from one MJPEG response."""

        scene_id = quote(scene.scene_id, safe="")
        request = Request(
            f"{self.base_url}/v1/scenes/{scene_id}/camera/stream.mjpg",
            headers={
                "Accept": "multipart/x-mixed-replace",
                "Authorization": f"Bearer {self._bearer_token}",
                "X-Scene-Lease": scene.lease_token,
            },
            method="GET",
        )
        try:
            with self._opener.open(request, timeout=float(timeout) + 2.0) as response:
                status = int(response.status)
                if status != HTTPStatus.OK:
                    raise WorldWorkerError(
                        f"World Worker camera stream returned HTTP {status}",
                        status=status,
                    )
                boundary = self._mjpeg_boundary(response.headers.get("Content-Type", ""))
                while True:
                    marker = response.readline(256)
                    if marker == b"":
                        return
                    if len(marker) > 255:
                        raise WorldWorkerError("World Worker camera boundary is invalid")
                    marker = marker.rstrip(b"\r\n")
                    if not marker:
                        continue
                    if marker == b"--" + boundary + b"--":
                        return
                    if marker != b"--" + boundary:
                        raise WorldWorkerError("World Worker camera boundary is invalid")

                    part_headers: dict[str, str] = {}
                    for _ in range(32):
                        line = response.readline(8193)
                        if not line:
                            raise WorldWorkerError("World Worker camera part ended early")
                        if len(line) > 8192:
                            raise WorldWorkerError("World Worker camera header is too large")
                        if line in {b"\r\n", b"\n"}:
                            break
                        try:
                            name, value = line.decode("ascii").rstrip("\r\n").split(":", 1)
                        except (UnicodeError, ValueError) as error:
                            raise WorldWorkerError(
                                "World Worker camera header is invalid"
                            ) from error
                        normalized = name.strip().lower()
                        if not normalized or normalized in part_headers:
                            raise WorldWorkerError("World Worker camera header is invalid")
                        part_headers[normalized] = value.strip()
                    else:
                        raise WorldWorkerError("World Worker camera part has too many headers")

                    if part_headers.get("content-type", "").lower() != "image/jpeg":
                        raise WorldWorkerError("World Worker camera part is not JPEG")
                    try:
                        length = int(part_headers["content-length"])
                    except (KeyError, ValueError) as error:
                        raise WorldWorkerError(
                            "World Worker camera content length is invalid"
                        ) from error
                    if not 0 < length <= _MAX_RESPONSE_BYTES:
                        raise WorldWorkerError("World Worker camera frame exceeds the size limit")
                    raw = self._read_exact(response, length)
                    if self._read_exact(response, 2) != b"\r\n":
                        raise WorldWorkerError("World Worker camera part terminator is invalid")
                    yield self._camera_frame_from_headers(raw, part_headers, status=status)
        except HTTPError as error:
            body = error.read(_MAX_RESPONSE_BYTES + 1)
            parsed = self._decode(body, status=error.code, allow_error=True)
            error_body = parsed.get("error")
            message = (
                str(error_body.get("message", error.reason))
                if isinstance(error_body, Mapping)
                else str(error.reason)
            )
            raise WorldWorkerError(message, status=error.code) from error
        except (URLError, TimeoutError, socket.timeout, OSError) as error:
            reason = getattr(error, "reason", error)
            raise WorldWorkerError(
                f"World Worker camera stream is unreachable: {reason}"
            ) from error

    @staticmethod
    def _read_exact(stream: Any, length: int) -> bytes:
        chunks: list[bytes] = []
        remaining = length
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                raise WorldWorkerError("World Worker camera part ended early")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    @staticmethod
    def _mjpeg_boundary(content_type: str) -> bytes:
        pieces = [piece.strip() for piece in str(content_type).split(";")]
        if not pieces or pieces[0].lower() != "multipart/x-mixed-replace":
            raise WorldWorkerError("World Worker camera stream has an invalid content type")
        raw_boundary = next(
            (
                piece.split("=", 1)[1].strip().strip('"')
                for piece in pieces[1:]
                if piece.lower().startswith("boundary=")
            ),
            "",
        )
        if (
            not raw_boundary
            or len(raw_boundary) > 70
            or any(
                character not in "-_0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                for character in raw_boundary
            )
        ):
            raise WorldWorkerError("World Worker camera stream boundary is invalid")
        return raw_boundary.encode("ascii")

    @staticmethod
    def _camera_frame_from_headers(
        raw: bytes,
        headers: Mapping[str, Any],
        *,
        status: int,
    ) -> WorldWorkerCameraFrame:
        if not raw.startswith(b"\xff\xd8") or not raw.endswith(b"\xff\xd9"):
            raise WorldWorkerError("World Worker camera response is not JPEG", status=status)

        def header(name: str) -> Any:
            direct = headers.get(name)
            return direct if direct is not None else headers.get(name.lower())

        try:
            transform_payload = json.loads(header("X-Camera-Transform"))
            location = transform_payload["location"]
            rotation = transform_payload["rotation"]
            transform = (
                float(location["x"]),
                float(location["y"]),
                float(location["z"]),
                float(rotation["pitch"]),
                float(rotation["yaw"]),
                float(rotation["roll"]),
            )
            sequence = header("X-Camera-Sequence")
            if sequence is None:
                sequence = header("X-CARLA-Sequence")
            return WorldWorkerCameraFrame(
                sequence=int(sequence),
                frame=int(header("X-CARLA-Frame")),
                timestamp=float(header("X-CARLA-Timestamp")),
                received_monotonic=time.monotonic(),
                width=int(header("X-Camera-Width")),
                height=int(header("X-Camera-Height")),
                fov=float(header("X-Camera-FOV")),
                transform=transform,
                jpeg=raw,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise WorldWorkerError("World Worker camera metadata is invalid") from error

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


__all__ = [
    "WorldWorkerCameraFrame",
    "WorldWorkerCameraStream",
    "WorldWorkerClient",
    "WorldWorkerError",
    "WorldWorkerScene",
]
