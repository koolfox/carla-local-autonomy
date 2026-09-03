"""Standalone authenticated bridge to CARLA's official PythonAPI.

The worker deliberately exposes a small HTTP/JSON allow-list rather than a
generic CARLA RPC proxy.  It owns at most one asynchronous-world scene and all
actors created for that scene.  The official :mod:`carla` package and optional
route planner are imported lazily so importing this module remains safe on
non-CARLA hosts and in unit tests.

Run this file by path on the CARLA host.  Do not install the research project::

    py -3.12 carla_vision\\native\\world_worker.py --help

Direct-file execution uses only Python's standard library plus the official
``carla`` module already available to the selected interpreter.
"""

from __future__ import annotations

import argparse
import hmac
import importlib
import ipaddress
import json
import math
import os
import random
import re
import secrets
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlparse

SCHEMA_VERSION = "1.0"
WORKER_API_REVISION = 5
EXPECTED_CARLA_VERSION = "0.9.16"
DEFAULT_BIND = "127.0.0.1"
DEFAULT_PORT = 8766
DEFAULT_CARLA_HOST = "127.0.0.1"
DEFAULT_CARLA_PORT = 2000
DEFAULT_TRAFFIC_MANAGER_PORT = 8000
DEFAULT_MAP_LOAD_TIMEOUT = 60.0
DEFAULT_MAP_RECONNECT_SECONDS = 30.0
TOKEN_ENVIRONMENT_VARIABLE = "CARLA_WORLD_WORKER_TOKEN"

_MAX_BODY_BYTES = 64 * 1024
_MAP_NAME = re.compile(r"^[A-Za-z0-9_./-]{1,160}$")
_VEHICLE_BLUEPRINT = re.compile(r"^vehicle\.[A-Za-z0-9_.-]{1,150}$")
_COLOR = re.compile(r"^\d{1,3},\d{1,3},\d{1,3}$")
_SCENE_PATH = re.compile(
    r"^/v1/scenes/(?P<scene_id>[A-Za-z0-9_-]{16,128})/"
    r"(?P<action>start|heartbeat|control|mode|weather|configure|camera|camera_orbit|camera_pause|camera_resume|waypoints|stop)$"
)
_CAMERA_FRAME_PATH = re.compile(
    r"^/v1/scenes/(?P<scene_id>[A-Za-z0-9_-]{16,128})/camera/frame\.jpg$"
)
_CAMERA_STREAM_PATH = re.compile(
    r"^/v1/scenes/(?P<scene_id>[A-Za-z0-9_-]{16,128})/camera/stream\.mjpg$"
)
_MJPEG_BOUNDARY = "carla-frame"
_JPEG_QUALITY = 90
_ACTIVE_SCENE_STATES = frozenset({"prepared", "running", "stopping"})
_ROUTE_MODES = frozenset({"free", "random_destination"})
_CONTROL_MODES = frozenset({"manual", "autopilot"})
_GARAGE_CAMERA_PRESETS = frozenset({"orbit", "front", "rear", "top", "cockpit"})
_GARAGE_EXTERIOR_PRESETS: dict[str, tuple[float, float, float]] = {
    "front": (0.0, -8.0, 6.5),
    "rear": (180.0, -8.0, 6.5),
    "top": (0.0, -70.0, 8.0),
}
_SIMULATOR_SEED_MODULUS = 2**31 - 1

# Keep the bridge executable as a single file on the CARLA host.  These small
# presets are intentionally embedded here instead of importing the research
# package, so ``python world_worker.py`` needs only the standard library and
# CARLA's official PythonAPI.
WEATHER_PRESETS: dict[str, dict[str, Any]] = {
    "clear-day": {
        "light": "day",
        "cloudiness": 5.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 5.0,
        "sun_azimuth_angle": 35.0,
        "sun_altitude_angle": 55.0,
        "fog_density": 0.0,
        "fog_distance": 0.0,
        "wetness": 0.0,
        "fog_falloff": 0.2,
        "scattering_intensity": 1.0,
        "mie_scattering_scale": 0.03,
        "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 0.0,
    },
    "cloudy-day": {
        "light": "day",
        "cloudiness": 75.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 20.0,
        "sun_azimuth_angle": 120.0,
        "sun_altitude_angle": 40.0,
        "fog_density": 2.0,
        "fog_distance": 80.0,
        "wetness": 0.0,
        "fog_falloff": 0.2,
        "scattering_intensity": 1.0,
        "mie_scattering_scale": 0.05,
        "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 0.0,
    },
    "soft-rain-sunset": {
        "light": "sunset",
        "cloudiness": 70.0,
        "precipitation": 20.0,
        "precipitation_deposits": 25.0,
        "wind_intensity": 35.0,
        "sun_azimuth_angle": 250.0,
        "sun_altitude_angle": 8.0,
        "fog_density": 4.0,
        "fog_distance": 40.0,
        "wetness": 55.0,
        "fog_falloff": 0.5,
        "scattering_intensity": 1.0,
        "mie_scattering_scale": 0.08,
        "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 0.0,
    },
    "heavy-rain": {
        "light": "day",
        "cloudiness": 95.0,
        "precipitation": 80.0,
        "precipitation_deposits": 70.0,
        "wind_intensity": 60.0,
        "sun_azimuth_angle": 180.0,
        "sun_altitude_angle": 25.0,
        "fog_density": 12.0,
        "fog_distance": 30.0,
        "wetness": 90.0,
        "fog_falloff": 0.5,
        "scattering_intensity": 1.2,
        "mie_scattering_scale": 0.1,
        "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 0.0,
    },
    "fog-night": {
        "light": "night",
        "cloudiness": 65.0,
        "precipitation": 0.0,
        "precipitation_deposits": 10.0,
        "wind_intensity": 8.0,
        "sun_azimuth_angle": 315.0,
        "sun_altitude_angle": -35.0,
        "fog_density": 45.0,
        "fog_distance": 12.0,
        "wetness": 20.0,
        "fog_falloff": 0.7,
        "scattering_intensity": 1.5,
        "mie_scattering_scale": 0.2,
        "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 0.0,
    },
    "wet-day": {
        "light": "day",
        "cloudiness": 45.0,
        "precipitation": 0.0,
        "precipitation_deposits": 45.0,
        "wind_intensity": 15.0,
        "sun_azimuth_angle": 160.0,
        "sun_altitude_angle": 35.0,
        "fog_density": 3.0,
        "fog_distance": 60.0,
        "wetness": 70.0,
        "fog_falloff": 0.2,
        "scattering_intensity": 1.0,
        "mie_scattering_scale": 0.05,
        "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 0.0,
    },
}


def _prop_transform(x: float, y: float, z: float = 0.0) -> dict[str, float]:
    return {
        "x": x,
        "y": y,
        "z": z,
        "pitch": 0.0,
        "yaw": 0.0,
        "roll": 0.0,
    }


PROP_PRESETS: dict[str, tuple[dict[str, Any], ...]] = {
    "none": (),
    "cones": (
        {
            "blueprint_id": "static.prop.trafficcone01",
            "relative_to": "ego_start",
            "transform": _prop_transform(24.0, 1.8),
        },
        {
            "blueprint_id": "static.prop.trafficcone02",
            "relative_to": "ego_start",
            "transform": _prop_transform(30.0, 1.8),
        },
    ),
    "construction": (
        {
            "blueprint_id": "static.prop.warningconstruction",
            "relative_to": "ego_start",
            "transform": _prop_transform(28.0, 3.5),
        },
        {
            "blueprint_id": "static.prop.streetbarrier",
            "relative_to": "ego_start",
            "transform": _prop_transform(34.0, 2.8),
        },
        {
            "blueprint_id": "static.prop.trafficcone01",
            "relative_to": "ego_start",
            "transform": _prop_transform(25.0, 1.8),
        },
    ),
    "accident": (
        {
            "blueprint_id": "static.prop.warningaccident",
            "relative_to": "ego_start",
            "transform": _prop_transform(30.0, 3.0),
        },
        {
            "blueprint_id": "static.prop.dirtdebris01",
            "relative_to": "ego_start",
            "transform": _prop_transform(36.0, 1.5),
        },
    ),
}


class WorkerError(Exception):
    """HTTP-safe error raised by the strict worker application."""

    def __init__(self, status: int | HTTPStatus, code: str, message: str) -> None:
        super().__init__(message)
        self.status = HTTPStatus(status)
        self.code = str(code)
        self.message = str(message)


def _strict_keys(
    raw: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str] = frozenset(),
    name: str,
) -> None:
    keys = {str(key) for key in raw}
    missing = sorted(required - keys)
    unknown = sorted(keys - allowed)
    if missing:
        raise WorkerError(
            HTTPStatus.BAD_REQUEST,
            "missing_fields",
            f"{name} is missing fields: {', '.join(missing)}",
        )
    if unknown:
        raise WorkerError(
            HTTPStatus.BAD_REQUEST,
            "unknown_fields",
            f"{name} has unknown fields: {', '.join(unknown)}",
        )


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerError(HTTPStatus.BAD_REQUEST, "invalid_field", f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise WorkerError(
            HTTPStatus.BAD_REQUEST,
            "invalid_field",
            f"{name} must be in [{minimum}, {maximum}]",
        )
    return value


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerError(HTTPStatus.BAD_REQUEST, "invalid_field", f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise WorkerError(
            HTTPStatus.BAD_REQUEST,
            "invalid_field",
            f"{name} must be finite and in [{minimum}, {maximum}]",
        )
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise WorkerError(HTTPStatus.BAD_REQUEST, "invalid_field", f"{name} must be boolean")
    return value


def _text(value: Any, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str):
        raise WorkerError(HTTPStatus.BAD_REQUEST, "invalid_field", f"{name} must be a string")
    result = value.strip()
    if not result or len(result) > maximum:
        raise WorkerError(
            HTTPStatus.BAD_REQUEST,
            "invalid_field",
            f"{name} must contain 1-{maximum} characters",
        )
    return result


def _map_short_name(value: str) -> str:
    return (
        str(value).replace("\\", "/").rstrip("/").rsplit("/", maxsplit=1)[-1].removesuffix(".umap")
    )


def _label(identifier: str, prefix: str = "") -> str:
    value = identifier.removeprefix(prefix).replace("_", " ").replace(".", " · ")
    return value.title()


def _json_location(location: Any) -> dict[str, float]:
    return {
        "x": float(location.x),
        "y": float(location.y),
        "z": float(location.z),
    }


def _json_transform(transform: Any) -> dict[str, Any]:
    rotation = transform.rotation
    return {
        "location": _json_location(transform.location),
        "rotation": {
            "pitch": float(rotation.pitch),
            "yaw": float(rotation.yaw),
            "roll": float(rotation.roll),
        },
    }


def _default_carla_loader() -> Any:
    return importlib.import_module("carla")


def _default_route_planner_loader() -> Callable[[Any], Any] | None:
    try:
        module = importlib.import_module("agents.navigation.global_route_planner")
    except ImportError:
        return None
    planner_type = getattr(module, "GlobalRoutePlanner", None)
    if planner_type is None:
        return None

    def create(map_object: Any) -> Any:
        try:
            return planner_type(map_object, sampling_resolution=2.0)
        except TypeError:
            return planner_type(map_object, 2.0)

    return create


@dataclass(frozen=True)
class SceneConfig:
    """Validated allow-listed scene creation request."""

    map_name: str = "current"
    weather_preset: str = "keep"
    vehicle_blueprint: str = "vehicle.tesla.model3"
    color: str | None = None
    seed: int = 0
    traffic_count: int = 0
    walker_count: int = 0
    prop_preset: str = "none"
    route_mode: str = "free"
    initial_control_mode: str = "manual"
    pedestrian_crossing_factor: float = 0.2
    speed_difference_percent: float = 12.0
    following_distance_metres: float = 2.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        allowed = {
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
            "pedestrian_crossing_factor",
            "speed_difference_percent",
            "following_distance_metres",
        }
        _strict_keys(raw, allowed=allowed, name="scene prepare request")

        map_name = str(raw.get("map_name", "current")).strip()
        if map_name != "current" and not _MAP_NAME.fullmatch(map_name):
            raise WorkerError(
                HTTPStatus.BAD_REQUEST,
                "invalid_field",
                "map_name must be 'current' or an exact catalog map identifier",
            )
        weather = str(raw.get("weather_preset", "keep")).strip()
        if weather != "keep" and weather not in WEATHER_PRESETS:
            raise WorkerError(
                HTTPStatus.BAD_REQUEST,
                "invalid_field",
                f"unknown weather_preset {weather!r}",
            )
        blueprint = str(raw.get("vehicle_blueprint", "vehicle.tesla.model3")).strip()
        if not _VEHICLE_BLUEPRINT.fullmatch(blueprint):
            raise WorkerError(
                HTTPStatus.BAD_REQUEST,
                "invalid_field",
                "vehicle_blueprint must be an exact vehicle.* identifier",
            )

        color_value = raw.get("color")
        color = None if color_value in {None, ""} else str(color_value).strip()
        if color is not None:
            if not _COLOR.fullmatch(color) or any(
                not 0 <= int(item) <= 255 for item in color.split(",")
            ):
                raise WorkerError(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_field",
                    "color must be a catalog RGB triplet such as '255,0,0' or null",
                )

        prop_preset = str(raw.get("prop_preset", "none")).strip()
        if prop_preset not in PROP_PRESETS:
            raise WorkerError(
                HTTPStatus.BAD_REQUEST,
                "invalid_field",
                f"unknown prop_preset {prop_preset!r}",
            )
        route_mode = str(raw.get("route_mode", "free")).strip()
        if route_mode not in _ROUTE_MODES:
            raise WorkerError(
                HTTPStatus.BAD_REQUEST,
                "invalid_field",
                "route_mode must be free or random_destination",
            )
        control_mode = str(raw.get("initial_control_mode", "manual")).strip()
        if control_mode not in _CONTROL_MODES:
            raise WorkerError(
                HTTPStatus.BAD_REQUEST,
                "invalid_field",
                "initial_control_mode must be manual or autopilot",
            )

        return cls(
            map_name=map_name,
            weather_preset=weather,
            vehicle_blueprint=blueprint,
            color=color,
            seed=_integer(raw.get("seed", 0), "seed", 0, 2**63 - 1),
            traffic_count=_integer(raw.get("traffic_count", 0), "traffic_count", 0, 250),
            walker_count=_integer(raw.get("walker_count", 0), "walker_count", 0, 250),
            prop_preset=prop_preset,
            route_mode=route_mode,
            initial_control_mode=control_mode,
            pedestrian_crossing_factor=_number(
                raw.get("pedestrian_crossing_factor", 0.2),
                "pedestrian_crossing_factor",
                0.0,
                1.0,
            ),
            speed_difference_percent=_number(
                raw.get("speed_difference_percent", 12.0),
                "speed_difference_percent",
                -100.0,
                100.0,
            ),
            following_distance_metres=_number(
                raw.get("following_distance_metres", 2.0),
                "following_distance_metres",
                0.1,
                20.0,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "map_name": self.map_name,
            "weather_preset": self.weather_preset,
            "vehicle_blueprint": self.vehicle_blueprint,
            "color": self.color,
            "seed": self.seed,
            "traffic_count": self.traffic_count,
            "walker_count": self.walker_count,
            "prop_preset": self.prop_preset,
            "route_mode": self.route_mode,
            "initial_control_mode": self.initial_control_mode,
            "pedestrian_crossing_factor": self.pedestrian_crossing_factor,
            "speed_difference_percent": self.speed_difference_percent,
            "following_distance_metres": self.following_distance_metres,
        }


@dataclass
class OwnedActor:
    actor: Any
    actor_id: int
    type_id: str
    kind: str
    role_name: str | None = None


@dataclass(frozen=True)
class ActorSpawnRequest:
    """A prepared spawn; CARLA commands snapshot mutable blueprint attributes."""

    blueprint: Any
    transform: Any
    kind: str
    role_name: str | None
    parent: Any = None
    command: Any = None


@dataclass(frozen=True)
class CompressedCameraConfig:
    mode: str
    width: int
    height: int
    fps: float
    fov: float
    yaw: float = 325.0
    pitch: float = -10.0
    distance: float = 6.5

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        allowed = {
            "lease_token",
            "mode",
            "width",
            "height",
            "fps",
            "fov",
            "yaw",
            "pitch",
            "distance",
        }
        _strict_keys(
            raw,
            allowed=allowed,
            required={"lease_token", "mode", "width", "height", "fps", "fov"},
            name="compressed camera request",
        )
        mode = str(raw["mode"]).strip()
        if mode not in {"garage", "drive"}:
            raise WorkerError(
                HTTPStatus.BAD_REQUEST, "invalid_field", "camera mode must be garage or drive"
            )
        width = _integer(raw["width"], "width", 320, 1920)
        height = _integer(raw["height"], "height", 180, 1080)
        fps = _number(raw["fps"], "fps", 1.0, 60.0)
        fov = _number(raw["fov"], "fov", 30.0, 150.0)
        yaw = _number(raw.get("yaw", 325.0), "yaw", -3600.0, 3600.0)
        pitch = _number(raw.get("pitch", -10.0), "pitch", -25.0, 15.0)
        distance = _number(raw.get("distance", 6.5), "distance", 3.5, 10.0)
        return cls(
            mode=mode,
            width=width,
            height=height,
            fps=fps,
            fov=fov,
            yaw=yaw,
            pitch=pitch,
            distance=distance,
        )


def _camera_encoder_modules_present() -> bool:
    """Report whether the optional in-memory encoder looks importable.

    The actual import remains lazy because the standalone worker must still
    start, serve health, and manage non-camera scenes with only CARLA's
    official PythonAPI installed.
    """

    try:
        return all(importlib.util.find_spec(name) is not None for name in ("numpy", "cv2"))
    except (ImportError, AttributeError, ValueError):
        return False


def _load_in_memory_jpeg_encoder() -> Callable[[Any, int], bytes]:
    """Load NumPy/OpenCV only when a compressed camera is requested."""

    try:
        numpy = importlib.import_module("numpy")
        cv2 = importlib.import_module("cv2")
    except (ImportError, OSError) as error:
        raise WorkerError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "camera_encoder_unavailable",
            "in-memory camera streaming requires numpy and OpenCV on the CARLA host; "
            "install them in the Worker interpreter with "
            "'python -m pip install numpy opencv-python-headless'",
        ) from error

    def encode(image: Any, quality: int) -> bytes:
        width = int(getattr(image, "width", 0))
        height = int(getattr(image, "height", 0))
        if width <= 0 or height <= 0:
            raise RuntimeError("CARLA camera image has invalid dimensions")
        raw_data = getattr(image, "raw_data", None)
        if raw_data is None:
            raise RuntimeError("CARLA camera image does not expose raw_data")
        pixels = numpy.frombuffer(raw_data, dtype=numpy.uint8)
        expected = width * height * 4
        if int(pixels.size) != expected:
            raise RuntimeError(
                f"CARLA BGRA buffer has {int(pixels.size)} bytes; expected {expected}"
            )
        # CARLA emits BGRA. OpenCV's JPEG encoder needs BGR, so dropping alpha
        # is sufficient and avoids an extra color-conversion allocation.
        bgr = pixels.reshape((height, width, 4))[:, :, :3]
        options = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        ok, encoded = cv2.imencode(".jpg", bgr, options)
        if not ok:
            raise RuntimeError("OpenCV could not encode the CARLA frame as JPEG")
        payload = encoded.tobytes()
        if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
            raise RuntimeError("OpenCV produced an invalid JPEG frame")
        return payload

    return encode


class CompressedCameraRelay:
    """Keep only the newest in-memory JPEG produced beside the simulator."""

    _STOP_DRAIN_SECONDS = 0.4
    _ENCODER_JOIN_TIMEOUT_SECONDS = 2.0

    def __init__(
        self,
        sensor: Any,
        *,
        jpeg_encoder: Callable[[Any, int], bytes],
        jpeg_quality: int = _JPEG_QUALITY,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.sensor = sensor
        self._jpeg_encoder = jpeg_encoder
        self._jpeg_quality = int(jpeg_quality)
        self._clock = clock
        self._condition = threading.Condition()
        self._lifecycle_lock = threading.RLock()
        self._closed = False
        self._paused = False
        self._close_complete = False
        self._stopped_at: float | None = None
        self._pending_image: Any | None = None
        self._encoder_thread: threading.Thread | None = None
        self._sequence = -1
        self._jpeg: bytes | None = None
        self._metadata: dict[str, Any] = {}
        self._error: str | None = None
        self._frames_received = 0
        self._frames_encoded = 0
        self._frames_dropped_pending = 0
        self._frames_replaced = 0
        self._encoded_bytes_total = 0
        self._encode_seconds_total = 0.0
        self._encode_completed_at: list[float] = []

    def listen(self) -> None:
        # Do not let close() stop the sensor before listen() registers its
        # callback, or join a thread whose start is still pending.
        with self._lifecycle_lock:
            with self._condition:
                if self._closed:
                    raise RuntimeError("cannot listen on a closed camera relay")
                if self._encoder_thread is not None:
                    raise RuntimeError("camera relay is already listening")
                encoder_thread = threading.Thread(
                    target=self._encode_loop,
                    name=f"carla-camera-encoder-{int(self.sensor.id)}",
                    daemon=True,
                )
                self._encoder_thread = encoder_thread
            try:
                encoder_thread.start()
                self.sensor.listen(self._on_image)
            except BaseException as error:
                try:
                    self.close()
                except BaseException as cleanup_error:
                    error.add_note(f"camera relay cleanup failed: {cleanup_error}")
                raise

    def pause(self) -> None:
        """Stop CARLA sensor delivery without destroying the camera actor."""

        with self._lifecycle_lock:
            with self._condition:
                if self._closed:
                    raise RuntimeError("cannot pause a closed camera relay")
                if self._paused:
                    return
                self._paused = True
                self._pending_image = None
                self._condition.notify_all()
            try:
                self.sensor.stop()
            except BaseException:
                with self._condition:
                    self._paused = False
                    self._condition.notify_all()
                raise

    def resume(self) -> None:
        """Resume the same CARLA sensor subscription after :meth:`pause`."""

        with self._lifecycle_lock:
            with self._condition:
                if self._closed:
                    raise RuntimeError("cannot resume a closed camera relay")
                if not self._paused:
                    return
            try:
                self.sensor.listen(self._on_image)
            except BaseException:
                # Keep the relay visibly paused when CARLA could not subscribe.
                raise
            with self._condition:
                self._paused = False
                self._condition.notify_all()

    def _on_image(self, image: Any) -> None:
        with self._condition:
            if self._closed or self._paused:
                return
            self._frames_received += 1
            if self._pending_image is not None:
                self._frames_dropped_pending += 1
            self._pending_image = image
            self._condition.notify_all()

    def _encode_loop(self) -> None:
        while True:
            with self._condition:
                if self._closed:
                    return
                while not self._closed and self._pending_image is None:
                    self._condition.wait()
                if self._closed:
                    return
                image = self._pending_image
                self._pending_image = None

            started_at = self._clock()
            try:
                payload = self._jpeg_encoder(image, self._jpeg_quality)
                completed_at = self._clock()
                encode_seconds = max(0.0, completed_at - started_at)
                transform = getattr(image, "transform", None)
                metadata = {
                    "frame": int(getattr(image, "frame", 0)),
                    "timestamp": float(getattr(image, "timestamp", 0.0)),
                    "width": int(getattr(image, "width", 0)),
                    "height": int(getattr(image, "height", 0)),
                    "fov": float(getattr(image, "fov", 0.0)),
                    "transform": None if transform is None else _json_transform(transform),
                }
                with self._condition:
                    if self._closed:
                        return
                    self._sequence += 1
                    if self._jpeg is not None:
                        self._frames_replaced += 1
                    self._jpeg = payload
                    self._metadata = metadata
                    self._error = None
                    self._frames_encoded += 1
                    self._encoded_bytes_total += len(payload)
                    self._encode_seconds_total += encode_seconds
                    self._encode_completed_at.append(completed_at)
                    cutoff = completed_at - 5.0
                    while (
                        len(self._encode_completed_at) > 2 and self._encode_completed_at[0] < cutoff
                    ):
                        self._encode_completed_at.pop(0)
                    self._condition.notify_all()
            except Exception as error:
                with self._condition:
                    if self._closed:
                        return
                    self._error = f"{type(error).__name__}: {error}"
                    self._condition.notify_all()

    def wait(self, after_sequence: int, timeout: float) -> tuple[int, bytes, dict[str, Any]]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._closed and self._sequence <= after_sequence and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise WorkerError(
                        HTTPStatus.REQUEST_TIMEOUT,
                        "camera_timeout",
                        "timed out waiting for a compressed camera frame",
                    )
                self._condition.wait(remaining)
            if self._error is not None:
                raise WorkerError(
                    HTTPStatus.SERVICE_UNAVAILABLE, "camera_encode_failed", self._error
                )
            if self._closed or self._jpeg is None:
                raise WorkerError(
                    HTTPStatus.CONFLICT, "camera_inactive", "compressed camera is not active"
                )
            return self._sequence, self._jpeg, dict(self._metadata)

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            completion_times = self._encode_completed_at
            actual_fps = 0.0
            if len(completion_times) >= 2:
                elapsed = completion_times[-1] - completion_times[0]
                if elapsed > 0.0:
                    actual_fps = (len(completion_times) - 1) / elapsed
            average_encode_ms = 0.0
            if self._frames_encoded:
                average_encode_ms = self._encode_seconds_total * 1000.0 / self._frames_encoded
            return {
                "actor_id": int(self.sensor.id),
                "paused": self._paused,
                "sequence": self._sequence,
                "error": self._error,
                "telemetry": {
                    "jpeg_quality": self._jpeg_quality,
                    "frames_received": self._frames_received,
                    "frames_encoded": self._frames_encoded,
                    "frames_dropped_pending": self._frames_dropped_pending,
                    "frames_replaced": self._frames_replaced,
                    "encoded_bytes_total": self._encoded_bytes_total,
                    "latest_jpeg_bytes": 0 if self._jpeg is None else len(self._jpeg),
                    "average_encode_ms": round(average_encode_ms, 3),
                    "actual_fps_5s": round(actual_fps, 3),
                },
                **self._metadata,
            }

    def close(self) -> None:
        if self._encoder_thread is threading.current_thread():
            raise RuntimeError("camera relay cannot be closed by its encoder thread")
        # Every concurrent caller must observe the completed drain, not just
        # the flag that prevents callbacks from handing off another image.
        with self._lifecycle_lock:
            if self._close_complete:
                return
            with self._condition:
                self._closed = True
                self._pending_image = None
                self._condition.notify_all()
            if self._stopped_at is None:

                def confirmed_dead() -> bool:
                    try:
                        return self.sensor.is_alive is False
                    except Exception:
                        return False

                # A map change or external destroy can invalidate the sensor
                # before cleanup. Unknown state is not proof of detachment.
                if not confirmed_dead() and not self._paused:
                    try:
                        self.sensor.stop()
                    except Exception:
                        # Destruction may race stop(); still drain below when
                        # the sensor explicitly confirms it is no longer alive.
                        if not confirmed_dead():
                            raise
                self._stopped_at = time.monotonic()
            encoder_thread = self._encoder_thread
            if encoder_thread is not None and encoder_thread.is_alive():
                encoder_thread.join(timeout=self._ENCODER_JOIN_TIMEOUT_SECONDS)
                if encoder_thread.is_alive():
                    raise RuntimeError("camera encoder did not stop before the teardown timeout")
            # CARLA manual_control.py (ffd9d275c) allows 0.4s after stop() for
            # queued native callbacks to drain before the caller destroys the
            # sensor. Time spent joining the encoder counts toward this grace.
            remaining = self._STOP_DRAIN_SECONDS - (time.monotonic() - self._stopped_at)
            if remaining > 0.0:
                time.sleep(remaining)
            self._close_complete = True


@dataclass
class SceneLease:
    scene_id: str
    lease_token: str
    config: SceneConfig
    client: Any
    world: Any
    traffic_manager: Any
    episode_id: int
    episode_marker: tuple[Any, ...]
    map_name: str
    original_weather: Any
    ego: Any
    spawn_index: int
    owned_actors: list[OwnedActor]
    vehicle_actors: list[Any] = field(default_factory=list)
    walker_actors: list[Any] = field(default_factory=list)
    walker_controllers: list[Any] = field(default_factory=list)
    prop_actors: list[Any] = field(default_factory=list)
    status: str = "prepared"
    control_mode: str = "manual"
    weather_preset: str = "keep"
    route: dict[str, Any] = field(default_factory=dict)
    destination: dict[str, Any] | None = None
    route_locations: list[Any] = field(default_factory=list)
    waypoint_map: Any | None = None
    lease_deadline: float = 0.0
    last_control_at: float | None = None
    last_control_sequence: int = -1
    deadman_active: bool = True
    stop_reason: str | None = None
    cleanup_guard_passed: bool | None = None
    cleanup_errors: list[str] = field(default_factory=list)
    camera_relay: CompressedCameraRelay | None = None
    camera_config: CompressedCameraConfig | None = None


class WorldWorker:
    """Thread-safe CARLA scene owner used by the HTTP adapter."""

    def __init__(
        self,
        *,
        carla_host: str = DEFAULT_CARLA_HOST,
        carla_port: int = DEFAULT_CARLA_PORT,
        traffic_manager_port: int = DEFAULT_TRAFFIC_MANAGER_PORT,
        timeout: float = 5.0,
        map_load_timeout: float = DEFAULT_MAP_LOAD_TIMEOUT,
        map_reconnect_seconds: float = DEFAULT_MAP_RECONNECT_SECONDS,
        lease_seconds: float = 30.0,
        control_timeout: float = 0.75,
        expected_carla_version: str = EXPECTED_CARLA_VERSION,
        carla_loader: Callable[[], Any] = _default_carla_loader,
        route_planner_loader: Callable[[], Callable[[Any], Any] | None] = (
            _default_route_planner_loader
        ),
        clock: Callable[[], float] = time.monotonic,
        start_monitor: bool = True,
        monitor_period: float = 0.05,
        map_process_runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        if not 1 <= int(carla_port) <= 65535:
            raise ValueError("carla_port must be in [1, 65535]")
        if not 1 <= int(traffic_manager_port) <= 65535:
            raise ValueError("traffic_manager_port must be in [1, 65535]")
        if not 0.1 <= float(timeout) <= 300.0:
            raise ValueError("timeout must be in [0.1, 300]")
        if not 5.0 <= float(map_load_timeout) <= 600.0:
            raise ValueError("map_load_timeout must be in [5, 600]")
        if not 1.0 <= float(map_reconnect_seconds) <= 300.0:
            raise ValueError("map_reconnect_seconds must be in [1, 300]")
        if not 2.0 <= float(lease_seconds) <= 3600.0:
            raise ValueError("lease_seconds must be in [2, 3600]")
        if not 0.1 <= float(control_timeout) <= 10.0:
            raise ValueError("control_timeout must be in [0.1, 10]")
        if not 0.01 <= float(monitor_period) <= 1.0:
            raise ValueError("monitor_period must be in [0.01, 1]")

        self.carla_host = str(carla_host)
        self.carla_port = int(carla_port)
        self.traffic_manager_port = int(traffic_manager_port)
        self.timeout = float(timeout)
        self.map_load_timeout = float(map_load_timeout)
        self.map_reconnect_seconds = float(map_reconnect_seconds)
        self.lease_seconds = float(lease_seconds)
        self.control_timeout = float(control_timeout)
        self.expected_carla_version = str(expected_carla_version)
        self._carla_loader = carla_loader
        self._route_planner_loader = route_planner_loader
        self._clock = clock
        self._monitor_period = float(monitor_period)
        self._map_process_runner = map_process_runner
        self._lock = threading.RLock()
        self._client: Any | None = None
        self._carla: Any | None = None
        self._route_planner_factory: Callable[[Any], Any] | None = None
        self._route_planner_checked = False
        self._scene: SceneLease | None = None
        self._last_scene: dict[str, Any] | None = None
        self._closed = False
        self._monitor_stop = threading.Event()
        self._monitor: threading.Thread | None = None
        if start_monitor:
            self._monitor = threading.Thread(
                target=self._monitor_loop,
                name="carla-world-worker-safety",
                daemon=True,
            )
            self._monitor.start()

    def _ensure_client(self) -> tuple[Any, Any, Any]:
        try:
            if self._client is None:
                self._carla = self._carla_loader()
                self._client = self._carla.Client(self.carla_host, self.carla_port)
                self._client.set_timeout(self.timeout)
            world = self._client.get_world()
            client_version = str(self._client.get_client_version())
            server_version = str(self._client.get_server_version())
        except Exception as error:
            # A CARLA episode transition can invalidate an existing Client.
            # Never retain a transport that already failed; the next request
            # gets a fresh version-checked connection.
            self._client = None
            self._carla = None
            raise WorkerError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "carla_unavailable",
                f"CARLA connection failed: {type(error).__name__}: {error}",
            ) from error
        if self.expected_carla_version and (
            client_version != self.expected_carla_version
            or server_version != self.expected_carla_version
        ):
            raise WorkerError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "carla_version_mismatch",
                "CARLA client and server must both match "
                f"{self.expected_carla_version}; client={client_version}, server={server_version}",
            )
        return self._client, world, self._carla

    def _invalidate_client(self) -> None:
        self._client = None
        self._carla = None

    @staticmethod
    def _world_matches_target(world: Any, target: str) -> bool:
        actual = str(world.get_map().name)
        return target in {actual, _map_short_name(actual)} or _map_short_name(target) == (
            _map_short_name(actual)
        )

    def _run_isolated_map_load(self, target: str) -> str:
        """Run the crash-prone native ``load_world`` call outside this server.

        On Windows a native PythonAPI failure can terminate the calling Python
        process. The authenticated listener therefore never invokes
        ``Client.load_world`` itself. A short-lived child may fail or time out;
        the parent then reconciles against CARLA's authoritative current map.
        """

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--internal-map-load",
            "--carla-host",
            self.carla_host,
            "--carla-port",
            str(self.carla_port),
            "--timeout",
            str(self.map_load_timeout),
            "--expected-carla-version",
            self.expected_carla_version,
            "--target-map",
            target,
        ]
        environment = dict(os.environ)
        # The child is not an HTTP service and never needs the bearer secret.
        environment.pop(TOKEN_ENVIRONMENT_VARIABLE, None)
        try:
            completed = self._map_process_runner(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.map_load_timeout + 10.0,
                check=False,
                env=environment,
            )
        except subprocess.TimeoutExpired:
            return "isolated map loader exceeded its bounded timeout"
        except Exception as error:
            return f"isolated map loader could not start: {type(error).__name__}: {error}"

        returncode = int(getattr(completed, "returncode", 1))
        if returncode == 0:
            return "isolated map loader completed"
        stderr = str(getattr(completed, "stderr", "")).strip().replace("\r", " ").replace("\n", " ")
        if len(stderr) > 800:
            stderr = stderr[-800:]
        detail = f"isolated map loader exited with code {returncode}"
        return detail if not stderr else f"{detail}: {stderr}"

    def _reconnect_after_map_load(self, target: str, diagnostic: str) -> tuple[Any, Any]:
        deadline = time.monotonic() + self.map_reconnect_seconds
        last_error = diagnostic
        while True:
            self._invalidate_client()
            try:
                client, world, _ = self._ensure_client()
                if self._world_matches_target(world, target):
                    return client, world
                last_error = (
                    f"CARLA reports {_map_short_name(str(world.get_map().name))!r}, "
                    f"expected {_map_short_name(target)!r}"
                )
            except WorkerError as error:
                last_error = str(error)
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(0.5, remaining))
        raise WorkerError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "map_load_failed",
            f"CARLA did not stabilize on map {_map_short_name(target)!r}: {last_error}",
        )

    def _load_world_isolated(self, target: str) -> tuple[Any, Any]:
        diagnostic = self._run_isolated_map_load(target)
        # Discard the connection created before the episode transition even if
        # the child reported failure. A timed-out/crashed child may still have
        # successfully initiated the authoritative CARLA map reload.
        self._invalidate_client()
        return self._reconnect_after_map_load(target, diagnostic)

    def _planner_factory(self) -> Callable[[Any], Any] | None:
        if not self._route_planner_checked:
            try:
                self._route_planner_factory = self._route_planner_loader()
            except Exception:
                self._route_planner_factory = None
            self._route_planner_checked = True
        return self._route_planner_factory

    def _capabilities(self, client: Any | None = None) -> dict[str, bool]:
        # Do not instantiate/cache a process-local TrafficManager merely to
        # report capabilities. The actual post-reload instance is validated
        # in ``prepare`` before a random route is planned.
        random_route = client is not None and self._planner_factory() is not None
        in_memory_encoder = _camera_encoder_modules_present()
        return {
            "map_reload": True,
            "weather": True,
            "ego_vehicle": True,
            "traffic_manager": True,
            "walkers": True,
            "scene_props": True,
            "manual_control": True,
            "autopilot": True,
            "random_route": random_route,
            "lease": True,
            "manual_deadman": True,
            "asynchronous_world": True,
            # High-resolution remote video must never fall back silently to
            # CARLA's raw BGRA stream.  Advertise the compressed lanes only
            # when this exact Worker interpreter can load its encoder.
            "compressed_camera_relay": in_memory_encoder,
            "persistent_mjpeg_camera_relay": in_memory_encoder,
            "prepared_scene_handoff": in_memory_encoder,
            "prepared_scene_reconfigure": True,
            "waypoint_teacher": True,
            "camera_pause_resume": True,
            "in_memory_jpeg_encoder_available": in_memory_encoder,
            "camera_60_fps": in_memory_encoder,
            "garage_camera_presets": True,
            "garage_vehicle_autoframing": True,
            "world_dynamics_controls": True,
            "exact_scene_population": True,
            "bounded_scene_cleanup": True,
            "batch_scene_cleanup": True,
            "nonblocking_health": True,
            "idempotent_scene_stop": True,
        }

    @staticmethod
    def _carla_facts(client: Any, world: Any, *, host: str, port: int) -> dict[str, Any]:
        return {
            "connected": True,
            "host": host,
            "port": port,
            "client_version": str(client.get_client_version()),
            "server_version": str(client.get_server_version()),
            "current_map": _map_short_name(str(world.get_map().name)),
        }

    def health(self) -> dict[str, Any]:
        """Return authenticated readiness without exposing credentials."""

        # Long scene prepare/cleanup operations intentionally retain the world
        # lifecycle lock. Health must still answer immediately so a healthy,
        # busy Worker is never misreported as disconnected by the Operator.
        if not self._lock.acquire(blocking=False):
            client = self._client
            return {
                "schema_version": SCHEMA_VERSION,
                "worker_api_revision": WORKER_API_REVISION,
                "status": "busy",
                "ready": True,
                "error_code": None,
                "carla": {
                    "connected": client is not None,
                    "host": self.carla_host,
                    "port": self.carla_port,
                    "client_version": None,
                    "server_version": None,
                    "current_map": None,
                },
                "active_scene": self._scene_summary(self._scene),
                "capabilities": self._capabilities(client),
            }
        try:
            try:
                client, world, _ = self._ensure_client()
                facts = self._carla_facts(
                    client,
                    world,
                    host=self.carla_host,
                    port=self.carla_port,
                )
                capabilities = self._capabilities(client)
                status = "ready"
                ready = True
                error_code = None
            except WorkerError as error:
                facts = {
                    "connected": False,
                    "host": self.carla_host,
                    "port": self.carla_port,
                    "client_version": None,
                    "server_version": None,
                    "current_map": None,
                }
                capabilities = self._capabilities(None)
                capabilities["random_route"] = False
                status = "unavailable"
                ready = False
                error_code = error.code
            return {
                "schema_version": SCHEMA_VERSION,
                "worker_api_revision": WORKER_API_REVISION,
                "status": status,
                "ready": ready,
                "error_code": error_code,
                "carla": facts,
                "active_scene": self._scene_summary(self._scene),
                "capabilities": capabilities,
            }
        finally:
            self._lock.release()

    def catalog(self) -> dict[str, Any]:
        with self._lock:
            client, world, _ = self._ensure_client()
            try:
                available_maps = list(client.get_available_maps())
                library = world.get_blueprint_library()
                definitions = list(library.filter("vehicle.*"))
            except Exception as error:
                raise WorkerError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "catalog_unavailable",
                    f"CARLA catalog query failed: {type(error).__name__}: {error}",
                ) from error

            map_ids = {_map_short_name(str(value)) for value in available_maps}
            map_ids.add(_map_short_name(str(world.get_map().name)))
            vehicles: list[dict[str, Any]] = []
            for blueprint in sorted(definitions, key=lambda value: str(value.id)):
                identifier = str(blueprint.id)
                colors: list[str] = []
                try:
                    if blueprint.has_attribute("color"):
                        colors = [
                            str(value)
                            for value in blueprint.get_attribute("color").recommended_values
                        ]
                except Exception:
                    colors = []
                vehicles.append(
                    {
                        "id": identifier,
                        "label": _label(identifier, "vehicle."),
                        "colors": colors,
                    }
                )
            return {
                "schema_version": SCHEMA_VERSION,
                "worker_api_revision": WORKER_API_REVISION,
                "status": "ok",
                "carla": self._carla_facts(
                    client,
                    world,
                    host=self.carla_host,
                    port=self.carla_port,
                ),
                "maps": [
                    {"id": identifier, "label": _label(identifier)}
                    for identifier in sorted(map_ids)
                ],
                "vehicles": vehicles,
                "weather_presets": [
                    {"id": "keep", "label": "Keep Current Weather"},
                    *[
                        {"id": identifier, "label": _label(identifier)}
                        for identifier in WEATHER_PRESETS
                    ],
                ],
                "prop_presets": [
                    {"id": identifier, "label": _label(identifier)} for identifier in PROP_PRESETS
                ],
                "route_modes": [
                    {"id": "free", "label": "Free Drive"},
                    {"id": "random_destination", "label": "Random Destination"},
                ],
                "control_modes": [
                    {"id": "manual", "label": "Manual"},
                    {"id": "autopilot", "label": "Autopilot"},
                ],
                "capabilities": self._capabilities(client),
            }

    def current_scene(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": SCHEMA_VERSION,
                "worker_api_revision": WORKER_API_REVISION,
                "status": "idle" if self._scene is None else self._scene.status,
                "scene": None if self._scene is None else self._scene_snapshot(self._scene),
            }

    def _available_map_target(self, client: Any, world: Any, requested: str) -> str | None:
        if requested == "current":
            return None
        current_name = str(world.get_map().name)
        if requested in {current_name, _map_short_name(current_name)}:
            return None
        available = list(client.get_available_maps())
        matches = [
            str(value)
            for value in available
            if requested in {str(value), _map_short_name(str(value))}
        ]
        if len(matches) != 1:
            raise WorkerError(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "map_unavailable",
                f"map_name {requested!r} is not one exact catalog map",
            )
        return matches[0]

    @staticmethod
    def _episode_marker(world: Any) -> tuple[Any, ...]:
        map_name = _map_short_name(str(world.get_map().name))
        world_identifier = getattr(world, "id", None)
        if world_identifier is not None:
            return ("world", str(world_identifier), map_name)
        try:
            return ("spectator", int(world.get_spectator().id), map_name)
        except Exception:
            return ("object", id(world), map_name)

    @staticmethod
    def _episode_id(world: Any) -> int:
        value = getattr(world, "id", None)
        if isinstance(value, bool):
            value = None
        try:
            result = int(value)
        except (TypeError, ValueError) as error:
            raise WorkerError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "episode_id_unavailable",
                "CARLA world does not expose an authoritative integer episode id",
            ) from error
        if result < 0:
            raise WorkerError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "episode_id_unavailable",
                "CARLA world episode id must be non-negative",
            )
        return result

    @staticmethod
    def _ensure_async_world(world: Any) -> None:
        try:
            synchronous = bool(world.get_settings().synchronous_mode)
        except Exception as error:
            raise WorkerError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "world_settings_unavailable",
                f"could not inspect CARLA world settings: {error}",
            ) from error
        if synchronous:
            raise WorkerError(
                HTTPStatus.CONFLICT,
                "synchronous_world_owned_elsewhere",
                "World Worker requires an asynchronous CARLA world and never calls world.tick()",
            )

    def _weather_object(self, preset: str) -> Any:
        assert self._carla is not None
        weather = self._carla.WeatherParameters()
        for name, value in WEATHER_PRESETS[preset].items():
            if name == "light":
                continue
            if not hasattr(weather, name):
                raise WorkerError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "weather_contract_mismatch",
                    f"CARLA WeatherParameters lacks {name!r}",
                )
            setattr(weather, name, float(value))
        return weather

    def _apply_weather(self, world: Any, preset: str) -> None:
        if preset == "keep":
            return
        try:
            world.set_weather(self._weather_object(preset))
        except WorkerError:
            raise
        except Exception as error:
            raise WorkerError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "weather_failed",
                f"CARLA weather update failed: {type(error).__name__}: {error}",
            ) from error

    @staticmethod
    def _blueprint(library: Any, identifier: str) -> Any:
        try:
            blueprint = library.find(identifier)
        except Exception as error:
            raise WorkerError(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "blueprint_unavailable",
                f"blueprint {identifier!r} is unavailable",
            ) from error
        if blueprint is None:
            raise WorkerError(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "blueprint_unavailable",
                f"blueprint {identifier!r} is unavailable",
            )
        return blueprint

    @staticmethod
    def _set_role(blueprint: Any, role_name: str) -> str | None:
        try:
            if blueprint.has_attribute("role_name"):
                blueprint.set_attribute("role_name", role_name)
                return role_name
        except Exception:
            pass
        return None

    @staticmethod
    def _set_random_attribute(
        blueprint: Any,
        attribute_name: str,
        rng: random.Random,
    ) -> None:
        try:
            if not blueprint.has_attribute(attribute_name):
                return
            choices = list(blueprint.get_attribute(attribute_name).recommended_values)
            if choices:
                blueprint.set_attribute(attribute_name, str(rng.choice(choices)))
        except Exception:
            return

    @staticmethod
    def _set_color(blueprint: Any, color: str | None) -> None:
        if color is None:
            return
        try:
            if not blueprint.has_attribute("color"):
                raise ValueError("color attribute is absent")
            attribute = blueprint.get_attribute("color")
            recommended = {str(value) for value in attribute.recommended_values}
            if color not in recommended:
                raise ValueError("color is not recommended")
            blueprint.set_attribute("color", color)
        except Exception as error:
            raise WorkerError(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "color_unavailable",
                f"selected vehicle does not support catalog color {color!r}",
            ) from error

    @staticmethod
    def _record_actor(
        owned: list[OwnedActor],
        actor: Any,
        *,
        kind: str,
        role_name: str | None = None,
    ) -> None:
        owned.append(
            OwnedActor(
                actor=actor,
                actor_id=int(actor.id),
                type_id=str(actor.type_id),
                kind=kind,
                role_name=role_name,
            )
        )

    @staticmethod
    def _actor_is_confirmed_absent(world: Any, actor: Any) -> bool:
        if not bool(getattr(actor, "is_alive", True)):
            return True
        try:
            current = world.get_actor(int(actor.id))
        except Exception:
            return False
        return current is None or not bool(getattr(current, "is_alive", True))

    @classmethod
    def _discard_actor_group(
        cls,
        world: Any,
        owned: list[OwnedActor],
        *actors: Any,
    ) -> None:
        """Best-effort rollback without forgetting actors CARLA still owns."""

        destroyed_actor_ids: set[int] = set()
        for actor in actors:
            if actor is None:
                continue
            if cls._actor_is_confirmed_absent(world, actor):
                destroyed_actor_ids.add(int(actor.id))
                continue
            try:
                if str(getattr(actor, "type_id", "")).startswith("controller.ai.walker"):
                    actor.stop()
            except Exception:
                pass
            try:
                destroyed = actor.destroy()
            except Exception:
                destroyed = False
            if destroyed is True or cls._actor_is_confirmed_absent(world, actor):
                destroyed_actor_ids.add(int(actor.id))
        owned[:] = [item for item in owned if item.actor_id not in destroyed_actor_ids]

    @staticmethod
    def _registered_actor_ids(world: Any, actors: Sequence[Any]) -> set[int]:
        live_candidates = {
            int(actor.id) for actor in actors if bool(getattr(actor, "is_alive", True))
        }
        if not live_candidates:
            return set()
        if hasattr(world, "get_actors"):
            try:
                return {
                    int(actor.id)
                    for actor in world.get_actors(list(live_candidates))
                    if bool(getattr(actor, "is_alive", True))
                }
            except Exception:
                return set()
        registered: set[int] = set()
        for actor_id in live_candidates:
            try:
                current = world.get_actor(actor_id)
                if current is not None and bool(getattr(current, "is_alive", True)):
                    registered.add(actor_id)
            except Exception:
                continue
        return registered

    def _rollback_population(self, world: Any, owned: list[OwnedActor], *actors: Any) -> None:
        self._discard_actor_group(world, owned, *actors)
        rejected_ids = {int(actor.id) for actor in actors if actor is not None}
        if any(item.actor_id in rejected_ids for item in owned):
            raise RuntimeError(
                "CARLA population rollback was not confirmed; refusing replacement actors"
            )

    def _spawn_ego(
        self,
        world: Any,
        config: SceneConfig,
        scene_id: str,
        spawn_points: list[Any],
        rng: random.Random,
        owned: list[OwnedActor],
    ) -> tuple[Any, int]:
        library = world.get_blueprint_library()
        blueprint = self._blueprint(library, config.vehicle_blueprint)
        role_name = self._set_role(blueprint, f"world_worker_{scene_id}")
        self._set_color(blueprint, config.color)
        indices = list(range(len(spawn_points)))
        rng.shuffle(indices)
        for index in indices[: min(80, len(indices))]:
            try:
                actor = world.try_spawn_actor(blueprint, spawn_points[index])
            except Exception:
                actor = None
            if actor is None:
                continue
            self._record_actor(owned, actor, kind="ego", role_name=role_name)
            self._apply_full_brake(actor)
            return actor, index
        raise WorkerError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "ego_spawn_failed",
            "could not spawn the ego vehicle at any official map spawn point",
        )

    @staticmethod
    def _safe_vehicle_blueprints(library: Any) -> list[Any]:
        candidates = sorted(library.filter("vehicle.*"), key=lambda item: str(item.id))
        safe: list[Any] = []
        for blueprint in candidates:
            try:
                if blueprint.has_attribute("base_type"):
                    if str(blueprint.get_attribute("base_type")) == "car":
                        safe.append(blueprint)
                elif not any(
                    token in str(blueprint.id) for token in ("bike", "motorcycle", "microlino")
                ):
                    safe.append(blueprint)
            except Exception:
                safe.append(blueprint)
        return safe or candidates

    def _supports_spawn_batches(self) -> bool:
        commands = getattr(self._carla, "command", None)
        return callable(getattr(commands, "SpawnActor", None)) and callable(
            getattr(self._client, "apply_batch_sync", None)
        )

    def _spawn_request(
        self,
        blueprint: Any,
        transform: Any,
        *,
        kind: str,
        role_name: str | None = None,
        parent: Any = None,
    ) -> ActorSpawnRequest:
        command = None
        if self._supports_spawn_batches():
            spawn = self._carla.command.SpawnActor
            command = (
                spawn(blueprint, transform)
                if parent is None
                else spawn(blueprint, transform, int(parent.id))
            )
        return ActorSpawnRequest(blueprint, transform, kind, role_name, parent, command)

    def _wait_population_snapshot(self, world: Any) -> None:
        """Observe a natural tick; this live worker must never tick the server."""

        wait = getattr(world, "wait_for_tick", None)
        if callable(wait):
            try:
                wait(seconds=min(self.timeout, 5.0))
            except TypeError:
                wait(min(self.timeout, 5.0))

    def _recover_uncertain_spawns(
        self, world: Any, requests: Sequence[ActorSpawnRequest], owned: list[OwnedActor]
    ) -> None:
        """Retain identifiable actors for rollback, never replay a timed-out batch."""

        try:
            self._wait_population_snapshot(world)
            actors = list(world.get_actors())
        except Exception:
            return
        known = {item.actor_id for item in owned}
        for actor in actors:
            if int(actor.id) in known:
                continue
            for request in requests:
                if str(actor.type_id) != str(request.blueprint.id):
                    continue
                if request.parent is not None:
                    parent = getattr(actor, "parent", None)
                    matches = parent is not None and int(parent.id) == int(request.parent.id)
                else:
                    matches = request.role_name is not None and (
                        self._actor_attribute(actor, "role_name") == request.role_name
                    )
                if matches:
                    self._record_actor(owned, actor, kind=request.kind, role_name=request.role_name)
                    known.add(int(actor.id))
                    break

    def _spawn_owned_batch(
        self, world: Any, requests: Sequence[ActorSpawnRequest], owned: list[OwnedActor]
    ) -> list[Any | None]:
        """Return position-matched actors, retaining every successful ID for cleanup."""

        if not requests:
            return []
        if requests[0].command is None:
            # Compatibility clients have no command API. Callers prepare one
            # request at a time so blueprint mutations cannot leak across spawns.
            assert len(requests) == 1
            request = requests[0]
            try:
                if request.parent is None:
                    actor = world.try_spawn_actor(request.blueprint, request.transform)
                else:
                    actor = world.try_spawn_actor(
                        request.blueprint, request.transform, attach_to=request.parent
                    )
            except Exception:
                self._recover_uncertain_spawns(world, [request], owned)
                raise
            if actor is not None:
                self._record_actor(owned, actor, kind=request.kind, role_name=request.role_name)
            return [actor]

        try:
            # Spawn only: a chained autopilot error must not hide a created ID.
            responses = list(
                self._client.apply_batch_sync([item.command for item in requests], False)
            )
        except Exception:
            self._recover_uncertain_spawns(world, requests, owned)
            raise

        spawned: dict[int, OwnedActor] = {}
        malformed = len(responses) != len(requests)
        for index, (request, response) in enumerate(zip(requests, responses, strict=False)):
            if str(getattr(response, "error", "") or "").strip():
                continue
            try:
                actor_id = int(response.actor_id)
                if actor_id <= 0 or any(item.actor_id == actor_id for item in owned):
                    raise ValueError("invalid or repeated actor ID")
            except (AttributeError, TypeError, ValueError):
                malformed = True
                continue
            item = OwnedActor(
                actor=None,
                actor_id=actor_id,
                type_id=str(request.blueprint.id),
                kind=request.kind,
                role_name=request.role_name,
            )
            owned.append(item)
            spawned[index] = item
        if malformed:
            self._recover_uncertain_spawns(world, requests, owned)
            raise RuntimeError("CARLA spawn batch returned an incomplete or invalid response")
        if not spawned:
            return [None] * len(requests)

        # get_actors reads the episode snapshot, which may predate the command.
        # Register IDs above *before* waiting, so a failed barrier can roll back.
        self._wait_population_snapshot(world)
        actors, errors = self._bulk_current_actors(
            world, [item.actor_id for item in spawned.values()]
        )
        if errors:
            raise RuntimeError("CARLA spawn snapshot lookup failed")
        result: list[Any | None] = [None] * len(requests)
        for index, item in spawned.items():
            actor = actors.get(item.actor_id)
            if actor is not None:
                self._validate_owned_actor_identity(actor, item)
                item.actor = actor
                result[index] = actor
        if any(actor is None for index, actor in enumerate(result) if index in spawned):
            # A successful command with no observed actor is uncertain, not a
            # failed spawn. Abort/rollback instead of creating duplicate NPCs.
            raise RuntimeError("CARLA spawned actors were not visible after the snapshot barrier")
        return result

    def _spawn_traffic(
        self,
        scene_id: str,
        world: Any,
        traffic_manager: Any,
        spawn_points: list[Any],
        ego_spawn_index: int,
        count: int,
        rng: random.Random,
        owned: list[OwnedActor],
    ) -> list[Any]:
        if count == 0:
            return []
        blueprints = self._safe_vehicle_blueprints(world.get_blueprint_library())
        if not blueprints:
            raise WorkerError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "traffic_blueprints_unavailable",
                "no safe NPC vehicle blueprints are available",
            )
        indices = [index for index in range(len(spawn_points)) if index != ego_spawn_index]
        rng.shuffle(indices)
        actors: list[Any] = []
        batch_size = 32 if self._supports_spawn_batches() else 1
        offset = 0
        while offset < len(indices) and len(actors) < count:
            selected = indices[offset : offset + min(batch_size, count - len(actors))]
            offset += len(selected)
            requests: list[ActorSpawnRequest] = []
            for index in selected:
                blueprint = rng.choice(blueprints)
                role_name = self._set_role(blueprint, f"world_worker_npc_{scene_id}")
                self._set_random_attribute(blueprint, "color", rng)
                self._set_random_attribute(blueprint, "driver_id", rng)
                requests.append(
                    self._spawn_request(
                        blueprint, spawn_points[index], kind="traffic", role_name=role_name
                    )
                )
            for actor in self._spawn_owned_batch(world, requests, owned):
                if actor is None:
                    continue
                try:
                    actor.set_autopilot(True, int(traffic_manager.get_port()))
                    if hasattr(traffic_manager, "update_vehicle_lights"):
                        traffic_manager.update_vehicle_lights(actor, True)
                except Exception:
                    self._rollback_population(world, owned, actor)
                    continue
                actors.append(actor)
        return actors

    @staticmethod
    def _walker_speed(blueprint: Any, running: bool) -> float:
        try:
            values = list(blueprint.get_attribute("speed").recommended_values)
        except Exception:
            return 2.8 if running else 1.4
        if not values:
            return 2.8 if running else 1.4
        index = 2 if running and len(values) > 2 else min(1, len(values) - 1)
        try:
            return float(values[index])
        except (TypeError, ValueError):
            return 2.8 if running else 1.4

    def _spawn_walkers(
        self,
        scene_id: str,
        world: Any,
        count: int,
        rng: random.Random,
        owned: list[OwnedActor],
    ) -> tuple[list[Any], list[Any]]:
        if count == 0:
            return [], []
        assert self._carla is not None
        library = world.get_blueprint_library()
        blueprints = sorted(library.filter("walker.pedestrian.*"), key=lambda item: str(item.id))
        controller_blueprint = self._blueprint(library, "controller.ai.walker")
        if not blueprints:
            raise WorkerError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "walker_blueprints_unavailable",
                "no walker blueprints are available",
            )
        walkers: list[Any] = []
        controllers: list[Any] = []

        # Dense walker creation can invalidate a newly attached parent before
        # WalkerAIController.start() toggles its physics state. Retry failed
        # pairs independently instead of aborting the whole scene with CARLA's
        # opaque "actor not found in registry" error.
        activation_batch_size = 24
        batch_spawning = self._supports_spawn_batches()
        maximum_rounds = max(3, math.ceil(count / activation_batch_size) * 3)
        for _round in range(maximum_rounds):
            needed = count - len(walkers)
            if needed <= 0:
                break
            batch_target = min(activation_batch_size, needed)
            pending: list[tuple[Any, Any, float]] = []
            requests: list[ActorSpawnRequest] = []
            speeds: list[float] = []
            for _ in range(batch_target * 3):
                if len(pending) + len(requests) >= batch_target:
                    break
                location = world.get_random_location_from_navigation()
                if location is None:
                    continue
                blueprint = rng.choice(blueprints)
                try:
                    if blueprint.has_attribute("is_invincible"):
                        blueprint.set_attribute("is_invincible", "false")
                except Exception:
                    pass
                role_name = self._set_role(blueprint, f"world_worker_walker_{scene_id}")
                transform = self._carla.Transform(location)
                request = self._spawn_request(
                    blueprint, transform, kind="walker", role_name=role_name
                )
                if batch_spawning:
                    requests.append(request)
                    speeds.append(self._walker_speed(blueprint, rng.random() < 0.05))
                    continue
                walker = self._spawn_owned_batch(world, [request], owned)[0]
                if walker is None:
                    continue
                controller_request = self._spawn_request(
                    controller_blueprint,
                    self._carla.Transform(),
                    kind="walker_controller",
                    parent=walker,
                )
                controller = self._spawn_owned_batch(world, [controller_request], owned)[0]
                if controller is None:
                    self._rollback_population(world, owned, walker)
                    continue
                pending.append(
                    (walker, controller, self._walker_speed(blueprint, rng.random() < 0.05))
                )

            if batch_spawning:
                spawned = self._spawn_owned_batch(world, requests, owned)
                pairs = [
                    (walker, speed)
                    for walker, speed in zip(spawned, speeds, strict=True)
                    if walker is not None
                ]
                controller_requests = [
                    self._spawn_request(
                        controller_blueprint,
                        self._carla.Transform(),
                        kind="walker_controller",
                        parent=walker,
                    )
                    for walker, _speed in pairs
                ]
                attached = self._spawn_owned_batch(world, controller_requests, owned)
                for (walker, speed), controller in zip(pairs, attached, strict=True):
                    if controller is None:
                        self._rollback_population(world, owned, walker)
                    else:
                        pending.append((walker, controller, speed))

            barrier_ready = True
            if pending and not batch_spawning and hasattr(world, "wait_for_tick"):
                try:
                    world.wait_for_tick(seconds=min(self.timeout, 5.0))
                except TypeError:
                    try:
                        world.wait_for_tick(min(self.timeout, 5.0))
                    except Exception:
                        barrier_ready = False
                except Exception:
                    barrier_ready = False
            if not barrier_ready:
                for walker, controller, _speed in pending:
                    self._rollback_population(world, owned, controller, walker)
                continue
            for walker, controller, speed in pending:
                try:
                    controller.start()
                    destination = None
                    for _ in range(3):
                        destination = world.get_random_location_from_navigation()
                        if destination is not None:
                            break
                    if destination is None:
                        raise RuntimeError("navigation mesh returned no walker destination")
                    controller.go_to_location(destination)
                    controller.set_max_speed(speed)
                except Exception:
                    self._rollback_population(world, owned, controller, walker)
                    continue
                walkers.append(walker)
                controllers.append(controller)
        return walkers, controllers

    @staticmethod
    def _compose_prop_transform(carla: Any, origin: Any, relative: Mapping[str, Any]) -> Any:
        pitch = math.radians(float(origin.rotation.pitch))
        yaw = math.radians(float(origin.rotation.yaw))
        roll = math.radians(float(origin.rotation.roll))
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        matrix = (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )
        local = (
            float(relative.get("x", 0.0)),
            float(relative.get("y", 0.0)),
            float(relative.get("z", 0.0)),
        )
        translated = [
            origin_value
            + sum(
                coefficient * component for coefficient, component in zip(row, local, strict=True)
            )
            for origin_value, row in zip(
                (origin.location.x, origin.location.y, origin.location.z), matrix, strict=True
            )
        ]
        return carla.Transform(
            carla.Location(x=translated[0], y=translated[1], z=translated[2]),
            carla.Rotation(
                pitch=float(origin.rotation.pitch) + float(relative.get("pitch", 0.0)),
                yaw=float(origin.rotation.yaw) + float(relative.get("yaw", 0.0)),
                roll=float(origin.rotation.roll) + float(relative.get("roll", 0.0)),
            ),
        )

    def _spawn_props(
        self,
        scene_id: str,
        world: Any,
        preset: str,
        ego_transform: Any,
        owned: list[OwnedActor],
    ) -> list[Any]:
        assert self._carla is not None
        library = world.get_blueprint_library()
        actors: list[Any] = []
        for item in PROP_PRESETS[preset]:
            identifier = str(item["blueprint_id"])
            blueprint = self._blueprint(library, identifier)
            role_name = self._set_role(blueprint, f"world_worker_prop_{scene_id}")
            transform = self._compose_prop_transform(
                self._carla,
                ego_transform,
                item["transform"],
            )
            try:
                actor = self._spawn_owned_batch(
                    world,
                    [self._spawn_request(blueprint, transform, kind="prop", role_name=role_name)],
                    owned,
                )[0]
            except Exception as error:
                raise WorkerError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "prop_spawn_failed",
                    f"could not spawn prop {identifier!r}: {error}",
                ) from error
            if actor is None:
                raise WorkerError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "prop_spawn_failed",
                    f"could not spawn prop {identifier!r}",
                )
            actors.append(actor)
        return actors

    @staticmethod
    def _distance(first: Any, second: Any) -> float:
        if hasattr(first, "distance"):
            return float(first.distance(second))
        return math.sqrt(
            (float(first.x) - float(second.x)) ** 2
            + (float(first.y) - float(second.y)) ** 2
            + (float(first.z) - float(second.z)) ** 2
        )

    def _plan_random_route(
        self,
        world: Any,
        ego: Any,
        spawn_points: list[Any],
        spawn_index: int,
        rng: random.Random,
    ) -> tuple[dict[str, Any], dict[str, Any], list[Any]]:
        factory = self._planner_factory()
        if factory is None:
            raise WorkerError(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "random_route_unavailable",
                "GlobalRoutePlanner is unavailable on the World Worker host",
            )
        candidates = [index for index in range(len(spawn_points)) if index != spawn_index]
        origin = ego.get_location()
        candidates.sort(
            key=lambda index: (self._distance(origin, spawn_points[index].location), index),
            reverse=True,
        )
        farthest = candidates[: max(1, min(16, len(candidates)))]
        rng.shuffle(farthest)
        planner = factory(world.get_map())
        errors: list[str] = []
        for destination_index in farthest:
            destination_transform = spawn_points[destination_index]
            try:
                traced = list(planner.trace_route(origin, destination_transform.location))
                locations = [item[0].transform.location for item in traced]
            except Exception as error:
                errors.append(f"destination {destination_index}: {error}")
                continue
            if len(locations) < 2:
                errors.append(f"destination {destination_index}: route is trivial")
                continue
            destination = {
                "spawn_index": destination_index,
                "transform": _json_transform(destination_transform),
            }
            route = {
                "mode": "random_destination",
                "provider": "GlobalRoutePlanner.trace_route+TrafficManager.set_path",
                "planned": True,
                "enforced": False,
                "waypoint_count": len(locations),
            }
            return route, destination, locations
        detail = "; ".join(errors[-3:])
        raise WorkerError(
            HTTPStatus.UNPROCESSABLE_ENTITY,
            "random_route_failed",
            f"could not trace a non-trivial route to a random destination: {detail}",
        )

    def prepare(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        config = SceneConfig.from_mapping(raw)
        with self._lock:
            if self._closed:
                raise WorkerError(
                    HTTPStatus.SERVICE_UNAVAILABLE, "worker_closed", "worker is closed"
                )
            if self._scene is not None and self._scene.status in _ACTIVE_SCENE_STATES:
                raise WorkerError(
                    HTTPStatus.CONFLICT,
                    "scene_active",
                    "another leased scene is already active",
                )
            client, current_world, _ = self._ensure_client()
            target = self._available_map_target(client, current_world, config.map_name)
            if target is None:
                world = current_world
            else:
                client, world = self._load_world_isolated(target)
            self._ensure_async_world(world)
            traffic_manager: Any | None = None
            try:
                traffic_manager = client.get_trafficmanager(self.traffic_manager_port)
                if config.route_mode == "random_destination" and (
                    self._planner_factory() is None or not hasattr(traffic_manager, "set_path")
                ):
                    raise WorkerError(
                        HTTPStatus.UNPROCESSABLE_ENTITY,
                        "random_route_unavailable",
                        "random_destination requires GlobalRoutePlanner and "
                        "TrafficManager.set_path",
                    )
                traffic_manager.set_synchronous_mode(False)
                simulator_seed = config.seed % _SIMULATOR_SEED_MODULUS
                if hasattr(traffic_manager, "set_random_device_seed"):
                    traffic_manager.set_random_device_seed(simulator_seed)
                if hasattr(traffic_manager, "set_global_distance_to_leading_vehicle"):
                    traffic_manager.set_global_distance_to_leading_vehicle(
                        config.following_distance_metres
                    )
                if hasattr(traffic_manager, "global_percentage_speed_difference"):
                    traffic_manager.global_percentage_speed_difference(
                        config.speed_difference_percent
                    )
                if hasattr(world, "set_pedestrians_seed"):
                    world.set_pedestrians_seed((simulator_seed + 1) % _SIMULATOR_SEED_MODULUS)
                if hasattr(world, "set_pedestrians_cross_factor"):
                    world.set_pedestrians_cross_factor(config.pedestrian_crossing_factor)
                original_weather = world.get_weather()
            except WorkerError:
                self._release_traffic_manager(traffic_manager)
                raise
            except Exception as error:
                self._release_traffic_manager(traffic_manager)
                raise WorkerError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "world_prepare_failed",
                    f"CARLA world preparation failed: {type(error).__name__}: {error}",
                ) from error

            scene_id = secrets.token_urlsafe(18).replace("-", "_")
            lease_token = secrets.token_urlsafe(32)
            role_rng = random.Random(config.seed)
            owned: list[OwnedActor] = []
            partial: SceneLease | None = None
            try:
                self._apply_weather(world, config.weather_preset)
                spawn_points = list(world.get_map().get_spawn_points())
                if not spawn_points:
                    raise WorkerError(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        "spawn_points_unavailable",
                        "active CARLA map exposes no vehicle spawn points",
                    )
                maximum_traffic = max(0, len(spawn_points) - 1)
                if config.traffic_count > maximum_traffic:
                    raise WorkerError(
                        HTTPStatus.UNPROCESSABLE_ENTITY,
                        "scene_population_capacity",
                        "requested traffic_count "
                        f"{config.traffic_count} exceeds this map's maximum of "
                        f"{maximum_traffic} after reserving one ego spawn point",
                    )
                ego, spawn_index = self._spawn_ego(
                    world,
                    config,
                    scene_id,
                    spawn_points,
                    role_rng,
                    owned,
                )
                partial = SceneLease(
                    scene_id=scene_id,
                    lease_token=lease_token,
                    config=config,
                    client=client,
                    world=world,
                    traffic_manager=traffic_manager,
                    episode_id=self._episode_id(world),
                    episode_marker=self._episode_marker(world),
                    map_name=_map_short_name(str(world.get_map().name)),
                    original_weather=original_weather,
                    ego=ego,
                    spawn_index=spawn_index,
                    owned_actors=owned,
                    control_mode=config.initial_control_mode,
                    weather_preset=config.weather_preset,
                    lease_deadline=self._clock() + self.lease_seconds,
                )
                # Reserve the deterministic ego-relative fixed scene before
                # filling the map with stochastic traffic and walkers.
                partial.prop_actors = self._spawn_props(
                    scene_id,
                    world,
                    config.prop_preset,
                    ego.get_transform(),
                    owned,
                )
                partial.vehicle_actors = self._spawn_traffic(
                    scene_id,
                    world,
                    traffic_manager,
                    spawn_points,
                    spawn_index,
                    config.traffic_count,
                    role_rng,
                    owned,
                )
                partial.walker_actors, partial.walker_controllers = self._spawn_walkers(
                    scene_id,
                    world,
                    config.walker_count,
                    role_rng,
                    owned,
                )
                registered_actor_ids = self._registered_actor_ids(
                    world,
                    [
                        *partial.vehicle_actors,
                        *partial.walker_actors,
                        *partial.walker_controllers,
                    ],
                )
                verified_vehicles: list[Any] = []
                for actor in partial.vehicle_actors:
                    if int(actor.id) in registered_actor_ids:
                        verified_vehicles.append(actor)
                    else:
                        self._discard_actor_group(world, owned, actor)
                partial.vehicle_actors = verified_vehicles
                verified_walkers: list[Any] = []
                verified_controllers: list[Any] = []
                for walker, controller in zip(
                    partial.walker_actors,
                    partial.walker_controllers,
                    strict=True,
                ):
                    if (
                        int(walker.id) in registered_actor_ids
                        and int(controller.id) in registered_actor_ids
                    ):
                        verified_walkers.append(walker)
                        verified_controllers.append(controller)
                    else:
                        self._discard_actor_group(world, owned, controller, walker)
                partial.walker_actors = verified_walkers
                partial.walker_controllers = verified_controllers
                actual_traffic = len(partial.vehicle_actors)
                actual_walkers = len(partial.walker_actors)
                if actual_traffic != config.traffic_count or actual_walkers != config.walker_count:
                    raise WorkerError(
                        HTTPStatus.UNPROCESSABLE_ENTITY,
                        "scene_population_shortfall",
                        "CARLA could not create the exact requested population: "
                        f"traffic {actual_traffic}/{config.traffic_count}, "
                        f"walkers {actual_walkers}/{config.walker_count}. "
                        "No partial scene was retained; reduce the counts or change map/seed.",
                    )
                if config.route_mode == "random_destination":
                    partial.route, partial.destination, partial.route_locations = (
                        self._plan_random_route(
                            world,
                            ego,
                            spawn_points,
                            spawn_index,
                            role_rng,
                        )
                    )
                else:
                    partial.route = {
                        "mode": "free",
                        "provider": "TrafficManager"
                        if config.initial_control_mode == "autopilot"
                        else None,
                        "planned": False,
                        "enforced": False,
                        "waypoint_count": 0,
                    }
            except Exception as error:
                if partial is None:
                    partial = SceneLease(
                        scene_id=scene_id,
                        lease_token=lease_token,
                        config=config,
                        client=client,
                        world=world,
                        traffic_manager=traffic_manager,
                        episode_id=self._episode_id(world),
                        episode_marker=self._episode_marker(world),
                        map_name=_map_short_name(str(world.get_map().name)),
                        original_weather=original_weather,
                        ego=owned[0].actor if owned else None,
                        spawn_index=-1,
                        owned_actors=owned,
                        lease_deadline=self._clock(),
                    )
                self._cleanup_resources(partial, reason="prepare_failed")
                if partial.cleanup_errors:
                    raise WorkerError(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        "scene_prepare_failed",
                        f"CARLA scene preparation failed: {type(error).__name__}: {error}. "
                        "Cleanup could not be fully confirmed: "
                        + "; ".join(partial.cleanup_errors),
                    ) from error
                if isinstance(error, WorkerError):
                    raise
                raise WorkerError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "scene_prepare_failed",
                    f"CARLA scene preparation failed: {type(error).__name__}: {error}",
                ) from error

            partial.lease_deadline = self._clock() + self.lease_seconds
            self._scene = partial
            return self._scene_response(partial, "prepared")

    @staticmethod
    def _lease_token(raw: Mapping[str, Any], *, allowed: set[str]) -> str:
        _strict_keys(
            raw,
            allowed=allowed,
            required={"lease_token"},
            name="scene request",
        )
        return _text(raw["lease_token"], "lease_token", maximum=256)

    def _require_scene(self, scene_id: str, lease_token: str) -> SceneLease:
        scene = self._scene
        if scene is None or scene.scene_id != scene_id:
            raise WorkerError(HTTPStatus.NOT_FOUND, "scene_not_found", "active scene was not found")
        if not hmac.compare_digest(scene.lease_token, lease_token):
            raise WorkerError(HTTPStatus.CONFLICT, "lease_mismatch", "scene lease does not match")
        if scene.status not in _ACTIVE_SCENE_STATES:
            raise WorkerError(HTTPStatus.CONFLICT, "scene_inactive", "scene is not active")
        return scene

    def _refresh_lease(self, scene: SceneLease) -> None:
        scene.lease_deadline = self._clock() + self.lease_seconds

    def _remove_scene_actor_subset(self, scene: SceneLease, owned: list[OwnedActor]) -> None:
        """Use the normal verified batch cleanup, without touching other actors."""

        if not owned:
            return
        world = scene.client.get_world()
        if self._episode_marker(world) != scene.episode_marker:
            raise WorkerError(HTTPStatus.CONFLICT, "episode_changed", "CARLA episode changed")
        for item in owned:
            if item.kind == "walker_controller" and item.actor is not None:
                current = world.get_actor(item.actor_id)
                if current is not None:
                    self._validate_owned_actor_identity(current, item)
                    current.stop()
        subset = replace(scene, owned_actors=list(owned), cleanup_errors=[])
        self._destroy_owned_actors(subset, world)
        if subset.cleanup_errors:
            raise WorkerError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "scene_reconfigure_cleanup_failed",
                "Affected actor cleanup could not be confirmed: "
                + "; ".join(subset.cleanup_errors),
            )
        removed = {item.actor_id for item in owned}
        scene.owned_actors[:] = [
            item for item in scene.owned_actors if item.actor_id not in removed
        ]

    def _replace_prepared_ego(self, scene: SceneLease, config: SceneConfig) -> None:
        """Replace only the parked ego, restoring its old blueprint on failure."""

        original = scene.config
        origin = scene.ego.get_transform()
        original_color = self._actor_attribute(scene.ego, "color") or original.color
        old_owned = next(item for item in scene.owned_actors if item.kind == "ego")
        try:
            self._remove_scene_actor_subset(scene, [old_owned])
        except Exception:
            self._cleanup_resources(scene, reason="ego_reconfigure_failed")
            raise
        scene.ego = None

        def spawn(selected: SceneConfig) -> Any:
            blueprint = self._blueprint(
                scene.world.get_blueprint_library(), selected.vehicle_blueprint
            )
            self._set_color(blueprint, selected.color)
            role_name = self._set_role(blueprint, f"world_worker_{scene.scene_id}")
            request = self._spawn_request(blueprint, origin, kind="ego", role_name=role_name)
            actor = self._spawn_owned_batch(scene.world, [request], scene.owned_actors)[0]
            if actor is None:
                raise RuntimeError("CARLA rejected ego spawn at its existing location")
            scene.ego = actor
            self._apply_full_brake(actor)
            return actor

        try:
            spawn(config)
        except Exception as error:
            try:
                self._remove_scene_actor_subset(
                    scene, [item for item in scene.owned_actors if item.kind == "ego"]
                )
                scene.ego = None
                spawn(replace(original, color=original_color))
            except Exception as rollback_error:
                self._cleanup_resources(scene, reason="ego_reconfigure_failed")
                raise WorkerError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "ego_reconfigure_failed",
                    f"Ego replacement and restoration failed; scene stopped: {rollback_error}",
                ) from error
            raise WorkerError(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "ego_reconfigure_failed",
                f"Ego replacement failed; original vehicle restored: {error}",
            ) from error
        scene.config = replace(
            scene.config, vehicle_blueprint=config.vehicle_blueprint, color=config.color
        )

    def configure(self, scene_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        """Apply prepared-scene deltas using the official CARLA population APIs.

        Map/seed changes still need prepare. Existing actors, camera and lease
        survive all other supported changes; an unconfirmed destructive failure
        stops the scene instead of advertising a fictional applied configuration.
        """

        fields = set(SceneConfig().as_dict())
        lease_token = self._lease_token(raw, allowed=fields | {"lease_token"})
        config = SceneConfig.from_mapping(
            {key: value for key, value in raw.items() if key != "lease_token"}
        )
        with self._lock:
            scene = self._require_scene(scene_id, lease_token)
            if scene.status != "prepared":
                raise WorkerError(
                    HTTPStatus.CONFLICT,
                    "scene_not_prepared",
                    "only a prepared scene can be reconfigured",
                )
            if self._clock() >= scene.lease_deadline:
                raise WorkerError(HTTPStatus.CONFLICT, "lease_expired", "scene lease expired")
            world = scene.client.get_world()
            if (
                self._episode_id(world) != scene.episode_id
                or self._episode_marker(world) != scene.episode_marker
            ):
                raise WorkerError(
                    HTTPStatus.CONFLICT,
                    "episode_changed",
                    "CARLA episode changed before scene reconfiguration",
                )
            original = scene.config
            if config.map_name != original.map_name or config.seed != original.seed:
                raise WorkerError(
                    HTTPStatus.CONFLICT,
                    "scene_reconfigure_unsupported",
                    "map or seed changes require a new prepared scene",
                )
            try:
                ego_owned = next(item for item in scene.owned_actors if item.kind == "ego")
                current_ego = world.get_actor(ego_owned.actor_id)
                if current_ego is None or not bool(getattr(current_ego, "is_alive", True)):
                    raise RuntimeError("owned ego is no longer alive")
                if scene.ego is None or int(scene.ego.id) != ego_owned.actor_id:
                    raise RuntimeError("owned ego identity does not match the scene")
                self._validate_owned_actor_identity(current_ego, ego_owned)
                population_ids = [
                    int(actor.id)
                    for actor in [
                        *scene.vehicle_actors,
                        *scene.walker_actors,
                        *scene.walker_controllers,
                        *scene.prop_actors,
                    ]
                ]
                population, errors = self._bulk_current_actors(world, population_ids)
                if errors or set(population_ids) != set(population):
                    raise RuntimeError("owned scene population is no longer fully registered")
                for item in scene.owned_actors:
                    if item.actor_id in population:
                        self._validate_owned_actor_identity(population[item.actor_id], item)
            except Exception as error:
                raise WorkerError(
                    HTTPStatus.CONFLICT,
                    "scene_identity_changed",
                    f"scene ego ownership check failed: {error}",
                ) from error
            change_ego = (config.vehicle_blueprint, config.color) != (
                original.vehicle_blueprint,
                original.color,
            )
            if change_ego:
                if scene.camera_relay is not None and (
                    scene.camera_config is None
                    or scene.camera_config.mode != "garage"
                    or getattr(scene.camera_relay.sensor, "parent", None) is not None
                ):
                    raise WorkerError(
                        HTTPStatus.CONFLICT,
                        "scene_reconfigure_unsupported",
                        "ego replacement with an attached camera needs a new scene",
                    )
                blueprint = self._blueprint(world.get_blueprint_library(), config.vehicle_blueprint)
                self._set_color(blueprint, config.color)
            if config.prop_preset != original.prop_preset:
                for item in PROP_PRESETS[config.prop_preset]:
                    self._blueprint(world.get_blueprint_library(), str(item["blueprint_id"]))
            tm = scene.traffic_manager
            dynamics = (
                ("speed_difference_percent", tm, "global_percentage_speed_difference"),
                ("following_distance_metres", tm, "set_global_distance_to_leading_vehicle"),
                ("pedestrian_crossing_factor", world, "set_pedestrians_cross_factor"),
            )
            for field_name, target, method in dynamics:
                if getattr(config, field_name) != getattr(original, field_name):
                    if not callable(getattr(target, method, None)):
                        raise WorkerError(
                            HTTPStatus.CONFLICT,
                            "scene_reconfigure_unsupported",
                            f"CARLA does not expose {method}",
                        )
            spawn_points = list(world.get_map().get_spawn_points())
            if config.traffic_count > max(0, len(spawn_points) - 1):
                raise WorkerError(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    "scene_population_capacity",
                    "requested traffic exceeds available map spawn points",
                )
            if config.route_mode == "random_destination" and (
                self._planner_factory() is None or not hasattr(tm, "set_path")
            ):
                raise WorkerError(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    "random_route_unavailable",
                    "random_destination requires a route planner and set_path",
                )

            rng = random.Random(config.seed)
            uncertain = False
            try:
                if change_ego:
                    self._replace_prepared_ego(scene, config)
                if config.weather_preset != scene.config.weather_preset:
                    self._apply_weather(world, config.weather_preset)
                    if config.weather_preset != "keep":
                        scene.weather_preset = config.weather_preset
                    scene.config = replace(scene.config, weather_preset=config.weather_preset)
                # generate_traffic.py sets this before controller destinations.
                # Existing walkers also need fresh go_to_location calls to use
                # the new crossing factor in their next navigation route.
                for field_name, target, method in dynamics:
                    if getattr(config, field_name) == getattr(scene.config, field_name):
                        continue
                    previous_value = getattr(scene.config, field_name)
                    try:
                        getattr(target, method)(getattr(config, field_name))
                        if field_name == "pedestrian_crossing_factor":
                            for controller in scene.walker_controllers:
                                destination = world.get_random_location_from_navigation()
                                if destination is None:
                                    raise RuntimeError(
                                        "navigation mesh returned no walker destination"
                                    )
                                controller.go_to_location(destination)
                    except Exception:
                        # A failed destination refresh must be retried on the
                        # next Apply. Restore the old factor before retaining
                        # its confirmed config value.
                        try:
                            getattr(target, method)(previous_value)
                        except Exception:
                            uncertain = True
                        raise
                    scene.config = replace(
                        scene.config, **{field_name: getattr(config, field_name)}
                    )
                for kind, requested, actors in (
                    ("traffic", config.traffic_count, scene.vehicle_actors),
                    ("walker", config.walker_count, scene.walker_actors),
                ):
                    if requested < len(actors):
                        removed_ids = {int(actor.id) for actor in actors[requested:]}
                        if kind == "walker":
                            removed_ids.update(
                                int(actor.id) for actor in scene.walker_controllers[requested:]
                            )
                        uncertain = True
                        self._remove_scene_actor_subset(
                            scene,
                            [item for item in scene.owned_actors if item.actor_id in removed_ids],
                        )
                        del actors[requested:]
                        if kind == "walker":
                            del scene.walker_controllers[requested:]
                        scene.config = replace(scene.config, **{f"{kind}_count": len(actors)})
                        uncertain = False
                    elif requested > len(actors):
                        previous_ids = {item.actor_id for item in scene.owned_actors}
                        try:
                            count = requested - len(actors)
                            if kind == "traffic":
                                additions = self._spawn_traffic(
                                    scene_id,
                                    world,
                                    tm,
                                    spawn_points,
                                    scene.spawn_index,
                                    count,
                                    rng,
                                    scene.owned_actors,
                                )
                                controllers: list[Any] = []
                            else:
                                additions, controllers = self._spawn_walkers(
                                    scene_id, world, count, rng, scene.owned_actors
                                )
                            registered = self._registered_actor_ids(
                                world, [*additions, *controllers]
                            )
                            if len(additions) != count or any(
                                int(actor.id) not in registered
                                for actor in [*additions, *controllers]
                            ):
                                raise WorkerError(
                                    HTTPStatus.UNPROCESSABLE_ENTITY,
                                    "scene_population_shortfall",
                                    f"CARLA created {len(additions)}/{count} additional {kind} actors; "
                                    "new actors rolled back, existing population retained",
                                )
                        except Exception:
                            uncertain = True
                            self._remove_scene_actor_subset(
                                scene,
                                [
                                    item
                                    for item in scene.owned_actors
                                    if item.actor_id not in previous_ids
                                ],
                            )
                            uncertain = False
                            raise
                        actors.extend(additions)
                        if kind == "walker":
                            scene.walker_controllers.extend(controllers)
                        scene.config = replace(scene.config, **{f"{kind}_count": len(actors)})
                if config.prop_preset != scene.config.prop_preset:
                    uncertain = True
                    self._remove_scene_actor_subset(
                        scene, [item for item in scene.owned_actors if item.kind == "prop"]
                    )
                    scene.prop_actors = []
                    scene.config = replace(scene.config, prop_preset="none")
                    try:
                        scene.prop_actors = self._spawn_props(
                            scene_id,
                            world,
                            config.prop_preset,
                            scene.ego.get_transform(),
                            scene.owned_actors,
                        )
                    except Exception:
                        self._remove_scene_actor_subset(
                            scene, [item for item in scene.owned_actors if item.kind == "prop"]
                        )
                        uncertain = False
                        raise
                    scene.config = replace(scene.config, prop_preset=config.prop_preset)
                    uncertain = False
                if config.route_mode != scene.config.route_mode:
                    if config.route_mode == "random_destination":
                        route, destination, locations = self._plan_random_route(
                            world, scene.ego, spawn_points, scene.spawn_index, rng
                        )
                    else:
                        route = {
                            "mode": "free",
                            "provider": None,
                            "planned": False,
                            "enforced": False,
                            "waypoint_count": 0,
                        }
                        destination, locations = None, []
                    scene.route, scene.destination, scene.route_locations = (
                        route,
                        destination,
                        locations,
                    )
                    scene.config = replace(scene.config, route_mode=config.route_mode)
                # PREPARED means parked; autopilot is enabled only by start().
                scene.control_mode = config.initial_control_mode
                scene.config = replace(
                    scene.config, initial_control_mode=config.initial_control_mode
                )
                if scene.config.route_mode == "free":
                    scene.route["provider"] = (
                        "TrafficManager" if scene.control_mode == "autopilot" else None
                    )
                self._refresh_lease(scene)
                return self._scene_response(scene, "prepared")
            except Exception as error:
                if uncertain:
                    self._cleanup_resources(scene, reason="scene_reconfigure_failed")
                if isinstance(error, WorkerError):
                    raise
                raise WorkerError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "scene_reconfigure_failed",
                    f"CARLA scene reconfiguration failed: {error}",
                ) from error
            finally:
                # This mutation owns the normal lock, so heartbeats cannot run
                # during a dense delta; its work must not expire its own lease.
                self._refresh_lease(scene)

    def _apply_full_brake(self, ego: Any) -> None:
        assert self._carla is not None
        ego.apply_control(
            self._carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=1.0,
                hand_brake=False,
                reverse=False,
            )
        )

    def _enable_autopilot(self, scene: SceneLease) -> None:
        try:
            scene.ego.set_autopilot(True, int(scene.traffic_manager.get_port()))
            if hasattr(scene.traffic_manager, "update_vehicle_lights"):
                scene.traffic_manager.update_vehicle_lights(scene.ego, True)
            if scene.config.route_mode == "random_destination":
                if not scene.route_locations or not hasattr(scene.traffic_manager, "set_path"):
                    raise RuntimeError("prepared random route cannot be enforced")
                scene.traffic_manager.set_path(scene.ego, list(scene.route_locations))
                scene.route["enforced"] = True
        except Exception as error:
            try:
                scene.ego.set_autopilot(False, int(scene.traffic_manager.get_port()))
                self._apply_full_brake(scene.ego)
            except Exception:
                pass
            scene.route["enforced"] = False
            raise WorkerError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "autopilot_failed",
                f"CARLA Traffic Manager autopilot failed: {type(error).__name__}: {error}",
            ) from error

    def _enable_manual(self, scene: SceneLease) -> None:
        try:
            scene.ego.set_autopilot(False, int(scene.traffic_manager.get_port()))
            self._apply_full_brake(scene.ego)
        except Exception as error:
            raise WorkerError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "manual_mode_failed",
                f"could not take authoritative manual control: {type(error).__name__}: {error}",
            ) from error
        scene.route["enforced"] = False
        scene.deadman_active = True
        scene.last_control_at = self._clock()

    def _garage_camera_transform(
        self,
        ego: Any,
        *,
        yaw: float,
        pitch: float,
        distance: float,
        width: int | None = None,
        height: int | None = None,
        fov: float | None = None,
        preset: str = "orbit",
    ) -> Any:
        assert self._carla is not None
        vehicle = ego.get_transform()
        if preset in _GARAGE_EXTERIOR_PRESETS:
            yaw, pitch, distance = _GARAGE_EXTERIOR_PRESETS[preset]
        bounds = self._garage_vehicle_bounds(ego, vehicle)
        if preset == "cockpit":
            return self._garage_cockpit_transform(vehicle, bounds)

        if bounds is not None and width is not None and height is not None and fov is not None:
            target, _local_center, extent = bounds
            try:
                aspect = float(width) / float(height)
                horizontal_half_fov = math.radians(float(fov)) / 2.0
                vertical_half_fov = math.atan(math.tan(horizontal_half_fov) / aspect)
                azimuth = math.radians(float(yaw))
                pitch_radians = math.radians(float(pitch))
                extent_x, extent_y, extent_z = extent
                half_width = abs(math.sin(azimuth)) * extent_x + abs(math.cos(azimuth)) * extent_y
                half_depth = abs(math.cos(azimuth)) * extent_x + abs(math.sin(azimuth)) * extent_y
                projected_height = (
                    abs(math.sin(pitch_radians)) * half_depth
                    + abs(math.cos(pitch_radians)) * extent_z
                )
                projected_depth = (
                    abs(math.cos(pitch_radians)) * half_depth
                    + abs(math.sin(pitch_radians)) * extent_z
                )
                minimum_fit = projected_depth + max(
                    half_width / math.tan(horizontal_half_fov),
                    projected_height / math.tan(vertical_half_fov),
                )
                zoom_margin = 1.02 + ((float(distance) - 3.5) / 6.5) * 0.43
                eye_distance = minimum_fit * zoom_margin
                if not math.isfinite(eye_distance) or eye_distance <= 0.0:
                    raise ValueError("computed Garage camera distance is invalid")
                return self._garage_exterior_transform(
                    vehicle,
                    target=target,
                    yaw=yaw,
                    pitch=pitch,
                    distance=eye_distance,
                )
            except (AttributeError, TypeError, ValueError, ZeroDivisionError):
                pass

        target = self._carla.Location(
            x=float(vehicle.location.x),
            y=float(vehicle.location.y),
            z=float(vehicle.location.z) + 0.9,
        )
        return self._garage_exterior_transform(
            vehicle,
            target=target,
            yaw=yaw,
            pitch=pitch,
            distance=distance,
        )

    def _garage_vehicle_bounds(
        self,
        ego: Any,
        vehicle: Any,
    ) -> (
        tuple[
            Any,
            tuple[float, float, float],
            tuple[float, float, float],
        ]
        | None
    ):
        """Return a validated world-space bounds centre and local half-extents."""

        try:
            bounds = ego.bounding_box
            center = bounds.location
            extent = bounds.extent
            center_values = (float(center.x), float(center.y), float(center.z))
            extent_values = (float(extent.x), float(extent.y), float(extent.z))
            if not all(math.isfinite(value) for value in (*center_values, *extent_values)):
                return None
            if any(value <= 0.0 for value in extent_values):
                return None
            target = self._garage_local_to_world(vehicle, center_values)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        return target, center_values, extent_values

    def _garage_local_to_world(
        self,
        vehicle: Any,
        local: tuple[float, float, float],
    ) -> Any:
        assert self._carla is not None
        pitch = math.radians(float(vehicle.rotation.pitch))
        yaw = math.radians(float(vehicle.rotation.yaw))
        roll = math.radians(float(vehicle.rotation.roll))
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        matrix = (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )
        translated = [
            origin
            + sum(coefficient * offset for coefficient, offset in zip(row, local, strict=True))
            for origin, row in zip(
                (vehicle.location.x, vehicle.location.y, vehicle.location.z),
                matrix,
                strict=True,
            )
        ]
        return self._carla.Location(x=translated[0], y=translated[1], z=translated[2])

    def _garage_cockpit_transform(
        self,
        vehicle: Any,
        bounds: tuple[
            Any,
            tuple[float, float, float],
            tuple[float, float, float],
        ]
        | None,
    ) -> Any:
        assert self._carla is not None
        if bounds is None:
            local_eye = (0.35, 0.0, 1.25)
        else:
            _target, center, extent = bounds
            local_eye = (
                center[0] + extent[0] * 0.5,
                center[1] - extent[1] * 0.25,
                center[2] + extent[2] * 0.65,
            )
        location = self._garage_local_to_world(vehicle, local_eye)
        rotation = self._carla.Rotation(
            pitch=float(vehicle.rotation.pitch),
            yaw=float(vehicle.rotation.yaw),
            roll=float(vehicle.rotation.roll),
        )
        return self._carla.Transform(location, rotation)

    def _garage_exterior_transform(
        self,
        vehicle: Any,
        *,
        target: Any,
        yaw: float,
        pitch: float,
        distance: float,
    ) -> Any:
        assert self._carla is not None
        bearing = math.radians(float(vehicle.rotation.yaw) + yaw)
        pitch_radians = math.radians(pitch)
        horizontal = distance * math.cos(pitch_radians)
        location = self._carla.Location(
            x=float(target.x) + horizontal * math.cos(bearing),
            y=float(target.y) + horizontal * math.sin(bearing),
            z=float(target.z) - distance * math.sin(pitch_radians),
        )
        rotation = self._carla.Rotation(
            pitch=pitch,
            yaw=float(vehicle.rotation.yaw) + yaw + 180.0,
            roll=0.0,
        )
        return self._carla.Transform(location, rotation)

    def camera(self, scene_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        config = CompressedCameraConfig.from_mapping(raw)
        lease_token = _text(raw["lease_token"], "lease_token", maximum=256)
        with self._lock:
            scene = self._require_scene(scene_id, lease_token)
            if scene.camera_relay is not None:
                if scene.status != "prepared" or scene.camera_config == config:
                    raise WorkerError(
                        HTTPStatus.CONFLICT,
                        "camera_already_active",
                        "compressed camera is already active for this scene",
                    )
                if self._episode_marker(scene.client.get_world()) != scene.episode_marker:
                    raise WorkerError(
                        HTTPStatus.CONFLICT,
                        "episode_changed",
                        "CARLA episode changed before camera handoff",
                    )
            # Resolve the optional encoder before spawning an actor so a
            # missing or broken OpenCV installation cannot leak a CARLA sensor.
            jpeg_encoder = _load_in_memory_jpeg_encoder()
            if scene.camera_relay is not None:
                relay = scene.camera_relay
                owned = next(
                    item for item in scene.owned_actors if item.actor_id == relay.sensor.id
                )
                relay.close()
                self._destroy_owned_actor(scene.world, owned)
                scene.owned_actors.remove(owned)
                scene.camera_relay = None
                scene.camera_config = None
            assert self._carla is not None
            blueprint = scene.world.get_blueprint_library().find("sensor.camera.rgb")
            attributes = {
                "role_name": "world_worker_camera",
                "image_size_x": str(config.width),
                "image_size_y": str(config.height),
                "sensor_tick": str(1.0 / config.fps),
                "fov": str(config.fov),
                "motion_blur_intensity": "0.0",
                "motion_blur_max_distortion": "0.0",
                "enable_postprocess_effects": "true",
                "gamma": "2.2",
            }
            for name, value in attributes.items():
                if blueprint.has_attribute(name):
                    blueprint.set_attribute(name, value)
            if config.mode == "garage":
                transform = self._garage_camera_transform(
                    scene.ego,
                    yaw=config.yaw,
                    pitch=config.pitch,
                    distance=config.distance,
                    width=config.width,
                    height=config.height,
                    fov=config.fov,
                )
                sensor = scene.world.spawn_actor(blueprint, transform)
            else:
                transform = self._carla.Transform(
                    self._carla.Location(x=1.5, y=0.0, z=1.7),
                    self._carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
                )
                sensor = scene.world.spawn_actor(blueprint, transform, attach_to=scene.ego)
            owned = OwnedActor(
                actor=sensor,
                actor_id=int(sensor.id),
                type_id=str(sensor.type_id),
                kind="compressed_camera",
                role_name="world_worker_camera",
            )
            scene.owned_actors.append(owned)
            # Preserve detail at 30 FPS; the high-refresh profile trades a
            # small amount of JPEG quality for encode time and LAN headroom.
            jpeg_quality = 85 if config.fps > 30.0 else _JPEG_QUALITY
            relay = CompressedCameraRelay(
                sensor,
                jpeg_encoder=jpeg_encoder,
                jpeg_quality=jpeg_quality,
            )
            scene.camera_relay = relay
            scene.camera_config = config
            try:
                relay.listen()
            except BaseException as error:
                try:
                    relay.close()
                except BaseException as cleanup_error:
                    # Keep the drain guard and sensor ownership for a later
                    # stop retry. Destroying while callbacks run is unsafe.
                    error.add_note(f"camera relay shutdown failed: {cleanup_error}")
                    raise error from cleanup_error
                scene.camera_relay = None
                scene.camera_config = None
                try:
                    self._destroy_owned_actor(scene.world, owned)
                except BaseException as cleanup_error:
                    error.add_note(f"camera activation rollback failed: {cleanup_error}")
                else:
                    scene.owned_actors.remove(owned)
                raise
            self._refresh_lease(scene)
            return {
                "schema_version": SCHEMA_VERSION,
                "worker_api_revision": WORKER_API_REVISION,
                "status": "starting",
                "scene_id": scene.scene_id,
                "camera": {
                    "actor_id": int(sensor.id),
                    "mode": config.mode,
                    "width": config.width,
                    "height": config.height,
                    "fps": config.fps,
                    "fov": config.fov,
                },
            }

    def camera_orbit(self, scene_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {"lease_token", "yaw", "pitch", "distance", "preset"}
        lease_token = self._lease_token(raw, allowed=allowed)
        _strict_keys(
            raw,
            allowed=allowed,
            required={"lease_token", "yaw", "pitch", "distance"},
            name="camera orbit request",
        )
        yaw = _number(raw["yaw"], "yaw", -3600.0, 3600.0)
        pitch = _number(raw["pitch"], "pitch", -25.0, 15.0)
        distance = _number(raw["distance"], "distance", 3.5, 10.0)
        preset = _text(raw.get("preset", "orbit"), "preset", maximum=16)
        if preset not in _GARAGE_CAMERA_PRESETS:
            choices = ", ".join(sorted(_GARAGE_CAMERA_PRESETS))
            raise WorkerError(
                HTTPStatus.BAD_REQUEST,
                "invalid_field",
                f"preset must be one of: {choices}",
            )
        with self._lock:
            scene = self._require_scene(scene_id, lease_token)
            relay = scene.camera_relay
            if relay is None:
                raise WorkerError(
                    HTTPStatus.CONFLICT, "camera_inactive", "compressed camera is not active"
                )
            if scene.camera_config is None or scene.camera_config.mode != "garage":
                raise WorkerError(
                    HTTPStatus.CONFLICT,
                    "camera_mode_conflict",
                    "camera orbit and Garage presets require an active Garage camera",
                )
            relay.sensor.set_transform(
                self._garage_camera_transform(
                    scene.ego,
                    yaw=yaw,
                    pitch=pitch,
                    distance=distance,
                    width=None if scene.camera_config is None else scene.camera_config.width,
                    height=None if scene.camera_config is None else scene.camera_config.height,
                    fov=None if scene.camera_config is None else scene.camera_config.fov,
                    preset=preset,
                )
            )
            self._refresh_lease(scene)
            return {
                "schema_version": SCHEMA_VERSION,
                "worker_api_revision": WORKER_API_REVISION,
                "status": "running",
                "scene_id": scene.scene_id,
                "camera": relay.snapshot(),
            }

    def _set_camera_subscription(
        self,
        scene_id: str,
        raw: Mapping[str, Any],
        *,
        listening: bool,
    ) -> dict[str, Any]:
        lease_token = self._lease_token(raw, allowed={"lease_token"})
        with self._lock:
            scene = self._require_scene(scene_id, lease_token)
            if scene.status not in {"prepared", "running"}:
                raise WorkerError(
                    HTTPStatus.CONFLICT, "scene_inactive", "scene is stopping"
                )
            relay = scene.camera_relay
            if relay is None:
                raise WorkerError(
                    HTTPStatus.NOT_FOUND, "camera_inactive", "compressed camera is not active"
                )

        # Sensor subscription calls can touch CARLA's streaming client. Keep
        # them outside the scene mutation lock so heartbeat/control remain live.
        try:
            if listening:
                relay.resume()
            else:
                relay.pause()
        except Exception as error:
            raise WorkerError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "camera_subscription_failed",
                f"CARLA camera subscription change failed: {type(error).__name__}: {error}",
            ) from error

        with self._lock:
            current = self._require_scene(scene_id, lease_token)
            if current is not scene or current.camera_relay is not relay:
                raise WorkerError(
                    HTTPStatus.CONFLICT,
                    "camera_changed",
                    "camera changed while its subscription was updated",
                )
            self._refresh_lease(current)
            return self._scene_response(current, current.status)

    def camera_pause(self, scene_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        """Pause the live RGB subscription for acceptance/diagnostics."""

        return self._set_camera_subscription(scene_id, raw, listening=False)

    def camera_resume(self, scene_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        """Resume the same live RGB sensor after a diagnostic pause."""

        return self._set_camera_subscription(scene_id, raw, listening=True)

    def camera_frame(
        self,
        scene_id: str,
        lease_token: str,
        *,
        after_sequence: int,
        timeout: float,
    ) -> tuple[int, bytes, dict[str, Any]]:
        with self._lock:
            scene = self._require_scene(scene_id, lease_token)
            relay = scene.camera_relay
            if relay is None:
                raise WorkerError(
                    HTTPStatus.NOT_FOUND, "camera_inactive", "compressed camera is not active"
                )
            self._refresh_lease(scene)
        return relay.wait(after_sequence, timeout)

    def waypoints(self, scene_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        """Read bounded map geometry for a teacher overlay, never vehicle control.

        Lease and scene identity are snapshotted under the mutation lock. CARLA/TM
        reads run after releasing it so optional teacher rendering cannot delay
        control, heartbeat, weather, mode changes, or scene teardown.
        """

        lease_token = self._lease_token(
            raw, allowed={"lease_token", "camera_location", "source_frame"}
        )
        camera_location = raw.get("camera_location")
        if camera_location is not None:
            if not isinstance(camera_location, Mapping):
                raise WorkerError(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_field",
                    "camera_location must be an object",
                )
            _strict_keys(
                camera_location,
                allowed={"x", "y", "z"},
                required={"x", "y", "z"},
                name="camera location",
            )
            camera_location = {
                axis: _number(
                    camera_location[axis], f"camera_location.{axis}", -1e7, 1e7
                )
                for axis in ("x", "y", "z")
            }
        source_frame = raw.get("source_frame")
        if source_frame is not None:
            source_frame = _integer(source_frame, "source_frame", 0, 2**63 - 1)

        if not self._lock.acquire(blocking=False):
            raise WorkerError(
                HTTPStatus.CONFLICT,
                "worker_busy",
                "World Worker is updating the scene",
            )
        try:
            scene = self._require_scene(scene_id, lease_token)
            if scene.status not in {"prepared", "running"}:
                raise WorkerError(
                    HTTPStatus.CONFLICT, "scene_inactive", "scene is stopping"
                )
            if self._clock() >= scene.lease_deadline:
                raise WorkerError(
                    HTTPStatus.CONFLICT, "lease_expired", "scene lease expired"
                )
            if self._episode_id(scene.world) != scene.episode_id:
                raise WorkerError(
                    HTTPStatus.CONFLICT, "episode_changed", "CARLA episode changed"
                )
            try:
                owned = next(item for item in scene.owned_actors if item.kind == "ego")
                ego = scene.ego
                if (
                    ego is None
                    or not bool(getattr(ego, "is_alive", True))
                    or int(ego.id) != owned.actor_id
                ):
                    raise RuntimeError("owned ego is no longer available")
                self._validate_owned_actor_identity(ego, owned)
            except Exception as error:
                raise WorkerError(
                    HTTPStatus.CONFLICT,
                    "scene_identity_changed",
                    "scene ego identity changed",
                ) from error

            world = scene.world
            route_locations = list(scene.route_locations)
            traffic_manager = scene.traffic_manager
            scene_status = str(scene.status)
            control_mode = str(scene.control_mode)
            episode_id = int(scene.episode_id)
            response_scene_id = str(scene.scene_id)
            waypoint_map = scene.waypoint_map
        finally:
            self._lock.release()

        try:
            origin = (
                ego.get_location()
                if camera_location is None
                else self._carla.Location(**camera_location)
            )
        except Exception as error:
            raise WorkerError(
                HTTPStatus.CONFLICT,
                "scene_identity_changed",
                "scene ego is no longer available",
            ) from error

        sampled_frame: int | None = None
        sampled_timestamp: float | None = None
        try:
            snapshot = world.get_snapshot()
            sampled_frame = int(snapshot.frame)
            sampled_timestamp = float(snapshot.timestamp.elapsed_seconds)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass

        source = "planned_route"
        locations = route_locations
        note = "Fixed map route; not a road prediction or collision-free path."
        if len(locations) < 2:
            locations = []
            get_actions = getattr(traffic_manager, "get_all_actions", None)
            if (
                scene_status == "running"
                and control_mode == "autopilot"
                and callable(get_actions)
            ):
                try:
                    actions = get_actions(ego)
                    locations = [action[1].transform.location for action in actions[:512]]
                except Exception:
                    # Teacher loss is optional and must never fail RGB or actuation.
                    locations = []
            if len(locations) >= 2:
                source = "traffic_manager"
                note = "Upcoming TM actions sampled now, not frame-matched route intent."
            else:
                source = "lane_centerline"
                note = "Lane centerline only; stops at a branch and is not the TM route."
                try:
                    if waypoint_map is None:
                        waypoint_map = world.get_map()
                        # Cache opportunistically; never wait for the mutation lock.
                        if self._lock.acquire(blocking=False):
                            try:
                                current = self._scene
                                if current is scene and current.waypoint_map is None:
                                    current.waypoint_map = waypoint_map
                            finally:
                                self._lock.release()
                    waypoint = waypoint_map.get_waypoint(origin)
                    locations = []
                    for _ in range(64):
                        if waypoint is None:
                            break
                        locations.append(waypoint.transform.location)
                        candidates = waypoint.next(2.0)
                        if len(candidates) != 1:
                            break
                        waypoint = candidates[0]
                except Exception as error:
                    raise WorkerError(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        "waypoint_teacher_unavailable",
                        "CARLA lane geometry is unavailable for the teacher overlay",
                    ) from error

        points = self._bounded_waypoint_locations(locations, origin)

        # Refuse a stale sample if lifecycle ownership changed while the optional
        # read was in flight. This final check is non-blocking by design.
        if not self._lock.acquire(blocking=False):
            raise WorkerError(
                HTTPStatus.CONFLICT,
                "worker_busy",
                "World Worker changed while waypoint geometry was sampled",
            )
        try:
            current = self._require_scene(scene_id, lease_token)
            if (
                current is not scene
                or current.status not in {"prepared", "running"}
                or int(current.episode_id) != episode_id
            ):
                raise WorkerError(
                    HTTPStatus.CONFLICT,
                    "episode_changed",
                    "CARLA scene changed while waypoint geometry was sampled",
                )
            if self._clock() >= current.lease_deadline:
                raise WorkerError(
                    HTTPStatus.CONFLICT, "lease_expired", "scene lease expired"
                )
        finally:
            self._lock.release()

        return {
            "schema_version": SCHEMA_VERSION,
            "worker_api_revision": WORKER_API_REVISION,
            "scene_id": response_scene_id,
            "episode_id": episode_id,
            "source": source,
            "coordinate_frame": "carla_world_metres",
            "teacher_only": True,
            "model_input": False,
            "controls_vehicle": False,
            "route_frame_matched": False,
            "source_frame": source_frame,
            "sampled_frame": sampled_frame,
            "sampled_timestamp": sampled_timestamp,
            "points": points,
            "note": note,
        }

    @classmethod
    def _bounded_waypoint_locations(
        cls, locations: list[Any], origin: Any
    ) -> list[dict[str, float]]:
        """Keep nearby forward map geometry, at most 64 points and 100 metres."""

        if not locations:
            return []
        nearest = min(
            range(len(locations)), key=lambda index: cls._distance(origin, locations[index])
        )
        points: list[dict[str, float]] = []
        previous = None
        distance = 0.0
        for location in locations[max(0, nearest - 1) :]:
            point = _json_location(location)
            if not all(math.isfinite(value) and abs(value) <= 1e7 for value in point.values()):
                break
            if previous is not None:
                step = cls._distance(previous, location)
                if step < 0.05:
                    continue
                distance += step
                if distance > 100.0:
                    break
            points.append(point)
            previous = location
            if len(points) == 64:
                break
        return points

    def start(self, scene_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        lease_token = self._lease_token(raw, allowed={"lease_token"})
        with self._lock:
            scene = self._require_scene(scene_id, lease_token)
            if scene.status != "prepared":
                raise WorkerError(
                    HTTPStatus.CONFLICT,
                    "invalid_scene_state",
                    "only a prepared scene can start",
                )
            if scene.control_mode == "autopilot":
                self._enable_autopilot(scene)
                scene.deadman_active = False
            else:
                self._enable_manual(scene)
            scene.status = "running"
            self._refresh_lease(scene)
            return self._scene_response(scene, "running")

    def heartbeat(self, scene_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        lease_token = self._lease_token(raw, allowed={"lease_token"})
        with self._lock:
            scene = self._require_scene(scene_id, lease_token)
            self._refresh_lease(scene)
            return self._scene_response(scene, scene.status)

    def control(self, scene_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {
            "lease_token",
            "sequence",
            "throttle",
            "steer",
            "brake",
            "hand_brake",
            "reverse",
        }
        lease_token = self._lease_token(raw, allowed=allowed)
        required = allowed
        _strict_keys(raw, allowed=allowed, required=required, name="manual control request")
        sequence = _integer(raw["sequence"], "sequence", 0, 2**63 - 1)
        throttle = _number(raw["throttle"], "throttle", 0.0, 1.0)
        steer = _number(raw["steer"], "steer", -1.0, 1.0)
        brake = _number(raw["brake"], "brake", 0.0, 1.0)
        hand_brake = _boolean(raw["hand_brake"], "hand_brake")
        reverse = _boolean(raw["reverse"], "reverse")
        if brake > 0.01 or hand_brake:
            throttle = 0.0

        with self._lock:
            scene = self._require_scene(scene_id, lease_token)
            if scene.status != "running" or scene.control_mode != "manual":
                raise WorkerError(
                    HTTPStatus.CONFLICT,
                    "manual_control_unavailable",
                    "manual control requires a running scene in manual mode",
                )
            if sequence <= scene.last_control_sequence:
                raise WorkerError(
                    HTTPStatus.CONFLICT,
                    "stale_control_sequence",
                    "control sequence must be newer than the last accepted sequence",
                )
            assert self._carla is not None
            try:
                scene.ego.apply_control(
                    self._carla.VehicleControl(
                        throttle=throttle,
                        steer=steer,
                        brake=brake,
                        hand_brake=hand_brake,
                        reverse=reverse,
                    )
                )
            except Exception as error:
                raise WorkerError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "control_failed",
                    f"CARLA vehicle control failed: {type(error).__name__}: {error}",
                ) from error
            scene.last_control_sequence = sequence
            scene.last_control_at = self._clock()
            scene.deadman_active = False
            self._refresh_lease(scene)
            return self._scene_response(scene, "running")

    def mode(self, scene_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {"lease_token", "control_mode"}
        lease_token = self._lease_token(raw, allowed=allowed)
        _strict_keys(raw, allowed=allowed, required=allowed, name="mode request")
        control_mode = str(raw["control_mode"]).strip()
        if control_mode not in _CONTROL_MODES:
            raise WorkerError(
                HTTPStatus.BAD_REQUEST,
                "invalid_field",
                "control_mode must be manual or autopilot",
            )
        with self._lock:
            scene = self._require_scene(scene_id, lease_token)
            if scene.status not in {"prepared", "running"}:
                raise WorkerError(
                    HTTPStatus.CONFLICT,
                    "invalid_scene_state",
                    "control mode cannot change in the current scene state",
                )
            if scene.control_mode != control_mode:
                if scene.status == "running":
                    if control_mode == "autopilot":
                        self._enable_autopilot(scene)
                        scene.deadman_active = False
                    else:
                        self._enable_manual(scene)
                scene.control_mode = control_mode
            self._refresh_lease(scene)
            return self._scene_response(scene, scene.status)

    def weather(self, scene_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {"lease_token", "weather_preset"}
        lease_token = self._lease_token(raw, allowed=allowed)
        _strict_keys(raw, allowed=allowed, required=allowed, name="weather request")
        preset = str(raw["weather_preset"]).strip()
        if preset not in WEATHER_PRESETS:
            raise WorkerError(
                HTTPStatus.BAD_REQUEST,
                "invalid_field",
                "weather_preset must be one named non-'keep' preset",
            )
        with self._lock:
            scene = self._require_scene(scene_id, lease_token)
            self._apply_weather(scene.world, preset)
            scene.weather_preset = preset
            scene.config = replace(scene.config, weather_preset=preset)
            self._refresh_lease(scene)
            return self._scene_response(scene, scene.status)

    def stop(self, scene_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        lease_token = self._lease_token(raw, allowed={"lease_token"})
        with self._lock:
            if self._scene is None and self._last_scene is not None:
                previous_scene_id = str(self._last_scene.get("scene_id", ""))
                if previous_scene_id == scene_id:
                    previous_token = str(self._last_scene.get("lease_token", ""))
                    if not hmac.compare_digest(previous_token, lease_token):
                        raise WorkerError(
                            HTTPStatus.CONFLICT,
                            "lease_mismatch",
                            "scene lease does not match",
                        )
                    return {
                        "schema_version": SCHEMA_VERSION,
                        "worker_api_revision": WORKER_API_REVISION,
                        "status": "stopped",
                        "scene": dict(self._last_scene),
                    }
            scene = self._require_scene(scene_id, lease_token)
            snapshot = self._cleanup_resources(scene, reason="operator_stop")
            self._last_scene = snapshot
            self._scene = None
            return {
                "schema_version": SCHEMA_VERSION,
                "worker_api_revision": WORKER_API_REVISION,
                "status": "stopped",
                "scene": snapshot,
            }

    def _scene_summary(self, scene: SceneLease | None) -> dict[str, Any] | None:
        if scene is None:
            return None
        return {
            "scene_id": scene.scene_id,
            "status": scene.status,
            "map_name": scene.map_name,
            "ego_actor_id": None if scene.ego is None else int(scene.ego.id),
            "control_mode": scene.control_mode,
            "route_mode": scene.config.route_mode,
        }

    def _scene_snapshot(self, scene: SceneLease) -> dict[str, Any]:
        now = self._clock()
        input_age = None if scene.last_control_at is None else max(0.0, now - scene.last_control_at)
        return {
            "scene_id": scene.scene_id,
            "lease_token": scene.lease_token,
            "status": scene.status,
            "episode_id": scene.episode_id,
            "config": scene.config.as_dict(),
            "ego_actor_id": None if scene.ego is None else int(scene.ego.id),
            "map_name": scene.map_name,
            "spawn_index": scene.spawn_index,
            "route_mode": scene.config.route_mode,
            "route": dict(scene.route),
            "destination": scene.destination,
            "control_mode": scene.control_mode,
            "weather_preset": scene.weather_preset,
            "traffic_count_requested": scene.config.traffic_count,
            "traffic_count": len(scene.vehicle_actors),
            "walker_count_requested": scene.config.walker_count,
            "walker_count": len(scene.walker_actors),
            "pedestrian_crossing_factor": scene.config.pedestrian_crossing_factor,
            "speed_difference_percent": scene.config.speed_difference_percent,
            "following_distance_metres": scene.config.following_distance_metres,
            "prop_actor_ids": [int(actor.id) for actor in scene.prop_actors],
            "lease_expires_in_seconds": max(0.0, scene.lease_deadline - now),
            "control_input_age_seconds": input_age,
            "deadman_active": scene.deadman_active,
            "last_control_sequence": scene.last_control_sequence,
            "stop_reason": scene.stop_reason,
            "cleanup_guard_passed": scene.cleanup_guard_passed,
            "cleanup_errors": list(scene.cleanup_errors),
            "camera": None if scene.camera_relay is None else scene.camera_relay.snapshot(),
            "camera_config": None
            if scene.camera_config is None
            else {
                name: getattr(scene.camera_config, name)
                for name in ("mode", "width", "height", "fps", "fov", "yaw", "pitch", "distance")
            },
            "capabilities": self._capabilities(scene.client),
        }

    def _scene_response(self, scene: SceneLease, status: str) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "worker_api_revision": WORKER_API_REVISION,
            "status": status,
            "scene": self._scene_snapshot(scene),
        }

    @staticmethod
    def _actor_attribute(actor: Any, name: str) -> str | None:
        try:
            attributes = dict(actor.attributes)
        except Exception:
            return None
        value = attributes.get(name)
        return None if value is None else str(value)

    def _destroy_owned_actor(self, current_world: Any, owned: OwnedActor) -> None:
        try:
            current = current_world.get_actor(owned.actor_id)
        except Exception:
            current = owned.actor
        if current is None:
            if owned.actor is None:
                raise RuntimeError(
                    f"spawned actor {owned.actor_id} has no snapshot proxy; "
                    "command cleanup is required"
                )
            return
        self._validate_owned_actor_identity(current, owned)
        if hasattr(current, "is_alive") and not bool(current.is_alive):
            return
        try:
            destroyed = current.destroy()
        except Exception as error:
            if self._actor_is_confirmed_absent(current_world, current):
                return
            raise RuntimeError(
                f"actor {owned.actor_id} destroy raised while actor remains registered: "
                f"{type(error).__name__}: {error}"
            ) from error
        if destroyed is True or self._actor_is_confirmed_absent(current_world, current):
            return
        raise RuntimeError(
            f"actor {owned.actor_id} destroy was not confirmed and actor remains registered"
        )

    def _validate_owned_actor_identity(self, current: Any, owned: OwnedActor) -> None:
        if str(getattr(current, "type_id", "")) != owned.type_id:
            raise RuntimeError(f"actor {owned.actor_id} type changed; refusing destroy")
        if owned.role_name is not None:
            current_role = self._actor_attribute(current, "role_name")
            if current_role != owned.role_name:
                raise RuntimeError(f"actor {owned.actor_id} role changed; refusing destroy")

    @staticmethod
    def _bulk_current_actors(
        current_world: Any,
        actor_ids: Sequence[int],
    ) -> tuple[dict[int, Any], dict[int, str]]:
        unique_ids = list(dict.fromkeys(int(actor_id) for actor_id in actor_ids))
        if not unique_ids:
            return {}, {}
        get_actors = getattr(current_world, "get_actors", None)
        bulk_error: str | None = None
        if callable(get_actors):
            try:
                actors = get_actors(unique_ids)
                current = {
                    int(actor.id): actor
                    for actor in actors
                    if actor is not None and bool(getattr(actor, "is_alive", True))
                }
            except Exception as error:
                bulk_error = f"{type(error).__name__}: {error}"
            else:
                return current, {}
        current: dict[int, Any] = {}
        lookup_errors: dict[int, str] = {}
        for actor_id in unique_ids:
            try:
                actor = current_world.get_actor(actor_id)
            except Exception as error:
                detail = f"{type(error).__name__}: {error}"
                if bulk_error is not None:
                    detail = f"bulk lookup failed ({bulk_error}); actor lookup failed ({detail})"
                lookup_errors[actor_id] = detail
                continue
            if actor is not None and bool(getattr(actor, "is_alive", True)):
                current[actor_id] = actor
        return current, lookup_errors

    def _destroy_owned_actors(self, scene: SceneLease, current_world: Any) -> None:
        """Destroy a verified scene population in one CARLA command batch."""

        ordered: list[OwnedActor] = []
        seen: set[int] = set()
        for owned in reversed(scene.owned_actors):
            if owned.actor_id in seen:
                continue
            seen.add(owned.actor_id)
            ordered.append(owned)
        current_by_id, lookup_errors = self._bulk_current_actors(
            current_world,
            [owned.actor_id for owned in ordered],
        )
        verified: list[OwnedActor] = []
        for owned in ordered:
            # A successful SpawnActor response establishes ownership even when
            # a failed snapshot barrier prevented obtaining a proxy. IDs are
            # never reused inside an episode; cleanup's episode guard has
            # already passed. Retain these IDs for authoritative batch destroy.
            if owned.actor is None and owned.actor_id not in current_by_id:
                verified.append(owned)
                continue
            if owned.actor_id in lookup_errors:
                # A failed lookup is not proof that an actor is absent. Use the
                # retained, ownership-recorded proxy as the conservative
                # compatibility path so the scene cannot silently leak actors.
                try:
                    self._destroy_owned_actor(current_world, owned)
                except Exception as error:
                    scene.cleanup_errors.append(
                        f"destroy {owned.kind} {owned.actor_id}: "
                        f"lookup failed ({lookup_errors[owned.actor_id]}); {error}"
                    )
                continue
            current = current_by_id.get(owned.actor_id)
            if current is None:
                continue
            try:
                self._validate_owned_actor_identity(current, owned)
            except Exception as error:
                scene.cleanup_errors.append(f"destroy {owned.kind} {owned.actor_id}: {error}")
                continue
            verified.append(owned)
        if not verified:
            return

        command_module = None if self._carla is None else getattr(self._carla, "command", None)
        destroy_actor = getattr(command_module, "DestroyActor", None)
        apply_batch_sync = getattr(scene.client, "apply_batch_sync", None)
        if not callable(destroy_actor) or not callable(apply_batch_sync):
            for owned in verified:
                try:
                    self._destroy_owned_actor(current_world, owned)
                except Exception as error:
                    scene.cleanup_errors.append(f"destroy {owned.kind} {owned.actor_id}: {error}")
            return

        try:
            responses = list(
                apply_batch_sync(
                    [destroy_actor(owned.actor_id) for owned in verified],
                    False,
                )
            )
        except Exception as error:
            scene.cleanup_errors.append(
                f"batch actor destroy failed; using compatibility cleanup: {error}"
            )
            for owned in verified:
                try:
                    self._destroy_owned_actor(current_world, owned)
                except Exception as fallback_error:
                    scene.cleanup_errors.append(
                        f"destroy {owned.kind} {owned.actor_id}: {fallback_error}"
                    )
            return

        response_errors: dict[int, str] = {}
        for index, owned in enumerate(verified):
            if index >= len(responses):
                response_errors[owned.actor_id] = "batch response was missing"
                continue
            message = str(getattr(responses[index], "error", "") or "").strip()
            if message:
                response_errors[owned.actor_id] = message

        # ``World.get_actors(ids)`` reads LibCarla's latest episode snapshot,
        # not the authoritative command response.  In an asynchronous world
        # that snapshot can still contain every actor immediately after a
        # successful batch.  Wait for one natural server tick before dropping
        # the owned proxies or checking the commands that returned an error.
        wait_for_tick = getattr(current_world, "wait_for_tick", None)
        if callable(wait_for_tick):
            barrier_timeout = min(self.timeout, 5.0)
            try:
                wait_for_tick(seconds=barrier_timeout)
            except TypeError:
                try:
                    wait_for_tick(barrier_timeout)
                except Exception as error:
                    scene.cleanup_errors.append(
                        f"batch destroy snapshot barrier failed: {type(error).__name__}: {error}"
                    )
            except Exception as error:
                scene.cleanup_errors.append(
                    f"batch destroy snapshot barrier failed: {type(error).__name__}: {error}"
                )

        # An empty command response is CARLA's authoritative success result.
        # Re-query only failed or missing responses after the snapshot barrier;
        # this keeps idempotent "not found" destroys quiet when the actor is
        # now absent while retaining genuine survivor diagnostics.
        if not response_errors:
            return
        survivors, verification_errors = self._bulk_current_actors(
            current_world,
            list(response_errors),
        )
        for owned in verified:
            if owned.actor_id not in response_errors:
                continue
            detail = response_errors[owned.actor_id]
            if owned.actor_id in verification_errors:
                scene.cleanup_errors.append(
                    f"destroy {owned.kind} {owned.actor_id}: verification failed "
                    f"({verification_errors[owned.actor_id]}); batch response: {detail}"
                )
                continue
            if owned.actor_id not in survivors:
                if owned.actor is None:
                    scene.cleanup_errors.append(
                        f"destroy unobserved {owned.kind} {owned.actor_id}: {detail}"
                    )
                continue
            scene.cleanup_errors.append(f"destroy {owned.kind} {owned.actor_id}: {detail}")

    @staticmethod
    def _release_traffic_manager(
        traffic_manager: Any | None,
        errors: list[str] | None = None,
    ) -> None:
        if traffic_manager is None:
            return
        try:
            traffic_manager.set_synchronous_mode(False)
        except Exception as error:
            if errors is not None:
                errors.append(f"Traffic Manager async restore failed: {error}")
        # CARLA 0.9.16's native TrafficManager.shut_down() performs an
        # unbounded internal thread join. On Windows it can retain the GIL and
        # freeze every HTTP handler even while the simulator RPC remains
        # healthy. A scene lease owns actors, not the process-scoped Traffic
        # Manager connection, so restore async mode and reuse that connection.

    def _cleanup_resources(self, scene: SceneLease, *, reason: str) -> dict[str, Any]:
        if scene.status == "stopped":
            return self._scene_snapshot(scene)
        scene.status = "stopping"
        scene.stop_reason = reason
        try:
            current_world = scene.client.get_world()
            same_episode = self._episode_marker(current_world) == scene.episode_marker
        except Exception as error:
            current_world = scene.world
            same_episode = False
            scene.cleanup_errors.append(f"episode guard query failed: {error}")
        scene.cleanup_guard_passed = same_episode
        if not same_episode:
            if scene.camera_relay is not None:
                scene.camera_relay.close()
                scene.camera_relay = None
            scene.camera_config = None
            self._release_traffic_manager(scene.traffic_manager, scene.cleanup_errors)
            scene.cleanup_errors.append(
                "CARLA episode changed; actor and weather cleanup intentionally skipped"
            )
            scene.status = "stopped"
            return self._scene_snapshot(scene)

        control_actor_ids = [
            *([] if scene.ego is None else [int(scene.ego.id)]),
            *(int(controller.id) for controller in scene.walker_controllers),
        ]
        live_control_actors, control_lookup_errors = self._bulk_current_actors(
            current_world,
            control_actor_ids,
        )
        owned_by_id = {owned.actor_id: owned for owned in scene.owned_actors}
        if scene.ego is not None:
            ego_id = int(scene.ego.id)
            ego = live_control_actors.get(ego_id)
            if ego_id in control_lookup_errors:
                scene.cleanup_errors.append(
                    f"ego stop lookup failed: {control_lookup_errors[ego_id]}"
                )
            if ego is not None:
                try:
                    self._validate_owned_actor_identity(ego, owned_by_id[ego_id])
                except Exception as error:
                    scene.cleanup_errors.append(f"ego stop failed: {error}")
                else:
                    try:
                        ego.set_autopilot(False, int(scene.traffic_manager.get_port()))
                        scene.route["enforced"] = False
                    except Exception as error:
                        scene.cleanup_errors.append(f"ego autopilot disable failed: {error}")
                    try:
                        self._apply_full_brake(ego)
                        scene.deadman_active = True
                    except Exception as error:
                        scene.cleanup_errors.append(f"ego brake failed: {error}")
        for controller in scene.walker_controllers:
            controller_id = int(controller.id)
            current = live_control_actors.get(controller_id)
            if controller_id in control_lookup_errors:
                scene.cleanup_errors.append(
                    "walker controller stop lookup failed "
                    f"{controller_id}: {control_lookup_errors[controller_id]}"
                )
            if current is None:
                continue
            try:
                self._validate_owned_actor_identity(current, owned_by_id[controller_id])
                current.stop()
            except Exception as error:
                scene.cleanup_errors.append(f"walker controller stop failed: {error}")
        # Apply the guarded safety stop before camera detach can fail. Keep all
        # actor ownership intact until the relay has finished draining so a
        # failed detach remains retryable without leaving the ego driving.
        if scene.camera_relay is not None:
            scene.camera_relay.close()
            scene.camera_relay = None
        scene.camera_config = None
        self._destroy_owned_actors(scene, current_world)
        try:
            current_world.set_weather(scene.original_weather)
        except Exception as error:
            scene.cleanup_errors.append(f"weather restore failed: {error}")
        self._release_traffic_manager(scene.traffic_manager, scene.cleanup_errors)
        scene.route["enforced"] = False
        scene.deadman_active = True
        scene.status = "stopped"
        return self._scene_snapshot(scene)

    def enforce_timeouts(self) -> None:
        """Apply deadman and lease safety; public for deterministic tests."""

        with self._lock:
            scene = self._scene
            if scene is None:
                return
            now = self._clock()
            if now >= scene.lease_deadline:
                snapshot = self._cleanup_resources(scene, reason="lease_expired")
                self._last_scene = snapshot
                self._scene = None
                return
            if (
                scene.status == "running"
                and scene.control_mode == "manual"
                and scene.last_control_at is not None
                and now - scene.last_control_at >= self.control_timeout
                and not scene.deadman_active
            ):
                try:
                    self._apply_full_brake(scene.ego)
                    scene.deadman_active = True
                except Exception as error:
                    scene.cleanup_errors.append(f"manual deadman brake failed: {error}")

    def _monitor_loop(self) -> None:
        while not self._monitor_stop.wait(self._monitor_period):
            try:
                self.enforce_timeouts()
            except Exception:
                # The monitor must stay alive; operational failures are retained
                # on the scene wherever a safe state update is possible.
                continue

    def close(self) -> None:
        self._monitor_stop.set()
        if self._monitor is not None and self._monitor is not threading.current_thread():
            self._monitor.join(timeout=2.0)
        with self._lock:
            if self._closed and self._scene is None:
                return
            self._closed = True
            if self._scene is not None:
                snapshot = self._cleanup_resources(self._scene, reason="worker_shutdown")
                self._last_scene = snapshot
                self._scene = None


class WorldWorkerHTTPServer(ThreadingHTTPServer):
    """Threading HTTP adapter retaining the authenticated application."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        worker: WorldWorker,
        *,
        token: str,
        log_requests: bool = False,
    ) -> None:
        self.worker = worker
        self.token = token
        self.log_requests = bool(log_requests)
        super().__init__(address, WorldWorkerRequestHandler)


class WorldWorkerRequestHandler(BaseHTTPRequestHandler):
    server: WorldWorkerHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        if self.server.log_requests:
            super().log_message(format, *args)

    def _authorized(self) -> bool:
        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {self.server.token}"
        return hmac.compare_digest(supplied, expected)

    def _send_json(self, status: int | HTTPStatus, payload: Mapping[str, Any]) -> None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _send_jpeg(self, sequence: int, payload: bytes, metadata: Mapping[str, Any]) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Camera-Sequence", str(sequence))
        self.send_header("X-CARLA-Sequence", str(sequence))
        self.send_header("X-CARLA-Frame", str(metadata.get("frame", 0)))
        self.send_header("X-CARLA-Timestamp", str(metadata.get("timestamp", 0.0)))
        self.send_header("X-Camera-Width", str(metadata.get("width", 0)))
        self.send_header("X-Camera-Height", str(metadata.get("height", 0)))
        self.send_header("X-Camera-FOV", str(metadata.get("fov", 0.0)))
        self.send_header(
            "X-Camera-Transform",
            json.dumps(metadata.get("transform"), separators=(",", ":")),
        )
        self.end_headers()
        self.wfile.write(payload)

    @staticmethod
    def _mjpeg_part(sequence: int, payload: bytes, metadata: Mapping[str, Any]) -> bytes:
        transform = json.dumps(metadata.get("transform"), separators=(",", ":"))
        headers = (
            f"--{_MJPEG_BOUNDARY}\r\n"
            "Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(payload)}\r\n"
            f"X-CARLA-Sequence: {sequence}\r\n"
            f"X-Camera-Sequence: {sequence}\r\n"
            f"X-CARLA-Frame: {metadata.get('frame', 0)}\r\n"
            f"X-CARLA-Timestamp: {metadata.get('timestamp', 0.0)}\r\n"
            f"X-Camera-Width: {metadata.get('width', 0)}\r\n"
            f"X-Camera-Height: {metadata.get('height', 0)}\r\n"
            f"X-Camera-FOV: {metadata.get('fov', 0.0)}\r\n"
            f"X-Camera-Transform: {transform}\r\n"
            "\r\n"
        ).encode("ascii")
        return headers + payload + b"\r\n"

    def _send_mjpeg(self, scene_id: str, lease_token: str) -> None:
        # Fetch one frame before committing HTTP headers, so invalid scene
        # credentials and startup encoder errors retain the normal JSON error
        # envelope.
        sequence, payload, metadata = self.server.worker.camera_frame(
            scene_id,
            lease_token,
            after_sequence=-1,
            timeout=5.0,
        )
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY}",
        )
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        while True:
            try:
                self.wfile.write(self._mjpeg_part(sequence, payload, metadata))
                self.wfile.flush()
                while True:
                    try:
                        sequence, payload, metadata = self.server.worker.camera_frame(
                            scene_id,
                            lease_token,
                            after_sequence=sequence,
                            timeout=5.0,
                        )
                        break
                    except WorkerError as error:
                        # A temporary CARLA stall must not tear down a long-lived
                        # stream. All other scene/lease failures end it cleanly.
                        if error.code != "camera_timeout":
                            return
            except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                return

    def _error(self, error: WorkerError) -> None:
        self._send_json(
            error.status,
            {
                "schema_version": SCHEMA_VERSION,
                "worker_api_revision": WORKER_API_REVISION,
                "error": {"code": error.code, "message": error.message},
            },
        )

    def _authenticate(self) -> bool:
        if self._authorized():
            return True
        self._error(
            WorkerError(
                HTTPStatus.UNAUTHORIZED,
                "unauthorized",
                "a valid Bearer token is required",
            )
        )
        return False

    def _body(self) -> Mapping[str, Any]:
        content_type = self.headers.get("Content-Type", "").split(";", maxsplit=1)[0].strip()
        if content_type != "application/json":
            raise WorkerError(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "content_type_required",
                "Content-Type must be application/json",
            )
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise WorkerError(
                HTTPStatus.BAD_REQUEST,
                "invalid_content_length",
                "Content-Length must be an integer",
            ) from error
        if not 0 <= length <= _MAX_BODY_BYTES:
            raise WorkerError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "body_too_large",
                f"request body exceeds {_MAX_BODY_BYTES} bytes",
            )
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw or b"{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise WorkerError(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                "request body must be valid JSON",
            ) from error
        if not isinstance(value, Mapping):
            raise WorkerError(
                HTTPStatus.BAD_REQUEST,
                "invalid_json_type",
                "request body must be a JSON object",
            )
        return value

    def do_GET(self) -> None:  # noqa: N802
        if not self._authenticate():
            return
        try:
            parsed = urlparse(self.path)
            if parsed.query or parsed.fragment:
                raise WorkerError(
                    HTTPStatus.BAD_REQUEST,
                    "query_not_supported",
                    "query strings are not supported",
                )
            camera_match = _CAMERA_FRAME_PATH.fullmatch(parsed.path)
            if camera_match is not None:
                lease_token = self.headers.get("X-Scene-Lease", "")
                try:
                    after_sequence = int(self.headers.get("X-Camera-After", "-1"))
                    timeout = float(self.headers.get("X-Camera-Timeout", "5"))
                except ValueError as error:
                    raise WorkerError(
                        HTTPStatus.BAD_REQUEST,
                        "invalid_camera_header",
                        "camera sequence and timeout headers must be numeric",
                    ) from error
                if after_sequence < -1 or not 0.1 <= timeout <= 15.0:
                    raise WorkerError(
                        HTTPStatus.BAD_REQUEST,
                        "invalid_camera_header",
                        "camera after must be >= -1 and timeout must be in [0.1, 15]",
                    )
                sequence, payload, metadata = self.server.worker.camera_frame(
                    camera_match.group("scene_id"),
                    lease_token,
                    after_sequence=after_sequence,
                    timeout=timeout,
                )
                self._send_jpeg(sequence, payload, metadata)
            elif (stream_match := _CAMERA_STREAM_PATH.fullmatch(parsed.path)) is not None:
                self._send_mjpeg(
                    stream_match.group("scene_id"),
                    self.headers.get("X-Scene-Lease", ""),
                )
            elif parsed.path == "/v1/health":
                self._send_json(HTTPStatus.OK, self.server.worker.health())
            elif parsed.path == "/v1/catalog":
                self._send_json(HTTPStatus.OK, self.server.worker.catalog())
            elif parsed.path == "/v1/scenes/current":
                self._send_json(HTTPStatus.OK, self.server.worker.current_scene())
            else:
                raise WorkerError(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
        except (BrokenPipeError, ConnectionResetError):
            return
        except WorkerError as error:
            self._error(error)
        except Exception:
            self._error(
                WorkerError(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "internal_error",
                    "internal World Worker error",
                )
            )

    def do_POST(self) -> None:  # noqa: N802
        if not self._authenticate():
            return
        try:
            parsed = urlparse(self.path)
            if parsed.query or parsed.fragment:
                raise WorkerError(
                    HTTPStatus.BAD_REQUEST,
                    "query_not_supported",
                    "query strings are not supported",
                )
            body = self._body()
            if parsed.path == "/v1/scenes/prepare":
                result = self.server.worker.prepare(body)
                self._send_json(HTTPStatus.CREATED, result)
                return
            match = _SCENE_PATH.fullmatch(parsed.path)
            if match is None:
                raise WorkerError(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
            scene_id = match.group("scene_id")
            action = match.group("action")
            operation = getattr(self.server.worker, action)
            result = operation(scene_id, body)
            self._send_json(HTTPStatus.OK, result)
        except (BrokenPipeError, ConnectionResetError):
            return
        except WorkerError as error:
            self._error(error)
        except Exception:
            self._error(
                WorkerError(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "internal_error",
                    "internal World Worker error",
                )
            )


def _is_loopback_bind(bind: str) -> bool:
    if bind.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(bind).is_loopback
    except ValueError:
        return False


def _validate_token(token: str, *, lan: bool) -> str:
    if not isinstance(token, str):
        raise ValueError("World Worker token must be text")
    value = token.strip()
    minimum = 32 if lan else 16
    if len(value) < minimum:
        raise ValueError(f"World Worker token must contain at least {minimum} characters")
    if any(character.isspace() or ord(character) < 33 for character in value):
        raise ValueError("World Worker token must not contain whitespace or control characters")
    return value


def _load_token(*, token_file: str | None, bind: str) -> tuple[str, str]:
    environment_value = os.environ.get(TOKEN_ENVIRONMENT_VARIABLE)
    if token_file and environment_value:
        raise ValueError(f"use either --token-file or {TOKEN_ENVIRONMENT_VARIABLE}, not both")
    if token_file:
        path = Path(token_file).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError("--token-file must identify a regular file")
        token = path.read_text(encoding="utf-8").strip()
        source = "file"
    elif environment_value:
        token = environment_value.strip()
        source = "environment"
    else:
        raise ValueError(
            f"set {TOKEN_ENVIRONMENT_VARIABLE} or pass --token-file; plaintext token CLI args are unsupported"
        )
    return _validate_token(token, lan=not _is_loopback_bind(bind)), source


def create_server(
    *,
    bind: str,
    port: int,
    token: str,
    worker: WorldWorker,
    allow_lan: bool = False,
    log_requests: bool = False,
) -> WorldWorkerHTTPServer:
    loopback = _is_loopback_bind(bind)
    if not loopback and not allow_lan:
        raise ValueError("non-loopback bind requires --allow-lan")
    if not 0 <= int(port) <= 65535:
        raise ValueError("port must be in [0, 65535]")
    checked_token = _validate_token(token, lan=not loopback)
    return WorldWorkerHTTPServer(
        (bind, int(port)),
        worker,
        token=checked_token,
        log_requests=log_requests,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the authenticated CARLA 0.9.16 LAN World Worker"
    )
    parser.add_argument("--bind", default=DEFAULT_BIND)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--allow-lan", action="store_true")
    parser.add_argument("--token-file")
    parser.add_argument("--carla-host", default=DEFAULT_CARLA_HOST)
    parser.add_argument("--carla-port", type=int, default=DEFAULT_CARLA_PORT)
    parser.add_argument("--traffic-manager-port", type=int, default=DEFAULT_TRAFFIC_MANAGER_PORT)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--map-load-timeout", type=float, default=DEFAULT_MAP_LOAD_TIMEOUT)
    parser.add_argument(
        "--map-reconnect-seconds",
        type=float,
        default=DEFAULT_MAP_RECONNECT_SECONDS,
    )
    parser.add_argument("--lease-seconds", type=float, default=30.0)
    parser.add_argument("--control-timeout", type=float, default=0.75)
    parser.add_argument("--expected-carla-version", default=EXPECTED_CARLA_VERSION)
    parser.add_argument("--log-requests", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be in [0, 65535]")
    if not 1 <= args.carla_port <= 65535:
        parser.error("--carla-port must be in [1, 65535]")
    if not 1 <= args.traffic_manager_port <= 65535:
        parser.error("--traffic-manager-port must be in [1, 65535]")
    if not 0.1 <= args.timeout <= 300.0:
        parser.error("--timeout must be in [0.1, 300]")
    if not 5.0 <= args.map_load_timeout <= 600.0:
        parser.error("--map-load-timeout must be in [5, 600]")
    if not 1.0 <= args.map_reconnect_seconds <= 300.0:
        parser.error("--map-reconnect-seconds must be in [1, 300]")
    if not 2.0 <= args.lease_seconds <= 3600.0:
        parser.error("--lease-seconds must be in [2, 3600]")
    if not 0.1 <= args.control_timeout <= 10.0:
        parser.error("--control-timeout must be in [0.1, 10]")
    if not _is_loopback_bind(args.bind) and not args.allow_lan:
        parser.error("non-loopback --bind requires --allow-lan")
    return args


def _parse_internal_map_load_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse the private one-shot child contract used for map transitions."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--carla-host", required=True)
    parser.add_argument("--carla-port", type=int, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--expected-carla-version", required=True)
    parser.add_argument("--target-map", required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.carla_port <= 65535:
        parser.error("--carla-port must be in [1, 65535]")
    if not 5.0 <= args.timeout <= 600.0:
        parser.error("--timeout must be in [5, 600]")
    target = str(args.target_map).strip()
    if not _MAP_NAME.fullmatch(target):
        parser.error("--target-map is invalid")
    args.target_map = target
    return args


def _internal_map_load_main(argv: Sequence[str]) -> int:
    """Run only CARLA's crash-prone ``load_world`` RPC in a child process."""

    args = _parse_internal_map_load_args(argv)
    try:
        carla = _default_carla_loader()
        client = carla.Client(args.carla_host, args.carla_port)
        client.set_timeout(args.timeout)
        client_version = str(client.get_client_version())
        server_version = str(client.get_server_version())
        expected = str(args.expected_carla_version)
        if expected and (client_version != expected or server_version != expected):
            raise RuntimeError(
                "CARLA client/server version mismatch: "
                f"expected={expected}, client={client_version}, server={server_version}"
            )
        world = client.load_world(args.target_map)
        actual = str(world.get_map().name)
        if _map_short_name(actual) != _map_short_name(args.target_map):
            raise RuntimeError(
                f"CARLA loaded {_map_short_name(actual)!r}, "
                f"expected {_map_short_name(args.target_map)!r}"
            )
    except Exception as error:
        detail = f"{type(error).__name__}: {error}".replace("\r", " ").replace("\n", " ")
        print(f"isolated CARLA map load failed: {detail[:1200]}", file=sys.stderr, flush=True)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    child_argv = list(sys.argv[1:] if argv is None else argv)
    if child_argv[:1] == ["--internal-map-load"]:
        return _internal_map_load_main(child_argv[1:])
    args = parse_args(argv)
    try:
        token, token_source = _load_token(token_file=args.token_file, bind=args.bind)
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(f"World Worker token configuration failed: {error}") from error
    worker = WorldWorker(
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        traffic_manager_port=args.traffic_manager_port,
        timeout=args.timeout,
        map_load_timeout=args.map_load_timeout,
        map_reconnect_seconds=args.map_reconnect_seconds,
        lease_seconds=args.lease_seconds,
        control_timeout=args.control_timeout,
        expected_carla_version=args.expected_carla_version,
    )
    server = create_server(
        bind=args.bind,
        port=args.port,
        token=token,
        worker=worker,
        allow_lan=args.allow_lan,
        log_requests=args.log_requests,
    )
    address, port = server.server_address[:2]
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "worker_api_revision": WORKER_API_REVISION,
                "status": "listening",
                "bind": address,
                "port": port,
                "carla_endpoint": {
                    "host": args.carla_host,
                    "port": args.carla_port,
                },
                "map_load_timeout": args.map_load_timeout,
                "map_reconnect_seconds": args.map_reconnect_seconds,
                "token_source": token_source,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        worker.close()
    return 0


__all__ = [
    "DEFAULT_BIND",
    "DEFAULT_PORT",
    "EXPECTED_CARLA_VERSION",
    "SCHEMA_VERSION",
    "SceneConfig",
    "WorkerError",
    "WorldWorker",
    "WorldWorkerHTTPServer",
    "WorldWorkerRequestHandler",
    "create_server",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    raise SystemExit(main())
