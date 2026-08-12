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
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlparse

SCHEMA_VERSION = "1.0"
EXPECTED_CARLA_VERSION = "0.9.16"
DEFAULT_BIND = "127.0.0.1"
DEFAULT_PORT = 8766
DEFAULT_CARLA_HOST = "127.0.0.1"
DEFAULT_CARLA_PORT = 2000
DEFAULT_TRAFFIC_MANAGER_PORT = 8000
TOKEN_ENVIRONMENT_VARIABLE = "CARLA_WORLD_WORKER_TOKEN"

_MAX_BODY_BYTES = 64 * 1024
_MAP_NAME = re.compile(r"^[A-Za-z0-9_./-]{1,160}$")
_VEHICLE_BLUEPRINT = re.compile(r"^vehicle\.[A-Za-z0-9_.-]{1,150}$")
_COLOR = re.compile(r"^\d{1,3},\d{1,3},\d{1,3}$")
_SCENE_PATH = re.compile(
    r"^/v1/scenes/(?P<scene_id>[A-Za-z0-9_-]{16,128})/"
    r"(?P<action>start|heartbeat|control|mode|weather|stop)$"
)
_ACTIVE_SCENE_STATES = frozenset({"prepared", "running", "stopping"})
_ROUTE_MODES = frozenset({"free", "random_destination"})
_CONTROL_MODES = frozenset({"manual", "autopilot"})
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
        }


@dataclass
class OwnedActor:
    actor: Any
    actor_id: int
    type_id: str
    kind: str
    role_name: str | None = None


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
    lease_deadline: float = 0.0
    last_control_at: float | None = None
    last_control_sequence: int = -1
    deadman_active: bool = True
    stop_reason: str | None = None
    cleanup_guard_passed: bool | None = None
    cleanup_errors: list[str] = field(default_factory=list)


class WorldWorker:
    """Thread-safe CARLA scene owner used by the HTTP adapter."""

    def __init__(
        self,
        *,
        carla_host: str = DEFAULT_CARLA_HOST,
        carla_port: int = DEFAULT_CARLA_PORT,
        traffic_manager_port: int = DEFAULT_TRAFFIC_MANAGER_PORT,
        timeout: float = 5.0,
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
    ) -> None:
        if not 1 <= int(carla_port) <= 65535:
            raise ValueError("carla_port must be in [1, 65535]")
        if not 1 <= int(traffic_manager_port) <= 65535:
            raise ValueError("traffic_manager_port must be in [1, 65535]")
        if not 0.1 <= float(timeout) <= 300.0:
            raise ValueError("timeout must be in [0.1, 300]")
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
        self.lease_seconds = float(lease_seconds)
        self.control_timeout = float(control_timeout)
        self.expected_carla_version = str(expected_carla_version)
        self._carla_loader = carla_loader
        self._route_planner_loader = route_planner_loader
        self._clock = clock
        self._monitor_period = float(monitor_period)
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

    def _planner_factory(self) -> Callable[[Any], Any] | None:
        if not self._route_planner_checked:
            try:
                self._route_planner_factory = self._route_planner_loader()
            except Exception:
                self._route_planner_factory = None
            self._route_planner_checked = True
        return self._route_planner_factory

    def _capabilities(self, client: Any | None = None) -> dict[str, bool]:
        random_route = False
        if client is not None and self._planner_factory() is not None:
            try:
                random_route = hasattr(
                    client.get_trafficmanager(self.traffic_manager_port), "set_path"
                )
            except Exception:
                random_route = False
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

        with self._lock:
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
                "status": status,
                "ready": ready,
                "error_code": error_code,
                "carla": facts,
                "active_scene": self._scene_summary(self._scene),
                "capabilities": capabilities,
            }

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
        for index in indices:
            if len(actors) >= count:
                break
            blueprint = rng.choice(blueprints)
            role_name = self._set_role(blueprint, f"world_worker_npc_{scene_id}")
            self._set_random_attribute(blueprint, "color", rng)
            self._set_random_attribute(blueprint, "driver_id", rng)
            try:
                actor = world.try_spawn_actor(blueprint, spawn_points[index])
            except Exception:
                actor = None
            if actor is None:
                continue
            self._record_actor(owned, actor, kind="traffic", role_name=role_name)
            actor.set_autopilot(True, int(traffic_manager.get_port()))
            if hasattr(traffic_manager, "update_vehicle_lights"):
                traffic_manager.update_vehicle_lights(actor, True)
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
        speeds: list[float] = []
        for _ in range(count * 3):
            if len(walkers) >= count:
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
            try:
                walker = world.try_spawn_actor(blueprint, transform)
            except Exception:
                walker = None
            if walker is None:
                continue
            self._record_actor(owned, walker, kind="walker", role_name=role_name)
            try:
                controller = world.try_spawn_actor(
                    controller_blueprint,
                    self._carla.Transform(),
                    attach_to=walker,
                )
            except Exception:
                controller = None
            if controller is None:
                try:
                    walker.destroy()
                finally:
                    owned.pop()
                continue
            self._record_actor(owned, controller, kind="walker_controller")
            walkers.append(walker)
            controllers.append(controller)
            speeds.append(self._walker_speed(blueprint, rng.random() < 0.05))

        if controllers and hasattr(world, "wait_for_tick"):
            try:
                world.wait_for_tick(seconds=min(self.timeout, 5.0))
            except TypeError:
                world.wait_for_tick(min(self.timeout, 5.0))
            except Exception:
                pass
        for controller, speed in zip(controllers, speeds, strict=True):
            controller.start()
            destination = world.get_random_location_from_navigation()
            if destination is not None:
                controller.go_to_location(destination)
            controller.set_max_speed(speed)
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
                actor = world.try_spawn_actor(blueprint, transform)
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
            self._record_actor(owned, actor, kind="prop", role_name=role_name)
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
            traffic_manager = client.get_trafficmanager(self.traffic_manager_port)
            capabilities = self._capabilities(client)
            if config.route_mode == "random_destination" and not capabilities["random_route"]:
                raise WorkerError(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    "random_route_unavailable",
                    "random_destination requires GlobalRoutePlanner and TrafficManager.set_path",
                )

            target = self._available_map_target(client, current_world, config.map_name)
            try:
                world = current_world if target is None else client.load_world(target)
            except Exception as error:
                raise WorkerError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "map_load_failed",
                    f"CARLA map load failed: {type(error).__name__}: {error}",
                ) from error
            self._ensure_async_world(world)
            try:
                traffic_manager = client.get_trafficmanager(self.traffic_manager_port)
                traffic_manager.set_synchronous_mode(False)
                simulator_seed = config.seed % _SIMULATOR_SEED_MODULUS
                if hasattr(traffic_manager, "set_random_device_seed"):
                    traffic_manager.set_random_device_seed(simulator_seed)
                if hasattr(traffic_manager, "set_global_distance_to_leading_vehicle"):
                    traffic_manager.set_global_distance_to_leading_vehicle(2.0)
                if hasattr(traffic_manager, "global_percentage_speed_difference"):
                    traffic_manager.global_percentage_speed_difference(12.0)
                if hasattr(world, "set_pedestrians_seed"):
                    world.set_pedestrians_seed((simulator_seed + 1) % _SIMULATOR_SEED_MODULUS)
                if hasattr(world, "set_pedestrians_cross_factor"):
                    world.set_pedestrians_cross_factor(0.2)
                original_weather = world.get_weather()
            except Exception as error:
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
                partial.prop_actors = self._spawn_props(
                    scene_id,
                    world,
                    config.prop_preset,
                    ego.get_transform(),
                    owned,
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
            self._refresh_lease(scene)
            return self._scene_response(scene, scene.status)

    def stop(self, scene_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        lease_token = self._lease_token(raw, allowed={"lease_token"})
        with self._lock:
            scene = self._require_scene(scene_id, lease_token)
            snapshot = self._cleanup_resources(scene, reason="operator_stop")
            self._last_scene = snapshot
            self._scene = None
            return {
                "schema_version": SCHEMA_VERSION,
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
            "ego_actor_id": None if scene.ego is None else int(scene.ego.id),
            "map_name": scene.map_name,
            "spawn_index": scene.spawn_index,
            "route_mode": scene.config.route_mode,
            "route": dict(scene.route),
            "destination": scene.destination,
            "control_mode": scene.control_mode,
            "weather_preset": scene.weather_preset,
            "traffic_count": len(scene.vehicle_actors),
            "walker_count": len(scene.walker_actors),
            "prop_actor_ids": [int(actor.id) for actor in scene.prop_actors],
            "lease_expires_in_seconds": max(0.0, scene.lease_deadline - now),
            "control_input_age_seconds": input_age,
            "deadman_active": scene.deadman_active,
            "last_control_sequence": scene.last_control_sequence,
            "stop_reason": scene.stop_reason,
            "cleanup_guard_passed": scene.cleanup_guard_passed,
            "cleanup_errors": list(scene.cleanup_errors),
            "capabilities": self._capabilities(scene.client),
        }

    def _scene_response(self, scene: SceneLease, status: str) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
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
            return
        if str(getattr(current, "type_id", "")) != owned.type_id:
            raise RuntimeError(f"actor {owned.actor_id} type changed; refusing destroy")
        if owned.role_name is not None:
            current_role = self._actor_attribute(current, "role_name")
            if current_role != owned.role_name:
                raise RuntimeError(f"actor {owned.actor_id} role changed; refusing destroy")
        if hasattr(current, "is_alive") and not bool(current.is_alive):
            return
        current.destroy()

    def _cleanup_resources(self, scene: SceneLease, *, reason: str) -> dict[str, Any]:
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
            scene.cleanup_errors.append(
                "CARLA episode changed; actor and weather cleanup intentionally skipped"
            )
            scene.status = "stopped"
            return self._scene_snapshot(scene)

        if scene.ego is not None:
            try:
                scene.ego.set_autopilot(False, int(scene.traffic_manager.get_port()))
                self._apply_full_brake(scene.ego)
            except Exception as error:
                scene.cleanup_errors.append(f"ego stop failed: {error}")
        for controller in scene.walker_controllers:
            try:
                controller.stop()
            except Exception as error:
                scene.cleanup_errors.append(f"walker controller stop failed: {error}")
        for owned in reversed(scene.owned_actors):
            try:
                self._destroy_owned_actor(current_world, owned)
            except Exception as error:
                scene.cleanup_errors.append(f"destroy {owned.kind} {owned.actor_id}: {error}")
        try:
            current_world.set_weather(scene.original_weather)
        except Exception as error:
            scene.cleanup_errors.append(f"weather restore failed: {error}")
        try:
            scene.traffic_manager.set_synchronous_mode(False)
        except Exception as error:
            scene.cleanup_errors.append(f"Traffic Manager async restore failed: {error}")
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
            if self._closed:
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

    def _error(self, error: WorkerError) -> None:
        self._send_json(
            error.status,
            {
                "schema_version": SCHEMA_VERSION,
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
            if parsed.path == "/v1/health":
                self._send_json(HTTPStatus.OK, self.server.worker.health())
            elif parsed.path == "/v1/catalog":
                self._send_json(HTTPStatus.OK, self.server.worker.catalog())
            elif parsed.path == "/v1/scenes/current":
                self._send_json(HTTPStatus.OK, self.server.worker.current_scene())
            else:
                raise WorkerError(HTTPStatus.NOT_FOUND, "route_not_found", "route not found")
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
    if not 2.0 <= args.lease_seconds <= 3600.0:
        parser.error("--lease-seconds must be in [2, 3600]")
    if not 0.1 <= args.control_timeout <= 10.0:
        parser.error("--control-timeout must be in [0.1, 10]")
    if not _is_loopback_bind(args.bind) and not args.allow_lan:
        parser.error("non-loopback --bind requires --allow-lan")
    return args


def main(argv: Sequence[str] | None = None) -> int:
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
                "status": "listening",
                "bind": address,
                "port": port,
                "carla_endpoint": {
                    "host": args.carla_host,
                    "port": args.carla_port,
                },
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
