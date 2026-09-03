"""Small real-CARLA preview used by the Garage vehicle selector.

The preview deliberately reuses the native World Worker for world and ego
ownership.  This process owns only one unparented RGB camera and its stream.
It never exposes a generic CARLA RPC endpoint and it never actuates the ego.
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import cv2

from ..bridge import (
    CarlaCameraStream,
    CarlaRpc,
    garage_camera_preset_transform,
    garage_orbit_camera_transform,
    spawn_unparented_rgb_camera,
)
from .drive_contracts import DriveStartConfig
from .world_worker_client import (
    WorldWorkerCameraStream,
    WorldWorkerClient,
    WorldWorkerScene,
)

_ACTIVE_DRIVE_STATES = frozenset({"starting", "running", "stopping"})
_VEHICLE_BLUEPRINT = re.compile(r"^vehicle\.[A-Za-z0-9_.-]{1,150}$")
_MAP_NAME = re.compile(r"^[A-Za-z0-9_./-]{1,160}$")
_PRESET_NAME = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_COLOR = re.compile(r"^\d{1,3},\d{1,3},\d{1,3}$")
_HEARTBEAT_SECONDS = 2.0
_HEARTBEAT_FAILURE_LIMIT = 3
_EXPECTED_CARLA_VERSION = "0.9.16"
_CAMERA_PROFILES: dict[str, tuple[int, int, float]] = {
    "balanced": (1280, 720, 30.0),
    "high-refresh": (1280, 720, 60.0),
    "detail": (1920, 1080, 30.0),
    "compatibility": (640, 384, 10.0),
}


def _strict_keys(
    raw: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    name: str,
) -> None:
    keys = {str(key) for key in raw}
    unknown = sorted(keys - allowed)
    missing = sorted(required - keys)
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(missing)}")


def _integer(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _number(
    value: Any,
    *,
    name: str,
    minimum: float = -math.inf,
    maximum: float = math.inf,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return result


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _jpeg(image: Any, quality: int = 88) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError("could not encode Garage preview JPEG")
    return encoded.tobytes()


def _serialized_actor_attribute(actor: Any, name: str) -> str | None:
    """Read one attribute from CARLA's serialized actor definition."""

    try:
        attributes = actor[2][2]
    except (IndexError, TypeError):
        return None
    if not isinstance(attributes, list):
        return None
    for attribute in attributes:
        if isinstance(attribute, list) and len(attribute) >= 3 and str(attribute[0]) == name:
            return str(attribute[2])
    return None


@dataclass(frozen=True)
class GaragePreviewConfig:
    map_name: str
    weather_preset: str
    vehicle_blueprint: str
    color: str | None
    seed: int
    traffic_count: int
    walker_count: int
    prop_preset: str
    pedestrian_crossing_factor: float = 0.2
    speed_difference_percent: float = 12.0
    following_distance_metres: float = 2.0
    spectator_mirror: bool = False
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    profile: str = "balanced"
    fov: float = 65.0
    yaw: float = 325.0
    pitch: float = -10.0
    distance: float = 6.5
    route_mode: str = "free"

    def scene_payload(self) -> dict[str, Any]:
        values = {
            name: getattr(self, name)
            for name in (
                "map_name",
                "weather_preset",
                "vehicle_blueprint",
                "color",
                "seed",
                "traffic_count",
                "walker_count",
                "prop_preset",
                "route_mode",
            )
        }
        values["initial_control_mode"] = "manual"
        for name, default in (
            ("pedestrian_crossing_factor", 0.2),
            ("speed_difference_percent", 12.0),
            ("following_distance_metres", 2.0),
        ):
            if getattr(self, name) != default:
                values[name] = getattr(self, name)
        return values

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GaragePreviewConfig":
        required = {
            "map_name",
            "weather_preset",
            "vehicle_blueprint",
            "color",
            "seed",
            "traffic_count",
            "walker_count",
            "prop_preset",
        }
        allowed = required | {
            "pedestrian_crossing_factor",
            "speed_difference_percent",
            "following_distance_metres",
            "spectator_mirror",
            "profile",
            "fov",
            "route_mode",
        }
        _strict_keys(raw, allowed=allowed, required=required, name="Garage preview request")

        map_name = str(raw["map_name"]).strip()
        if map_name != "current" and not _MAP_NAME.fullmatch(map_name):
            raise ValueError("map_name must be 'current' or a bounded CARLA map identifier")
        weather = str(raw["weather_preset"]).strip()
        if weather != "keep" and not _PRESET_NAME.fullmatch(weather):
            raise ValueError("weather_preset must be a bounded preset identifier")
        vehicle = str(raw["vehicle_blueprint"]).strip()
        if not _VEHICLE_BLUEPRINT.fullmatch(vehicle):
            raise ValueError("vehicle_blueprint must be a bounded vehicle.* identifier")

        raw_color = raw.get("color")
        color = None if raw_color in {None, ""} else str(raw_color).strip()
        if color is not None and (
            not _COLOR.fullmatch(color)
            or any(not 0 <= int(channel) <= 255 for channel in color.split(","))
        ):
            raise ValueError("color must be null or an RGB triplet such as '255,0,0'")

        prop_preset = str(raw["prop_preset"]).strip()
        if not _PRESET_NAME.fullmatch(prop_preset):
            raise ValueError("prop_preset must be a bounded preset identifier")

        profile = str(raw.get("profile", "balanced")).strip().lower()
        route_mode = str(raw.get("route_mode", "free")).strip()
        if route_mode not in {"free", "random_destination"}:
            raise ValueError("route_mode must be free or random_destination")
        try:
            width, height, fps = _CAMERA_PROFILES[profile]
        except KeyError as error:
            choices = ", ".join(_CAMERA_PROFILES)
            raise ValueError(f"profile must be one of {choices}") from error

        return cls(
            map_name=map_name,
            weather_preset=weather,
            vehicle_blueprint=vehicle,
            color=color,
            seed=_integer(raw["seed"], name="seed", minimum=0, maximum=2**63 - 1),
            traffic_count=_integer(
                raw["traffic_count"], name="traffic_count", minimum=0, maximum=250
            ),
            walker_count=_integer(raw["walker_count"], name="walker_count", minimum=0, maximum=250),
            prop_preset=prop_preset,
            pedestrian_crossing_factor=_number(
                raw.get("pedestrian_crossing_factor", 0.2),
                name="pedestrian_crossing_factor",
                minimum=0.0,
                maximum=1.0,
            ),
            speed_difference_percent=_number(
                raw.get("speed_difference_percent", 12.0),
                name="speed_difference_percent",
                minimum=-100.0,
                maximum=100.0,
            ),
            following_distance_metres=_number(
                raw.get("following_distance_metres", 2.0),
                name="following_distance_metres",
                minimum=0.1,
                maximum=20.0,
            ),
            spectator_mirror=_boolean(raw.get("spectator_mirror", False), name="spectator_mirror"),
            width=width,
            height=height,
            fps=fps,
            profile=profile,
            route_mode=route_mode,
            fov=_number(
                raw.get("fov", 65.0),
                name="fov",
                minimum=30.0,
                maximum=150.0,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the exact bounded configuration sent to the preview runtime."""

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
            "pedestrian_crossing_factor": self.pedestrian_crossing_factor,
            "speed_difference_percent": self.speed_difference_percent,
            "following_distance_metres": self.following_distance_metres,
            "spectator_mirror": self.spectator_mirror,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "profile": self.profile,
            "fov": self.fov,
            "yaw": self.yaw,
            "pitch": self.pitch,
            "distance": self.distance,
        }


@dataclass(frozen=True)
class GarageOrbitRequest:
    sequence: int
    yaw: float
    pitch: float
    distance: float
    preset: str = "orbit"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GarageOrbitRequest":
        fields = {"sequence", "yaw", "pitch", "distance", "preset"}
        _strict_keys(
            raw,
            allowed=fields,
            required={"sequence", "yaw", "pitch", "distance"},
            name="Garage orbit request",
        )
        preset = str(raw.get("preset", "orbit")).strip().lower()
        if preset not in {"orbit", "front", "rear", "top", "cockpit"}:
            raise ValueError("preset must be orbit, front, rear, top, or cockpit")
        return cls(
            sequence=_integer(raw["sequence"], name="sequence", minimum=0, maximum=2**63 - 1),
            yaw=_number(raw["yaw"], name="yaw") % 360.0,
            pitch=min(15.0, max(-25.0, _number(raw["pitch"], name="pitch"))),
            distance=min(10.0, max(3.5, _number(raw["distance"], name="distance"))),
            preset=preset,
        )


class GaragePreviewSession:
    """Own one parked worker ego and one Mac-side RGB preview camera."""

    def __init__(
        self,
        config: GaragePreviewConfig,
        *,
        carla_host: str,
        carla_port: int,
        world_worker: WorldWorkerClient,
        rpc_factory: Callable[..., CarlaRpc] = CarlaRpc,
        stream_factory: Callable[..., CarlaCameraStream] = CarlaCameraStream,
    ) -> None:
        self.config = config
        self.carla_host = str(carla_host)
        self.carla_port = int(carla_port)
        self.world_worker = world_worker
        self._rpc_factory = rpc_factory
        self._stream_factory = stream_factory
        self._lock = threading.RLock()
        self._frame_condition = threading.Condition(self._lock)
        self._lifecycle_lock = threading.RLock()
        self._worker_request_lock = threading.Lock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._frame_stop = threading.Event()
        self._frame_thread: threading.Thread | None = None
        self._scene: WorldWorkerScene | None = None
        self._rpc: CarlaRpc | None = None
        self._stream: CarlaCameraStream | WorldWorkerCameraStream | None = None
        self._worker_camera = False
        self._camera_transport = "pending"
        self._camera_width = config.width
        self._camera_height = config.height
        self._episode_id: int | None = None
        self._vehicle_id: int | None = None
        self._camera_id: int | None = None
        self._camera_type: str | None = None
        self._spectator_id: int | None = None
        self._spectator_transform: list[Any] | None = None
        self._spectator_mirror_active = False
        self._status = "starting"
        self._error: str | None = None
        self._cleanup_errors: list[str] = []
        self._yaw = config.yaw
        self._pitch = config.pitch
        self._distance = config.distance
        self._camera_preset = "orbit"
        self._last_orbit_sequence = -1
        self._frame_sequence = -1
        self._source_frame_sequence = -1
        self._frame_sequence_offset = 0
        self._updating = False
        self._configuration_confirmed = True
        self._reader_needs_restart = False
        self._jpeg: bytes | None = None
        self._frame_received_monotonic: float | None = None
        self._frame_arrivals: deque[float] = deque(maxlen=300)
        self._closed = False

    def start(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Garage preview session is already closed")
            try:
                with self._worker_request_lock:
                    scene = self.world_worker.prepare_scene(self.config.scene_payload())
                if scene.ego_actor_id is None or scene.episode_id is None:
                    raise RuntimeError(
                        "World Worker prepared Garage preview without an ego episode"
                    )
                with self._lock:
                    self._scene = scene
                    self._episode_id = scene.episode_id
                    self._vehicle_id = scene.ego_actor_id
                self._start_heartbeat()

                rpc = self._rpc_factory(self.carla_host, self.carla_port, timeout=5.0)
                self._rpc = rpc
                server_version = str(rpc.value_call("version"))
                if server_version != _EXPECTED_CARLA_VERSION:
                    raise RuntimeError(
                        "Garage preview requires CARLA "
                        f"{_EXPECTED_CARLA_VERSION}, got {server_version}"
                    )
                if rpc.episode_id() != scene.episode_id:
                    raise RuntimeError("Garage preview camera episode does not match World Worker")
                vehicle = rpc.actor(scene.ego_actor_id)
                if vehicle is None or not str(vehicle[2][1]).startswith("vehicle."):
                    raise RuntimeError(
                        "World Worker Garage preview ego did not verify as a vehicle"
                    )
                vehicle_transform = rpc.actor_transform(scene.ego_actor_id, "VehicleMesh")
                camera_transform = garage_orbit_camera_transform(
                    vehicle_transform,
                    azimuth_degrees=self._yaw,
                    pitch_degrees=self._pitch,
                    distance=self._distance,
                )
                worker_camera = bool(scene.capabilities.get("compressed_camera_relay"))
                if worker_camera:
                    camera_response = self.world_worker.start_camera(
                        scene,
                        mode="garage",
                        width=self.config.width,
                        height=self.config.height,
                        fps=self.config.fps,
                        fov=self.config.fov,
                        yaw=self._yaw,
                        pitch=self._pitch,
                        distance=self._distance,
                    )
                    camera_payload = camera_response.get("camera")
                    if not isinstance(camera_payload, Mapping):
                        raise RuntimeError("World Worker camera response is malformed")
                    camera_id = int(camera_payload["actor_id"])
                    stream = WorldWorkerCameraStream(
                        self.world_worker,
                        scene,
                        timeout=max(8.0, 4.0 / self.config.fps),
                    )
                    camera_transport = str(getattr(stream, "transport", "worker_jpeg_long_poll"))
                else:
                    if self.config.profile != "compatibility":
                        raise RuntimeError(
                            "the selected Garage video profile requires the persistent "
                            "compressed World Worker camera; use Compatibility mode"
                        )
                    # An older Worker leaves the camera on CARLA's raw BGRA
                    # stream. Cap only that compatibility lane so the first
                    # frame can cross an ordinary LAN; upgraded Workers retain
                    # the selected worker-side compressed camera profile.
                    raw_width = min(self.config.width, 960)
                    raw_height = min(self.config.height, 540)
                    camera = spawn_unparented_rgb_camera(
                        rpc,
                        camera_transform,
                        role_name="garage_preview",
                        width=raw_width,
                        height=raw_height,
                        sensor_tick=1.0 / self.config.fps,
                        fov=self.config.fov,
                    )
                    if not isinstance(camera, list) or len(camera) < 6 or not camera[5]:
                        raise RuntimeError("Garage preview camera did not expose a stream token")
                    camera_id = int(camera[0])
                    verified_camera = rpc.actor(camera_id)
                    if verified_camera is None:
                        raise RuntimeError("Garage preview camera could not be verified")
                    camera_type = str(verified_camera[2][1])
                    if camera_type != "sensor.camera.rgb":
                        raise RuntimeError(
                            f"Garage preview actor {camera_id} has unexpected type {camera_type!r}"
                        )
                    camera_role = _serialized_actor_attribute(verified_camera, "role_name")
                    if camera_role != "garage_preview":
                        raise RuntimeError("Garage preview camera role_name could not be verified")
                    stream = self._stream_factory(
                        self.carla_host,
                        camera[5],
                        timeout=max(8.0, 4.0 / self.config.fps),
                    )
                    camera_transport = "carla_raw_bgra_fallback"
                with self._lock:
                    self._camera_id = camera_id
                    self._camera_type = "sensor.camera.rgb"
                    self._worker_camera = worker_camera
                    self._camera_transport = camera_transport
                    self._camera_width = self.config.width if worker_camera else raw_width
                    self._camera_height = self.config.height if worker_camera else raw_height
                # Town*_Opt can stream tiles around the spectator. Until the
                # Worker-side camera relay is deployed, keep the server view at
                # the Garage camera so the raw sensor's tile remains loaded.
                # The exact previous spectator transform is restored on close.
                mirror_spectator = self.config.spectator_mirror or not worker_camera
                if mirror_spectator:
                    try:
                        spectator = rpc.spectator()
                        spectator_id = int(spectator[0])
                        spectator_transform = rpc.actor_transform(spectator_id, "Camera")
                        rpc.set_actor_transform(spectator_id, camera_transform)
                        with self._lock:
                            self._spectator_id = spectator_id
                            self._spectator_transform = spectator_transform
                            self._spectator_mirror_active = True
                    except Exception as error:
                        self._cleanup_errors.append(f"spectator preview setup: {error}")
                self._stream = stream
                frame = stream.wait_for_frame(timeout=20.0)
                self._cache_frame(frame)
                self._start_frame_pump()
                with self._lock:
                    self._status = "running"
                return self.snapshot()
            except BaseException as error:
                with self._lock:
                    self._error = f"{type(error).__name__}: {error}"
                    self._status = "failed"
                self.close(reason="start_failed")
                raise

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            scene = self._scene
            source_fps = 0.0
            if len(self._frame_arrivals) >= 2:
                elapsed = self._frame_arrivals[-1] - self._frame_arrivals[0]
                if elapsed > 0.0:
                    source_fps = (len(self._frame_arrivals) - 1) / elapsed
            frame_age = (
                None
                if self._frame_received_monotonic is None
                else max(0.0, time.monotonic() - self._frame_received_monotonic)
            )
            stale_after = max(0.25, 3.0 / self.config.fps)
            stream = {
                "transport": self._camera_transport,
                "profile": self.config.profile,
                "resolution": f"{self._camera_width}x{self._camera_height}",
                "target_fps": self.config.fps,
                "source_fps": round(source_fps, 1),
                "frame_age_seconds": None if frame_age is None else round(frame_age, 3),
                "stale": frame_age is None or frame_age > stale_after,
            }
            return {
                "schema_version": "1.0",
                "applied_config": self.config.as_dict(),
                "status": self._status,
                "active": self._status in {"starting", "running"} and not self._closed,
                "updating": self._updating,
                "configuration_confirmed": self._configuration_confirmed,
                "frame_sequence": self._frame_sequence,
                "yaw": self._yaw,
                "pitch": self._pitch,
                "distance": self._distance,
                "camera_preset": self._camera_preset,
                "error": self._error,
                "cleanup_errors": list(self._cleanup_errors),
                "map": None if scene is None else scene.map_name,
                "weather_preset": self.config.weather_preset,
                "vehicle_blueprint": self.config.vehicle_blueprint,
                "color": self.config.color,
                "vehicle_id": self._vehicle_id,
                "camera_id": self._camera_id,
                "camera_transport": self._camera_transport,
                "camera_resolution": f"{self._camera_width}x{self._camera_height}",
                "camera_profile": self.config.profile,
                "camera_fov": self.config.fov,
                "camera_target_fps": self.config.fps,
                "camera_source_fps": stream["source_fps"],
                "camera_frame_age_seconds": stream["frame_age_seconds"],
                "camera_stale": stream["stale"],
                "stream": stream,
                "traffic_count": (
                    self.config.traffic_count
                    if scene is None or scene.traffic_count is None
                    else scene.traffic_count
                ),
                "walker_count": (
                    self.config.walker_count
                    if scene is None or scene.walker_count is None
                    else scene.walker_count
                ),
                "traffic_count_requested": self.config.traffic_count,
                "walker_count_requested": self.config.walker_count,
                "pedestrian_crossing_factor": (
                    self.config.pedestrian_crossing_factor
                    if scene is None or scene.pedestrian_crossing_factor is None
                    else scene.pedestrian_crossing_factor
                ),
                "speed_difference_percent": (
                    self.config.speed_difference_percent
                    if scene is None or scene.speed_difference_percent is None
                    else scene.speed_difference_percent
                ),
                "following_distance_metres": (
                    self.config.following_distance_metres
                    if scene is None or scene.following_distance_metres is None
                    else scene.following_distance_metres
                ),
                "prop_preset": self.config.prop_preset,
                "spectator_mirror": self._spectator_mirror_active,
            }

    def frame(self) -> tuple[int, bytes]:
        with self._lifecycle_lock:
            with self._lock:
                if self._status != "running" or self._closed:
                    raise FileNotFoundError("Garage preview frame is not available")
                stream = self._stream
            if stream is None:
                raise FileNotFoundError("Garage preview frame is not available")
            with self._lock:
                if self._jpeg is None:
                    raise FileNotFoundError("Garage preview frame is not ready")
                return self._frame_sequence, self._jpeg

    def wait_for_frame(self, after_sequence: int, *, timeout: float = 5.0) -> tuple[int, bytes]:
        with self._frame_condition:
            ready = self._frame_condition.wait_for(
                lambda: (
                    self._frame_sequence > after_sequence
                    or self._closed
                    or self._status == "failed"
                ),
                timeout=timeout,
            )
            if not ready:
                raise TimeoutError("timed out waiting for Garage preview frame")
            if self._frame_sequence > after_sequence and self._jpeg is not None:
                return self._frame_sequence, self._jpeg
            raise EOFError("Garage preview stream ended")

    def orbit(self, request: GarageOrbitRequest) -> dict[str, Any]:
        with self._lifecycle_lock:
            with self._lock:
                if self._status != "running" or self._closed:
                    raise RuntimeError("Garage preview is not running")
                if request.sequence <= self._last_orbit_sequence:
                    raise ValueError("orbit sequence must be newer than the previous request")
                rpc = self._rpc
                episode_id = self._episode_id
                vehicle_id = self._vehicle_id
                camera_id = self._camera_id
                worker_camera = self._worker_camera
            if rpc is None or episode_id is None or vehicle_id is None or camera_id is None:
                raise RuntimeError("Garage preview actors are not ready")
            if rpc.episode_id() != episode_id:
                raise RuntimeError("CARLA episode changed while Garage preview was active")
            vehicle_transform = rpc.actor_transform(vehicle_id, "VehicleMesh")
            transform = garage_camera_preset_transform(
                vehicle_transform,
                request.preset,
                azimuth_degrees=request.yaw,
                pitch_degrees=request.pitch,
                distance=request.distance,
            )
            if worker_camera:
                with self._worker_request_lock:
                    scene = self._scene
                    if scene is None:
                        raise RuntimeError("Garage preview Worker scene is not ready")
                    self.world_worker.orbit_camera(
                        scene,
                        yaw=request.yaw,
                        pitch=request.pitch,
                        distance=request.distance,
                        preset=request.preset,
                    )
            else:
                camera = rpc.actor(camera_id)
                if (
                    camera is None
                    or str(camera[2][1]) != "sensor.camera.rgb"
                    or _serialized_actor_attribute(camera, "role_name") != "garage_preview"
                ):
                    raise RuntimeError("Garage preview camera ownership could not be verified")
                rpc.set_actor_transform(camera_id, transform)
            with self._lock:
                self._last_orbit_sequence = request.sequence
                self._yaw = request.yaw
                self._pitch = request.pitch
                self._distance = request.distance
                self._camera_preset = request.preset
                spectator_id = self._spectator_id
                spectator_active = self._spectator_mirror_active
            if spectator_id is not None and spectator_active:
                try:
                    current_spectator = rpc.spectator()
                    if int(current_spectator[0]) != spectator_id:
                        raise RuntimeError("CARLA spectator actor changed")
                    rpc.set_actor_transform(spectator_id, transform)
                except Exception as error:
                    with self._lock:
                        self._spectator_mirror_active = False
                    self._cleanup_errors.append(f"spectator preview orbit: {error}")
            return self.snapshot()

    def update_weather(self, weather_preset: str) -> dict[str, Any]:
        """Update weather without replacing the active scene or camera."""

        preset = str(weather_preset).strip()
        if preset != "keep" and not _PRESET_NAME.fullmatch(preset):
            raise ValueError("weather_preset must be a bounded preset identifier")
        with self._lifecycle_lock:
            with self._lock:
                if self._status != "running" or self._closed:
                    raise RuntimeError("Garage preview is not running")
                if preset == self.config.weather_preset:
                    return self.snapshot()
                scene = self._scene
            if scene is None:
                raise RuntimeError("Garage preview Worker scene is not ready")
            if preset != "keep":
                with self._worker_request_lock:
                    updated = self.world_worker.weather(scene, preset)
                with self._lock:
                    self._scene = updated
            with self._lock:
                self.config = replace(self.config, weather_preset=preset)
                return self.snapshot()

    def can_reconfigure(self, config: GaragePreviewConfig) -> bool:
        """Only map/seed changes (or an older Worker) require a new scene."""
        with self._lock:
            scene = self._scene
            if (
                self._closed
                or self._status != "running"
                or scene is None
                or scene.status != "prepared"
                or not self._worker_camera
            ):
                return False
            if config.map_name != self.config.map_name or config.seed != self.config.seed:
                return False
            if config.scene_payload() != self.config.scene_payload() and not scene.capabilities.get(
                "prepared_scene_reconfigure"
            ):
                return False
            camera_changed = any(
                getattr(config, key) != getattr(self.config, key)
                for key in ("width", "height", "fps", "fov")
            )
            return not camera_changed or bool(scene.capabilities.get("prepared_scene_handoff"))

    def reconfigure(self, config: GaragePreviewConfig) -> dict[str, Any]:
        """Mutate the prepared scene in place; keep its browser stream and lease."""
        with self._lifecycle_lock:
            if not self.can_reconfigure(config):
                raise RuntimeError("Garage update requires a new prepared scene")
            with self._lock:
                self._updating = True
                previous = self.config
                scene = self._scene
            assert scene is not None
            try:
                if config.scene_payload() != previous.scene_payload():
                    with self._worker_request_lock:
                        updated = self.world_worker.configure_scene(scene, config.scene_payload())
                        if (
                            updated.scene_id != scene.scene_id
                            or updated.episode_id != scene.episode_id
                            or updated.status != "prepared"
                            or updated.ego_actor_id is None
                        ):
                            raise RuntimeError("Worker did not preserve the prepared Garage scene")
                        with self._lock:
                            self._scene = updated
                            self._vehicle_id = updated.ego_actor_id
                            # Native mutations are already committed. Preserve truthful
                            # scene settings even if a following camera replacement fails.
                            self.config = replace(
                                config,
                                **{
                                    key: getattr(previous, key)
                                    for key in (
                                        "width",
                                        "height",
                                        "fps",
                                        "fov",
                                        "profile",
                                        "spectator_mirror",
                                        "yaw",
                                        "pitch",
                                        "distance",
                                    )
                                },
                            )
                    scene = updated
                camera_changed = self._camera_id is None or any(
                    getattr(config, key) != getattr(previous, key)
                    for key in ("width", "height", "fps", "fov")
                )
                if camera_changed:
                    self._replace_camera(config, scene)
                if (
                    camera_changed
                    or config.vehicle_blueprint != previous.vehicle_blueprint
                    or config.color != previous.color
                    or self._error is not None
                ):
                    # Preserve the user's selected view; use the Worker's existing
                    # vehicle-size-aware camera transform, not a second geometry model.
                    with self._worker_request_lock:
                        self.world_worker.orbit_camera(
                            scene,
                            yaw=self._yaw,
                            pitch=self._pitch,
                            distance=self._distance,
                            preset=self._camera_preset,
                        )
                if config.spectator_mirror != previous.spectator_mirror:
                    self._update_spectator(config.spectator_mirror)
                elif self._spectator_mirror_active:
                    self._update_spectator(True)
                with self._lock:
                    self.config = config
                    self._error = None
                    self._configuration_confirmed = True
            except BaseException as error:
                # Native operations commit individual successful deltas. A
                # timeout is not proof that no actors/settings were changed.
                self._configuration_confirmed = False
                try:
                    self.ensure_configuration_confirmed()
                except Exception as readback_error:
                    error.add_note(f"Worker configuration readback: {readback_error}")
                with self._lock:
                    self._error = f"Garage update failed: {type(error).__name__}: {error}"
                raise
            finally:
                # Consume the pump failure and end the update atomically. The
                # pump uses this same lock, so it cannot leave a restart flag
                # stranded after the mutation owner has finished.
                with self._lock:
                    restart_reader = (
                        self._reader_needs_restart
                        and self._status == "running"
                        and self._camera_id is not None
                    )
                    self._reader_needs_restart = False
                    self._updating = False
                if restart_reader:
                    self._restart_camera_reader()
            result = self.snapshot()
            result["configure_action"] = "updated"
            return result

    def ensure_configuration_confirmed(self) -> None:
        """Recover a partial/ambiguous apply before another update or Drive."""
        if self._configuration_confirmed:
            return
        with self._worker_request_lock:
            payload = self.world_worker.current_scene()
            if payload.get("status") == "preparing":
                raise RuntimeError("World Worker is still applying Garage settings")
            updated = WorldWorkerScene.from_response(payload)
            scene = self._scene
            if (
                scene is None
                or updated.scene_id != scene.scene_id
                or updated.lease_token != scene.lease_token
                or updated.episode_id != scene.episode_id
            ):
                raise RuntimeError("Worker scene ownership changed during Garage update")
            if updated.status != "prepared" or updated.ego_actor_id is None:
                self._scene = updated
                self._status = "failed"
                raise RuntimeError("World Worker no longer has a prepared Garage scene")
            raw = payload["scene"]
            confirmed = raw.get("config")
            if not isinstance(confirmed, Mapping):
                raise RuntimeError("World Worker did not return confirmed Garage settings")
            values = self.config.as_dict()
            for key in ("width", "height", "fps", "yaw", "pitch", "distance"):
                values.pop(key)
            for key in self.config.scene_payload():
                if key != "initial_control_mode":
                    values[key] = confirmed[key]
            # Include default-valued dynamics too: scene_payload omits those
            # for compatibility with older prepare endpoints.
            for key in (
                "pedestrian_crossing_factor",
                "speed_difference_percent",
                "following_distance_metres",
            ):
                values[key] = confirmed[key]
            camera = raw.get("camera_config")
            if isinstance(camera, Mapping):
                for profile, dimensions in _CAMERA_PROFILES.items():
                    if dimensions == (camera.get("width"), camera.get("height"), camera.get("fps")):
                        values["profile"] = profile
                        break
                else:
                    raise RuntimeError("Worker camera profile changed outside Garage")
                values["fov"] = camera["fov"]
            config = GaragePreviewConfig.from_mapping(values)
            with self._lock:
                self._scene = updated
                self._vehicle_id = updated.ego_actor_id
                self.config = config
                if isinstance(camera, Mapping):
                    self._camera_width, self._camera_height = config.width, config.height
                    self._camera_id = int(raw["camera"]["actor_id"])
                elif "camera_config" in raw:
                    # Native camera replacement can fail after old sensor
                    # destruction. Retry must create a sensor even when the
                    # operator selects the previous profile again.
                    self._camera_id = None
                    self._reader_needs_restart = False
                    self._frame_stop.set()
                self._configuration_confirmed = True

    def _replace_camera(self, config: GaragePreviewConfig, scene: WorldWorkerScene) -> None:
        self._stop_camera_reader()
        self._reader_needs_restart = True
        with self._worker_request_lock:
            response = self.world_worker.start_camera(
                scene,
                mode="garage",
                width=config.width,
                height=config.height,
                fps=config.fps,
                fov=config.fov,
                yaw=self._yaw,
                pitch=self._pitch,
                distance=self._distance,
            )
        camera = response.get("camera")
        if not isinstance(camera, Mapping):
            raise RuntimeError("World Worker camera response is malformed")
        with self._lock:
            self._camera_id = int(camera["actor_id"])
            self._camera_width = config.width
            self._camera_height = config.height
            self.config = replace(
                self.config,
                **{
                    key: getattr(config, key)
                    for key in (
                        "width",
                        "height",
                        "fps",
                        "fov",
                        "profile",
                    )
                },
            )
        self._restart_camera_reader()

    def _stop_camera_reader(self) -> None:
        self._frame_stop.set()
        old_stream = self._stream
        if old_stream is not None:
            old_stream.close()
        old_pump = self._frame_thread
        if old_pump is not None:
            old_pump.join(timeout=2.0)
            if old_pump.is_alive():
                raise TimeoutError("Garage frame reader did not stop for camera replacement")

    def _restart_camera_reader(self) -> None:
        self._stop_camera_reader()
        scene = self._scene
        if scene is None:
            raise RuntimeError("Garage scene is unavailable for camera stream")
        stream = WorldWorkerCameraStream(
            self.world_worker, scene, timeout=max(8.0, 4.0 / self.config.fps)
        )
        with self._lock:
            self._stream = stream
            self._frame_sequence_offset = self._frame_sequence + 1
            self._source_frame_sequence = -1
            self._frame_received_monotonic = None
            self._frame_arrivals.clear()
            self._frame_stop = threading.Event()
            self._reader_needs_restart = False
        # The browser retains the last good JPEG until this exact replacement
        # produces a frame. Do not block Apply on a second first-frame wait.
        self._start_frame_pump()

    def _update_spectator(self, enabled: bool) -> None:
        rpc = self._rpc
        if rpc is None or self._camera_id is None or rpc.episode_id() != self._episode_id:
            raise RuntimeError("Garage spectator episode is no longer available")
        if enabled:
            spectator_id = int(rpc.spectator()[0])
            if self._spectator_mirror_active and self._spectator_id != spectator_id:
                raise RuntimeError("Garage spectator actor changed")
            original = (
                self._spectator_transform
                if self._spectator_mirror_active
                else rpc.actor_transform(spectator_id, "Camera")
            )
            transform = rpc.actor_transform(self._camera_id, "Camera")
            rpc.set_actor_transform(spectator_id, transform)
            self._spectator_id = spectator_id
            self._spectator_transform = original
        elif self._spectator_id is not None and self._spectator_transform is not None:
            if int(rpc.spectator()[0]) != self._spectator_id:
                raise RuntimeError("Garage spectator actor changed")
            rpc.set_actor_transform(self._spectator_id, self._spectator_transform)
            self._spectator_id = None
            self._spectator_transform = None
        self._spectator_mirror_active = enabled

    def _start_heartbeat(self) -> None:
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="garage-preview-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _cache_frame(self, frame: Any) -> None:
        # The Garage is a visual selection surface, not a detector transport.
        # Preserve CARLA texture/detail instead of applying the old aggressive
        # JPEG compression before the frame fills a desktop or tablet screen.
        jpeg = getattr(frame, "jpeg", None)
        payload = bytes(jpeg) if isinstance(jpeg, bytes) else _jpeg(frame.bgr(), quality=94)
        with self._frame_condition:
            if frame.sequence > self._source_frame_sequence:
                self._source_frame_sequence = frame.sequence
                self._frame_sequence = self._frame_sequence_offset + frame.sequence
                self._jpeg = payload
                received = time.monotonic()
                self._frame_received_monotonic = received
                self._frame_arrivals.append(received)
                self._frame_condition.notify_all()

    def _start_frame_pump(self) -> None:
        self._frame_thread = threading.Thread(
            target=self._frame_loop,
            name="garage-preview-frames",
            daemon=True,
        )
        self._frame_thread.start()

    def _frame_loop(self) -> None:
        last_frame_at = time.monotonic()
        stale_after = max(5.0, 20.0 / self.config.fps)
        while not self._frame_stop.is_set():
            with self._lock:
                stream = self._stream
                sequence = self._source_frame_sequence
                closed = self._closed
            if stream is None or closed:
                return
            try:
                frame = stream.wait_for_frame(
                    after_sequence=sequence,
                    timeout=max(1.0, 4.0 / self.config.fps),
                )
                self._cache_frame(frame)
                last_frame_at = time.monotonic()
            except TimeoutError:
                if self._frame_stop.is_set():
                    return
                # The legacy raw-BGRA stream can pause for many seconds on a
                # loaded remote CARLA host. Keep the last valid Garage frame
                # visible and continue waiting; only the new worker-side JPEG
                # transport has a bounded freshness contract.
                if (
                    self._updating
                    or not self._worker_camera
                    or time.monotonic() - last_frame_at < stale_after
                ):
                    if self._updating:
                        last_frame_at = time.monotonic()
                    continue
                error = TimeoutError(
                    f"no Garage camera frame arrived for {stale_after:.1f} seconds"
                )
                with self._frame_condition:
                    self._error = f"Garage preview camera failed: {error}"
                    self._status = "failed"
                    self._frame_condition.notify_all()
                self.close(reason="camera_stale")
                return
            except BaseException as error:
                if self._frame_stop.is_set():
                    return
                with self._lock:
                    if self._updating:
                        # A dense native delta can exhaust transport retries.
                        # Restart only the reader, never queue scene cleanup.
                        self._reader_needs_restart = True
                        return
                with self._frame_condition:
                    self._error = f"Garage preview camera failed: {type(error).__name__}: {error}"
                    self._status = "failed"
                    self._frame_condition.notify_all()
                self.close(reason="camera_failed")
                return

    def _heartbeat_loop(self) -> None:
        consecutive_failures = 0
        transient_error: str | None = None
        while not self._heartbeat_stop.wait(_HEARTBEAT_SECONDS):
            try:
                with self._worker_request_lock:
                    with self._lock:
                        scene = self._scene
                        closed = self._closed
                    if scene is None or closed:
                        return
                    updated = self.world_worker.heartbeat(scene)
                    with self._lock:
                        self._scene = updated
                        if transient_error is not None and self._error == transient_error:
                            self._error = None
                    consecutive_failures = 0
                    transient_error = None
            except BaseException as error:
                if self._heartbeat_stop.is_set():
                    return
                consecutive_failures += 1
                message = (
                    "World Worker heartbeat failed: "
                    f"{type(error).__name__}: {error} "
                    f"({consecutive_failures}/{_HEARTBEAT_FAILURE_LIMIT})"
                )
                with self._lock:
                    self._error = message
                transient_error = message
                if consecutive_failures < _HEARTBEAT_FAILURE_LIMIT:
                    continue
                with self._lock:
                    self._status = "failed"
                self.close(reason="heartbeat_failed")
                return

    def close(self, *, reason: str = "operator_stop", release_scene: bool = True) -> dict[str, Any]:
        with self._lifecycle_lock:
            with self._lock:
                if self._closed and (
                    not release_scene or self._scene is None or self._scene.status == "stopped"
                ):
                    return self.snapshot()
                self._closed = True
                if self._status not in {"failed", "stopped"}:
                    self._status = "stopping"
                stream = self._stream
                rpc = self._rpc
                episode_id = self._episode_id
                camera_id = self._camera_id
                camera_type = self._camera_type
                worker_camera = self._worker_camera
                spectator_id = self._spectator_id
                spectator_transform = self._spectator_transform
            self._heartbeat_stop.set()
            self._frame_stop.set()
            with self._frame_condition:
                self._frame_condition.notify_all()
            heartbeat = self._heartbeat_thread
            if heartbeat is not None and heartbeat is not threading.current_thread():
                heartbeat.join(timeout=2.0)

            if stream is not None:
                try:
                    stream.close()
                except Exception as error:
                    self._cleanup_errors.append(f"camera stream close: {error}")
            frame_thread = self._frame_thread
            if frame_thread is not None and frame_thread is not threading.current_thread():
                frame_thread.join(timeout=2.0)
            if rpc is not None:
                try:
                    same_episode = episode_id is not None and rpc.episode_id() == episode_id
                except Exception as error:
                    same_episode = False
                    self._cleanup_errors.append(f"episode cleanup check: {error}")
                if same_episode and camera_id is not None and not worker_camera:
                    try:
                        actor = rpc.actor(camera_id)
                        if actor is not None:
                            actor_type = str(actor[2][1])
                            actor_role = _serialized_actor_attribute(actor, "role_name")
                            if (
                                actor_type != camera_type
                                or actor_type != "sensor.camera.rgb"
                                or actor_role != "garage_preview"
                            ):
                                self._cleanup_errors.append(
                                    "skip destroy camera "
                                    f"{camera_id}: ownership changed to {actor_type}/{actor_role}"
                                )
                            else:
                                rpc.destroy_actor(camera_id)
                    except Exception as error:
                        self._cleanup_errors.append(f"destroy camera {camera_id}: {error}")
                elif camera_id is not None and not worker_camera:
                    self._cleanup_errors.append(
                        "episode changed; skipped Garage preview camera destruction"
                    )
                if same_episode and spectator_id is not None and spectator_transform is not None:
                    try:
                        current_spectator = rpc.spectator()
                        if int(current_spectator[0]) == spectator_id:
                            rpc.set_actor_transform(spectator_id, spectator_transform)
                    except Exception as error:
                        self._cleanup_errors.append(f"spectator preview restore: {error}")
                try:
                    rpc.close()
                except Exception as error:
                    self._cleanup_errors.append(f"RPC close: {error}")

            with self._worker_request_lock:
                with self._lock:
                    scene = self._scene
                if scene is not None and release_scene:
                    try:
                        stopped = self.world_worker.stop_scene(scene)
                        with self._lock:
                            self._scene = stopped
                        self._cleanup_errors.extend(
                            f"World Worker cleanup: {error}" for error in stopped.cleanup_errors
                        )
                    except Exception as error:
                        self._cleanup_errors.append(f"World Worker stop ({reason}): {error}")
            with self._lock:
                self._stream = None
                self._rpc = None
                if self._status != "failed":
                    self._status = "stopped"
                return self.snapshot()

    def take_for_drive(self, config: DriveStartConfig) -> WorldWorkerScene | None:
        """Transfer the parked lease, not its local preview transports."""

        with self._lifecycle_lock:
            with self._lock:
                scene = self._scene
                if (
                    self._closed
                    or self._status != "running"
                    or scene is None
                    or scene.status != "prepared"
                    or self._cleanup_errors
                    or not self._configuration_confirmed
                    or self._camera_id is None
                    or scene.ego_actor_id is None
                    or scene.episode_id is None
                    or not self._worker_camera
                    or not scene.capabilities.get("prepared_scene_handoff")
                    or config.experiment_preset != "free_drive"
                ):
                    return None
                scene_fields = (
                    "map_name",
                    "weather_preset",
                    "vehicle_blueprint",
                    "color",
                    "seed",
                    "traffic_count",
                    "walker_count",
                    "prop_preset",
                    "route_mode",
                    "pedestrian_crossing_factor",
                    "speed_difference_percent",
                    "following_distance_metres",
                )
                if any(getattr(self.config, key) != getattr(config, key) for key in scene_fields):
                    return None
            self.close(reason="drive_handoff", release_scene=False)
            with self._worker_request_lock:
                try:
                    if self._cleanup_errors:
                        raise RuntimeError("Garage preview transports did not close cleanly")
                    # This also orders handoff after the final preview heartbeat
                    # and gives the new owner a full lease interval to start.
                    renewed = self.world_worker.heartbeat(scene)
                    if (
                        renewed.status != "prepared"
                        or renewed.ego_actor_id != scene.ego_actor_id
                        or renewed.episode_id != scene.episode_id
                    ):
                        raise RuntimeError("Garage scene changed before handoff")
                    scene = renewed
                except BaseException as error:
                    try:
                        stopped = self.world_worker.stop_scene(scene)
                    except BaseException as cleanup_error:
                        error.add_note(f"Garage handoff cleanup failed: {cleanup_error}")
                    else:
                        with self._lock:
                            self._scene = stopped
                    raise
                with self._lock:
                    self._scene = None
                return scene


class GaragePreviewManager:
    """Coordinate one Garage preview with the one active Drive session."""

    def __init__(
        self,
        *,
        carla_host: str,
        carla_port: int,
        world_worker: WorldWorkerClient | None,
        drive_state: Callable[[], Mapping[str, Any]],
        world_mode_lock: threading.RLock,
        session_factory: Callable[..., GaragePreviewSession] = GaragePreviewSession,
    ) -> None:
        self.carla_host = str(carla_host)
        self.carla_port = int(carla_port)
        self.world_worker = world_worker
        self._drive_state = drive_state
        self._world_mode_lock = world_mode_lock
        self._session_factory = session_factory
        self._lock = threading.RLock()
        self._configure_condition = threading.Condition()
        self._configure_running = False
        self._configure_requested_revision = 0
        self._configure_completed_revision = 0
        self._configure_requested_config: GaragePreviewConfig | None = None
        self._configure_shutdown = False
        self._configure_outcomes: dict[
            int,
            tuple[GaragePreviewConfig | None, dict[str, Any] | None, BaseException | None],
        ] = {}
        self._session: GaragePreviewSession | None = None
        self._last_error: str | None = None

    def preparation_status(self) -> dict[str, Any] | None:
        """Read Worker preparation progress without taking the Garage scene lock."""

        worker = self.world_worker
        if worker is None:
            return None
        try:
            payload = worker.current_scene()
        except Exception:
            # Progress telemetry must never become a second failure path. The
            # authoritative configure call will surface transport/CARLA errors.
            return None
        preparation = payload.get("preparation")
        if not isinstance(preparation, Mapping):
            return None
        return dict(preparation)

    def state(self) -> dict[str, Any]:
        with self._lock:
            session = self._session
            last_error = self._last_error
        if session is None:
            return {
                "schema_version": "1.0",
                "status": "idle",
                "active": False,
                "frame_sequence": -1,
                "yaw": 325.0,
                "pitch": -10.0,
                "distance": 6.5,
                "error": last_error,
                "cleanup_errors": [],
                "applied_config": None,
                "map": None,
                "weather_preset": None,
                "vehicle_blueprint": None,
                "color": None,
                "vehicle_id": None,
                "camera_id": None,
                "traffic_count": 0,
                "walker_count": 0,
                "prop_preset": None,
                "spectator_mirror": False,
            }
        return session.snapshot()

    def configure(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        config = GaragePreviewConfig.from_mapping(raw)
        if self.world_worker is None:
            raise RuntimeError("Garage preview requires a configured World Worker")
        with self._configure_condition:
            if self._configure_shutdown:
                raise RuntimeError("Garage preview manager is shutting down")
            self._configure_requested_revision += 1
            request_revision = self._configure_requested_revision
            self._configure_requested_config = config
            runner = not self._configure_running
            if runner:
                self._configure_running = True
            self._configure_condition.notify_all()

        if runner:
            self._drain_configure_requests()

        with self._configure_condition:
            while request_revision not in self._configure_outcomes:
                self._configure_condition.wait()
            applied_config, result, error = self._configure_outcomes.pop(request_revision)
        if error is not None:
            raise error
        if result is None:
            raise RuntimeError("Garage configuration completed without a result")
        response = dict(result)
        if applied_config != config:
            response["requested_config_superseded"] = True
        return response

    def _drain_configure_requests(self) -> None:
        """Apply one in-flight configuration and then only the newest pending one."""

        while True:
            with self._configure_condition:
                if self._configure_completed_revision >= self._configure_requested_revision:
                    self._configure_running = False
                    self._configure_condition.notify_all()
                    return
                target_revision = self._configure_requested_revision
                config = self._configure_requested_config
            if config is None:
                error: BaseException | None = RuntimeError(
                    "Garage configuration queue lost its pending request"
                )
                result: dict[str, Any] | None = None
            else:
                try:
                    result = self._apply_config(config, request_revision=target_revision)
                    error = None
                except BaseException as caught:
                    result = None
                    error = caught

            with self._configure_condition:
                if target_revision <= self._configure_completed_revision:
                    # Stop or shutdown cancelled this request while it waited
                    # for the CARLA world-mode lock.
                    if self._configure_completed_revision >= self._configure_requested_revision:
                        self._configure_running = False
                        self._configure_condition.notify_all()
                        return
                    continue
                if self._configure_requested_revision != target_revision:
                    # A newer request supersedes both this result and this
                    # failure. Intermediate settings are deliberately skipped.
                    continue
                first_revision = self._configure_completed_revision + 1
                outcome_config = config
                for revision in range(first_revision, target_revision + 1):
                    self._configure_outcomes[revision] = (
                        outcome_config,
                        None if result is None else dict(result),
                        error,
                    )
                self._configure_completed_revision = target_revision
                self._configure_running = False
                self._configure_condition.notify_all()
                return

    def _apply_config(
        self,
        config: GaragePreviewConfig,
        *,
        request_revision: int,
    ) -> dict[str, Any]:
        with self._world_mode_lock:
            with self._configure_condition:
                if request_revision <= self._configure_completed_revision:
                    raise RuntimeError("Garage configuration was cancelled")
                if self._configure_shutdown:
                    raise RuntimeError("Garage preview manager is shutting down")
            drive_status = str(self._drive_state().get("status", "idle"))
            if drive_status in _ACTIVE_DRIVE_STATES:
                raise RuntimeError("end the active Drive before starting Garage preview")
            with self._lock:
                current = self._session
            current_active = current is not None and current.snapshot().get("active") is True
            if current_active:
                confirm = getattr(current, "ensure_configuration_confirmed", None)
                if callable(confirm):
                    confirm()
                if current.config == config and not current.snapshot().get("error"):
                    result = dict(current.snapshot())
                    result["configure_action"] = "noop"
                    return result
                can_update = getattr(current, "can_reconfigure", None)
                if callable(can_update) and can_update(config):
                    return dict(current.reconfigure(config))
                weather_only = (
                    replace(
                        config,
                        weather_preset=current.config.weather_preset,
                    )
                    == current.config
                )
                if weather_only and config.weather_preset != current.config.weather_preset:
                    result = dict(current.update_weather(config.weather_preset))
                    result["configure_action"] = "weather"
                    return result
            self._stop_locked(reason="reconfigure")
            session = self._session_factory(
                config,
                carla_host=self.carla_host,
                carla_port=self.carla_port,
                world_worker=self.world_worker,
            )
            with self._lock:
                self._session = session
                self._last_error = None
            try:
                result = dict(session.start())
                result["configure_action"] = "restarted" if current_active else "started"
                return result
            except BaseException as error:
                with self._lock:
                    self._last_error = f"{type(error).__name__}: {error}"
                raise

    def orbit(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        request = GarageOrbitRequest.from_mapping(raw)
        with self._lock:
            session = self._session
        if session is None:
            raise RuntimeError("Garage preview is not active")
        return session.orbit(request)

    def frame(self) -> tuple[int, bytes]:
        with self._lock:
            session = self._session
        if session is None:
            raise FileNotFoundError("Garage preview is not active")
        return session.frame()

    def subscribe(self) -> GaragePreviewSession:
        with self._lock:
            session = self._session
        if session is None or not session.snapshot()["active"]:
            raise FileNotFoundError("Garage preview is not active")
        return session

    def stop(self, raw: Mapping[str, Any] | None = None) -> dict[str, Any]:
        body = {} if raw is None else raw
        _strict_keys(body, allowed=set(), required=set(), name="Garage preview stop request")
        with self._world_mode_lock:
            self._cancel_configure_requests(reason="operator stop")
            self._stop_locked(reason="operator_stop")
            return self.state()

    def stop_for_drive(self) -> None:
        with self._world_mode_lock:
            self._cancel_configure_requests(reason="Drive startup")
            self._stop_locked(reason="drive_start")

    def take_for_drive(self, config: DriveStartConfig) -> WorldWorkerScene | None:
        with self._world_mode_lock:
            self._cancel_configure_requests(reason="Drive startup")
            with self._lock:
                session = self._session
            if session is None:
                return None
            session.ensure_configuration_confirmed()
            if (
                session.world_worker is not self.world_worker
                or session.carla_host != self.carla_host
                or session.carla_port != self.carla_port
            ):
                return None
            scene = session.take_for_drive(config)
            if scene is not None:
                with self._lock:
                    self._session = None
                    self._last_error = None
            return scene

    def shutdown(self) -> None:
        with self._world_mode_lock:
            self._cancel_configure_requests(
                reason="Operator server shutdown",
                permanent=True,
            )
            self._stop_locked(reason="operator_server_shutdown")

    def _cancel_configure_requests(self, *, reason: str, permanent: bool = False) -> None:
        with self._configure_condition:
            if permanent:
                self._configure_shutdown = True
            first_revision = self._configure_completed_revision + 1
            final_revision = self._configure_requested_revision
            for revision in range(first_revision, final_revision + 1):
                self._configure_outcomes[revision] = (
                    None,
                    None,
                    RuntimeError(f"Garage configuration cancelled by {reason}"),
                )
            self._configure_completed_revision = final_revision
            self._configure_condition.notify_all()

    def _stop_locked(self, *, reason: str) -> None:
        with self._lock:
            session = self._session
        if session is None:
            return
        snapshot = session.close(reason=reason)
        cleanup_errors = snapshot.get("cleanup_errors", [])
        with self._lock:
            if cleanup_errors:
                self._last_error = "; ".join(str(value) for value in cleanup_errors)
            self._session = None


__all__ = [
    "GarageOrbitRequest",
    "GaragePreviewConfig",
    "GaragePreviewManager",
    "GaragePreviewSession",
]
