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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import cv2

from ..bridge import (
    CarlaCameraStream,
    CarlaRpc,
    garage_camera_preset_transform,
    garage_orbit_camera_transform,
    spawn_unparented_rgb_camera,
)
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
_EXPECTED_CARLA_VERSION = "0.9.16"


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


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
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
    spectator_mirror: bool = False
    width: int = 1920
    height: int = 1080
    fps: float = 10.0
    fov: float = 65.0
    yaw: float = 325.0
    pitch: float = -10.0
    distance: float = 6.5

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GaragePreviewConfig":
        allowed = {
            "map_name",
            "weather_preset",
            "vehicle_blueprint",
            "color",
            "seed",
            "traffic_count",
            "walker_count",
            "prop_preset",
            "spectator_mirror",
        }
        required = allowed - {"spectator_mirror"}
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
            spectator_mirror=_boolean(raw.get("spectator_mirror", False), name="spectator_mirror"),
        )


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
        self._jpeg: bytes | None = None
        self._closed = False

    def start(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Garage preview session is already closed")
            try:
                with self._worker_request_lock:
                    scene = self.world_worker.prepare_scene(
                        {
                            "map_name": self.config.map_name,
                            "weather_preset": self.config.weather_preset,
                            "vehicle_blueprint": self.config.vehicle_blueprint,
                            "color": self.config.color,
                            "seed": self.config.seed,
                            "traffic_count": self.config.traffic_count,
                            "walker_count": self.config.walker_count,
                            "prop_preset": self.config.prop_preset,
                            "route_mode": "free",
                            "initial_control_mode": "manual",
                        }
                    )
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
                else:
                    # An older Worker leaves the camera on CARLA's raw BGRA
                    # stream. Cap only that compatibility lane so the first
                    # frame can cross an ordinary LAN; upgraded Workers retain
                    # the requested 1080p worker-side JPEG relay.
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
                with self._lock:
                    self._camera_id = camera_id
                    self._camera_type = "sensor.camera.rgb"
                    self._worker_camera = worker_camera
                    self._camera_transport = (
                        "worker_jpeg" if worker_camera else "carla_raw_bgra_fallback"
                    )
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
            return {
                "schema_version": "1.0",
                "status": self._status,
                "active": self._status in {"starting", "running"} and not self._closed,
                "frame_sequence": self._frame_sequence,
                "yaw": self._yaw,
                "pitch": self._pitch,
                "distance": self._distance,
                "camera_preset": self._camera_preset,
                "error": self._error,
                "cleanup_errors": list(self._cleanup_errors),
                "map": None if self._scene is None else self._scene.map_name,
                "weather_preset": self.config.weather_preset,
                "vehicle_blueprint": self.config.vehicle_blueprint,
                "color": self.config.color,
                "vehicle_id": self._vehicle_id,
                "camera_id": self._camera_id,
                "camera_transport": self._camera_transport,
                "camera_resolution": f"{self._camera_width}x{self._camera_height}",
                "traffic_count": self.config.traffic_count,
                "walker_count": self.config.walker_count,
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
            if frame.sequence > self._frame_sequence:
                self._frame_sequence = frame.sequence
                self._jpeg = payload
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
                sequence = self._frame_sequence
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
                # The legacy raw-BGRA stream can pause for many seconds on a
                # loaded remote CARLA host. Keep the last valid Garage frame
                # visible and continue waiting; only the new worker-side JPEG
                # transport has a bounded freshness contract.
                if not self._worker_camera or time.monotonic() - last_frame_at < stale_after:
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
                with self._frame_condition:
                    self._error = f"Garage preview camera failed: {type(error).__name__}: {error}"
                    self._status = "failed"
                    self._frame_condition.notify_all()
                self.close(reason="camera_failed")
                return

    def _heartbeat_loop(self) -> None:
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
            except BaseException as error:
                with self._lock:
                    self._error = f"World Worker heartbeat failed: {type(error).__name__}: {error}"
                    self._status = "failed"
                self.close(reason="heartbeat_failed")
                return

    def close(self, *, reason: str = "operator_stop") -> dict[str, Any]:
        with self._lifecycle_lock:
            with self._lock:
                if self._closed:
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
                if scene is not None:
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
        self._session: GaragePreviewSession | None = None
        self._last_error: str | None = None

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
        with self._world_mode_lock:
            drive_status = str(self._drive_state().get("status", "idle"))
            if drive_status in _ACTIVE_DRIVE_STATES:
                raise RuntimeError("end the active Drive before starting Garage preview")
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
                return session.start()
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
            self._stop_locked(reason="operator_stop")
            return self.state()

    def stop_for_drive(self) -> None:
        with self._world_mode_lock:
            self._stop_locked(reason="drive_start")

    def shutdown(self) -> None:
        with self._world_mode_lock:
            self._stop_locked(reason="operator_server_shutdown")

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
