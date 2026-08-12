from __future__ import annotations

import math
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any

import msgpack
import numpy as np


class CarlaError(RuntimeError):
    """Raised when CARLA rejects an RPC call or returns malformed data."""


class CarlaRpc:
    """Small CARLA 0.9.16 RPC client implemented on top of MessagePack-RPC."""

    def __init__(self, host: str, port: int = 2000, timeout: float = 2.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket = socket.create_connection((host, port), timeout=timeout)
        self._socket.settimeout(timeout)
        self._unpacker = msgpack.Unpacker(raw=False, strict_map_key=False)
        self._request_id = 0
        self._lock = threading.Lock()

    def close(self) -> None:
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._socket.close()

    def __enter__(self) -> "CarlaRpc":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def raw_call(self, method: str, *args: Any) -> Any:
        """Call a CARLA endpoint and return its serialized Response<T> wrapper."""

        with self._lock:
            self._request_id += 1
            request_id = self._request_id
            request = [0, request_id, method, [[False], *args]]
            payload = msgpack.packb(
                request,
                use_bin_type=True,
                use_single_float=True,
            )
            self._socket.sendall(payload)

            while True:
                for response in self._unpacker:
                    if not isinstance(response, list) or len(response) != 4:
                        continue
                    if response[1] != request_id:
                        continue
                    if response[2] is not None:
                        raise CarlaError(f"{method}: {response[2]}")
                    return response[3]

                chunk = self._socket.recv(1 << 20)
                if not chunk:
                    raise CarlaError(f"{method}: CARLA closed the RPC connection")
                self._unpacker.feed(chunk)

    def async_call(self, method: str, *args: Any) -> None:
        """Send an RPC request without waiting, matching CARLA Client::AsyncCall."""

        with self._lock:
            self._request_id += 1
            request = [0, self._request_id, method, [[False], *args]]
            self._socket.sendall(
                msgpack.packb(
                    request,
                    use_bin_type=True,
                    use_single_float=True,
                )
            )

    def value_call(self, method: str, *args: Any) -> Any:
        """Call an endpoint returning CARLA Response<T>."""

        wrapped = self.raw_call(method, *args)
        if not isinstance(wrapped, list) or len(wrapped) != 1:
            raise CarlaError(f"{method}: malformed response {wrapped!r}")
        variant = wrapped[0]
        if not isinstance(variant, list) or len(variant) != 2:
            raise CarlaError(f"{method}: malformed value response {wrapped!r}")
        if variant[0] == 0:
            raise CarlaError(f"{method}: {variant[1]!r}")
        if variant[0] != 1:
            raise CarlaError(f"{method}: unknown response variant {variant!r}")
        return variant[1]

    def void_call(self, method: str, *args: Any) -> None:
        """Call an endpoint returning CARLA Response<void>."""

        wrapped = self.raw_call(method, *args)
        if wrapped == [[False]]:
            return
        if (
            isinstance(wrapped, list)
            and len(wrapped) == 1
            and isinstance(wrapped[0], list)
            and wrapped[0]
            and wrapped[0][0] is True
        ):
            raise CarlaError(f"{method}: {wrapped[0][1:]!r}")
        raise CarlaError(f"{method}: malformed void response {wrapped!r}")

    def actor(self, actor_id: int) -> list[Any] | None:
        actors = self.value_call("get_actors_by_id", [actor_id])
        return actors[0] if actors else None

    def spectator(self) -> list[Any]:
        """Return CARLA's server-side spectator actor."""

        spectator = self.value_call("get_spectator")
        if not isinstance(spectator, list) or not spectator:
            raise CarlaError(f"get_spectator: malformed actor {spectator!r}")
        return spectator

    def episode_id(self) -> int:
        """Return the current CARLA episode identifier."""

        episode = self.value_call("get_episode_info")
        if not isinstance(episode, list) or not episode:
            raise CarlaError(f"get_episode_info: malformed value {episode!r}")
        return int(episode[0])

    def actor_transform(self, actor_id: int, component: str = "VehicleMesh") -> list[Any]:
        return self.value_call("get_actor_component_world_transform", actor_id, component)

    def telemetry(self, actor_id: int) -> "VehicleTelemetry":
        value = self.value_call("get_telemetry_data", actor_id)
        return VehicleTelemetry(
            speed=float(value[0]),
            steer=float(value[1]),
            throttle=float(value[2]),
            brake=float(value[3]),
            rpm=float(value[4]),
            gear=int(value[5]),
        )

    def apply_vehicle_control(self, actor_id: int, control: list[Any]) -> None:
        self.void_call("apply_control_to_vehicle", actor_id, control)

    def apply_vehicle_control_async(self, actor_id: int, control: list[Any]) -> None:
        self.async_call("apply_control_to_vehicle", actor_id, control)

    def set_actor_transform(
        self,
        actor_id: int,
        transform: list[list[float]],
    ) -> None:
        """Move an actor and consume CARLA's response before returning."""

        self.void_call("set_actor_transform", actor_id, transform)

    def destroy_actor(self, actor_id: int) -> None:
        """Destroy an actor created by this client."""

        if not bool(self.value_call("destroy_actor", actor_id)):
            raise CarlaError(f"destroy_actor: actor {actor_id} was not destroyed")


@dataclass(frozen=True)
class VehicleTelemetry:
    speed: float
    steer: float
    throttle: float
    brake: float
    rpm: float
    gear: int


@dataclass(frozen=True)
class CarlaImageFrame:
    sequence: int
    sensor_type: int
    frame: int
    timestamp: float
    transform: tuple[float, float, float, float, float, float]
    width: int
    height: int
    fov: float
    bgra: bytes
    received_monotonic: float

    def bgra_array(self) -> np.ndarray:
        """Return a writable copy of the exact BGRA sensor payload."""

        pixels = np.frombuffer(self.bgra, dtype=np.uint8)
        return pixels.reshape(self.height, self.width, 4).copy()

    def bgr(self) -> np.ndarray:
        return self.bgra_array()[:, :, :3]


def vehicle_transform_from_front_camera(
    frame: CarlaImageFrame,
    forward_offset: float = 1.5,
    height_offset: float = 1.7,
) -> list[list[float]]:
    """Recover the parent vehicle pose from the known rigid camera mounting."""

    x, y, z, pitch, yaw, roll = frame.transform
    pitch_radians = math.radians(pitch)
    yaw_radians = math.radians(yaw)
    roll_radians = math.radians(roll)
    cp, sp = math.cos(pitch_radians), math.sin(pitch_radians)
    cy, sy = math.cos(yaw_radians), math.sin(yaw_radians)
    cr, sr = math.cos(roll_radians), math.sin(roll_radians)
    offset_x = forward_offset * cp * cy + height_offset * (-cy * sp * cr - sy * sr)
    offset_y = forward_offset * cp * sy + height_offset * (-sy * sp * cr + cy * sr)
    offset_z = forward_offset * sp + height_offset * cp * cr
    return [
        [
            x - offset_x,
            y - offset_y,
            z - offset_z,
        ],
        [pitch, yaw, roll],
    ]


def spectator_chase_transform(
    vehicle_transform: list[list[float]],
    *,
    distance: float = 7.0,
    height: float = 3.0,
    pitch: float = -15.0,
) -> list[list[float]]:
    """Place the CARLA spectator behind and above a vehicle transform."""

    try:
        location, rotation = vehicle_transform
        x, y, z = (float(value) for value in location)
        vehicle_pitch, yaw, roll = (float(value) for value in rotation)
        distance = float(distance)
        height = float(height)
        pitch = float(pitch)
    except (TypeError, ValueError, IndexError) as error:
        raise ValueError("vehicle transform must contain location and rotation triples") from error
    values = (x, y, z, vehicle_pitch, yaw, roll, distance, height, pitch)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("spectator chase transform values must be finite")
    if distance <= 0.0 or height <= 0.0:
        raise ValueError("spectator chase distance and height must be positive")
    pitch_radians = math.radians(vehicle_pitch)
    yaw_radians = math.radians(yaw)
    forward_xy = math.cos(pitch_radians)
    return [
        [
            x - forward_xy * math.cos(yaw_radians) * distance,
            y - forward_xy * math.sin(yaw_radians) * distance,
            z + height,
        ],
        [pitch, yaw, 0.0],
    ]


def _transform_triples(
    transform: Any,
    *,
    name: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Validate and normalize CARLA's serialized transform representation."""

    try:
        location, rotation = transform
        x, y, z = location
        pitch, yaw, roll = rotation
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain location and rotation triples") from error
    raw_values = (x, y, z, pitch, yaw, roll)
    if any(isinstance(value, bool) for value in raw_values):
        raise ValueError(f"{name} values must be finite numbers")
    try:
        values = tuple(float(value) for value in raw_values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} values must be finite numbers") from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} values must be finite numbers")
    return values[:3], values[3:]


def garage_orbit_camera_transform(
    vehicle_transform: Any,
    *,
    azimuth_degrees: float,
    pitch_degrees: float = -10.0,
    distance: float = 6.5,
    target_height: float = 0.9,
) -> list[list[float]]:
    """Place a CARLA RGB camera on a vehicle-centred showroom orbit.

    Azimuth zero places the camera on the vehicle's positive local X axis and
    points it inward. Increasing azimuth moves the camera toward positive local
    Y. ``distance`` is the three-dimensional eye-to-target distance, not merely
    the horizontal radius.
    """

    location, rotation = _transform_triples(vehicle_transform, name="vehicle transform")
    try:
        azimuth = float(azimuth_degrees)
        pitch = float(pitch_degrees)
        eye_distance = float(distance)
        look_height = float(target_height)
    except (TypeError, ValueError) as error:
        raise ValueError("garage orbit values must be finite numbers") from error
    if any(
        isinstance(value, bool)
        for value in (azimuth_degrees, pitch_degrees, distance, target_height)
    ) or not all(math.isfinite(value) for value in (azimuth, pitch, eye_distance, look_height)):
        raise ValueError("garage orbit values must be finite numbers")
    if not 3.5 <= eye_distance <= 10.0:
        raise ValueError("garage orbit distance must be in [3.5, 10.0] metres")
    if not -25.0 <= pitch <= 15.0:
        raise ValueError("garage orbit pitch must be in [-25, 15] degrees")
    if not 0.0 <= look_height <= 5.0:
        raise ValueError("garage orbit target height must be in [0, 5] metres")

    x, y, z = location
    vehicle_yaw = rotation[1]
    bearing_radians = math.radians(vehicle_yaw + azimuth)
    pitch_radians = math.radians(pitch)
    horizontal_radius = eye_distance * math.cos(pitch_radians)
    target_z = z + look_height
    camera_yaw = vehicle_yaw + azimuth + 180.0
    return [
        [
            x + horizontal_radius * math.cos(bearing_radians),
            y + horizontal_radius * math.sin(bearing_radians),
            target_z - eye_distance * math.sin(pitch_radians),
        ],
        [pitch, camera_yaw, 0.0],
    ]


GARAGE_CAMERA_PRESETS: dict[str, dict[str, float]] = {
    "front": {
        "azimuth_degrees": 0.0,
        "pitch_degrees": -8.0,
        "distance": 6.5,
        "target_height": 0.9,
    },
    "rear": {
        "azimuth_degrees": 180.0,
        "pitch_degrees": -8.0,
        "distance": 6.5,
        "target_height": 0.9,
    },
    "top": {
        "azimuth_degrees": 0.0,
        "pitch_degrees": -25.0,
        "distance": 8.0,
        "target_height": 0.9,
    },
}


def garage_camera_preset_transform(
    vehicle_transform: Any,
    preset: str,
    *,
    azimuth_degrees: float = 325.0,
    pitch_degrees: float = -10.0,
    distance: float = 6.5,
    target_height: float = 0.9,
) -> list[list[float]]:
    """Resolve a bounded Garage camera preset to a CARLA world transform.

    ``orbit`` uses the supplied continuous viewport values. Front, rear, and
    top are fixed exterior shortcuts. ``cockpit`` is a best-effort interior
    view whose camera follows the vehicle's full rotation.
    """

    name = str(preset).strip().lower()
    if name == "orbit":
        return garage_orbit_camera_transform(
            vehicle_transform,
            azimuth_degrees=azimuth_degrees,
            pitch_degrees=pitch_degrees,
            distance=distance,
            target_height=target_height,
        )
    if name in GARAGE_CAMERA_PRESETS:
        return garage_orbit_camera_transform(
            vehicle_transform,
            **GARAGE_CAMERA_PRESETS[name],
        )
    if name != "cockpit":
        choices = ", ".join(("orbit", *GARAGE_CAMERA_PRESETS, "cockpit"))
        raise ValueError(f"garage camera preset must be one of: {choices}")

    location, rotation = _transform_triples(vehicle_transform, name="vehicle transform")
    x, y, z = location
    pitch, yaw, roll = rotation
    pitch_radians = math.radians(pitch)
    yaw_radians = math.radians(yaw)
    roll_radians = math.radians(roll)
    cy, sy = math.cos(yaw_radians), math.sin(yaw_radians)
    cp, sp = math.cos(pitch_radians), math.sin(pitch_radians)
    cr, sr = math.cos(roll_radians), math.sin(roll_radians)
    matrix = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    local_position = (0.35, 0.0, 1.25)
    translated = [
        origin
        + sum(coefficient * offset for coefficient, offset in zip(row, local_position, strict=True))
        for origin, row in zip((x, y, z), matrix, strict=True)
    ]
    return [translated, [pitch, yaw, roll]]


@dataclass(frozen=True)
class StreamToken:
    stream_id: int
    port: int
    protocol: int
    address_type: int
    address_bytes: bytes

    @classmethod
    def parse(cls, raw: bytes | list[int]) -> "StreamToken":
        data = bytes(raw)
        if len(data) != 24:
            raise CarlaError(f"stream token is {len(data)} bytes; expected 24")
        return cls(*struct.unpack("<IHBB16s", data))

    def endpoint_host(self, fallback_host: str) -> str:
        if self.address_type == 1 and any(self.address_bytes[:4]):
            return socket.inet_ntoa(self.address_bytes[:4])
        return fallback_host


class CarlaCameraStream:
    """Background reader that drops old CARLA camera frames."""

    _SENSOR_HEADER = struct.Struct("<QQdffffff")
    _IMAGE_HEADER = struct.Struct("<IIf")
    _MAX_MESSAGE_SIZE = 128 * 1024 * 1024

    def __init__(
        self,
        fallback_host: str,
        token: bytes | list[int],
        timeout: float = 2.0,
    ) -> None:
        self.token = StreamToken.parse(token)
        if self.token.protocol != 1:
            raise CarlaError(f"camera token protocol {self.token.protocol} is not TCP")
        host = self.token.endpoint_host(fallback_host)
        self._socket = socket.create_connection((host, self.token.port), timeout=timeout)
        self._socket.settimeout(timeout)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._socket.sendall(struct.pack("<I", self.token.stream_id))

        self._condition = threading.Condition()
        self._latest: CarlaImageFrame | None = None
        self._sequence = 0
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"carla-camera-{self.token.stream_id}",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._closed = True
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._socket.close()
        self._thread.join(timeout=2.0)

    def __enter__(self) -> "CarlaCameraStream":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def latest(self) -> CarlaImageFrame | None:
        with self._condition:
            if self._error is not None:
                raise CarlaError(f"camera stream failed: {self._error}")
            return self._latest

    def wait_for_frame(
        self,
        after_sequence: int = -1,
        timeout: float = 3.0,
    ) -> CarlaImageFrame:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if self._error is not None:
                    raise CarlaError(f"camera stream failed: {self._error}")
                if self._latest is not None and self._latest.sequence > after_sequence:
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for a CARLA camera frame")
                self._condition.wait(remaining)

    def _recv_exact(self, count: int) -> bytes:
        chunks: list[bytes] = []
        remaining = count
        while remaining:
            chunk = self._socket.recv(remaining)
            if not chunk:
                raise CarlaError("CARLA camera stream closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _run(self) -> None:
        try:
            while not self._closed:
                size = struct.unpack("<I", self._recv_exact(4))[0]
                if size < 60 or size > self._MAX_MESSAGE_SIZE:
                    raise CarlaError(f"invalid camera message size {size}")
                payload = self._recv_exact(size)
                frame = self._decode(payload)
                with self._condition:
                    self._latest = frame
                    self._condition.notify_all()
        except BaseException as exc:
            if not self._closed:
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()

    def _decode(self, payload: bytes) -> CarlaImageFrame:
        sensor_values = self._SENSOR_HEADER.unpack_from(payload, 0)
        width, height, fov = self._IMAGE_HEADER.unpack_from(
            payload,
            self._SENSOR_HEADER.size,
        )
        offset = self._SENSOR_HEADER.size + self._IMAGE_HEADER.size
        pixels = payload[offset:]
        expected = width * height * 4
        if len(pixels) != expected:
            raise CarlaError(f"camera pixel payload is {len(pixels)} bytes; expected {expected}")
        self._sequence += 1
        return CarlaImageFrame(
            sequence=self._sequence,
            sensor_type=int(sensor_values[0]),
            frame=int(sensor_values[1]),
            timestamp=float(sensor_values[2]),
            transform=tuple(float(item) for item in sensor_values[3:9]),
            width=width,
            height=height,
            fov=float(fov),
            bgra=pixels,
            received_monotonic=time.monotonic(),
        )


def _actor_description(
    definition: list[Any],
    overrides: dict[str, str],
) -> list[Any]:
    uid, actor_id, _tags, attributes = definition
    values = []
    for attr_id, attr_type, value, _recommended, _modifiable, _restricted in attributes:
        values.append([attr_id, attr_type, overrides.get(attr_id, value)])
    return [uid, actor_id, values]


def spawn_unparented_rgb_camera(
    rpc: CarlaRpc,
    world_transform: Any,
    *,
    role_name: str = "garage_preview",
    width: int = 960,
    height: int = 540,
    sensor_tick: float = 0.1,
    fov: float = 65.0,
    attributes: dict[str, str] | None = None,
) -> list[Any]:
    """Spawn a world-space RGB camera for a non-driving Garage preview.

    The serialized CARLA actor is returned unchanged so callers can consume its
    actor ID at index ``0`` and stream token at index ``5``. A malformed camera
    response is destroyed before the validation error is raised.
    """

    location, rotation = _transform_triples(world_transform, name="camera world transform")
    normalized_transform = [list(location), list(rotation)]
    if not isinstance(role_name, str) or not role_name.strip() or len(role_name) > 128:
        raise ValueError("camera role_name must contain 1-128 characters")
    if any(character in role_name for character in "\x00\r\n"):
        raise ValueError("camera role_name contains a control character")
    for value, name, lower, upper in (
        (width, "camera width", 1, 8192),
        (height, "camera height", 1, 8192),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    try:
        tick = float(sensor_tick)
        field_of_view = float(fov)
    except (TypeError, ValueError) as error:
        raise ValueError("camera sensor_tick and fov must be finite numbers") from error
    if any(isinstance(value, bool) for value in (sensor_tick, fov)) or not all(
        math.isfinite(value) for value in (tick, field_of_view)
    ):
        raise ValueError("camera sensor_tick and fov must be finite numbers")
    if not 0.01 <= tick <= 10.0:
        raise ValueError("camera sensor_tick must be in [0.01, 10.0] seconds")
    if not 1.0 <= field_of_view <= 179.0:
        raise ValueError("camera fov must be in [1, 179] degrees")
    if attributes is not None and not isinstance(attributes, dict):
        raise TypeError("camera attributes must be a dictionary")

    definitions = rpc.value_call("get_actor_definitions")
    definition = next(
        (item for item in definitions if item[1] == "sensor.camera.rgb"),
        None,
    )
    if definition is None:
        raise CarlaError("CARLA blueprint 'sensor.camera.rgb' is unavailable")
    overrides = {
        "role_name": role_name,
        "sensor_tick": str(tick),
        "image_size_x": str(width),
        "image_size_y": str(height),
        "fov": str(field_of_view),
        "motion_blur_intensity": "0.0",
        "motion_blur_max_distortion": "0.0",
    }
    if attributes:
        overrides.update({str(key): str(value) for key, value in attributes.items()})
    description = _actor_description(definition, overrides)
    camera = rpc.value_call("spawn_actor", description, normalized_transform)

    actor_id: int | None = None
    try:
        if not isinstance(camera, list) or len(camera) < 6:
            raise CarlaError(f"spawned RGB camera is malformed: {camera!r}")
        if isinstance(camera[0], bool):
            raise ValueError("actor ID is boolean")
        actor_id = int(camera[0])
        if actor_id <= 0:
            raise ValueError("actor ID is not positive")
        definition_payload = camera[2]
        if (
            not isinstance(definition_payload, list)
            or len(definition_payload) < 2
            or str(definition_payload[1]) != "sensor.camera.rgb"
        ):
            raise ValueError("actor is not an RGB camera")
        StreamToken.parse(camera[5])
    except (CarlaError, TypeError, ValueError, IndexError) as error:
        if actor_id is not None:
            try:
                rpc.destroy_actor(actor_id)
            except Exception:
                pass
        if isinstance(error, CarlaError):
            raise
        raise CarlaError(f"spawned RGB camera is malformed: {error}") from error
    return camera


def spawn_camera(
    rpc: CarlaRpc,
    vehicle_id: int,
    sensor_type: str,
    *,
    role_name: str,
    width: int = 640,
    height: int = 384,
    sensor_tick: float = 0.1,
    fov: float = 90.0,
    relative_transform: list[list[float]] | None = None,
    attributes: dict[str, str] | None = None,
) -> list[Any]:
    """Attach a camera sensor and return its serialized CARLA actor.

    Only attributes present in the server-side blueprint are overridden.  This
    keeps one path usable for RGB and privileged annotation cameras while
    preserving identical geometry.
    """

    definitions = rpc.value_call("get_actor_definitions")
    definition = next(
        (item for item in definitions if item[1] == sensor_type),
        None,
    )
    if definition is None:
        raise CarlaError(f"CARLA blueprint {sensor_type!r} is unavailable")
    overrides = {
        "role_name": role_name,
        "sensor_tick": str(sensor_tick),
        "image_size_x": str(width),
        "image_size_y": str(height),
        "fov": str(fov),
        "motion_blur_intensity": "0.0",
        "motion_blur_max_distortion": "0.0",
    }
    if attributes:
        overrides.update({str(key): str(value) for key, value in attributes.items()})
    description = _actor_description(
        definition,
        overrides,
    )
    transform = (
        [[1.5, 0.0, 1.7], [0.0, 0.0, 0.0]] if relative_transform is None else relative_transform
    )
    return rpc.value_call(
        "spawn_actor_with_parent",
        description,
        transform,
        vehicle_id,
        0,  # AttachmentType::Rigid
        "",
    )


def spawn_front_camera(
    rpc: CarlaRpc,
    vehicle_id: int,
    width: int = 640,
    height: int = 384,
    sensor_tick: float = 0.1,
    fov: float = 90.0,
) -> list[Any]:
    """Attach an RGB camera to a CARLA vehicle and return its serialized actor."""

    return spawn_camera(
        rpc,
        vehicle_id,
        "sensor.camera.rgb",
        role_name="front",
        width=width,
        height=height,
        sensor_tick=sensor_tick,
        fov=fov,
    )
