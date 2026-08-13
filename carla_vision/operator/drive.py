"""Interactive, one-session CARLA drive engine for the local operator UI.

This module deliberately keeps model output advisory.  The only actuator input
is the fresh, complete browser control state; stale browser or camera input
falls back to braking, with :class:`SafeActuator` providing a second watchdog.
"""

from __future__ import annotations

import json
import math
import random
import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import cv2
import numpy as np

from ..artifacts import RunArtifactTracker, fingerprint_file
from ..bridge import (
    CarlaCameraStream,
    CarlaRpc,
    spawn_front_camera,
    spectator_chase_transform,
    vehicle_transform_from_front_camera,
)
from ..contracts import DetectorConfig, PerceptionResult
from ..controller import ControlCommand
from ..detectors.factory import create_detector
from ..display import OverlayRenderer
from ..perception import PerceptionWorker
from ..recording import AsyncVideoRecorder
from ..watchdog import SafeActuator
from .drive_contracts import DriveInput, DriveStartConfig, weather_payload
from .situations import PROP_PRESETS, WEATHER_PRESETS
from .world_worker_client import (
    WorldWorkerCameraStream,
    WorldWorkerClient,
    WorldWorkerScene,
)

_ACTIVE = frozenset({"starting", "running", "stopping"})
_TERMINAL = frozenset({"success", "failed"})
_BROWSER_LEASE_SECONDS = 0.40
_CONTROL_PERIOD_SECONDS = 0.05
_TELEMETRY_PERIOD_SECONDS = 0.20
_WORKER_HEARTBEAT_SECONDS = 0.50
HUMAN_MARKER_LABELS = frozenset(
    {
        "interesting",
        "false_detection",
        "missed_object",
        "autopilot_issue",
        "scene_issue",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _map_short_name(raw: str) -> str:
    return raw.rsplit("/", 1)[-1].removesuffix(".umap")


def _world_worker_health_ready(payload: Mapping[str, Any]) -> bool:
    ready = payload.get("ready")
    if isinstance(ready, bool):
        return ready
    return str(payload.get("status", "")).strip().lower() in {"ok", "ready"}


def _validate_camera_attachment(camera: list[Any], vehicle_id: int) -> None:
    """Validate the authoritative parent recorded in CARLA's actor serialization."""

    try:
        parent_id = int(camera[1])
    except (IndexError, TypeError, ValueError) as error:
        raise RuntimeError("spawned front camera has malformed parent metadata") from error
    if parent_id != vehicle_id:
        raise RuntimeError("spawned front camera is not attached to the ego vehicle")


def _jpeg(image: np.ndarray, quality: int = 92) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError("could not encode browser JPEG frame")
    return encoded.tobytes()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _json_line(stream: TextIO | None, payload: Mapping[str, Any]) -> None:
    if stream is None:
        return
    stream.write(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    stream.flush()


def _definition_attributes(definition: list[Any]) -> dict[str, list[Any]]:
    if not isinstance(definition, list) or len(definition) < 4:
        raise RuntimeError(f"malformed actor definition {definition!r}")
    return {str(attribute[0]): attribute for attribute in definition[3]}


def _vehicle_description(
    definition: list[Any],
    *,
    role_name: str,
    color: str | None,
) -> list[Any]:
    attributes = _definition_attributes(definition)
    overrides = {"role_name": role_name}
    if color is not None:
        color_attribute = attributes.get("color")
        if color_attribute is None or not bool(color_attribute[4]):
            raise ValueError("selected vehicle does not expose a modifiable color")
        recommended = {str(value) for value in color_attribute[3]}
        if color not in recommended:
            raise ValueError("selected color is not recommended by this vehicle blueprint")
        overrides["color"] = color

    serialized: list[list[Any]] = []
    for attribute in definition[3]:
        attribute_id, attribute_type, default = attribute[:3]
        serialized.append([attribute_id, attribute_type, overrides.get(str(attribute_id), default)])
    return [definition[0], definition[1], serialized]


def _spawn_vehicle(
    rpc: CarlaRpc,
    *,
    blueprint: str,
    color: str | None,
    spawn_points: list[list[Any]],
    seed: int,
) -> tuple[int, list[Any], int]:
    definitions = rpc.value_call("get_actor_definitions")
    definition = next(
        (item for item in definitions if str(item[1]) == blueprint),
        None,
    )
    if definition is None:
        raise ValueError(f"vehicle blueprint {blueprint!r} is unavailable")
    description = _vehicle_description(
        definition,
        role_name="research_drive_ego",
        color=color,
    )
    indices = list(range(len(spawn_points)))
    random.Random(seed).shuffle(indices)
    errors: list[str] = []
    for index in indices[: min(40, len(indices))]:
        transform = spawn_points[index]
        actor_id: int | None = None
        try:
            actor = rpc.value_call("spawn_actor", description, transform)
            actor_id = int(actor[0])
            verified = rpc.actor(actor_id)
            if verified is None or not str(verified[2][1]).startswith("vehicle."):
                raise RuntimeError(f"spawned actor {actor_id} did not verify as a vehicle")
            return actor_id, transform, index
        except Exception as error:
            if actor_id is not None:
                try:
                    rpc.destroy_actor(actor_id)
                except Exception:
                    pass
            errors.append(f"spawn {index}: {error}")
    detail = "; ".join(errors[-5:])
    raise RuntimeError(f"could not spawn ego at any random road start: {detail}")


def _compose_relative_transform(
    origin: list[Any],
    relative: Mapping[str, Any],
) -> list[list[float]]:
    location, rotation = origin
    x, y, z = (float(value) for value in location)
    pitch, yaw, roll = (float(value) for value in rotation)
    local = (
        float(relative.get("x", 0.0)),
        float(relative.get("y", 0.0)),
        float(relative.get("z", 0.0)),
    )
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    matrix = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    translated = [
        origin_value
        + sum(coefficient * component for coefficient, component in zip(row, local, strict=True))
        for origin_value, row in zip((x, y, z), matrix, strict=True)
    ]
    return [
        translated,
        [
            pitch + float(relative.get("pitch", 0.0)),
            yaw + float(relative.get("yaw", 0.0)),
            roll + float(relative.get("roll", 0.0)),
        ],
    ]


def _spawn_props(
    rpc: CarlaRpc,
    *,
    preset: str,
    ego_start: list[Any],
) -> list[tuple[int, str]]:
    recipe = PROP_PRESETS[preset]
    if not recipe:
        return []
    definitions = rpc.value_call("get_actor_definitions")
    by_id = {str(definition[1]): definition for definition in definitions}
    spawned: list[tuple[int, str]] = []
    try:
        for item in recipe:
            blueprint = str(item["blueprint_id"])
            definition = by_id.get(blueprint)
            if definition is None:
                raise RuntimeError(f"scene prop blueprint {blueprint!r} is unavailable")
            serialized = [
                definition[0],
                definition[1],
                [[attribute[0], attribute[1], attribute[2]] for attribute in definition[3]],
            ]
            transform = _compose_relative_transform(ego_start, item["transform"])
            actor = rpc.value_call("spawn_actor", serialized, transform)
            actor_id = int(actor[0])
            spawned.append((actor_id, blueprint))
            verified = rpc.actor(actor_id)
            if verified is None or str(verified[2][1]) != blueprint:
                raise RuntimeError(f"spawned scene prop {actor_id} did not verify as {blueprint}")
    except BaseException:
        for actor_id, _ in reversed(spawned):
            try:
                rpc.destroy_actor(actor_id)
            except Exception:
                pass
        raise
    return spawned


def _vehicle_catalog(definitions: list[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for definition in definitions:
        actor_id = str(definition[1])
        if not actor_id.startswith("vehicle."):
            continue
        attributes = _definition_attributes(definition)
        color = attributes.get("color")
        colors = [str(value) for value in color[3]] if color is not None else []
        readable = actor_id.removeprefix("vehicle.").replace("_", " ").replace(".", " · ")
        result.append(
            {
                "id": actor_id,
                "label": readable.title(),
                "colors": colors,
            }
        )
    result.sort(key=lambda item: (item["label"], item["id"]))
    return result


class DriveSession:
    """Own every resource for one interactive browser-controlled drive."""

    def __init__(
        self,
        config: DriveStartConfig,
        *,
        workspace: Path,
        world_worker: WorldWorkerClient | None = None,
    ) -> None:
        self.config = config
        self.workspace = workspace
        self._world_worker = world_worker
        if config.world_worker_enabled != (world_worker is not None):
            raise ValueError("drive configuration and World Worker availability disagree")
        self.session_id = config.run_id
        self._lock = threading.RLock()
        self._worker_request_lock = threading.Lock()
        self._frame_condition = threading.Condition(self._lock)
        self._status = "starting"
        self._error: str | None = None
        self._stop_reason: str | None = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"drive-session-{self.session_id}",
            daemon=True,
        )
        self._started_monotonic = time.monotonic()
        self._finished_monotonic: float | None = None
        self._last_input: tuple[DriveInput, float] | None = None
        self._last_input_sequence = -1
        self._emergency = False
        self._deadman_active = True
        self._control_source = "deadman"
        self._requested_weather: str | None = None
        self._weather_preset = config.weather_preset
        self._control_mode = config.initial_control_mode
        self._mode_history: list[dict[str, Any]] = [
            {
                "at": _utc_now(),
                "from": None,
                "to": config.initial_control_mode,
                "reason": "session_start",
            }
        ]
        self._pending_events: list[dict[str, Any]] = []
        self._worker_scene: WorldWorkerScene | None = None
        self._worker_scene_id: str | None = None
        self._worker_scene_stopped = False
        self._worker_cleanup_guard_passed: bool | None = None
        self._worker_control_sequence = 0
        self._worker_heartbeat_stop = threading.Event()
        self._worker_heartbeat_thread: threading.Thread | None = None
        self._worker_heartbeat_error: BaseException | None = None
        self._raw_jpeg: bytes | None = None
        self._overlay_jpeg: bytes | None = None
        self._raw_frame_sequence = -1
        self._overlay_frame_sequence = -1
        self._telemetry = {
            "speed": 0.0,
            "gear": 0,
            "throttle": 0.0,
            "steer": 0.0,
            "brake": 1.0,
        }
        self._map: str | None = None
        self._server_version: str | None = None
        self._vehicle_id: int | None = None
        self._camera_id: int | None = None
        self._prop_ids: list[int] = []
        self._spawn_index: int | None = None
        self._traffic_count_actual = 0
        self._walker_count_actual = 0
        self._route: dict[str, Any] = {}
        self._destination: Any = None
        self._detector_name: str | None = None
        self._recording = bool(config.record_video)
        self._output_path: str | None = None
        self._controls_written = 0
        self._detections_written = 0
        self._frames_seen = 0
        self._manual_commands = 0
        self._deadman_commands = 0
        self._marker_counts = {label: 0 for label in sorted(HUMAN_MARKER_LABELS)}
        self._cleanup_errors: list[str] = []

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> bool:
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def snapshot(self) -> dict[str, Any]:
        now = self._finished_monotonic or time.monotonic()
        with self._lock:
            input_age = (
                None
                if self._last_input is None
                else max(0.0, time.monotonic() - self._last_input[1])
            )
            return {
                "schema_version": "1.0",
                "status": self._status,
                "session_id": self.session_id,
                "run_id": self.config.run_id,
                "map": self._map,
                "server_version": self._server_version,
                "vehicle_id": self._vehicle_id,
                "camera_id": self._camera_id,
                "prop_ids": list(self._prop_ids),
                "spawn_index": self._spawn_index,
                "elapsed_seconds": max(0.0, now - self._started_monotonic),
                "telemetry": dict(self._telemetry),
                "input_age_seconds": input_age,
                "deadman_active": self._deadman_active,
                "control_source": self._control_source,
                "control_mode": self._control_mode,
                "world_worker": {
                    "enabled": self._world_worker is not None,
                    "scene_id": self._worker_scene_id,
                    "status": (None if self._worker_scene is None else self._worker_scene.status),
                    "cleanup_guard_passed": self._worker_cleanup_guard_passed,
                },
                "traffic_count_actual": self._traffic_count_actual,
                "walker_count_actual": self._walker_count_actual,
                "route": dict(self._route),
                "destination": self._destination,
                "detector": {
                    "enabled": self.config.detector_enabled,
                    "name": self._detector_name,
                    "advisory_only": True,
                    "actuated": False,
                },
                "recording": self._recording,
                "weather_preset": self._weather_preset,
                "output_path": self._output_path,
                "error": self._error,
                "stop_reason": self._stop_reason,
                "raw_frame_sequence": self._raw_frame_sequence,
                "overlay_frame_sequence": self._overlay_frame_sequence,
                "frames_seen": self._frames_seen,
                "controls_written": self._controls_written,
                "detections_written": self._detections_written,
                "experiment_preset": self.config.experiment_preset,
                "human_marker_counts": dict(self._marker_counts),
                "human_markers_written": sum(self._marker_counts.values()),
                "cleanup_errors": list(self._cleanup_errors),
            }

    def submit_control(self, control: DriveInput) -> dict[str, Any]:
        with self._lock:
            if self._status not in {"starting", "running"}:
                raise RuntimeError("drive session is not accepting controls")
            if control.session_id != self.session_id:
                raise ValueError("control session_id does not match the active drive")
            if control.sequence <= self._last_input_sequence:
                raise ValueError("control sequence must be newer than the previous input")
            self._last_input_sequence = control.sequence
            if self._control_mode == "manual":
                self._last_input = (control, time.monotonic())
        return self.snapshot()

    def emergency_stop(self) -> dict[str, Any]:
        worker = self._world_worker
        worker_scene: WorldWorkerScene | None
        with self._lock:
            self._emergency = True
            self._deadman_active = True
            self._control_source = "emergency_stop"
            worker_scene = self._worker_scene
        if worker is not None and worker_scene is not None and self._control_mode != "manual":
            with self._worker_request_lock:
                with self._lock:
                    worker_scene = self._worker_scene
                    previous_mode = self._control_mode
                if worker_scene is None:
                    raise RuntimeError("World Worker scene is not ready")
                updated = worker.mode(worker_scene, "manual")
                with self._lock:
                    self._worker_control_sequence += 1
                    worker_sequence = self._worker_control_sequence
                emergency_command = ControlCommand.service_brake()
                updated = worker.control(
                    updated,
                    {
                        "sequence": worker_sequence,
                        "throttle": emergency_command.throttle,
                        "steer": emergency_command.steer,
                        "brake": emergency_command.brake,
                        "hand_brake": emergency_command.hand_brake,
                        "reverse": emergency_command.reverse,
                    },
                )
                with self._lock:
                    self._worker_scene = updated
                    self._route = dict(updated.route)
                    self._destination = updated.destination
                    self._control_mode = "manual"
                    self._last_input = None
                self._record_mode_change(previous_mode, "manual", "emergency_stop")
        return self.snapshot()

    def request_mode(self, mode: str) -> dict[str, Any]:
        control_mode = str(mode).strip()
        if control_mode not in {"manual", "autopilot"}:
            raise ValueError("control mode must be manual or autopilot")
        with self._lock:
            if self._status != "running":
                raise RuntimeError("control mode can only change during a running drive")
            if self._emergency and control_mode != "manual":
                raise RuntimeError("emergency brake is latched; autopilot cannot be re-enabled")
            worker_scene = self._worker_scene
        if self._world_worker is None:
            if control_mode != "manual":
                raise RuntimeError("autopilot requires a configured World Worker")
            return self.snapshot()
        if worker_scene is None:
            raise RuntimeError("World Worker scene is not ready")
        with self._worker_request_lock:
            with self._lock:
                worker_scene = self._worker_scene
                previous_mode = self._control_mode
            if worker_scene is None:
                raise RuntimeError("World Worker scene is not ready")
            updated = self._world_worker.mode(worker_scene, control_mode)
            with self._lock:
                self._worker_scene = updated
                self._route = dict(updated.route)
                self._destination = updated.destination
                self._control_mode = control_mode
                self._last_input = None
                self._deadman_active = control_mode == "manual"
                self._control_source = (
                    "worker_autopilot" if control_mode == "autopilot" else "browser_deadman"
                )
            self._record_mode_change(previous_mode, control_mode, "operator_request")
        return self.snapshot()

    def request_weather(self, preset: str) -> dict[str, Any]:
        if preset not in WEATHER_PRESETS:
            raise ValueError(f"unknown weather preset {preset!r}")
        with self._lock:
            if self._status != "running":
                raise RuntimeError("weather can only change during a running drive")
            self._requested_weather = preset
        return self.snapshot()

    def mark_human_event(self, label: str, note: str | None = None) -> dict[str, Any]:
        marker = str(label).strip()
        if marker not in HUMAN_MARKER_LABELS:
            raise ValueError(
                "human marker label must be one of: " + ", ".join(sorted(HUMAN_MARKER_LABELS))
            )
        normalized_note = None if note is None else str(note).strip()
        if normalized_note == "":
            normalized_note = None
        if normalized_note is not None and (
            len(normalized_note) > 240
            or any(character in normalized_note for character in "\x00\r\n")
        ):
            raise ValueError("human marker note must be at most 240 single-line characters")
        with self._lock:
            if self._status != "running":
                raise RuntimeError("human moments can only be marked during a running drive")
            event = {
                "event": "human_moment_marked",
                "at": _utc_now(),
                "elapsed_seconds": time.monotonic() - self._started_monotonic,
                "label": marker,
                "note": normalized_note,
                "experiment_preset": self.config.experiment_preset,
                "control_mode": self._control_mode,
                "control_source": self._control_source,
                "raw_camera_sequence": self._raw_frame_sequence,
                "detector_sequence": self._overlay_frame_sequence,
                "telemetry": dict(self._telemetry),
                "model_output_actuated": False,
            }
            self._marker_counts[marker] += 1
            self._pending_events.append(event)
        return self.snapshot()

    def request_stop(self, reason: str = "operator_stop") -> dict[str, Any]:
        with self._lock:
            if self._status in _TERMINAL:
                return self.snapshot()
            self._status = "stopping"
            self._stop_reason = reason
            self._stop_event.set()
        return self.snapshot()

    def frame(self, view: str) -> tuple[int, bytes]:
        with self._lock:
            if view == "raw":
                sequence, payload = self._raw_frame_sequence, self._raw_jpeg
            elif view == "overlay":
                sequence, payload = self._overlay_frame_sequence, self._overlay_jpeg
            else:
                raise ValueError("drive frame view must be raw or overlay")
            if payload is None:
                raise FileNotFoundError(f"{view} drive frame is not ready")
            return sequence, payload

    def _set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def _cache_frame(self, view: str, sequence: int, payload: bytes) -> None:
        with self._frame_condition:
            if view == "raw":
                self._raw_frame_sequence = sequence
                self._raw_jpeg = payload
            else:
                self._overlay_frame_sequence = sequence
                self._overlay_jpeg = payload
            self._frame_condition.notify_all()

    def _record_mode_change(self, previous: str, current: str, reason: str) -> None:
        if previous == current:
            return
        event = {
            "event": "control_mode_changed",
            "at": _utc_now(),
            "from": previous,
            "to": current,
            "reason": reason,
        }
        with self._lock:
            self._mode_history.append(
                {key: value for key, value in event.items() if key != "event"}
            )
            self._pending_events.append(event)

    def _drain_pending_events(self, stream: TextIO | None) -> None:
        with self._lock:
            pending = list(self._pending_events)
            self._pending_events.clear()
        for event in pending:
            _json_line(stream, event)

    def _run(self) -> None:
        try:
            self._run_tracked()
        except BaseException as error:
            with self._lock:
                self._error = f"{type(error).__name__}: {error}"
                self._status = "failed"
                self._finished_monotonic = time.monotonic()
        else:
            with self._lock:
                self._status = "success"
                self._finished_monotonic = time.monotonic()

    def _begin_worker_scene(self) -> WorldWorkerScene:
        worker = self._world_worker
        if worker is None:
            raise RuntimeError("World Worker is not configured")
        with self._worker_request_lock:
            prepared = worker.prepare_scene(
                {
                    "map_name": self.config.map_name,
                    "weather_preset": self.config.weather_preset,
                    "vehicle_blueprint": self.config.vehicle_blueprint,
                    "color": self.config.color,
                    "seed": self.config.seed,
                    "traffic_count": self.config.traffic_count,
                    "walker_count": self.config.walker_count,
                    "prop_preset": self.config.prop_preset,
                    "route_mode": self.config.route_mode,
                    "initial_control_mode": self.config.initial_control_mode,
                }
            )
        with self._lock:
            self._worker_scene = prepared
            self._worker_scene_id = prepared.scene_id
            self._traffic_count_actual = prepared.traffic_count or 0
            self._walker_count_actual = prepared.walker_count or 0
            self._route = dict(prepared.route)
            self._destination = prepared.destination
        if prepared.ego_actor_id is None:
            raise RuntimeError("World Worker prepared a scene without an ego actor")
        if prepared.episode_id is None:
            raise RuntimeError("World Worker prepared a scene without an episode_id")
        if prepared.control_mode != self.config.initial_control_mode:
            raise RuntimeError("World Worker did not prepare the requested initial control mode")
        with self._lock:
            self._control_mode = self.config.initial_control_mode
            self._deadman_active = True
            self._control_source = "worker_prepared"
        self._start_worker_heartbeat()
        return prepared

    def _activate_worker_scene(self) -> WorldWorkerScene:
        worker = self._world_worker
        if worker is None:
            raise RuntimeError("World Worker is not configured")
        with self._worker_request_lock:
            with self._lock:
                scene = self._worker_scene
                heartbeat_error = self._worker_heartbeat_error
            if heartbeat_error is not None:
                raise RuntimeError(f"World Worker heartbeat failed: {heartbeat_error}")
            if scene is None:
                raise RuntimeError("World Worker scene is not ready")
            started = worker.start_scene(scene)
            if started.status != "running":
                raise RuntimeError("World Worker did not enter the running scene state")
            if started.ego_actor_id is None or started.episode_id is None:
                raise RuntimeError("World Worker running scene lost its authoritative actors")
            if started.control_mode != self.config.initial_control_mode:
                raise RuntimeError("World Worker did not apply the requested initial control mode")
            with self._lock:
                self._worker_scene = started
                self._route = dict(started.route)
                self._destination = started.destination
                self._control_mode = self.config.initial_control_mode
                self._deadman_active = self._control_mode == "manual"
                self._control_source = (
                    "worker_autopilot" if self._control_mode == "autopilot" else "browser_deadman"
                )
        return started

    def _start_worker_heartbeat(self) -> None:
        if self._world_worker is None or self._worker_heartbeat_thread is not None:
            return
        self._worker_heartbeat_thread = threading.Thread(
            target=self._worker_heartbeat_loop,
            name=f"world-worker-heartbeat-{self.session_id}",
            daemon=True,
        )
        self._worker_heartbeat_thread.start()

    def _worker_heartbeat_loop(self) -> None:
        assert self._world_worker is not None
        while not self._worker_heartbeat_stop.wait(_WORKER_HEARTBEAT_SECONDS):
            try:
                with self._worker_request_lock:
                    with self._lock:
                        scene = self._worker_scene
                        stopped = self._worker_scene_stopped
                    if scene is None or stopped:
                        return
                    updated = self._world_worker.heartbeat(scene)
                    with self._lock:
                        self._worker_scene = updated
            except BaseException as error:
                with self._lock:
                    self._worker_heartbeat_error = error
                self._stop_event.set()
                return

    def _stop_worker_heartbeat(self) -> None:
        self._worker_heartbeat_stop.set()
        thread = self._worker_heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def _stop_worker_scene(self) -> None:
        worker = self._world_worker
        with self._lock:
            scene = self._worker_scene
            stopped = self._worker_scene_stopped
        if worker is None or scene is None or stopped:
            return
        self._stop_worker_heartbeat()
        with self._worker_request_lock:
            with self._lock:
                scene = self._worker_scene
            if scene is None:
                return
            updated = worker.stop_scene(scene)
        with self._lock:
            self._worker_scene = updated
            self._worker_scene_stopped = True
            self._worker_cleanup_guard_passed = updated.cleanup_guard_passed
            self._route = dict(updated.route)
            self._destination = updated.destination
            for error in updated.cleanup_errors:
                self._cleanup_errors.append(f"World Worker cleanup: {error}")

    def _run_tracked(self) -> None:
        try:
            if self._world_worker is not None:
                self._begin_worker_scene()
            with CarlaRpc(self.config.host, self.config.port, timeout=4.0) as preflight_rpc:
                preflight_version = str(preflight_rpc.value_call("version"))
                preflight_map = str(preflight_rpc.value_call("get_map_info")[0])
            model_refs: tuple[Mapping[str, Any], ...] = ()
            if self.config.detector_enabled and self.config.weights is not None:
                model_refs = (
                    {
                        "kind": "advisory_object_detector",
                        "actuation_authorized": False,
                        **fingerprint_file(self.config.weights),
                    },
                )
            tracker = RunArtifactTracker(
                self.workspace / "runs",
                run_id=self.config.run_id,
                cli_args={"source": "operator_drive_console"},
                config=self.config.manifest_config(),
                repository_root=self.workspace,
                carla_endpoint={"host": self.config.host, "port": self.config.port},
                carla_version=preflight_version,
                carla_map=preflight_map,
                model_refs=model_refs,
            )
            with self._lock:
                self._output_path = tracker.run_dir.relative_to(self.workspace).as_posix()

            failure: BaseException | None = None
            with tracker:
                try:
                    self._execute(tracker)
                except BaseException as error:
                    failure = error
                    with self._lock:
                        self._error = f"{type(error).__name__}: {error}"
                finally:
                    if self._world_worker is not None and not self._worker_scene_stopped:
                        try:
                            self._stop_worker_scene()
                        except Exception as error:
                            self._cleanup_errors.append(f"World Worker cleanup: {error}")
                    self._finalize(tracker, failure)
                if failure is not None:
                    raise failure
        finally:
            if self._world_worker is not None and not self._worker_scene_stopped:
                try:
                    self._stop_worker_scene()
                except Exception as error:
                    self._cleanup_errors.append(f"World Worker final cleanup: {error}")

    def _execute(self, tracker: RunArtifactTracker) -> None:
        rpc: CarlaRpc | None = None
        stream: CarlaCameraStream | WorldWorkerCameraStream | None = None
        actuator: SafeActuator | None = None
        perception: PerceptionWorker | None = None
        raw_recorder: AsyncVideoRecorder | None = None
        overlay_recorder: AsyncVideoRecorder | None = None
        controls_stream: TextIO | None = None
        detections_stream: TextIO | None = None
        events_stream: TextIO | None = None
        episode_id: int | None = None
        owned_actor_ids: list[tuple[int, str]] = []
        original_weather: list[float] | None = None
        weather_changed = False
        spectator_id: int | None = None
        spectator_episode_id: int | None = None
        spectator_transform: list[Any] | None = None
        spectator_follow_active = False
        latest_raw: np.ndarray | None = None
        latest_overlay: np.ndarray | None = None
        last_camera_sequence = -1
        last_result_sequence = -1
        last_camera_received = 0.0
        last_control_at = 0.0
        last_telemetry_at = 0.0
        detector_failed = False

        config_path = tracker.artifact_path("config.json")
        controls_path = tracker.artifact_path("controls.jsonl")
        detections_path = tracker.artifact_path("detections.jsonl")
        events_path = tracker.artifact_path("events.jsonl")
        raw_video_path = tracker.artifact_path("raw-drive.mp4")
        overlay_video_path = tracker.artifact_path("model-overlay.mp4")

        _write_json(config_path, self.config.manifest_config())
        tracker.register_artifact(config_path, role="interactive_drive_configuration")

        try:
            controls_stream = controls_path.open("w", encoding="utf-8", buffering=1)
            detections_stream = detections_path.open("w", encoding="utf-8", buffering=1)
            events_stream = events_path.open("w", encoding="utf-8", buffering=1)
            rpc = CarlaRpc(self.config.host, self.config.port, timeout=5.0)
            self._server_version = str(rpc.value_call("version"))
            if self._server_version != "0.9.16":
                raise RuntimeError(
                    f"interactive bridge requires CARLA 0.9.16, got {self._server_version}"
                )
            episode_id = rpc.episode_id()
            map_info = rpc.value_call("get_map_info")
            if not isinstance(map_info, list) or len(map_info) < 2 or not map_info[1]:
                raise RuntimeError("CARLA map exposes no spawn points")
            self._map = _map_short_name(str(map_info[0]))
            spawn_transform: list[Any] | None = None
            if self._world_worker is not None:
                with self._lock:
                    worker_scene = self._worker_scene
                if worker_scene is None or worker_scene.ego_actor_id is None:
                    raise RuntimeError("World Worker scene is not ready")
                if worker_scene.episode_id != episode_id:
                    raise RuntimeError(
                        "World Worker scene episode does not match the camera bridge episode"
                    )
                if (
                    worker_scene.map_name not in {None, "current"}
                    and _map_short_name(worker_scene.map_name) != self._map
                ):
                    raise RuntimeError(
                        "World Worker scene map does not match the camera bridge map"
                    )
                vehicle_id = worker_scene.ego_actor_id
                verified_vehicle = rpc.actor(vehicle_id)
                if verified_vehicle is None or not str(verified_vehicle[2][1]).startswith(
                    "vehicle."
                ):
                    raise RuntimeError("World Worker ego actor did not verify as a vehicle")
                spawn_index = worker_scene.spawn_index
                self._vehicle_id = vehicle_id
                self._spawn_index = spawn_index
                self._prop_ids = list(worker_scene.prop_actor_ids)
            else:
                original_weather = [
                    float(value) for value in rpc.value_call("get_weather_parameters")
                ]
                if self.config.weather_preset != "keep":
                    rpc.void_call(
                        "set_weather_parameters",
                        weather_payload(self.config.weather_preset),
                    )
                    weather_changed = True
                vehicle_id, spawn_transform, spawn_index = _spawn_vehicle(
                    rpc,
                    blueprint=self.config.vehicle_blueprint,
                    color=self.config.color,
                    spawn_points=list(map_info[1]),
                    seed=self.config.seed,
                )
                self._vehicle_id = vehicle_id
                self._spawn_index = spawn_index
                owned_actor_ids.append((vehicle_id, "vehicle"))
                rpc.apply_vehicle_control(vehicle_id, ControlCommand.parked().as_carla())

                spawned_props = _spawn_props(
                    rpc,
                    preset=self.config.prop_preset,
                    ego_start=spawn_transform,
                )
                for prop_id, blueprint in spawned_props:
                    owned_actor_ids.append((prop_id, f"prop:{blueprint}"))
                    self._prop_ids.append(prop_id)

            compressed_camera = bool(
                self._world_worker is not None
                and worker_scene is not None
                and worker_scene.capabilities.get("compressed_camera_relay")
            )
            if compressed_camera:
                assert self._world_worker is not None
                assert worker_scene is not None
                with self._worker_request_lock:
                    camera_response = self._world_worker.start_camera(
                        worker_scene,
                        mode="drive",
                        width=self.config.width,
                        height=self.config.height,
                        fps=self.config.camera_fps,
                        fov=self.config.camera_fov,
                    )
                camera_payload = camera_response.get("camera")
                if not isinstance(camera_payload, Mapping):
                    raise RuntimeError("World Worker camera response is malformed")
                camera_id = int(camera_payload["actor_id"])
                stream = WorldWorkerCameraStream(
                    self._world_worker,
                    worker_scene,
                    timeout=max(8.0, 4.0 / self.config.camera_fps),
                )
            else:
                camera = spawn_front_camera(
                    rpc,
                    vehicle_id,
                    width=self.config.width,
                    height=self.config.height,
                    sensor_tick=1.0 / self.config.camera_fps,
                    fov=self.config.camera_fov,
                )
                camera_id = int(camera[0])
                owned_actor_ids.append((camera_id, "camera"))
                _validate_camera_attachment(camera, vehicle_id)
                if not camera[5]:
                    raise RuntimeError("spawned front camera did not expose a stream token")
                stream = CarlaCameraStream(
                    self.config.host,
                    camera[5],
                    timeout=max(8.0, 4.0 / self.config.camera_fps),
                )
            self._camera_id = camera_id
            stream.wait_for_frame(timeout=10.0)

            if self._world_worker is None:
                actuator = SafeActuator(
                    self.config.host,
                    self.config.port,
                    vehicle_id,
                    # The browser lease already substitutes full brake at 0.40s,
                    # while camera freshness has its own threshold. This independent
                    # process timeout is only the second boundary for a dead owner
                    # and must tolerate normal LAN/UE scheduling.
                    heartbeat_timeout=1.5,
                )
            if self.config.detector_enabled:
                detector = create_detector(
                    DetectorConfig(
                        backend=self.config.detector,
                        weights=self.config.weights,
                        device=self.config.device,
                        image_size=self.config.image_size,
                        confidence=self.config.confidence,
                    )
                )
                self._detector_name = detector.name
                perception = PerceptionWorker(detector)

            if self.config.record_video:
                raw_recorder = AsyncVideoRecorder(
                    raw_video_path,
                    frame_size=(self.config.width, self.config.height),
                    fps=self.config.camera_fps,
                )
                if self.config.detector_enabled:
                    overlay_recorder = AsyncVideoRecorder(
                        overlay_video_path,
                        frame_size=(self.config.width, self.config.height),
                        fps=self.config.camera_fps,
                    )

            if self.config.spectator_follow:
                try:
                    spectator = rpc.spectator()
                    spectator_id = int(spectator[0])
                    spectator_episode_id = episode_id
                    spectator_transform = rpc.actor_transform(spectator_id, "Camera")
                    spectator_follow_active = True
                except Exception as error:
                    self._cleanup_errors.append(f"spectator setup: {error}")
                    spectator_follow_active = False

            if self._world_worker is not None:
                self._activate_worker_scene()

            _json_line(
                events_stream,
                {
                    "event": "session_started",
                    "at": _utc_now(),
                    "episode_id": episode_id,
                    "map": self._map,
                    "vehicle_id": vehicle_id,
                    "camera_id": camera_id,
                    "prop_ids": list(self._prop_ids),
                    "prop_preset": self.config.prop_preset,
                    "spawn_index": spawn_index,
                    "spawn_transform": spawn_transform,
                    "weather_preset": self.config.weather_preset,
                    "world_worker_scene_id": self._worker_scene_id,
                    "camera_transport": ("worker_jpeg" if compressed_camera else "carla_raw_bgra"),
                    "control_mode": self._control_mode,
                    "model_output_actuated": False,
                },
            )
            self._set_status("running")

            renderer = OverlayRenderer(stale_after_seconds=2.0)
            while not self._stop_event.is_set():
                self._drain_pending_events(events_stream)
                now = time.monotonic()
                frame = stream.latest()
                if frame is not None and frame.sequence > last_camera_sequence:
                    last_camera_sequence = frame.sequence
                    last_camera_received = frame.received_monotonic
                    latest_raw = frame.bgr()
                    source_jpeg = getattr(frame, "jpeg", None)
                    browser_frame = (
                        bytes(source_jpeg) if isinstance(source_jpeg, bytes) else _jpeg(latest_raw)
                    )
                    self._cache_frame("raw", frame.sequence, browser_frame)
                    self._frames_seen += 1
                    if raw_recorder is not None:
                        raw_recorder.submit(frame.sequence, latest_raw)
                    if perception is not None and not detector_failed:
                        try:
                            perception.submit(frame)
                        except Exception as error:
                            detector_failed = True
                            self._cleanup_errors.append(f"detector submit: {error}")
                    if spectator_id is not None and spectator_follow_active:
                        try:
                            rpc.set_actor_transform(
                                spectator_id,
                                spectator_chase_transform(
                                    vehicle_transform_from_front_camera(frame)
                                ),
                            )
                        except Exception as error:
                            self._cleanup_errors.append(f"spectator follow: {error}")
                            spectator_follow_active = False

                if perception is not None and not detector_failed:
                    try:
                        result = perception.latest()
                    except Exception as error:
                        detector_failed = True
                        self._cleanup_errors.append(f"detector runtime: {error}")
                        result = None
                    if result is not None and result.sequence > last_result_sequence:
                        last_result_sequence = result.sequence
                        with self._lock:
                            control_mode = self._control_mode
                        latest_overlay = renderer.render(
                            result,
                            now_monotonic=result.completed_monotonic,
                            hud={
                                "CONTROL": (
                                    "CARLA / AUTOPILOT"
                                    if control_mode == "autopilot"
                                    else "HUMAN / BROWSER"
                                ),
                                "MODEL": "ADVISORY ONLY",
                            },
                        )
                        self._cache_frame("overlay", result.sequence, _jpeg(latest_overlay))
                        self._write_detections(detections_stream, result)

                if (
                    overlay_recorder is not None
                    and latest_overlay is not None
                    and last_camera_sequence >= 0
                ):
                    # Submit at camera cadence. Repeating the latest exact model
                    # frame preserves wall-clock duration when inference is slower.
                    overlay_recorder.submit(last_camera_sequence, latest_overlay)

                requested_weather: str | None
                with self._lock:
                    requested_weather = self._requested_weather
                    self._requested_weather = None
                if requested_weather is not None:
                    if self._world_worker is not None:
                        with self._worker_request_lock:
                            with self._lock:
                                worker_scene = self._worker_scene
                            if worker_scene is None:
                                raise RuntimeError("World Worker scene is not ready")
                            updated_scene = self._world_worker.weather(
                                worker_scene,
                                requested_weather,
                            )
                        with self._lock:
                            self._worker_scene = updated_scene
                    else:
                        rpc.void_call(
                            "set_weather_parameters",
                            weather_payload(requested_weather),
                        )
                        weather_changed = True
                    with self._lock:
                        self._weather_preset = requested_weather
                    _json_line(
                        events_stream,
                        {
                            "event": "weather_changed",
                            "at": _utc_now(),
                            "preset": requested_weather,
                        },
                    )

                if now - last_control_at >= _CONTROL_PERIOD_SECONDS:
                    with self._lock:
                        control_mode = self._control_mode
                    if control_mode == "manual":
                        camera_stale = (
                            last_camera_received <= 0.0
                            or now - last_camera_received > max(1.5, 4.0 / self.config.camera_fps)
                        )
                        command, source, input_age = self._command(
                            now,
                            camera_stale=camera_stale,
                        )
                        command_sent = True
                        if self._world_worker is not None:
                            with self._worker_request_lock:
                                with self._lock:
                                    if self._control_mode != "manual":
                                        command_sent = False
                                        worker_scene = None
                                        worker_sequence = 0
                                    else:
                                        worker_scene = self._worker_scene
                                        self._worker_control_sequence += 1
                                        worker_sequence = self._worker_control_sequence
                                if command_sent:
                                    if worker_scene is None:
                                        raise RuntimeError("World Worker scene is not ready")
                                    updated_scene = self._world_worker.control(
                                        worker_scene,
                                        {
                                            "sequence": worker_sequence,
                                            "throttle": command.throttle,
                                            "steer": command.steer,
                                            "brake": command.brake,
                                            "hand_brake": command.hand_brake,
                                            "reverse": command.reverse,
                                        },
                                    )
                                    with self._lock:
                                        self._worker_scene = updated_scene
                        else:
                            assert actuator is not None
                            actuator.send(command)
                        if command_sent:
                            self._write_control(
                                controls_stream,
                                command=command,
                                source=source,
                                input_age=input_age,
                                camera_sequence=last_camera_sequence,
                                detector_sequence=last_result_sequence,
                            )
                        else:
                            with self._lock:
                                self._deadman_active = False
                                self._control_source = "worker_autopilot"
                    else:
                        with self._lock:
                            self._deadman_active = False
                            self._control_source = "worker_autopilot"
                    last_control_at = now

                if now - last_telemetry_at >= _TELEMETRY_PERIOD_SECONDS:
                    telemetry = rpc.telemetry(vehicle_id)
                    with self._lock:
                        self._telemetry = {
                            "speed": telemetry.speed,
                            "gear": telemetry.gear,
                            "throttle": telemetry.throttle,
                            "steer": telemetry.steer,
                            "brake": telemetry.brake,
                        }
                    last_telemetry_at = now
                time.sleep(0.01)
            if self._world_worker is not None:
                with self._lock:
                    heartbeat_error = self._worker_heartbeat_error
                if heartbeat_error is not None:
                    raise RuntimeError(f"World Worker heartbeat failed: {heartbeat_error}")
        finally:
            self._set_status("stopping")
            self._drain_pending_events(events_stream)
            if self._world_worker is not None:
                try:
                    # The worker owns the ego/world. Stop that lease first; the
                    # camera is a locally-owned child and may already disappear
                    # with its parent, so raw cleanup below verifies before destroy.
                    self._stop_worker_scene()
                except Exception as error:
                    self._cleanup_errors.append(f"World Worker stop: {error}")
            if actuator is not None:
                try:
                    actuator.stop()
                except Exception as error:
                    self._cleanup_errors.append(f"actuator stop: {error}")
            if stream is not None:
                try:
                    stream.close()
                except Exception as error:
                    self._cleanup_errors.append(f"camera stream close: {error}")
            if perception is not None:
                try:
                    perception.close()
                except Exception as error:
                    self._cleanup_errors.append(f"detector close: {error}")
            for name, recorder in (("raw", raw_recorder), ("overlay", overlay_recorder)):
                if recorder is not None:
                    try:
                        recorder.close()
                    except Exception as error:
                        self._cleanup_errors.append(f"{name} recorder close: {error}")
            with self._lock:
                self._recording = False
            for stream_handle in (controls_stream, detections_stream, events_stream):
                if stream_handle is not None:
                    try:
                        stream_handle.close()
                    except Exception as error:
                        self._cleanup_errors.append(f"log close: {error}")

            if rpc is not None:
                same_episode = False
                try:
                    same_episode = episode_id is not None and rpc.episode_id() == episode_id
                except Exception as error:
                    self._cleanup_errors.append(f"episode cleanup check: {error}")
                if same_episode:
                    for actor_id, kind in reversed(owned_actor_ids):
                        try:
                            actor = rpc.actor(actor_id)
                            if actor is None:
                                continue
                            actor_type = str(actor[2][1])
                            expected = (
                                "sensor.camera."
                                if kind == "camera"
                                else kind.removeprefix("prop:")
                                if kind.startswith("prop:")
                                else "vehicle."
                            )
                            if not actor_type.startswith(expected):
                                self._cleanup_errors.append(
                                    f"skip destroy {actor_id}: type changed to {actor_type}"
                                )
                                continue
                            rpc.destroy_actor(actor_id)
                        except Exception as error:
                            self._cleanup_errors.append(f"destroy {kind} {actor_id}: {error}")
                    if (
                        spectator_transform is not None
                        and spectator_episode_id == episode_id
                        and spectator_id is not None
                    ):
                        try:
                            current_spectator = rpc.spectator()
                            if int(current_spectator[0]) == spectator_id:
                                rpc.set_actor_transform(spectator_id, spectator_transform)
                        except Exception as error:
                            self._cleanup_errors.append(f"spectator restore: {error}")
                    if original_weather is not None and weather_changed:
                        try:
                            rpc.void_call("set_weather_parameters", original_weather)
                        except Exception as error:
                            self._cleanup_errors.append(f"weather restore: {error}")
                else:
                    self._cleanup_errors.append(
                        "episode changed; skipped actor destruction and global-state restore"
                    )
                try:
                    rpc.close()
                except Exception as error:
                    self._cleanup_errors.append(f"RPC close: {error}")
            self._register_outputs(
                tracker,
                controls_path=controls_path,
                detections_path=detections_path,
                events_path=events_path,
                raw_video_path=raw_video_path,
                overlay_video_path=overlay_video_path,
                latest_raw=latest_raw,
                latest_overlay=latest_overlay,
            )

    def _register_outputs(
        self,
        tracker: RunArtifactTracker,
        *,
        controls_path: Path,
        detections_path: Path,
        events_path: Path,
        raw_video_path: Path,
        overlay_video_path: Path,
        latest_raw: np.ndarray | None,
        latest_overlay: np.ndarray | None,
    ) -> None:
        for path, role in (
            (controls_path, "browser_manual_controls"),
            (detections_path, "exact_frame_model_detections"),
            (events_path, "interactive_drive_events"),
            (raw_video_path, "raw_drive_video"),
            (overlay_video_path, "advisory_model_overlay_video"),
        ):
            if not path.is_file():
                continue
            try:
                tracker.register_artifact(
                    path,
                    role=role,
                    metadata={"model_output_actuated": False},
                )
            except Exception as error:
                self._cleanup_errors.append(f"register {role}: {error}")

        if latest_raw is not None:
            try:
                latest_raw_path = tracker.artifact_path("latest-raw.jpg")
                if not cv2.imwrite(str(latest_raw_path), latest_raw):
                    raise RuntimeError("OpenCV did not write latest raw frame")
                tracker.register_artifact(latest_raw_path, role="latest_raw_drive_frame")
            except Exception as error:
                self._cleanup_errors.append(f"latest raw frame: {error}")
        if latest_overlay is not None:
            try:
                latest_overlay_path = tracker.artifact_path("latest-overlay.jpg")
                if not cv2.imwrite(str(latest_overlay_path), latest_overlay):
                    raise RuntimeError("OpenCV did not write latest overlay frame")
                tracker.register_artifact(
                    latest_overlay_path,
                    role="latest_advisory_model_frame",
                    metadata={"model_output_actuated": False},
                )
            except Exception as error:
                self._cleanup_errors.append(f"latest overlay frame: {error}")

    def _command(
        self,
        now: float,
        *,
        camera_stale: bool,
    ) -> tuple[ControlCommand, str, float | None]:
        with self._lock:
            emergency = self._emergency
            latest = self._last_input
        age = None if latest is None else max(0.0, now - latest[1])
        if emergency:
            command = ControlCommand.service_brake()
            source = "emergency_stop"
        elif camera_stale:
            command = ControlCommand.service_brake()
            source = "camera_deadman"
        elif latest is None or age is None or age > _BROWSER_LEASE_SECONDS:
            command = ControlCommand.service_brake(steer=0.0 if latest is None else latest[0].steer)
            source = "browser_deadman"
        else:
            command = latest[0].command(max_throttle=self.config.max_throttle)
            source = "browser_manual"
            with self._lock:
                speed = abs(float(self._telemetry.get("speed", 0.0)))
            if command.reverse and speed > 0.5:
                command = ControlCommand.service_brake(steer=command.steer)
                source = "reverse_interlock"
        deadman = source != "browser_manual"
        with self._lock:
            self._deadman_active = deadman
            self._control_source = source
        if deadman:
            self._deadman_commands += 1
        else:
            self._manual_commands += 1
        return command, source, age

    def _write_control(
        self,
        stream: TextIO | None,
        *,
        command: ControlCommand,
        source: str,
        input_age: float | None,
        camera_sequence: int,
        detector_sequence: int,
    ) -> None:
        with self._lock:
            telemetry = dict(self._telemetry)
            requested_sequence = None if self._last_input is None else self._last_input[0].sequence
        _json_line(
            stream,
            {
                "at": _utc_now(),
                "elapsed_seconds": time.monotonic() - self._started_monotonic,
                "input_sequence": requested_sequence,
                "input_age_seconds": input_age,
                "source": source,
                "failsafe": source != "browser_manual",
                "applied": {
                    "throttle": command.throttle,
                    "steer": command.steer,
                    "brake": command.brake,
                    "hand_brake": command.hand_brake,
                    "reverse": command.reverse,
                },
                "telemetry": telemetry,
                "camera_sequence": camera_sequence,
                "detector_sequence": detector_sequence,
                "model_output_actuated": False,
            },
        )
        self._controls_written += 1

    def _write_detections(self, stream: TextIO | None, result: PerceptionResult) -> None:
        _json_line(
            stream,
            {
                "at": _utc_now(),
                "sequence": result.sequence,
                "carla_frame": result.carla_frame,
                "source_timestamp": result.source_timestamp,
                "inference_seconds": result.inference_seconds,
                "detector": result.detector_name,
                "model_output_actuated": False,
                "detections": [
                    {
                        "class_id": item.class_id,
                        "source_class_id": item.source_class_id,
                        "label": item.label,
                        "confidence": item.confidence,
                        "xyxy": list(item.xyxy),
                    }
                    for item in result.detections
                ],
            },
        )
        self._detections_written += 1

    def _finalize(
        self,
        tracker: RunArtifactTracker,
        failure: BaseException | None,
    ) -> None:
        summary_path = tracker.artifact_path("summary.json")
        with self._lock:
            mode_history = [dict(item) for item in self._mode_history]
        summary = {
            "schema_version": "1.0",
            "object_type": "interactive_drive_session_summary",
            "status": "failed" if failure is not None else "success",
            "run_id": self.config.run_id,
            "session_id": self.session_id,
            "map": self._map,
            "server_version": self._server_version,
            "vehicle_id": self._vehicle_id,
            "camera_id": self._camera_id,
            "prop_ids": list(self._prop_ids),
            "spawn_index": self._spawn_index,
            "world_worker_enabled": self._world_worker is not None,
            "world_worker_scene_id": self._worker_scene_id,
            "world_worker_cleanup_guard_passed": self._worker_cleanup_guard_passed,
            "control_mode": self._control_mode,
            "initial_control_mode": self.config.initial_control_mode,
            "final_control_mode": self._control_mode,
            "mode_history": mode_history,
            "route_mode": self.config.route_mode,
            "route": dict(self._route),
            "destination": self._destination,
            "traffic_count": self.config.traffic_count,
            "traffic_count_requested": self.config.traffic_count,
            "traffic_count_actual": self._traffic_count_actual,
            "walker_count": self.config.walker_count,
            "walker_count_requested": self.config.walker_count,
            "walker_count_actual": self._walker_count_actual,
            "duration_seconds": time.monotonic() - self._started_monotonic,
            "frames_seen": self._frames_seen,
            "controls_written": self._controls_written,
            "detections_written": self._detections_written,
            "manual_commands": self._manual_commands,
            "deadman_commands": self._deadman_commands,
            "experiment_preset": self.config.experiment_preset,
            "human_marker_counts": dict(self._marker_counts),
            "human_markers_written": sum(self._marker_counts.values()),
            "model_output_actuated": False,
            "recording_requested": self.config.record_video,
            "stop_reason": self._stop_reason,
            "failure": (
                None
                if failure is None
                else {"type": type(failure).__name__, "message": str(failure)}
            ),
            "cleanup_errors": list(self._cleanup_errors),
        }
        try:
            _write_json(summary_path, summary)
            tracker.register_artifact(summary_path, role="interactive_drive_summary")
        except Exception as error:
            self._cleanup_errors.append(f"summary finalization: {error}")


class DriveSessionManager:
    """Thread-safe one-active-drive coordinator used by HTTP request threads."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        carla_host: str,
        carla_port: int,
        session_factory: Callable[..., DriveSession] = DriveSession,
        rpc_factory: Callable[..., CarlaRpc] = CarlaRpc,
        world_worker: WorldWorkerClient | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve(strict=True)
        self.carla_host = str(carla_host)
        self.carla_port = int(carla_port)
        self._session_factory = session_factory
        self._rpc_factory = rpc_factory
        self._world_worker = world_worker
        self._lock = threading.RLock()
        self._session: DriveSession | None = None

    def catalog(self) -> dict[str, Any]:
        base = {
            "schema_version": "1.0",
            "connected": False,
            "host": self.carla_host,
            "port": self.carla_port,
            "server_version": None,
            "map": None,
            "maps": [],
            "vehicles": [],
            "spawn_count": 0,
            "weather_presets": [
                {"id": "keep", "label": "Keep current weather"},
                *[
                    {"id": preset, "label": preset.replace("-", " ").title()}
                    for preset in WEATHER_PRESETS
                ],
            ],
            "prop_presets": [
                {"id": preset, "label": preset.replace("-", " ").title()} for preset in PROP_PRESETS
            ],
            "capabilities": {
                "manual_drive": True,
                "random_road_start": True,
                "model_advisory": True,
                "recording": True,
                "weather": True,
                "scene_props": True,
                "spectator_follow": True,
                "map_reload": False,
                "random_route": False,
                "traffic_manager": False,
                "walkers": False,
                "autopilot": False,
                "native_worker": False,
            },
            "world_worker": {
                "configured": self._world_worker is not None,
                "connected": False,
            },
        }
        try:
            with self._rpc_factory(self.carla_host, self.carla_port, timeout=4.0) as rpc:
                map_info = rpc.value_call("get_map_info")
                base.update(
                    {
                        "connected": True,
                        "server_version": str(rpc.value_call("version")),
                        "map": _map_short_name(str(map_info[0])),
                        "maps": list(rpc.value_call("get_available_maps")),
                        "vehicles": _vehicle_catalog(rpc.value_call("get_actor_definitions")),
                        "spawn_count": len(map_info[1]),
                    }
                )
        except Exception as error:
            base["error"] = f"{type(error).__name__}: {error}"
        if self._world_worker is not None:
            try:
                health = self._world_worker.health()
                if not _world_worker_health_ready(health):
                    raise RuntimeError("World Worker reports that CARLA is unavailable")
                worker_payload = self._world_worker.catalog()
                worker_catalog = worker_payload.get("catalog", worker_payload)
                if not isinstance(worker_catalog, Mapping):
                    raise RuntimeError("World Worker catalog must be an object")
                worker_capabilities = worker_catalog.get("capabilities", {})
                if not isinstance(worker_capabilities, Mapping):
                    raise RuntimeError("World Worker capabilities must be an object")
                for key, value in worker_capabilities.items():
                    if isinstance(value, bool):
                        base["capabilities"][str(key)] = value
                base["capabilities"]["native_worker"] = True

                maps = worker_catalog.get("maps")
                if isinstance(maps, list):
                    base["maps"] = maps
                vehicles = worker_catalog.get("vehicles")
                if isinstance(vehicles, list):
                    base["vehicles"] = vehicles
                carla_facts = worker_catalog.get("carla", {})
                if not isinstance(carla_facts, Mapping):
                    carla_facts = {}
                current_map = worker_catalog.get(
                    "current_map",
                    worker_catalog.get("map", carla_facts.get("current_map")),
                )
                if isinstance(current_map, str) and current_map.strip():
                    base["map"] = _map_short_name(current_map.strip())
                worker_server_version = carla_facts.get("server_version")
                if isinstance(worker_server_version, str) and worker_server_version.strip():
                    base["server_version"] = worker_server_version.strip()
                spawn_count = worker_catalog.get("spawn_count")
                if (
                    isinstance(spawn_count, int)
                    and not isinstance(spawn_count, bool)
                    and spawn_count >= 0
                ):
                    base["spawn_count"] = spawn_count
                weather_presets = worker_catalog.get("weather_presets")
                if isinstance(weather_presets, list) and weather_presets:
                    base["weather_presets"] = weather_presets
                prop_presets = worker_catalog.get("prop_presets")
                if isinstance(prop_presets, list) and prop_presets:
                    base["prop_presets"] = prop_presets
                base["world_worker"] = {
                    "configured": True,
                    "connected": True,
                    "status": health.get("status", "reachable"),
                }
            except Exception as error:
                base["world_worker"] = {
                    "configured": True,
                    "connected": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        return base

    def start(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        active_world_worker: WorldWorkerClient | None = None
        if self._world_worker is not None:
            try:
                health = self._world_worker.health()
                if not _world_worker_health_ready(health):
                    raise RuntimeError("World Worker reports that CARLA is unavailable")
            except Exception:
                # A configured but unreachable worker must not silently claim
                # world features. Exact legacy defaults may still drive through
                # the existing raw bridge; non-default worker fields are rejected
                # by DriveStartConfig below.
                active_world_worker = None
            else:
                active_world_worker = self._world_worker
        config = DriveStartConfig.from_mapping(
            raw,
            workspace=self.workspace,
            expected_host=self.carla_host,
            expected_port=self.carla_port,
            world_worker_configured=active_world_worker is not None,
        )
        with self._lock:
            if self._session is not None and self._session.snapshot()["status"] in _ACTIVE:
                raise RuntimeError("another interactive drive session is already active")
            if active_world_worker is None:
                session = self._session_factory(config, workspace=self.workspace)
            else:
                session = self._session_factory(
                    config,
                    workspace=self.workspace,
                    world_worker=active_world_worker,
                )
            self._session = session
            session.start()
            return session.snapshot()

    def state(self) -> dict[str, Any]:
        with self._lock:
            session = self._session
        if session is None:
            return {
                "schema_version": "1.0",
                "status": "idle",
                "session_id": None,
                "run_id": None,
                "recording": False,
                "detector": {
                    "enabled": False,
                    "name": None,
                    "advisory_only": True,
                    "actuated": False,
                },
                "deadman_active": True,
                "error": None,
            }
        return session.snapshot()

    def control(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        control = DriveInput.from_mapping(raw)
        return self._require_session(control.session_id).submit_control(control)

    def emergency_stop(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        session_id = self._session_id(raw)
        return self._require_session(session_id).emergency_stop()

    def weather(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {"session_id", "preset"}
        unknown = sorted(str(key) for key in raw if str(key) not in allowed)
        if unknown:
            raise ValueError(f"weather request has unknown fields: {', '.join(unknown)}")
        if set(raw) != allowed:
            raise ValueError("weather request requires session_id and preset")
        session_id = str(raw["session_id"]).strip()
        preset = str(raw["preset"]).strip()
        return self._require_session(session_id).request_weather(preset)

    def mode(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {"session_id", "mode"}
        unknown = sorted(str(key) for key in raw if str(key) not in allowed)
        if unknown:
            raise ValueError(f"mode request has unknown fields: {', '.join(unknown)}")
        if set(raw) != allowed:
            raise ValueError("mode request requires session_id and mode")
        session_id = str(raw["session_id"]).strip()
        mode = str(raw["mode"]).strip()
        return self._require_session(session_id).request_mode(mode)

    def mark(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {"session_id", "label", "note"}
        unknown = sorted(str(key) for key in raw if str(key) not in allowed)
        if unknown:
            raise ValueError(f"human marker request has unknown fields: {', '.join(unknown)}")
        required = {"session_id", "label"}
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError("human marker request is missing fields: " + ", ".join(missing))
        session_id = str(raw["session_id"]).strip()
        label = str(raw["label"]).strip()
        note = raw.get("note")
        if note is not None and not isinstance(note, str):
            raise TypeError("human marker note must be a string or null")
        return self._require_session(session_id).mark_human_event(label, note)

    def stop(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        session_id = self._session_id(raw)
        session = self._require_session(session_id)
        session.request_stop("operator_stop")
        session.join(timeout=20.0)
        return session.snapshot()

    def frame(self, view: str) -> tuple[int, bytes]:
        with self._lock:
            session = self._session
        if session is None:
            raise FileNotFoundError("no interactive drive session exists")
        return session.frame(view)

    def shutdown(self) -> None:
        with self._lock:
            session = self._session
        if session is None:
            return
        if session.snapshot()["status"] in _ACTIVE:
            session.request_stop("operator_server_shutdown")
            session.join(timeout=20.0)

    @staticmethod
    def _session_id(raw: Mapping[str, Any]) -> str:
        if set(raw) != {"session_id"}:
            raise ValueError("request requires only session_id")
        session_id = str(raw["session_id"]).strip()
        if not session_id:
            raise ValueError("session_id is required")
        return session_id

    def _require_session(self, session_id: str) -> DriveSession:
        with self._lock:
            session = self._session
        if session is None or session.session_id != session_id:
            raise KeyError("interactive drive session was not found")
        return session


__all__ = ["DriveSession", "DriveSessionManager"]
