"""Additive autonomous-drive modes for the browser Garage.

The existing :mod:`carla_vision.operator.drive` engine remains the source of
truth for actor ownership, browser camera streaming, recording, deadman safety,
emergency stop, and cleanup.  This module only replaces the command source and,
for model modes, attaches a separate RGB sensor used exclusively by the policy.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import random
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from ..controller import ControlCommand
from ..model_driver import (
    ModelDriverConfig,
    ModelObservation,
    control_from_value,
    create_driving_model,
)
from .drive import (
    DriveSession,
    DriveSessionManager,
    _json_line,
    _utc_now,
    _write_json,
)
from .drive_contracts import DriveInput, DriveStartConfig
from .world_worker_client import WorldWorkerClient

_PRODUCTION_CONTROL_MODES = frozenset({"manual"})
_EXPERIMENTAL_CONTROL_MODES = frozenset({"behavior", "imitation", "voxel"})
_CONTROL_MODES = _PRODUCTION_CONTROL_MODES | _EXPERIMENTAL_CONTROL_MODES
_AUTONOMOUS_MODES = _EXPERIMENTAL_CONTROL_MODES
_IMITATION_FACTORY = "carla_vision.imitation.predictor:create_driver"
_VOXEL_FACTORY = "carla_vision.voxel.model_examples.temporal_flow:create_predictor"


def _workspace_file(
    workspace: Path,
    raw: Any,
    *,
    name: str,
    suffixes: tuple[str, ...],
    required: bool,
) -> Path | None:
    value = str(raw or "").strip()
    if not value:
        if required:
            raise ValueError(f"{name} is required")
        return None
    candidate = (workspace / value).expanduser().resolve(strict=True)
    candidate.relative_to(workspace)
    if not candidate.is_file() or candidate.suffix.lower() not in suffixes:
        allowed = ", ".join(suffixes)
        raise ValueError(f"{name} must be a workspace-contained file ending in {allowed}")
    return candidate


def _integer(raw: Any, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= raw <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return raw


def _number(raw: Any, *, name: str, minimum: float, maximum: float) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError(f"{name} must be a number")
    value = float(raw)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return value


def _boolean(raw: Any, *, name: str) -> bool:
    if not isinstance(raw, bool):
        raise TypeError(f"{name} must be a boolean")
    return raw


@dataclass(frozen=True)
class GarageDriveStartConfig:
    """Drive configuration layered on top of the existing browser contract."""

    base: DriveStartConfig
    control_mode: str = "manual"
    behavior: str = "normal"
    acknowledge_autonomy: bool = False
    traffic_vehicles: int = 0
    walkers: int = 0
    tm_port: int = 8000
    target_speed_kmh: float = 35.0
    policy_checkpoint: Path | None = None
    policy_device: str = "cpu"
    voxel_readiness_report: Path | None = None
    max_policy_errors: int = 3
    max_model_speed_kmh: float = 45.0
    max_steer_rate: float = 2.5

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    @property
    def autonomous(self) -> bool:
        return self.control_mode in _AUTONOMOUS_MODES

    @property
    def model_output_actuated(self) -> bool:
        return self.control_mode in {"imitation", "voxel"}

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        workspace: Path,
        expected_host: str,
        expected_port: int,
        world_worker_configured: bool = False,
        experimental_enabled: bool = False,
    ) -> "GarageDriveStartConfig":
        extra = {
            "control_mode",
            "behavior",
            "acknowledge_autonomy",
            "traffic_vehicles",
            "walkers",
            "tm_port",
            "target_speed_kmh",
            "policy_checkpoint",
            "policy_device",
            "voxel_readiness_report",
            "max_policy_errors",
            "max_model_speed_kmh",
            "max_steer_rate",
        }
        base_raw = {key: value for key, value in raw.items() if key not in extra}
        base = DriveStartConfig.from_mapping(
            base_raw,
            workspace=workspace,
            expected_host=expected_host,
            expected_port=expected_port,
            world_worker_configured=world_worker_configured,
        )
        mode = str(raw.get("control_mode", "manual")).strip().lower()
        if mode not in _CONTROL_MODES:
            raise ValueError(f"control_mode must be one of {', '.join(sorted(_CONTROL_MODES))}")
        if mode in _EXPERIMENTAL_CONTROL_MODES and not experimental_enabled:
            raise PermissionError(
                f"control mode {mode!r} is experimental and disabled; "
                "restart the Garage server with --enable-experimental to opt in"
            )
        if mode in _AUTONOMOUS_MODES and base.detector_enabled:
            base = replace(base, detector_enabled=False, weights=None)
        behavior = str(raw.get("behavior", "normal")).strip().lower()
        if behavior not in {"cautious", "normal", "aggressive"}:
            raise ValueError("behavior must be cautious, normal, or aggressive")
        acknowledgement = _boolean(
            raw.get("acknowledge_autonomy", False),
            name="acknowledge_autonomy",
        )
        if mode in _AUTONOMOUS_MODES and not acknowledgement:
            raise ValueError("autonomous drive modes require explicit operator acknowledgement")
        if mode in _AUTONOMOUS_MODES and base.initial_control_mode == "autopilot":
            raise ValueError("Garage autonomy cannot run while World Worker autopilot owns control")

        checkpoint = _workspace_file(
            workspace,
            raw.get("policy_checkpoint", ""),
            name="policy_checkpoint",
            suffixes=(".pt", ".pth", ".ckpt"),
            required=mode in {"imitation", "voxel"},
        )
        readiness = _workspace_file(
            workspace,
            raw.get("voxel_readiness_report", ""),
            name="voxel_readiness_report",
            suffixes=(".json",),
            required=False,
        )
        if mode not in {"imitation", "voxel"} and checkpoint is not None:
            raise ValueError("policy_checkpoint requires control_mode=imitation or voxel")
        if mode != "voxel" and readiness is not None:
            raise ValueError("voxel_readiness_report requires control_mode=voxel")
        device = str(raw.get("policy_device", "cpu")).strip()
        if not device or len(device) > 64:
            raise ValueError("policy_device must be a non-empty device name")

        return cls(
            base=base,
            control_mode=mode,
            behavior=behavior,
            acknowledge_autonomy=acknowledgement,
            traffic_vehicles=_integer(
                raw.get("traffic_vehicles", 0),
                name="traffic_vehicles",
                minimum=0,
                maximum=200,
            ),
            walkers=_integer(raw.get("walkers", 0), name="walkers", minimum=0, maximum=200),
            tm_port=_integer(raw.get("tm_port", 8000), name="tm_port", minimum=1, maximum=65535),
            target_speed_kmh=_number(
                raw.get("target_speed_kmh", 35.0),
                name="target_speed_kmh",
                minimum=5.0,
                maximum=120.0,
            ),
            policy_checkpoint=checkpoint,
            policy_device=device,
            voxel_readiness_report=readiness,
            max_policy_errors=_integer(
                raw.get("max_policy_errors", 3),
                name="max_policy_errors",
                minimum=1,
                maximum=20,
            ),
            max_model_speed_kmh=_number(
                raw.get("max_model_speed_kmh", 45.0),
                name="max_model_speed_kmh",
                minimum=5.0,
                maximum=160.0,
            ),
            max_steer_rate=_number(
                raw.get("max_steer_rate", 2.5),
                name="max_steer_rate",
                minimum=0.1,
                maximum=10.0,
            ),
        )

    def manifest_config(self) -> dict[str, Any]:
        payload = self.base.manifest_config()
        payload.update(
            {
                "garage_mode": self.control_mode,
                "control_owner": (
                    "world_worker_autopilot"
                    if self.control_mode == "manual"
                    and self.base.initial_control_mode == "autopilot"
                    else "browser_manual"
                    if self.control_mode == "manual"
                    else self.control_mode
                ),
                "autonomy_output_actuated": self.autonomous,
                "model_output_actuated": self.model_output_actuated,
                "operator_acknowledged_autonomy": self.acknowledge_autonomy,
                "behavior": self.behavior,
                "garage_traffic_vehicles": self.traffic_vehicles,
                "garage_walkers": self.walkers,
                "tm_port": self.tm_port,
                "target_speed_kmh": self.target_speed_kmh,
                "policy_checkpoint": (
                    str(self.policy_checkpoint) if self.policy_checkpoint is not None else None
                ),
                "policy_device": self.policy_device,
                "voxel_readiness_report": (
                    str(self.voxel_readiness_report)
                    if self.voxel_readiness_report is not None
                    else None
                ),
            }
        )
        return payload


@dataclass(frozen=True)
class _PolicyFrame:
    frame: int
    timestamp: float
    bgr: np.ndarray
    received_monotonic: float


class _PolicyCamera:
    def __init__(self, carla: Any, world: Any, ego: Any, config: GarageDriveStartConfig) -> None:
        self._lock = threading.Lock()
        self._latest: _PolicyFrame | None = None
        blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("role_name", "garage_policy_front")
        blueprint.set_attribute("image_size_x", str(config.width))
        blueprint.set_attribute("image_size_y", str(config.height))
        blueprint.set_attribute("fov", str(config.camera_fov))
        blueprint.set_attribute("sensor_tick", str(1.0 / config.camera_fps))
        if blueprint.has_attribute("motion_blur_intensity"):
            blueprint.set_attribute("motion_blur_intensity", "0.0")
        transform = carla.Transform(
            carla.Location(x=1.5, z=1.7),
            carla.Rotation(pitch=0.0),
        )
        self.actor = world.spawn_actor(
            blueprint,
            transform,
            attach_to=ego,
            attachment_type=carla.AttachmentType.Rigid,
        )
        self.actor.listen(self._callback)

    def _callback(self, image: Any) -> None:
        pixels = np.frombuffer(image.raw_data, dtype=np.uint8)
        bgr = pixels.reshape(int(image.height), int(image.width), 4)[:, :, :3].copy()
        frame = _PolicyFrame(
            frame=int(image.frame),
            timestamp=float(image.timestamp),
            bgr=bgr,
            received_monotonic=time.monotonic(),
        )
        with self._lock:
            self._latest = frame

    def latest(self) -> _PolicyFrame | None:
        with self._lock:
            frame = self._latest
        if frame is None:
            return None
        return _PolicyFrame(
            frame=frame.frame,
            timestamp=frame.timestamp,
            bgr=frame.bgr.copy(),
            received_monotonic=frame.received_monotonic,
        )

    def close(self) -> None:
        actor = getattr(self, "actor", None)
        if actor is None:
            return
        self.actor = None
        try:
            actor.stop()
        except Exception as error:
            _ = error
        try:
            actor.destroy()
        except Exception as error:
            _ = error


class _CarlaContext:
    def __init__(self, config: GarageDriveStartConfig, vehicle_id: int) -> None:
        self.carla = importlib.import_module("carla")
        self.client = self.carla.Client(config.host, config.port)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.ego = self.world.get_actor(int(vehicle_id))
        if self.ego is None:
            raise RuntimeError("Garage autonomy could not attach to the spawned ego vehicle")
        self.spawn_points = list(self.world.get_map().get_spawn_points())
        if not self.spawn_points:
            raise RuntimeError("active CARLA map exposes no spawn points")
        self.rng = random.Random(config.seed)


class _Population:
    def __init__(self) -> None:
        self.vehicles: list[Any] = []
        self.walkers: list[Any] = []
        self.controllers: list[Any] = []
        self.started = False
        self.error: str | None = None

    def start(self, context: _CarlaContext, config: GarageDriveStartConfig) -> None:
        if self.started:
            return
        self.started = True
        try:
            self._spawn_vehicles(context, config)
            self._spawn_walkers(context, config)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            self.close()
            raise

    @staticmethod
    def _safe_vehicle_blueprints(world: Any) -> list[Any]:
        blueprints = sorted(world.get_blueprint_library().filter("vehicle.*"), key=lambda bp: bp.id)
        cars: list[Any] = []
        for blueprint in blueprints:
            if blueprint.has_attribute("base_type"):
                if str(blueprint.get_attribute("base_type")) == "car":
                    cars.append(blueprint)
            elif not any(token in blueprint.id for token in ("bike", "motorcycle", "microlino")):
                cars.append(blueprint)
        return cars or blueprints

    def _spawn_vehicles(self, context: _CarlaContext, config: GarageDriveStartConfig) -> None:
        if config.traffic_vehicles <= 0:
            return
        traffic_manager = context.client.get_trafficmanager(config.tm_port)
        candidates = list(context.spawn_points)
        context.rng.shuffle(candidates)
        blueprints = self._safe_vehicle_blueprints(context.world)
        ego_location = context.ego.get_location()
        for transform in candidates:
            if len(self.vehicles) >= config.traffic_vehicles:
                break
            if ego_location.distance(transform.location) < 8.0:
                continue
            blueprint = context.rng.choice(blueprints)
            if blueprint.has_attribute("role_name"):
                blueprint.set_attribute("role_name", "garage_traffic")
            if blueprint.has_attribute("color"):
                colors = list(blueprint.get_attribute("color").recommended_values)
                if colors:
                    blueprint.set_attribute("color", context.rng.choice(colors))
            actor = context.world.try_spawn_actor(blueprint, transform)
            if actor is None:
                continue
            actor.set_autopilot(True, config.tm_port)
            try:
                traffic_manager.auto_lane_change(actor, True)
                traffic_manager.vehicle_percentage_speed_difference(
                    actor,
                    12.0 + context.rng.uniform(-7.0, 9.0),
                )
                traffic_manager.distance_to_leading_vehicle(actor, 2.0)
                traffic_manager.update_vehicle_lights(actor, True)
            except Exception as error:
                _ = error
            self.vehicles.append(actor)

    def _spawn_walkers(self, context: _CarlaContext, config: GarageDriveStartConfig) -> None:
        if config.walkers <= 0:
            return
        library = context.world.get_blueprint_library()
        walker_blueprints = sorted(library.filter("walker.pedestrian.*"), key=lambda bp: bp.id)
        controller_blueprint = library.find("controller.ai.walker")
        if not walker_blueprints or controller_blueprint is None:
            raise RuntimeError("walker blueprints are unavailable")
        for _ in range(config.walkers):
            location = context.world.get_random_location_from_navigation()
            if location is None:
                continue
            blueprint = context.rng.choice(walker_blueprints)
            if blueprint.has_attribute("is_invincible"):
                blueprint.set_attribute("is_invincible", "false")
            walker = context.world.try_spawn_actor(blueprint, context.carla.Transform(location))
            if walker is None:
                continue
            controller = context.world.try_spawn_actor(
                controller_blueprint,
                context.carla.Transform(),
                attach_to=walker,
            )
            if controller is None:
                walker.destroy()
                continue
            self.walkers.append(walker)
            self.controllers.append(controller)
            destination = context.world.get_random_location_from_navigation()
            controller.start()
            if destination is not None:
                controller.go_to_location(destination)
            controller.set_max_speed(1.4)

    def close(self) -> None:
        for controller in reversed(self.controllers):
            try:
                controller.stop()
            except Exception as error:
                _ = error
        for actor in reversed([*self.controllers, *self.walkers, *self.vehicles]):
            try:
                actor.destroy()
            except Exception as error:
                _ = error
        self.controllers.clear()
        self.walkers.clear()
        self.vehicles.clear()


def _command_from_carla(control: Any) -> ControlCommand:
    throttle = float(control.throttle)
    brake = float(control.brake)
    if brake > 0.01:
        throttle = 0.0
    return ControlCommand(
        throttle=max(0.0, min(1.0, throttle)),
        steer=max(-1.0, min(1.0, float(control.steer))),
        brake=max(0.0, min(1.0, brake)),
        hand_brake=bool(getattr(control, "hand_brake", False)),
        reverse=bool(getattr(control, "reverse", False)),
    )


class _BehaviorPolicy:
    def __init__(self, context: _CarlaContext, config: GarageDriveStartConfig) -> None:
        module = importlib.import_module("agents.navigation.behavior_agent")
        self.context = context
        self.config = config
        self.agent = module.BehaviorAgent(
            context.ego,
            behavior=config.behavior,
            opt_dict={"target_speed": float(config.target_speed_kmh)},
        )
        self.destination_index: int | None = None
        self._set_destination()

    def _set_destination(self) -> None:
        current = self.context.ego.get_location()
        candidates = [
            (index, transform)
            for index, transform in enumerate(self.context.spawn_points)
            if current.distance(transform.location) >= 80.0
        ] or list(enumerate(self.context.spawn_points))
        self.destination_index, destination = self.context.rng.choice(candidates)
        self.agent.set_destination(destination.location)

    def step(self, *_: Any, **__: Any) -> tuple[ControlCommand, str, bool, dict[str, Any]]:
        if self.agent.done():
            self._set_destination()
        command = _command_from_carla(self.agent.run_step(debug=False))
        return command, "behavior_agent", False, {"destination_index": self.destination_index}

    def close(self) -> None:
        return None


class _ImitationPolicy:
    def __init__(self, context: _CarlaContext, config: GarageDriveStartConfig) -> None:
        self.config = config
        self.camera = _PolicyCamera(context.carla, context.world, context.ego, config)
        self.model = create_driving_model(
            _IMITATION_FACTORY,
            ModelDriverConfig(
                checkpoint=config.policy_checkpoint,
                device=config.policy_device,
                options={},
            ),
        )
        self.model.reset()
        self.last_frame = -1
        self.last_command = ControlCommand.service_brake()
        self.last_command_at = 0.0
        self.previous_steer = 0.0
        self.errors = 0
        self.latched_error: str | None = None

    def step(
        self,
        *,
        now: float,
        speed_mps: float,
        dt_s: float,
    ) -> tuple[ControlCommand, str, bool, dict[str, Any]]:
        if self.latched_error is not None:
            return (
                ControlCommand.service_brake(),
                "imitation_error_latched",
                True,
                {"error": self.latched_error},
            )
        frame = self.camera.latest()
        stale_limit = max(0.5, 4.0 / self.config.camera_fps)
        if frame is None or now - frame.received_monotonic > stale_limit:
            return ControlCommand.service_brake(), "imitation_camera_deadman", True, {}
        if frame.frame == self.last_frame and now - self.last_command_at <= stale_limit:
            return self.last_command, "imitation_model", False, {"frame": frame.frame}
        try:
            value = self.model.predict(
                ModelObservation(
                    frame=frame.frame,
                    timestamp=frame.timestamp,
                    image_bgr=frame.bgr,
                    speed_mps=max(0.0, speed_mps),
                    dt_seconds=max(1e-3, dt_s),
                )
            )
            control = control_from_value(value)
            max_delta = self.config.max_steer_rate * max(1e-3, dt_s)
            steer = max(
                self.previous_steer - max_delta,
                min(self.previous_steer + max_delta, float(control.steer)),
            )
            throttle = float(control.throttle)
            brake = float(control.brake)
            speed_kmh = speed_mps * 3.6
            if speed_kmh > self.config.max_model_speed_kmh:
                throttle = 0.0
                brake = max(
                    brake,
                    min(0.65, 0.15 + (speed_kmh - self.config.max_model_speed_kmh) / 15.0),
                )
            if brake > 0.01:
                throttle = 0.0
            command = ControlCommand(
                throttle=throttle,
                steer=steer,
                brake=brake,
                hand_brake=bool(control.hand_brake),
                reverse=bool(control.reverse),
            )
            self.last_frame = frame.frame
            self.last_command = command
            self.last_command_at = now
            self.previous_steer = steer
            self.errors = 0
            return command, "imitation_model", False, {"frame": frame.frame}
        except Exception as error:
            self.errors += 1
            if self.errors >= self.config.max_policy_errors:
                self.latched_error = f"{type(error).__name__}: {error}"
            return (
                ControlCommand.service_brake(),
                "imitation_error",
                True,
                {
                    "error": f"{type(error).__name__}: {error}",
                    "consecutive_errors": self.errors,
                },
            )

    def close(self) -> None:
        try:
            self.model.close()
        finally:
            self.camera.close()


class _VoxelPolicy:
    def __init__(self, context: _CarlaContext, config: GarageDriveStartConfig) -> None:
        from ..voxel.actuation_runtime import VoxelActuationRuntime
        from ..voxel.actuation_supervisor import VoxelActuationSupervisorPolicy
        from ..voxel.contracts import VoxelGridSpec
        from ..voxel.models import CameraVoxelModelConfig, create_camera_voxel_predictor
        from ..voxel.shadow import ShadowPlannerConfig

        self.config = config
        self.behavior = _BehaviorPolicy(context, config)
        self.camera = _PolicyCamera(context.carla, context.world, context.ego, config)
        predictor = create_camera_voxel_predictor(
            _VOXEL_FACTORY,
            CameraVoxelModelConfig(
                checkpoint=config.policy_checkpoint,
                device=config.policy_device,
                options={},
            ),
        )
        readiness: dict[str, Any] | None = None
        if config.voxel_readiness_report is not None:
            payload = json.loads(config.voxel_readiness_report.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("voxel readiness report must contain a JSON object")
            readiness = payload
        self.runtime = VoxelActuationRuntime(
            predictor,
            VoxelGridSpec(),
            planner_config=ShadowPlannerConfig(),
            supervisor_policy=VoxelActuationSupervisorPolicy(),
            readiness_report=readiness,
        )
        self.runtime.reset()
        self.last_frame = -1
        self.last_result: dict[str, Any] | None = None
        self.last_decision_at = 0.0
        self.previous_steer = 0.0
        self.errors = 0
        self.latched_error: str | None = None

    def step(
        self,
        *,
        now: float,
        speed_mps: float,
        dt_s: float,
    ) -> tuple[ControlCommand, str, bool, dict[str, Any]]:
        if self.latched_error is not None:
            return (
                ControlCommand.service_brake(),
                "voxel_error_latched",
                True,
                {"error": self.latched_error},
            )
        base_command, _, base_failsafe, base_detail = self.behavior.step()
        if base_failsafe:
            return ControlCommand.service_brake(), "voxel_behavior_fail_closed", True, base_detail
        frame = self.camera.latest()
        stale_limit = max(0.5, 4.0 / self.config.camera_fps)
        if frame is None or now - frame.received_monotonic > stale_limit:
            return ControlCommand.service_brake(), "voxel_camera_deadman", True, {}
        try:
            if frame.frame != self.last_frame:
                result = self.runtime.step(
                    frame=frame.frame,
                    timestamp=frame.timestamp,
                    rgb=frame.bgr[:, :, ::-1].copy(),
                    speed_mps=max(0.0, speed_mps),
                    previous_steering=self.previous_steer,
                    dt_s=max(1e-3, dt_s),
                )
                self.last_frame = frame.frame
                self.last_result = result.as_dict()
                self.last_decision_at = now
                self.errors = 0
                if result.decision.actuation_authorized and not result.decision.emergency_brake:
                    self.previous_steer = float(result.decision.steering)
            if self.last_result is None:
                return ControlCommand.service_brake(), "voxel_warmup", True, {}
            decision = self.last_result["decision"]
            if now - self.last_decision_at > 0.20:
                return (
                    ControlCommand.service_brake(),
                    "voxel_prediction_stale",
                    True,
                    self.last_result,
                )
            if decision["emergency_brake"] or not decision["actuation_authorized"]:
                return (
                    ControlCommand.service_brake(),
                    f"voxel_emergency:{decision['reason']}",
                    True,
                    self.last_result,
                )
            command = ControlCommand(
                throttle=base_command.throttle,
                steer=float(decision["steering"]),
                brake=base_command.brake,
                hand_brake=base_command.hand_brake,
                reverse=base_command.reverse,
            )
            return command, "voxel_planner", False, self.last_result
        except Exception as error:
            self.errors += 1
            if self.errors >= self.config.max_policy_errors:
                self.latched_error = f"{type(error).__name__}: {error}"
            return (
                ControlCommand.service_brake(),
                "voxel_error",
                True,
                {
                    "error": f"{type(error).__name__}: {error}",
                    "consecutive_errors": self.errors,
                },
            )

    def close(self) -> None:
        try:
            self.runtime.close()
        finally:
            self.camera.close()
            self.behavior.close()


class GarageDriveSession(DriveSession):
    """DriveSession that preserves the existing engine and swaps only command ownership."""

    config: GarageDriveStartConfig

    def __init__(
        self,
        config: GarageDriveStartConfig,
        *,
        workspace: Path,
        world_worker: WorldWorkerClient | None = None,
    ) -> None:
        super().__init__(  # type: ignore[arg-type]
            config,
            workspace=workspace,
            world_worker=world_worker,
        )
        self._garage_context: _CarlaContext | None = None
        self._population = _Population()
        self._policy: Any | None = None
        self._policy_lock = threading.RLock()
        self._last_policy_step = time.monotonic()
        self._autonomy_commands = 0
        self._autonomy_failsafes = 0
        self._policy_detail: dict[str, Any] = {}
        self._garage_closed = False

    def snapshot(self) -> dict[str, Any]:
        payload = super().snapshot()
        payload["garage_mode"] = self.config.control_mode
        payload["autonomy"] = {
            "enabled": self.config.autonomous,
            "operator_acknowledged": self.config.acknowledge_autonomy,
            "model_output_actuated": self.config.model_output_actuated,
            "policy_ready": self._policy is not None,
            "commands": self._autonomy_commands,
            "failsafes": self._autonomy_failsafes,
            "detail": dict(self._policy_detail),
            "traffic_vehicle_count": len(self._population.vehicles),
            "walker_count": len(self._population.walkers),
            "population_error": self._population.error,
        }
        return payload

    def submit_control(self, control: DriveInput) -> dict[str, Any]:
        if self.config.control_mode != "manual":
            raise RuntimeError(
                "browser manual control is disabled while an autonomous mode owns control"
            )
        return super().submit_control(control)

    def request_mode(self, mode: str) -> dict[str, Any]:
        if self.config.autonomous and str(mode).strip() == "autopilot":
            raise RuntimeError("World Worker autopilot cannot take over from Garage autonomy")
        return super().request_mode(mode)

    def request_stop(self, reason: str = "operator_stop") -> dict[str, Any]:
        self._close_garage_extensions()
        return super().request_stop(reason)

    def _ensure_context(self) -> _CarlaContext:
        with self._policy_lock:
            if self._garage_context is None:
                if self._vehicle_id is None:
                    raise RuntimeError("ego vehicle is not ready for Garage autonomy")
                self._garage_context = _CarlaContext(self.config, self._vehicle_id)
            return self._garage_context

    def _ensure_extensions(self) -> None:
        if self._garage_closed:
            raise RuntimeError("Garage extensions are closing")
        if (
            self.config.control_mode == "manual"
            and self.config.traffic_vehicles == 0
            and self.config.walkers == 0
        ):
            return
        context = self._ensure_context()
        self._population.start(context, self.config)
        if self.config.control_mode == "manual" or self._policy is not None:
            return
        if self.config.control_mode == "behavior":
            self._policy = _BehaviorPolicy(context, self.config)
        elif self.config.control_mode == "imitation":
            self._policy = _ImitationPolicy(context, self.config)
        elif self.config.control_mode == "voxel":
            self._policy = _VoxelPolicy(context, self.config)
        else:  # pragma: no cover - validated by the config contract
            raise RuntimeError(f"unsupported Garage control mode {self.config.control_mode!r}")

    def _command(
        self,
        now: float,
        *,
        camera_stale: bool,
    ) -> tuple[ControlCommand, str, float | None]:
        if self.config.control_mode == "manual":
            try:
                with self._policy_lock:
                    self._ensure_extensions()
            except Exception as error:
                self._policy_detail = {"setup_error": f"{type(error).__name__}: {error}"}
                self._autonomy_failsafes += 1
                with self._lock:
                    self._deadman_active = True
                    self._control_source = "garage_setup_fail_closed"
                return ControlCommand.service_brake(), "garage_setup_fail_closed", None
            return super()._command(now, camera_stale=camera_stale)

        with self._lock:
            emergency = self._emergency
            speed_mps = abs(float(self._telemetry.get("speed", 0.0)))
        if emergency:
            command = ControlCommand.service_brake()
            source = "emergency_stop"
            failsafe = True
            detail: dict[str, Any] = {}
        elif camera_stale:
            command = ControlCommand.service_brake()
            source = "camera_deadman"
            failsafe = True
            detail = {}
        else:
            try:
                with self._policy_lock:
                    self._ensure_extensions()
                    if self._policy is None:
                        raise RuntimeError("autonomous policy did not initialize")
                    dt_s = max(1e-3, min(0.25, now - self._last_policy_step))
                    self._last_policy_step = now
                    command, source, failsafe, detail = self._policy.step(
                        now=now,
                        speed_mps=speed_mps,
                        dt_s=dt_s,
                    )
            except Exception as error:
                command = ControlCommand.service_brake()
                source = "autonomy_setup_fail_closed"
                failsafe = True
                detail = {"error": f"{type(error).__name__}: {error}"}
        with self._lock:
            self._deadman_active = bool(failsafe)
            self._control_source = source
        self._policy_detail = dict(detail)
        if failsafe:
            self._autonomy_failsafes += 1
        else:
            self._autonomy_commands += 1
        return command, source, None

    def _write_control(
        self,
        stream: Any,
        *,
        command: ControlCommand,
        source: str,
        input_age: float | None,
        camera_sequence: int,
        detector_sequence: int,
    ) -> None:
        if self.config.control_mode == "manual":
            super()._write_control(
                stream,
                command=command,
                source=source,
                input_age=input_age,
                camera_sequence=camera_sequence,
                detector_sequence=detector_sequence,
            )
            return
        with self._lock:
            telemetry = dict(self._telemetry)
        _json_line(
            stream,
            {
                "at": _utc_now(),
                "elapsed_seconds": time.monotonic() - self._started_monotonic,
                "input_sequence": None,
                "input_age_seconds": None,
                "source": source,
                "failsafe": bool(self._deadman_active),
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
                "control_mode": self._control_mode,
                "garage_mode": self.config.control_mode,
                "autonomy_output_actuated": True,
                "model_output_actuated": self.config.model_output_actuated,
                "policy_detail": dict(self._policy_detail),
            },
        )
        self._controls_written += 1

    def _register_outputs(
        self,
        tracker: Any,
        *,
        controls_path: Path,
        detections_path: Path,
        events_path: Path,
        raw_video_path: Path,
        overlay_video_path: Path,
        latest_raw: np.ndarray | None,
        latest_raw_jpeg: bytes | None,
        latest_overlay: np.ndarray | None,
    ) -> None:
        if self.config.control_mode == "manual":
            return super()._register_outputs(
                tracker,
                controls_path=controls_path,
                detections_path=detections_path,
                events_path=events_path,
                raw_video_path=raw_video_path,
                overlay_video_path=overlay_video_path,
                latest_raw=latest_raw,
                latest_raw_jpeg=latest_raw_jpeg,
                latest_overlay=latest_overlay,
            )
        if events_path.is_file():
            rewritten: list[str] = []
            for line in events_path.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    rewritten.append(line)
                    continue
                if event.get("event") == "session_started":
                    event["garage_mode"] = self.config.control_mode
                    event["autonomy_output_actuated"] = True
                    event["model_output_actuated"] = self.config.model_output_actuated
                rewritten.append(
                    json.dumps(event, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                )
            events_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

        roles = (
            (controls_path, "garage_autonomous_controls"),
            (detections_path, "exact_frame_model_detections"),
            (events_path, "interactive_drive_events"),
            (raw_video_path, "raw_drive_video"),
            (overlay_video_path, "advisory_model_overlay_video"),
        )
        for path, role in roles:
            if not path.is_file():
                continue
            try:
                tracker.register_artifact(
                    path,
                    role=role,
                    metadata={
                        "control_mode": self._control_mode,
                        "garage_mode": self.config.control_mode,
                        "autonomy_output_actuated": role == "garage_autonomous_controls",
                        "model_output_actuated": (
                            self.config.model_output_actuated
                            if role == "garage_autonomous_controls"
                            else False
                        ),
                    },
                )
            except Exception as error:
                self._cleanup_errors.append(f"register {role}: {error}")
        if latest_raw_jpeg is not None or latest_raw is not None:
            try:
                path = tracker.artifact_path("latest-raw.jpg")
                if latest_raw_jpeg is not None:
                    path.write_bytes(latest_raw_jpeg)
                elif latest_raw is not None and not cv2.imwrite(str(path), latest_raw):
                    raise RuntimeError("OpenCV did not write latest raw frame")
                tracker.register_artifact(path, role="latest_raw_drive_frame")
            except Exception as error:
                self._cleanup_errors.append(f"latest raw frame: {error}")
        if latest_overlay is not None:
            try:
                path = tracker.artifact_path("latest-overlay.jpg")
                if not cv2.imwrite(str(path), latest_overlay):
                    raise RuntimeError("OpenCV did not write latest overlay frame")
                tracker.register_artifact(
                    path,
                    role="latest_advisory_model_frame",
                    metadata={"model_output_actuated": False},
                )
            except Exception as error:
                self._cleanup_errors.append(f"latest overlay frame: {error}")

    def _finalize(self, tracker: Any, failure: BaseException | None) -> None:
        garage_traffic_actual = len(self._population.vehicles)
        garage_walkers_actual = len(self._population.walkers)
        population_error = self._population.error
        self._close_garage_extensions()
        summary_path = tracker.artifact_path("summary.json")
        super()._finalize(tracker, failure)
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary.update(
                {
                    "garage_mode": self.config.control_mode,
                    "garage_traffic_vehicles_requested": self.config.traffic_vehicles,
                    "garage_traffic_vehicles_actual": garage_traffic_actual,
                    "garage_walkers_requested": self.config.walkers,
                    "garage_walkers_actual": garage_walkers_actual,
                    "garage_population_error": population_error,
                    "autonomy_commands": self._autonomy_commands,
                    "autonomy_failsafes": self._autonomy_failsafes,
                    "autonomy_output_actuated": self.config.autonomous,
                    "model_output_actuated": self.config.model_output_actuated,
                    "operator_acknowledged_autonomy": self.config.acknowledge_autonomy,
                }
            )
            _write_json(summary_path, summary)
            tracker.register_artifact(summary_path, role="interactive_drive_summary")
        except Exception as error:
            self._cleanup_errors.append(f"Garage summary finalization: {error}")

    def _close_garage_extensions(self) -> None:
        with self._policy_lock:
            if self._garage_closed:
                return
            self._garage_closed = True
            policy, self._policy = self._policy, None
            try:
                if policy is not None:
                    policy.close()
            except Exception as error:
                self._cleanup_errors.append(f"garage policy close: {error}")
            try:
                self._population.close()
            except Exception as error:
                self._cleanup_errors.append(f"garage population close: {error}")


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False


class GarageDriveSessionManager(DriveSessionManager):
    """One-session manager that accepts the additive Garage drive contract."""

    def __init__(self, *, experimental_enabled: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.experimental_enabled = bool(experimental_enabled)

    def catalog(self) -> dict[str, Any]:
        payload = super().catalog()
        pythonapi = _module_available("carla")
        behavior_agent = _module_available("agents.navigation.behavior_agent")
        capabilities = payload.setdefault("capabilities", {})
        capabilities.update(
            {
                "garage_traffic_population": pythonapi,
                "garage_walker_population": pythonapi,
                "garage_experimental": self.experimental_enabled,
            }
        )
        payload["control_modes"] = [{"id": "manual", "label": "Manual", "available": True}]
        if self.experimental_enabled:
            capabilities.update(
                {
                    "garage_behavior_drive": pythonapi and behavior_agent,
                    "garage_imitation_drive": pythonapi,
                    "garage_voxel_drive": pythonapi and behavior_agent,
                }
            )
            payload["control_modes"].extend(
                [
                    {
                        "id": "behavior",
                        "label": "BehaviorAgent",
                        "available": pythonapi and behavior_agent,
                    },
                    {"id": "imitation", "label": "Imitation model", "available": pythonapi},
                    {
                        "id": "voxel",
                        "label": "Voxel planner",
                        "available": pythonapi and behavior_agent,
                    },
                ]
            )
        return payload

    def start(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        # Configuration is a stable deployment fact. Do not turn a transient
        # busy/health timeout into "Worker not configured" before the actual
        # scene operation gets its lifecycle timeout and authoritative error.
        active_world_worker = self._world_worker
        config = GarageDriveStartConfig.from_mapping(
            raw,
            workspace=self.workspace,
            expected_host=self.carla_host,
            expected_port=self.carla_port,
            world_worker_configured=active_world_worker is not None,
            experimental_enabled=self.experimental_enabled,
        )
        pythonapi = _module_available("carla")
        behavior_agent = _module_available("agents.navigation.behavior_agent")
        modes = {
            "manual": True,
            "behavior": self.experimental_enabled and pythonapi and behavior_agent,
            "imitation": self.experimental_enabled and pythonapi,
            "voxel": self.experimental_enabled and pythonapi and behavior_agent,
        }
        if not modes.get(config.control_mode, False):
            raise RuntimeError(
                f"control mode {config.control_mode!r} is unavailable in this operator environment"
            )
        if (config.traffic_vehicles or config.walkers) and not pythonapi:
            raise RuntimeError("traffic and pedestrians require CARLA PythonAPI")
        with self._lock:
            if self._session is not None and self._session.snapshot()["status"] in {
                "starting",
                "running",
                "stopping",
            }:
                raise RuntimeError("another interactive drive session is already active")
            session = GarageDriveSession(
                config,
                workspace=self.workspace,
                world_worker=active_world_worker,
            )
            self._session = session
            session.start()
            return session.snapshot()


__all__ = [
    "GarageDriveSession",
    "GarageDriveSessionManager",
    "GarageDriveStartConfig",
]
