"""Registered external driving-model integration for the Garage.

This module keeps the existing Garage safety/session engine and changes only the
model source. Model packages are resolved from the manifest-backed registry,
verified before actuation, and executed through the existing ModelDriver
contract. The legacy Imitation and Voxel modes remain untouched.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..controller import ControlCommand
from ..model_driver import (
    ModelDriverConfig,
    ModelObservation,
    control_from_value,
    create_driving_model,
)
from ..model_registry import ModelPackage, resolve_model_package
from .drive import _world_worker_health_ready
from .garage_drive import (
    GarageDriveSession,
    GarageDriveSessionManager,
    GarageDriveStartConfig,
    _PolicyCamera,
)
from .world_worker_client import WorldWorkerClient

_TORCHSCRIPT_FACTORY = "carla_vision.torchscript_driver:create_driver"


@dataclass(frozen=True)
class ExternalModelDriveStartConfig:
    """Garage config plus a verified model-package identity."""

    base: GarageDriveStartConfig
    model_package_id: str
    model_runtime: str
    model_factory: str
    model_options: Mapping[str, Any]
    model_artifact_sha256: str
    model_manifest_path: str
    model_requires_trusted_code: bool

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    @property
    def control_mode(self) -> str:
        return "model"

    @property
    def autonomous(self) -> bool:
        return True

    @property
    def model_output_actuated(self) -> bool:
        return True

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        workspace: Path,
        expected_host: str,
        expected_port: int,
        world_worker_configured: bool,
        experimental_enabled: bool,
    ) -> "ExternalModelDriveStartConfig":
        if not experimental_enabled:
            raise PermissionError(
                "control mode 'model' is experimental and disabled; restart the Garage "
                "server with --enable-experimental to opt in"
            )
        package_id = str(raw.get("model_package_id", "")).strip()
        if not package_id:
            raise ValueError("model_package_id is required for control_mode=model")
        trusted_ack = raw.get("model_trusted_code_acknowledged", False)
        if not isinstance(trusted_ack, bool):
            raise TypeError("model_trusted_code_acknowledged must be a boolean")

        resolved = resolve_model_package(workspace, package_id, required_role="driving_policy")
        package = resolved.package
        if package.requires_trusted_code and not trusted_ack:
            raise ValueError(
                "python_factory model packages require explicit trusted-code acknowledgement"
            )
        device = str(raw.get("policy_device", "cpu")).strip()
        if device not in package.devices:
            raise ValueError(
                f"model package {package_id!r} does not advertise device {device!r}; "
                f"supported: {', '.join(package.devices)}"
            )
        actual_sha256 = resolved.verify_artifact()
        factory = _factory_for(package)
        model_options = {
            "model_package_id": package.package_id,
            "model_package_version": package.version,
            "runtime": package.runtime,
            "image": dict(package.inputs.get("image", {}))
            if isinstance(package.inputs.get("image"), Mapping)
            else {},
            "speed": dict(package.inputs.get("speed", {}))
            if isinstance(package.inputs.get("speed"), Mapping)
            else {},
            "inputs": dict(package.inputs),
            "outputs": dict(package.outputs),
            "labels": package.labels,
            "source": package.source,
            "artifact_sha256": actual_sha256,
        }

        legacy = dict(raw)
        legacy.pop("model_package_id", None)
        legacy.pop("model_trusted_code_acknowledged", None)
        legacy["control_mode"] = "imitation"
        legacy["policy_checkpoint"] = resolved.artifact_path.relative_to(workspace).as_posix()
        base = GarageDriveStartConfig.from_mapping(
            legacy,
            workspace=workspace,
            expected_host=expected_host,
            expected_port=expected_port,
            world_worker_configured=world_worker_configured,
            experimental_enabled=experimental_enabled,
        )
        return cls(
            base=base,
            model_package_id=package.package_id,
            model_runtime=package.runtime,
            model_factory=factory,
            model_options=model_options,
            model_artifact_sha256=actual_sha256,
            model_manifest_path=package.manifest_path,
            model_requires_trusted_code=package.requires_trusted_code,
        )

    def manifest_config(self) -> dict[str, Any]:
        payload = self.base.manifest_config()
        payload.update(
            {
                "garage_mode": "model",
                "control_owner": "external_model",
                "autonomy_output_actuated": True,
                "model_output_actuated": True,
                "model_package_id": self.model_package_id,
                "model_runtime": self.model_runtime,
                "model_factory": self.model_factory,
                "model_artifact_sha256": self.model_artifact_sha256,
                "model_manifest_path": self.model_manifest_path,
                "model_requires_trusted_code": self.model_requires_trusted_code,
            }
        )
        return payload


def _factory_for(package: ModelPackage) -> str:
    if package.runtime == "torchscript_control_v1":
        return _TORCHSCRIPT_FACTORY
    if package.runtime == "python_factory" and package.factory is not None:
        return package.factory
    raise ValueError(
        f"driving model package {package.package_id!r} uses unsupported runtime {package.runtime!r}"
    )


class _ExternalModelPolicy:
    """Same fail-closed policy shell as Imitation, with a registry-selected factory."""

    def __init__(self, context: Any, config: ExternalModelDriveStartConfig) -> None:
        self.config = config
        self.camera = _PolicyCamera(context.carla, context.world, context.ego, config)
        self.model = create_driving_model(
            config.model_factory,
            ModelDriverConfig(
                checkpoint=config.policy_checkpoint,
                device=config.policy_device,
                options=config.model_options,
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
        detail = {
            "model_package_id": self.config.model_package_id,
            "model_runtime": self.config.model_runtime,
            "model_artifact_sha256": self.config.model_artifact_sha256,
        }
        if self.latched_error is not None:
            return ControlCommand.service_brake(), "model_error_latched", True, {
                **detail,
                "error": self.latched_error,
            }
        frame = self.camera.latest()
        stale_limit = max(0.5, 4.0 / self.config.camera_fps)
        if frame is None or now - frame.received_monotonic > stale_limit:
            return ControlCommand.service_brake(), "model_camera_deadman", True, detail
        if frame.frame == self.last_frame and now - self.last_command_at <= stale_limit:
            return self.last_command, "external_model", False, {**detail, "frame": frame.frame}
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
            return command, "external_model", False, {**detail, "frame": frame.frame}
        except Exception as error:
            self.errors += 1
            if self.errors >= self.config.max_policy_errors:
                self.latched_error = f"{type(error).__name__}: {error}"
            return ControlCommand.service_brake(), "model_error", True, {
                **detail,
                "error": f"{type(error).__name__}: {error}",
                "consecutive_errors": self.errors,
            }

    def close(self) -> None:
        try:
            self.model.close()
        finally:
            self.camera.close()


class ExternalModelDriveSession(GarageDriveSession):
    config: ExternalModelDriveStartConfig

    def _ensure_extensions(self) -> None:
        if self._garage_closed:
            raise RuntimeError("Garage extensions are closing")
        context = self._ensure_context()
        self._population.start(context, self.config)
        if self._policy is None:
            self._policy = _ExternalModelPolicy(context, self.config)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False


def _start_registered_model_session(
    manager: GarageDriveSessionManager,
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    if not _module_available("carla"):
        raise RuntimeError("registered external model driving requires CARLA PythonAPI")
    active_world_worker: WorldWorkerClient | None = None
    if manager._world_worker is not None:
        try:
            health = manager._world_worker.health()
            if not _world_worker_health_ready(health):
                raise RuntimeError("World Worker reports that CARLA is unavailable")
        except Exception:
            active_world_worker = None
        else:
            active_world_worker = manager._world_worker
    config = ExternalModelDriveStartConfig.from_mapping(
        raw,
        workspace=manager.workspace,
        expected_host=manager.carla_host,
        expected_port=manager.carla_port,
        world_worker_configured=active_world_worker is not None,
        experimental_enabled=manager.experimental_enabled,
    )
    catalog = manager.catalog()
    capabilities = catalog.get("capabilities", {})
    if not isinstance(capabilities, Mapping):
        raise RuntimeError("Garage capabilities are malformed")
    if config.traffic_vehicles and not bool(capabilities.get("garage_traffic_population")):
        raise RuntimeError("traffic population requires CARLA PythonAPI")
    if config.walkers and not bool(capabilities.get("garage_walker_population")):
        raise RuntimeError("pedestrian population requires CARLA PythonAPI")
    with manager._lock:
        if manager._session is not None and manager._session.snapshot()["status"] in {
            "starting",
            "running",
            "stopping",
        }:
            raise RuntimeError("another interactive drive session is already active")
        session = ExternalModelDriveSession(
            config,
            workspace=manager.workspace,
            world_worker=active_world_worker,
        )
        manager._session = session
        session.start()
        return session.snapshot()


def start_registered_model_session(
    manager: GarageDriveSessionManager,
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    """Start one registered model session while preserving preview/drive exclusion."""

    preview = getattr(manager, "_garage_preview", None)
    world_mode_lock = getattr(manager, "_garage_world_mode_lock", None)
    if preview is None or world_mode_lock is None:
        return _start_registered_model_session(manager, raw)
    with world_mode_lock:
        preview.stop_for_drive()
        return _start_registered_model_session(manager, raw)


__all__ = [
    "ExternalModelDriveSession",
    "ExternalModelDriveStartConfig",
    "start_registered_model_session",
]
