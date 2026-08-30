"""Registered external driving-model integration for the Garage.

This module keeps the existing Garage safety/session engine and changes only the
model source. Registered models consume the exact raw RGB stream already owned
by the Drive session, so remote World Worker deployments do not need a second
local CARLA PythonAPI camera. Model packages are verified before actuation and
executed through the existing ModelDriver contract.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from ..artifacts import fingerprint_file
from ..controller import ControlCommand
from ..model_driver import (
    ModelDriverConfig,
    ModelObservation,
    control_from_value,
    create_driving_model_from_factory,
)
from ..model_registry import ModelPackage, resolve_model_package
from .drive import _world_worker_health_ready
from .garage_drive import GarageDriveSession, GarageDriveSessionManager, GarageDriveStartConfig
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
    model_artifact_size_bytes: int
    model_manifest_path: str
    model_manifest_sha256: str
    model_manifest_size_bytes: int
    model_adapter_reference: Mapping[str, Any]
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
                "executable model packages require explicit trusted-code acknowledgement"
            )
        device = str(raw.get("policy_device", "cpu")).strip()
        if device not in package.devices:
            raise ValueError(
                f"model package {package_id!r} does not advertise device {device!r}; "
                f"supported: {', '.join(package.devices)}"
            )
        artifact_reference = resolved.verify_artifact_reference()
        manifest_reference = resolved.verify_manifest_reference()
        factory = _factory_for(package)
        adapter_reference = _adapter_reference(factory, package.runtime)
        if package.runtime == "torchscript_control_v1":
            model_options = dict(package.inputs)
        else:
            raw_options = package.inputs.get("options", {})
            if not isinstance(raw_options, Mapping):  # defensive; registry validated it
                raise TypeError("python_factory inputs.options must be an object")
            model_options = {str(key): value for key, value in raw_options.items()}

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
            model_artifact_sha256=str(artifact_reference["sha256"]),
            model_artifact_size_bytes=int(artifact_reference["size_bytes"]),
            model_manifest_path=package.manifest_path,
            model_manifest_sha256=str(manifest_reference["sha256"]),
            model_manifest_size_bytes=int(manifest_reference["size_bytes"]),
            model_adapter_reference=adapter_reference,
            model_requires_trusted_code=package.requires_trusted_code,
        )

    def manifest_config(self) -> dict[str, Any]:
        payload = self.base.manifest_config()
        payload.update(
            {
                "garage_mode": "model",
                "control_owner": "external_model",
                "runtime_sensor_contract": "existing_drive_front_rgb",
                "autonomy_output_actuated": True,
                "model_output_actuated": True,
                "model_package_id": self.model_package_id,
                "model_runtime": self.model_runtime,
                "model_factory": self.model_factory,
                "model_artifact_sha256": self.model_artifact_sha256,
                "model_artifact_size_bytes": self.model_artifact_size_bytes,
                "model_manifest_path": self.model_manifest_path,
                "model_manifest_sha256": self.model_manifest_sha256,
                "model_manifest_size_bytes": self.model_manifest_size_bytes,
                "model_adapter": dict(self.model_adapter_reference),
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


def _adapter_reference(factory: str, runtime: str) -> dict[str, Any]:
    if runtime == "torchscript_control_v1":
        path = Path(__file__).resolve().parent.parent / "torchscript_driver.py"
    else:
        module_name, _, _ = factory.partition(":")
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ModuleNotFoundError, AttributeError) as error:
            raise ValueError(f"model adapter module {module_name!r} is not importable") from error
        if spec is None or not spec.origin or spec.origin in {"built-in", "frozen"}:
            raise ValueError(f"model adapter module {module_name!r} has no fingerprintable source")
        path = Path(spec.origin).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"model adapter source is not a regular file: {path}")
    if path.suffix.lower() != ".py":
        raise ValueError("driving model adapter source must be a Python .py file")
    return {
        "kind": "driving_model_adapter",
        "runtime": runtime,
        "factory": factory,
        **fingerprint_file(path),
    }


def _assert_reference(reference: Mapping[str, Any], *, name: str) -> None:
    current = fingerprint_file(str(reference["path"]))
    if (
        current["sha256"] != reference["sha256"]
        or current["size_bytes"] != reference["size_bytes"]
    ):
        raise ValueError(f"{name} changed after the drive configuration was accepted")


def _load_verified_factory(
    reference: str,
    adapter_reference: Mapping[str, Any],
) -> tuple[Any, str]:
    """Execute exactly the fingerprinted adapter bytes under an isolated name."""

    module_name, _, attribute_name = reference.partition(":")
    path = Path(str(adapter_reference["path"])).expanduser().resolve(strict=True)
    if path.suffix.lower() != ".py":
        raise ValueError("driving model adapter source must be a Python .py file")
    before = path.stat()
    source = path.read_bytes()
    after = path.stat()
    before_signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_signature = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_signature != after_signature:
        raise RuntimeError(f"model adapter changed while reading: {path}")
    digest = hashlib.sha256(source).hexdigest()
    if digest != adapter_reference["sha256"] or len(source) != adapter_reference["size_bytes"]:
        raise ValueError("model adapter changed after the drive configuration was accepted")

    parent, _, leaf = module_name.rpartition(".")
    isolated_leaf = f"_carla_runtime_{leaf}_{digest[:16]}_{id(source):x}"
    isolated_name = f"{parent}.{isolated_leaf}" if parent else isolated_leaf
    module = types.ModuleType(isolated_name)
    module.__file__ = str(path)
    module.__package__ = parent
    module.__spec__ = importlib.util.spec_from_loader(isolated_name, loader=None, origin=str(path))
    sys.modules[isolated_name] = module
    try:
        code = compile(source, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
        factory = getattr(module, attribute_name, None)
        if not callable(factory):
            raise TypeError(f"model factory {reference!r} is not callable")
        return factory, isolated_name
    except BaseException:
        sys.modules.pop(isolated_name, None)
        raise


@dataclass(frozen=True)
class _DriveFrame:
    sequence: int
    received_monotonic: float
    bgr: np.ndarray


def _latest_drive_frame(session: GarageDriveSession) -> _DriveFrame | None:
    with session._frame_condition:
        sequence = int(session._raw_frame_sequence)
        jpeg = session._raw_jpeg
        received = session._frame_received_monotonic["raw"]
    if jpeg is None or received is None or sequence < 0:
        return None
    pixels = np.frombuffer(jpeg, dtype=np.uint8)
    bgr = cv2.imdecode(pixels, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Drive raw stream contains an invalid JPEG")
    return _DriveFrame(sequence=sequence, received_monotonic=float(received), bgr=bgr)


class _ExternalModelPolicy:
    """Fail-closed model policy using the existing Drive camera stream."""

    def __init__(self, session: GarageDriveSession, config: ExternalModelDriveStartConfig) -> None:
        self.session = session
        self.config = config
        self.model: Any | None = None
        self._adapter_module_name: str | None = None
        self._verify_package_files()
        try:
            factory, self._adapter_module_name = _load_verified_factory(
                config.model_factory,
                config.model_adapter_reference,
            )
            self.model = create_driving_model_from_factory(
                factory,
                ModelDriverConfig(
                    checkpoint=config.policy_checkpoint,
                    device=config.policy_device,
                    options=config.model_options,
                ),
            )
            self.model.reset()
            # Loading/importing can be slow.  Verify that neither the model nor
            # its executable adapter changed during that window.
            self._verify_package_files()
        except BaseException:
            if self.model is not None:
                try:
                    self.model.close()
                except Exception:
                    pass
            self.model = None
            if self._adapter_module_name is not None:
                sys.modules.pop(self._adapter_module_name, None)
                self._adapter_module_name = None
            raise
        self.last_frame = -1
        self.last_command = ControlCommand.service_brake()
        self.last_command_at = 0.0
        self.last_source = "model_warmup"
        self.last_failsafe = True
        self.last_detail: dict[str, Any] = {}
        self.previous_steer = 0.0
        self.errors = 0
        self.latched_error: str | None = None

    def _verify_package_files(self) -> None:
        artifact = {
            "path": str(self.config.policy_checkpoint),
            "sha256": self.config.model_artifact_sha256,
            "size_bytes": self.config.model_artifact_size_bytes,
        }
        manifest = {
            "path": str(self.session.workspace / self.config.model_manifest_path),
            "sha256": self.config.model_manifest_sha256,
            "size_bytes": self.config.model_manifest_size_bytes,
        }
        _assert_reference(artifact, name="model artifact")
        _assert_reference(manifest, name="model manifest")
        _assert_reference(self.config.model_adapter_reference, name="model adapter")

    def _remember(
        self,
        *,
        frame: int,
        command: ControlCommand,
        source: str,
        failsafe: bool,
        detail: Mapping[str, Any],
        at: float,
    ) -> None:
        self.last_frame = frame
        self.last_command = command
        self.last_command_at = at
        self.last_source = source
        self.last_failsafe = failsafe
        self.last_detail = dict(detail)

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
        try:
            frame = _latest_drive_frame(self.session)
        except Exception as error:
            self.errors += 1
            return ControlCommand.service_brake(), "model_frame_error", True, {
                **detail,
                "error": f"{type(error).__name__}: {error}",
            }
        stale_limit = max(0.5, 4.0 / self.config.camera_fps)
        if frame is None or now - frame.received_monotonic > stale_limit:
            frame_age = None if frame is None else max(0.0, now - frame.received_monotonic)
            return ControlCommand.service_brake(), "model_camera_deadman", True, {
                **detail,
                "frame_age_seconds": frame_age,
            }
        if frame.sequence == self.last_frame and now - self.last_command_at <= stale_limit:
            return self.last_command, self.last_source, self.last_failsafe, {
                **self.last_detail,
                "frame": frame.sequence,
                "reused": True,
            }
        try:
            inference_started = time.monotonic()
            if self.model is None:  # defensive; construction is fail-closed
                raise RuntimeError("external model is not loaded")
            value = self.model.predict(
                ModelObservation(
                    frame=frame.sequence,
                    timestamp=frame.received_monotonic,
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
            completed = time.monotonic()
            inference_latency = max(0.0, completed - inference_started)
            frame_age = max(0.0, completed - frame.received_monotonic)
            timing_detail = {
                **detail,
                "frame": frame.sequence,
                "inference_latency_seconds": inference_latency,
                "frame_age_seconds": frame_age,
            }
            if frame_age > stale_limit:
                brake_command = ControlCommand.service_brake()
                self._remember(
                    frame=frame.sequence,
                    command=brake_command,
                    source="model_inference_stale",
                    failsafe=True,
                    detail=timing_detail,
                    at=completed,
                )
                return brake_command, "model_inference_stale", True, timing_detail
            with self.session._lock:
                emergency = self.session._emergency
            if emergency:
                brake_command = ControlCommand.service_brake()
                self._remember(
                    frame=frame.sequence,
                    command=brake_command,
                    source="emergency_stop",
                    failsafe=True,
                    detail=timing_detail,
                    at=completed,
                )
                return brake_command, "emergency_stop", True, timing_detail
            self._remember(
                frame=frame.sequence,
                command=command,
                source="external_model",
                failsafe=False,
                detail=timing_detail,
                at=completed,
            )
            self.previous_steer = steer
            self.errors = 0
            return command, "external_model", False, timing_detail
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
        model, self.model = self.model, None
        try:
            if model is not None:
                model.close()
        finally:
            if self._adapter_module_name is not None:
                sys.modules.pop(self._adapter_module_name, None)
                self._adapter_module_name = None


class ExternalModelDriveSession(GarageDriveSession):
    config: ExternalModelDriveStartConfig

    def __init__(
        self,
        config: ExternalModelDriveStartConfig,
        *,
        workspace: Path,
        world_worker: WorldWorkerClient | None = None,
    ) -> None:
        super().__init__(config, workspace=workspace, world_worker=world_worker)
        self._model_setup_error: str | None = None

    def _model_references(self) -> tuple[Mapping[str, Any], ...]:
        manifest_reference = {
            "path": str(self.workspace / self.config.model_manifest_path),
            "sha256": self.config.model_manifest_sha256,
            "size_bytes": self.config.model_manifest_size_bytes,
        }
        artifact_reference = {
            "path": str(self.config.policy_checkpoint),
            "sha256": self.config.model_artifact_sha256,
            "size_bytes": self.config.model_artifact_size_bytes,
        }
        _assert_reference(manifest_reference, name="model manifest")
        _assert_reference(artifact_reference, name="model artifact")
        _assert_reference(self.config.model_adapter_reference, name="model adapter")
        return super()._model_references() + (
            {
                "kind": "actuating_model_package",
                "actuation_authorized": True,
                "package_id": self.config.model_package_id,
                "runtime": self.config.model_runtime,
                "manifest": manifest_reference,
                "artifact": artifact_reference,
                "adapter": dict(self.config.model_adapter_reference),
            },
        )

    def _ensure_extensions(self) -> None:
        if self._garage_closed:
            raise RuntimeError("Garage extensions are closing")
        if self.config.traffic_vehicles or self.config.walkers:
            context = self._ensure_context()
            self._population.start(context, self.config)
        if self._model_setup_error is not None:
            raise RuntimeError(f"external model setup failed (latched): {self._model_setup_error}")
        if self._policy is None:
            try:
                self._policy = _ExternalModelPolicy(self, self.config)
            except Exception as error:
                self._model_setup_error = f"{type(error).__name__}: {error}"
                raise RuntimeError(
                    f"external model setup failed (latched): {self._model_setup_error}"
                ) from error


def _start_registered_model_session(
    manager: GarageDriveSessionManager,
    raw: Mapping[str, Any],
) -> dict[str, Any]:
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
