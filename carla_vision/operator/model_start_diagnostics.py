"""Synchronous custom-model initialization diagnostics for Garage start.

The normal registered-model session defers policy construction until the drive
thread first requests Garage extensions. That is safe, but it means checkpoint
or adapter failures can arrive after ``POST /api/session/start`` has already
returned.

This module keeps the existing verified model-package loader, policy class and
Garage session. It changes only *when* the policy is constructed: once,
synchronously, before the session is published to the manager or started. The
same initialized policy is then reused by the live drive.
"""

from __future__ import annotations

import importlib.util
import time
import traceback
from dataclasses import dataclass
from typing import Any, Mapping

from .external_model_drive import (
    ExternalModelDriveSession,
    ExternalModelDriveStartConfig,
    _ExternalModelPolicy,
)
from .garage_drive import GarageDriveSessionManager

_MAX_TRACEBACK_CHARS = 24_000
_MAX_EXCEPTION_CHAIN = 8


def _exception_chain(error: BaseException) -> list[dict[str, str]]:
    chain: list[dict[str, str]] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and len(chain) < _MAX_EXCEPTION_CHAIN:
        identity = id(current)
        if identity in seen:
            break
        seen.add(identity)
        chain.append(
            {
                "type": type(current).__qualname__,
                "message": str(current),
            }
        )
        current = current.__cause__ or current.__context__
    return chain


def _traceback_text(error: BaseException) -> str:
    text = "".join(traceback.format_exception(error)).strip()
    if len(text) <= _MAX_TRACEBACK_CHARS:
        return text
    suffix = "\n... traceback truncated ..."
    return text[: _MAX_TRACEBACK_CHARS - len(suffix)] + suffix


@dataclass(frozen=True, slots=True)
class ModelInitializationDiagnostic:
    """Machine-readable evidence for one model construction attempt."""

    status: str
    package_id: str
    runtime: str
    factory: str
    device: str
    checkpoint_path: str
    artifact_sha256: str
    manifest_path: str
    manifest_sha256: str
    adapter_sha256: str
    duration_seconds: float
    exception: Mapping[str, str] | None = None
    exception_chain: tuple[Mapping[str, str], ...] = ()
    traceback_text: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "phase": "model_initialization",
            "status": self.status,
            "package_id": self.package_id,
            "runtime": self.runtime,
            "factory": self.factory,
            "device": self.device,
            "checkpoint_path": self.checkpoint_path,
            "artifact_sha256": self.artifact_sha256,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "adapter_sha256": self.adapter_sha256,
            "duration_seconds": self.duration_seconds,
        }
        if self.exception is not None:
            payload["exception"] = dict(self.exception)
        if self.exception_chain:
            payload["exception_chain"] = [dict(item) for item in self.exception_chain]
        if self.traceback_text is not None:
            payload["traceback"] = self.traceback_text
        return payload


def _diagnostic(
    config: ExternalModelDriveStartConfig,
    *,
    status: str,
    duration_seconds: float,
    error: BaseException | None = None,
) -> ModelInitializationDiagnostic:
    adapter_sha256 = str(config.model_adapter_reference.get("sha256", ""))
    exception = None
    chain: tuple[Mapping[str, str], ...] = ()
    traceback_text = None
    if error is not None:
        exception = {
            "type": type(error).__qualname__,
            "message": str(error),
        }
        chain = tuple(_exception_chain(error))
        traceback_text = _traceback_text(error)
    return ModelInitializationDiagnostic(
        status=status,
        package_id=config.model_package_id,
        runtime=config.model_runtime,
        factory=config.model_factory,
        device=config.policy_device,
        checkpoint_path=str(config.policy_checkpoint),
        artifact_sha256=config.model_artifact_sha256,
        manifest_path=config.model_manifest_path,
        manifest_sha256=config.model_manifest_sha256,
        adapter_sha256=adapter_sha256,
        duration_seconds=max(0.0, float(duration_seconds)),
        exception=exception,
        exception_chain=chain,
        traceback_text=traceback_text,
    )


class ModelInitializationError(RuntimeError):
    """Raised when a selected custom model cannot be constructed safely."""

    def __init__(self, diagnostic: ModelInitializationDiagnostic) -> None:
        self.diagnostic = diagnostic
        exception = diagnostic.exception or {"type": "RuntimeError", "message": "unknown error"}
        super().__init__(
            "custom model initialization failed "
            f"for package {diagnostic.package_id!r} "
            f"({diagnostic.runtime}, {diagnostic.device}): "
            f"{exception['type']}: {exception['message']}"
        )

    def as_dict(self) -> dict[str, Any]:
        return self.diagnostic.as_dict()


def _start_registered_model_session(
    manager: GarageDriveSessionManager,
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    active_world_worker = manager._world_worker
    config = ExternalModelDriveStartConfig.from_mapping(
        raw,
        workspace=manager.workspace,
        expected_host=manager.carla_host,
        expected_port=manager.carla_port,
        world_worker_configured=active_world_worker is not None,
        experimental_enabled=manager.experimental_enabled,
    )
    pythonapi = importlib.util.find_spec("carla") is not None
    if (config.traffic_vehicles or config.walkers) and not pythonapi:
        raise RuntimeError("traffic and pedestrians require CARLA PythonAPI")

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

        started = time.monotonic()
        try:
            policy = _ExternalModelPolicy(session, config)
        except BaseException as error:
            diagnostic = _diagnostic(
                config,
                status="failed",
                duration_seconds=time.monotonic() - started,
                error=error,
            )
            # Session construction itself does not publish/start CARLA resources,
            # but close the Garage extensions defensively if that ever changes.
            try:
                session._close_garage_extensions()
            except Exception:
                pass
            raise ModelInitializationError(diagnostic) from error

        diagnostic = _diagnostic(
            config,
            status="ready",
            duration_seconds=time.monotonic() - started,
        )
        session._policy = policy
        session._policy_detail = {"model_initialization": diagnostic.as_dict()}
        manager._session = session
        session.start()
        return session.snapshot()


def start_registered_model_session(
    manager: GarageDriveSessionManager,
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    """Initialize one registered model exactly once, then start the live session."""

    preview = getattr(manager, "_garage_preview", None)
    world_mode_lock = getattr(manager, "_garage_world_mode_lock", None)
    if preview is None or world_mode_lock is None:
        return _start_registered_model_session(manager, raw)
    with world_mode_lock:
        preview.stop_for_drive()
        return _start_registered_model_session(manager, raw)


__all__ = [
    "ModelInitializationDiagnostic",
    "ModelInitializationError",
    "start_registered_model_session",
]
