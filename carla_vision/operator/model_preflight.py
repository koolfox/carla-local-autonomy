"""Fail-fast diagnostics for registered driving-model packages.

The live Garage keeps a second, fail-closed model initialization path inside the
Drive session.  This module runs the same verified adapter bytes and model-driver
contract before a session is accepted so checkpoint/factory failures are visible
as structured diagnostics instead of surfacing only after the control loop has
started.

Preflight intentionally does not invent a synthetic model observation.  Custom
models own preprocessing and may require temporal/runtime state that is only
valid once real frames arrive.  The boundary therefore proves package
resolution, adapter execution, model construction, reset, integrity re-check,
and cleanup; inference remains guarded by the normal runtime deadman/latch.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..model_driver import ModelDriverConfig, create_driving_model_from_factory
from .external_model_drive import (
    ExternalModelDriveStartConfig,
    _assert_reference,
    _load_verified_factory,
)
from .garage_drive import GarageDriveSessionManager

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelPreflightDiagnostic:
    phase: str
    package_id: str | None
    runtime: str | None
    factory: str | None
    device: str | None
    artifact_path: str | None
    artifact_sha256: str | None
    exception_type: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "package_id": self.package_id,
            "runtime": self.runtime,
            "factory": self.factory,
            "device": self.device,
            "artifact": (
                {
                    "path": self.artifact_path,
                    "sha256": self.artifact_sha256,
                }
                if self.artifact_path is not None or self.artifact_sha256 is not None
                else None
            ),
            "exception": {
                "type": self.exception_type,
                "message": self.message,
            },
        }


class ModelPreflightError(RuntimeError):
    """Sanitized model initialization failure suitable for the local HTTP API."""

    def __init__(self, diagnostic: ModelPreflightDiagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__(
            f"custom model preflight failed during {diagnostic.phase}: "
            f"{diagnostic.exception_type}: {diagnostic.message}"
        )

    def response_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "type": type(self).__qualname__,
                "code": "model_preflight_failed",
                "message": str(self),
                "diagnostic": self.diagnostic.as_dict(),
            }
        }


def _relative_path(path: Path | None, workspace: Path) -> str | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve(strict=False)
    try:
        return resolved.relative_to(workspace).as_posix()
    except ValueError:
        return resolved.name


def _diagnostic(
    *,
    phase: str,
    error: Exception,
    workspace: Path,
    raw: Mapping[str, Any],
    config: ExternalModelDriveStartConfig | None,
) -> ModelPreflightDiagnostic:
    return ModelPreflightDiagnostic(
        phase=phase,
        package_id=(
            config.model_package_id
            if config is not None
            else str(raw.get("model_package_id", "")).strip() or None
        ),
        runtime=config.model_runtime if config is not None else None,
        factory=config.model_factory if config is not None else None,
        device=(
            config.policy_device
            if config is not None
            else str(raw.get("policy_device", "")).strip() or None
        ),
        artifact_path=(
            _relative_path(config.policy_checkpoint, workspace) if config is not None else None
        ),
        artifact_sha256=config.model_artifact_sha256 if config is not None else None,
        exception_type=type(error).__qualname__,
        message=str(error),
    )


def _raise_preflight_error(
    *,
    phase: str,
    error: Exception,
    workspace: Path,
    raw: Mapping[str, Any],
    config: ExternalModelDriveStartConfig | None,
) -> None:
    diagnostic = _diagnostic(
        phase=phase,
        error=error,
        workspace=workspace,
        raw=raw,
        config=config,
    )
    _LOGGER.exception(
        "Custom model preflight failed phase=%s package=%r runtime=%r factory=%r device=%r",
        diagnostic.phase,
        diagnostic.package_id,
        diagnostic.runtime,
        diagnostic.factory,
        diagnostic.device,
    )
    raise ModelPreflightError(diagnostic) from error


def _verify_config_references(
    config: ExternalModelDriveStartConfig,
    *,
    workspace: Path,
) -> None:
    if config.policy_checkpoint is None:
        raise ValueError("registered driving model package has no checkpoint artifact")
    _assert_reference(
        {
            "path": str(config.policy_checkpoint),
            "sha256": config.model_artifact_sha256,
            "size_bytes": config.model_artifact_size_bytes,
        },
        name="model artifact",
    )
    _assert_reference(
        {
            "path": str(workspace / config.model_manifest_path),
            "sha256": config.model_manifest_sha256,
            "size_bytes": config.model_manifest_size_bytes,
        },
        name="model manifest",
    )
    _assert_reference(config.model_adapter_reference, name="model adapter")


def preflight_registered_model_request(
    manager: GarageDriveSessionManager,
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    """Load/reset/close a registered model before accepting a Drive session.

    The returned mapping is safe to expose to the local browser.  Full Python
    tracebacks are emitted through :mod:`logging` and deliberately omitted from
    the response payload.
    """

    workspace = manager.workspace
    config: ExternalModelDriveStartConfig | None = None
    phase = "package_resolution"
    try:
        config = ExternalModelDriveStartConfig.from_mapping(
            raw,
            workspace=workspace,
            expected_host=manager.carla_host,
            expected_port=manager.carla_port,
            world_worker_configured=manager._world_worker is not None,
            experimental_enabled=manager.experimental_enabled,
        )
    except Exception as error:
        _raise_preflight_error(
            phase=phase,
            error=error,
            workspace=workspace,
            raw=raw,
            config=config,
        )

    model: Any | None = None
    isolated_module_name: str | None = None
    try:
        phase = "adapter_load"
        factory, isolated_module_name = _load_verified_factory(
            config.model_factory,
            config.model_adapter_reference,
        )

        phase = "model_initialization"
        model = create_driving_model_from_factory(
            factory,
            ModelDriverConfig(
                checkpoint=config.policy_checkpoint,
                device=config.policy_device,
                options=config.model_options,
            ),
        )

        phase = "model_reset"
        model.reset()

        phase = "integrity_recheck"
        _verify_config_references(config, workspace=workspace)

        phase = "model_cleanup"
        model.close()
        model = None
    except Exception as error:
        if model is not None:
            try:
                model.close()
            except Exception:
                _LOGGER.exception(
                    "Custom model preflight cleanup also failed package=%r",
                    config.model_package_id,
                )
        _raise_preflight_error(
            phase=phase,
            error=error,
            workspace=workspace,
            raw=raw,
            config=config,
        )
    finally:
        if isolated_module_name is not None:
            sys.modules.pop(isolated_module_name, None)

    return {
        "status": "ready",
        "phase": "complete",
        "package_id": config.model_package_id,
        "runtime": config.model_runtime,
        "factory": config.model_factory,
        "device": config.policy_device,
        "artifact": {
            "path": _relative_path(config.policy_checkpoint, workspace),
            "sha256": config.model_artifact_sha256,
        },
    }


__all__ = [
    "ModelPreflightDiagnostic",
    "ModelPreflightError",
    "preflight_registered_model_request",
]
