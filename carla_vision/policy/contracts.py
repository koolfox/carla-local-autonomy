"""Strict RGB-only policy contracts.

The policy never receives ``PerceptionResult`` because that object can retain a
privileged CARLA sensor transform.  ``VisionObservation`` copies only the RGB
image, RGB-derived detections, frame sequence/timing, and camera FOV.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from ..contracts import Detection, PerceptionResult

POLICY_CONTRACT_VERSION = "1.0"


@dataclass(frozen=True)
class VisionObservation:
    """The complete input available to a deployable vision policy."""

    sequence: int
    source_timestamp: float
    rgb_bgr: np.ndarray
    detections: tuple[Detection, ...]
    camera_fov_degrees: float | None

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("vision observation sequence must be non-negative")
        if not math.isfinite(self.source_timestamp):
            raise ValueError("vision observation timestamp must be finite")
        if (
            not isinstance(self.rgb_bgr, np.ndarray)
            or self.rgb_bgr.dtype != np.uint8
            or self.rgb_bgr.ndim != 3
            or self.rgb_bgr.shape[2] != 3
        ):
            raise ValueError("vision observation RGB must be a uint8 HxWx3 array")
        copied = np.ascontiguousarray(self.rgb_bgr.copy())
        copied.setflags(write=False)
        object.__setattr__(self, "rgb_bgr", copied)
        if not isinstance(self.detections, tuple) or any(
            not isinstance(item, Detection) for item in self.detections
        ):
            raise TypeError("vision observation detections must be a Detection tuple")
        if self.camera_fov_degrees is not None and (
            not math.isfinite(self.camera_fov_degrees)
            or not 1.0 <= self.camera_fov_degrees <= 179.0
        ):
            raise ValueError("camera FOV must be finite and in [1, 179] degrees")

    @classmethod
    def from_perception(cls, result: PerceptionResult) -> "VisionObservation":
        """Copy only the approved RGB lane from a perception result."""

        return cls(
            sequence=result.sequence,
            source_timestamp=result.source_timestamp,
            rgb_bgr=result.source_bgr,
            detections=result.detections,
            camera_fov_degrees=result.source_fov,
        )


@dataclass(frozen=True)
class VisionControlProposal:
    """A bounded non-actuating control proposal produced in shadow mode."""

    sequence: int
    throttle: float
    steer: float
    brake: float
    confidence: float
    reason: str
    state: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("policy proposal sequence must be non-negative")
        values = (self.throttle, self.steer, self.brake, self.confidence)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("policy proposal values must be finite")
        if not 0.0 <= self.throttle <= 1.0:
            raise ValueError("policy throttle must be in [0, 1]")
        if not -1.0 <= self.steer <= 1.0:
            raise ValueError("policy steer must be in [-1, 1]")
        if not 0.0 <= self.brake <= 1.0:
            raise ValueError("policy brake must be in [0, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("policy confidence must be in [0, 1]")
        if self.throttle > 0.05 and self.brake > 0.05:
            raise ValueError("policy cannot request throttle and brake together")
        if not self.reason.strip():
            raise ValueError("policy proposal reason must not be empty")
        if not isinstance(self.state, Mapping):
            raise TypeError("policy proposal state must be a mapping")

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "throttle": self.throttle,
            "steer": self.steer,
            "brake": self.brake,
            "confidence": self.confidence,
            "reason": self.reason,
            "state": dict(self.state),
        }


@dataclass(frozen=True)
class PolicyMetadata:
    """Auditable policy identity and declared input contract."""

    name: str
    backend: str
    contract_version: str
    input_fields: tuple[str, ...]
    temporal: bool
    checkpoint: str | None = None
    device: str = "cpu"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.backend.strip():
            raise ValueError("policy name and backend must not be empty")
        if self.contract_version != POLICY_CONTRACT_VERSION:
            raise ValueError(
                f"unsupported policy contract {self.contract_version!r}; "
                f"expected {POLICY_CONTRACT_VERSION!r}"
            )
        if not self.input_fields or len(self.input_fields) != len(set(self.input_fields)):
            raise ValueError("policy input_fields must be non-empty and unique")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "contract_version": self.contract_version,
            "input_fields": list(self.input_fields),
            "temporal": self.temporal,
            "checkpoint": self.checkpoint,
            "device": self.device,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class VisionPolicyConfig:
    """Serializable policy selection used by live-shadow runs."""

    backend: str
    factory: str | None = None
    checkpoint: Path | None = None
    device: str = "cpu"
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        backend = self.backend.strip().lower().replace("_", "-")
        if backend not in {"hazard-stop", "custom"}:
            raise ValueError("policy backend must be hazard-stop or custom")
        if backend == "custom" and not self.factory:
            raise ValueError("custom vision policy requires a module:callable factory")
        if backend != "custom" and self.factory is not None:
            raise ValueError("policy factory is allowed only for the custom backend")
        if not self.device.strip():
            raise ValueError("policy device must not be empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "factory": self.factory,
            "checkpoint": str(self.checkpoint) if self.checkpoint is not None else None,
            "device": self.device,
            "options": dict(self.options),
        }


@runtime_checkable
class VisionPolicy(Protocol):
    """Policy interface that cannot receive CARLA state through its signature."""

    @property
    def metadata(self) -> PolicyMetadata: ...

    def reset(self) -> None: ...

    def propose(self, observation: VisionObservation) -> VisionControlProposal: ...

    def close(self) -> None: ...


__all__ = [
    "POLICY_CONTRACT_VERSION",
    "PolicyMetadata",
    "VisionControlProposal",
    "VisionObservation",
    "VisionPolicy",
    "VisionPolicyConfig",
]
