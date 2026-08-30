"""Pluggable closed-loop driving-model contract for local CARLA simulations.

This module is deliberately independent of the CARLA Python module so model
configuration and unit tests can run on machines that do not have CARLA
installed.  A custom factory receives :class:`ModelDriverConfig` and returns
an object with ``reset()``, ``predict(observation)`` and ``close()`` methods.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class ModelDriverConfig:
    """Serializable configuration passed to a custom model-driver factory."""

    checkpoint: Path | None = None
    device: str = "cpu"
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.device.strip():
            raise ValueError("model device must not be empty")


@dataclass(frozen=True)
class ModelObservation:
    """Inputs available to an end-to-end driving model.

    The image is a copied, read-only BGR uint8 array.  Speed is included because
    most practical driving policies need a small amount of ego-state feedback.
    No CARLA actor, map, waypoint, or traffic-manager object is exposed.
    """

    frame: int
    timestamp: float
    image_bgr: np.ndarray
    speed_mps: float
    dt_seconds: float

    def __post_init__(self) -> None:
        if self.frame < 0:
            raise ValueError("model observation frame must be non-negative")
        if not math.isfinite(self.timestamp):
            raise ValueError("model observation timestamp must be finite")
        if not math.isfinite(self.speed_mps) or self.speed_mps < 0.0:
            raise ValueError("model observation speed must be finite and non-negative")
        if not math.isfinite(self.dt_seconds) or self.dt_seconds <= 0.0:
            raise ValueError("model observation dt must be finite and positive")
        if (
            not isinstance(self.image_bgr, np.ndarray)
            or self.image_bgr.dtype != np.uint8
            or self.image_bgr.ndim != 3
            or self.image_bgr.shape[2] != 3
        ):
            raise ValueError("model observation image must be uint8 HxWx3 BGR")
        image = np.ascontiguousarray(self.image_bgr.copy())
        image.setflags(write=False)
        object.__setattr__(self, "image_bgr", image)


@dataclass(frozen=True)
class ModelControl:
    """Validated model output that can be converted to ``carla.VehicleControl``."""

    throttle: float
    steer: float
    brake: float
    hand_brake: bool = False
    reverse: bool = False

    def __post_init__(self) -> None:
        values = (self.throttle, self.steer, self.brake)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("model controls must be finite")
        if not 0.0 <= float(self.throttle) <= 1.0:
            raise ValueError("model throttle must be in [0, 1]")
        if not -1.0 <= float(self.steer) <= 1.0:
            raise ValueError("model steer must be in [-1, 1]")
        if not 0.0 <= float(self.brake) <= 1.0:
            raise ValueError("model brake must be in [0, 1]")
        if float(self.throttle) > 0.05 and float(self.brake) > 0.05:
            raise ValueError("model cannot request throttle and brake together")

    @classmethod
    def stopped(cls) -> "ModelControl":
        return cls(throttle=0.0, steer=0.0, brake=1.0)


@runtime_checkable
class DrivingModel(Protocol):
    def reset(self) -> None: ...

    def predict(self, observation: ModelObservation) -> ModelControl | Mapping[str, Any]: ...

    def close(self) -> None: ...


def control_from_value(value: Any) -> ModelControl:
    """Normalize common model-return shapes into :class:`ModelControl`."""

    if isinstance(value, ModelControl):
        return value
    if isinstance(value, Mapping):
        return ModelControl(
            throttle=float(value.get("throttle", 0.0)),
            steer=float(value.get("steer", 0.0)),
            brake=float(value.get("brake", 0.0)),
            hand_brake=bool(value.get("hand_brake", False)),
            reverse=bool(value.get("reverse", False)),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) not in {3, 5}:
            raise ValueError("model control sequence must contain 3 or 5 values")
        return ModelControl(
            throttle=float(value[0]),
            steer=float(value[1]),
            brake=float(value[2]),
            hand_brake=bool(value[3]) if len(value) == 5 else False,
            reverse=bool(value[4]) if len(value) == 5 else False,
        )
    required = ("throttle", "steer", "brake")
    if all(hasattr(value, name) for name in required):
        return ModelControl(
            throttle=float(value.throttle),
            steer=float(value.steer),
            brake=float(value.brake),
            hand_brake=bool(getattr(value, "hand_brake", False)),
            reverse=bool(getattr(value, "reverse", False)),
        )
    raise TypeError(
        "model predict() must return ModelControl, a mapping, a 3/5-value sequence, "
        "or an object with throttle/steer/brake attributes"
    )


def _load_factory(reference: str) -> Callable[[ModelDriverConfig], Any]:
    module_name, separator, attribute_name = reference.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("model factory must use module:callable syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise TypeError(f"model factory {reference!r} is not callable")
    return factory


def create_driving_model(reference: str, config: ModelDriverConfig) -> DrivingModel:
    """Load and validate a custom driving model."""

    return create_driving_model_from_factory(_load_factory(reference), config)


def create_driving_model_from_factory(
    factory: Callable[[ModelDriverConfig], Any],
    config: ModelDriverConfig,
) -> DrivingModel:
    """Instantiate and validate an already-resolved model factory."""

    if not callable(factory):
        raise TypeError("driving model factory must be callable")
    candidate = factory(config)
    missing = [name for name in ("reset", "predict", "close") if not hasattr(candidate, name)]
    if missing:
        raise TypeError("driving model is missing required members: " + ", ".join(missing))
    if not all(callable(getattr(candidate, name)) for name in ("reset", "predict", "close")):
        raise TypeError("driving model reset, predict, and close members must be callable")
    return candidate


__all__ = [
    "DrivingModel",
    "ModelControl",
    "ModelDriverConfig",
    "ModelObservation",
    "control_from_value",
    "create_driving_model",
    "create_driving_model_from_factory",
]
