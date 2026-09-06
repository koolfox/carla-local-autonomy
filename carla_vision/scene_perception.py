"""Canonical, framework-neutral scene perception contracts.

The runtime model package owns model-specific preprocessing and inference. This
module defines the narrow boundary consumed by downstream planning, recording,
and visualization code. It intentionally contains no CARLA or model-framework
objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from .contracts import Detection
from .model_package_contracts import (
    CAPABILITY_DRIVABLE_AREA,
    CAPABILITY_LANE_MARKINGS,
    CAPABILITY_OBJECTS,
    CAPABILITY_TRAFFIC_LIGHT_STATE,
    CAPABILITY_TRAFFIC_LIGHTS,
    CAPABILITY_TRAFFIC_SIGNS,
    SCENE_PERCEPTION_CAPABILITIES,
    SCENE_PERCEPTION_INPUT_KIND,
    SCENE_PERCEPTION_OUTPUT_KIND,
)

ROAD_USER_CATEGORIES = frozenset({"vehicle", "pedestrian", "cyclist", "other_road_user"})
TRAFFIC_LIGHT_STATES = frozenset({"red", "yellow", "green", "off"})


def _confidence(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _point(value: tuple[float, float], name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two coordinates")
    x, y = (float(value[0]), float(value[1]))
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{name} coordinates must be finite")
    return x, y


def _text(value: str, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


@dataclass(frozen=True)
class RoadUserObservation:
    """A detected road user with a canonical coarse category."""

    detection: Detection
    category: str

    def __post_init__(self) -> None:
        category = _text(self.category, "category").lower()
        if category not in ROAD_USER_CATEGORIES:
            raise ValueError(
                "category must be one of: " + ", ".join(sorted(ROAD_USER_CATEGORIES))
            )
        object.__setattr__(self, "category", category)


@dataclass(frozen=True)
class TrafficLightObservation:
    """Traffic-light detection with optional model-predicted state.

    ``state=None`` means the model did not provide a state. Callers must not
    infer a state from the detection label when the package does not declare the
    ``traffic_light_state`` capability.
    """

    detection: Detection
    state: str | None = None
    state_confidence: float | None = None

    def __post_init__(self) -> None:
        if self.state is None:
            if self.state_confidence is not None:
                raise ValueError("state_confidence requires a traffic-light state")
            return
        state = _text(self.state, "state").lower()
        if state not in TRAFFIC_LIGHT_STATES:
            raise ValueError(
                "traffic-light state must be one of: "
                + ", ".join(sorted(TRAFFIC_LIGHT_STATES))
            )
        object.__setattr__(self, "state", state)
        if self.state_confidence is not None:
            object.__setattr__(
                self,
                "state_confidence",
                _confidence(self.state_confidence, "state_confidence"),
            )


@dataclass(frozen=True)
class LaneMarkingObservation:
    """Image-space lane or road-marking polyline."""

    label: str
    confidence: float
    points: tuple[tuple[float, float], ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _text(self.label, "label"))
        object.__setattr__(self, "confidence", _confidence(self.confidence, "confidence"))
        if len(self.points) < 2:
            raise ValueError("lane marking requires at least two points")
        object.__setattr__(
            self,
            "points",
            tuple(_point(point, f"points[{index}]") for index, point in enumerate(self.points)),
        )
        object.__setattr__(self, "attributes", dict(self.attributes))


@dataclass(frozen=True)
class DrivableAreaObservation:
    """Image-space polygon for model-predicted drivable surface."""

    confidence: float
    polygon: tuple[tuple[float, float], ...]
    label: str = "drivable"
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _text(self.label, "label"))
        object.__setattr__(self, "confidence", _confidence(self.confidence, "confidence"))
        if len(self.polygon) < 3:
            raise ValueError("drivable area requires at least three polygon points")
        object.__setattr__(
            self,
            "polygon",
            tuple(
                _point(point, f"polygon[{index}]") for index, point in enumerate(self.polygon)
            ),
        )
        object.__setattr__(self, "attributes", dict(self.attributes))


@dataclass(frozen=True)
class ScenePerception:
    """Canonical output for one scene-perception inference.

    Capability declarations are part of the value so downstream consumers can
    distinguish "no objects detected" from "this model does not provide object
    detections". Populated semantics that were not declared are rejected.
    """

    capabilities: tuple[str, ...]
    road_users: tuple[RoadUserObservation, ...] = ()
    traffic_lights: tuple[TrafficLightObservation, ...] = ()
    traffic_signs: tuple[Detection, ...] = ()
    lane_markings: tuple[LaneMarkingObservation, ...] = ()
    drivable_areas: tuple[DrivableAreaObservation, ...] = ()

    def __post_init__(self) -> None:
        normalized: list[str] = []
        for raw in self.capabilities:
            capability = _text(raw, "capability").lower()
            if capability not in SCENE_PERCEPTION_CAPABILITIES:
                raise ValueError(f"unsupported scene-perception capability {capability!r}")
            if capability in normalized:
                raise ValueError(f"duplicate scene-perception capability {capability!r}")
            normalized.append(capability)
        if not normalized:
            raise ValueError("scene perception must declare at least one capability")
        capabilities = frozenset(normalized)
        if (
            CAPABILITY_TRAFFIC_LIGHT_STATE in capabilities
            and CAPABILITY_TRAFFIC_LIGHTS not in capabilities
        ):
            raise ValueError("traffic_light_state capability requires traffic_lights")
        if self.road_users and CAPABILITY_OBJECTS not in capabilities:
            raise ValueError("road_users require the objects capability")
        if self.traffic_lights and CAPABILITY_TRAFFIC_LIGHTS not in capabilities:
            raise ValueError("traffic_lights require the traffic_lights capability")
        if self.traffic_signs and CAPABILITY_TRAFFIC_SIGNS not in capabilities:
            raise ValueError("traffic_signs require the traffic_signs capability")
        if self.lane_markings and CAPABILITY_LANE_MARKINGS not in capabilities:
            raise ValueError("lane_markings require the lane_markings capability")
        if self.drivable_areas and CAPABILITY_DRIVABLE_AREA not in capabilities:
            raise ValueError("drivable_areas require the drivable_area capability")
        if any(light.state is not None for light in self.traffic_lights) and (
            CAPABILITY_TRAFFIC_LIGHT_STATE not in capabilities
        ):
            raise ValueError(
                "traffic-light state values require the traffic_light_state capability"
            )
        object.__setattr__(self, "capabilities", tuple(normalized))


@dataclass(frozen=True)
class ScenePerceptionObservation:
    """Exact RGB frame passed to a scene-perception model."""

    sequence: int
    carla_frame: int
    source_timestamp: float
    source_received_monotonic: float
    image_bgr: np.ndarray

    def __post_init__(self) -> None:
        if self.sequence < 0 or self.carla_frame < 0:
            raise ValueError("frame identifiers must be non-negative")
        if not math.isfinite(self.source_timestamp):
            raise ValueError("source_timestamp must be finite")
        if not math.isfinite(self.source_received_monotonic):
            raise ValueError("source_received_monotonic must be finite")
        if (
            not isinstance(self.image_bgr, np.ndarray)
            or self.image_bgr.dtype != np.uint8
            or self.image_bgr.ndim != 3
            or self.image_bgr.shape[2] != 3
        ):
            raise ValueError("image_bgr must be a uint8 HxWx3 array")
        image = np.ascontiguousarray(self.image_bgr).copy()
        image.setflags(write=False)
        object.__setattr__(self, "image_bgr", image)


@dataclass(frozen=True)
class ScenePerceptionResult:
    """Canonical perception tied to the exact source frame and inference timing."""

    observation: ScenePerceptionObservation
    perception: ScenePerception
    inference_started_monotonic: float
    completed_monotonic: float
    model_name: str = "unknown"

    def __post_init__(self) -> None:
        if not math.isfinite(self.inference_started_monotonic) or not math.isfinite(
            self.completed_monotonic
        ):
            raise ValueError("inference timestamps must be finite")
        if not (
            self.observation.source_received_monotonic
            <= self.inference_started_monotonic
            <= self.completed_monotonic
        ):
            raise ValueError("inference timing must follow source frame receipt")
        object.__setattr__(self, "model_name", _text(self.model_name, "model_name"))

    @property
    def carla_frame(self) -> int:
        return self.observation.carla_frame

    @property
    def model_inference_seconds(self) -> float:
        return self.completed_monotonic - self.inference_started_monotonic

    @property
    def end_to_end_seconds(self) -> float:
        return self.completed_monotonic - self.observation.source_received_monotonic

    def age_seconds(self, now_monotonic: float) -> float:
        now = float(now_monotonic)
        if not math.isfinite(now):
            raise ValueError("now_monotonic must be finite")
        return max(0.0, now - self.observation.source_received_monotonic)


@runtime_checkable
class ScenePerceptionModel(Protocol):
    """Runtime boundary implemented by trusted scene-perception adapters."""

    def reset(self) -> None: ...

    def infer(self, observation: ScenePerceptionObservation) -> ScenePerception: ...

    def close(self) -> None: ...


__all__ = [
    "CAPABILITY_DRIVABLE_AREA",
    "CAPABILITY_LANE_MARKINGS",
    "CAPABILITY_OBJECTS",
    "CAPABILITY_TRAFFIC_LIGHTS",
    "CAPABILITY_TRAFFIC_LIGHT_STATE",
    "CAPABILITY_TRAFFIC_SIGNS",
    "DrivableAreaObservation",
    "LaneMarkingObservation",
    "ROAD_USER_CATEGORIES",
    "RoadUserObservation",
    "SCENE_PERCEPTION_CAPABILITIES",
    "SCENE_PERCEPTION_INPUT_KIND",
    "SCENE_PERCEPTION_OUTPUT_KIND",
    "ScenePerception",
    "ScenePerceptionModel",
    "ScenePerceptionObservation",
    "ScenePerceptionResult",
    "TRAFFIC_LIGHT_STATES",
    "TrafficLightObservation",
]
