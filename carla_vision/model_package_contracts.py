"""Pure, runtime-specific contracts for executable model packages.

Registry discovery and runtime adapters both use these validators. Keeping the
contract independent of PyTorch prevents a package from being advertised by
the UI only to fail later because discovery and execution interpreted its
manifest differently.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

CONTROL_OUTPUT_KIND = "vehicle_control_v1"
MODEL_OBSERVATION_INPUT_KIND = "model_observation_v1"
SCENE_PERCEPTION_INPUT_KIND = "scene_perception_observation_v1"
SCENE_PERCEPTION_OUTPUT_KIND = "scene_perception_v1"

CAPABILITY_OBJECTS = "objects"
CAPABILITY_TRAFFIC_LIGHTS = "traffic_lights"
CAPABILITY_TRAFFIC_LIGHT_STATE = "traffic_light_state"
CAPABILITY_TRAFFIC_SIGNS = "traffic_signs"
CAPABILITY_LANE_MARKINGS = "lane_markings"
CAPABILITY_DRIVABLE_AREA = "drivable_area"
SCENE_PERCEPTION_CAPABILITIES = frozenset(
    {
        CAPABILITY_OBJECTS,
        CAPABILITY_TRAFFIC_LIGHTS,
        CAPABILITY_TRAFFIC_LIGHT_STATE,
        CAPABILITY_TRAFFIC_SIGNS,
        CAPABILITY_LANE_MARKINGS,
        CAPABILITY_DRIVABLE_AREA,
    }
)


def _object(raw: Any, name: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{name} must be an object")
    return {str(key): value for key, value in raw.items()}


def _exact_keys(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{name} has unknown fields: " + ", ".join(unknown))


def _integer(raw: Any, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= raw <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return raw


def _number_triplet(raw: Any, name: str) -> list[float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"{name} must contain exactly 3 numbers")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw):
        raise TypeError(f"{name} must contain only numbers")
    values = [float(value) for value in raw]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain finite numbers")
    return values


def _string_sequence(raw: Any, name: str) -> list[str]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise TypeError(f"{name} must be a non-empty array of strings")
    values: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str):
            raise TypeError(f"{name}[{index}] must be a string")
        value = item.strip().lower()
        if not value:
            raise ValueError(f"{name}[{index}] must not be empty")
        if value in values:
            raise ValueError(f"{name} contains duplicate value {value!r}")
        values.append(value)
    return values


def normalize_control_outputs(raw: Any) -> dict[str, Any]:
    outputs = _object(raw, "outputs")
    _exact_keys(outputs, {"kind"}, "outputs")
    if outputs.get("kind") != CONTROL_OUTPUT_KIND:
        raise ValueError(f"outputs.kind must be {CONTROL_OUTPUT_KIND!r}")
    return {"kind": CONTROL_OUTPUT_KIND}


def normalize_torchscript_control_inputs(raw: Any) -> dict[str, Any]:
    inputs = _object(raw, "inputs")
    _exact_keys(inputs, {"image", "speed"}, "inputs")
    image = _object(inputs.get("image"), "inputs.image")
    _exact_keys(image, {"width", "height", "color", "mean", "std"}, "inputs.image")
    width = _integer(image.get("width"), "inputs.image.width", minimum=32, maximum=4096)
    height = _integer(image.get("height"), "inputs.image.height", minimum=32, maximum=4096)
    color = str(image.get("color", "rgb")).strip().lower()
    if color not in {"rgb", "bgr"}:
        raise ValueError("inputs.image.color must be rgb or bgr")
    mean = _number_triplet(image.get("mean", [0.0, 0.0, 0.0]), "inputs.image.mean")
    std = _number_triplet(image.get("std", [1.0, 1.0, 1.0]), "inputs.image.std")
    if any(value <= 0.0 for value in std):
        raise ValueError("inputs.image.std values must be positive")

    speed = _object(inputs.get("speed", {}), "inputs.speed")
    _exact_keys(speed, {"enabled", "unit"}, "inputs.speed")
    enabled = speed.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("inputs.speed.enabled must be a boolean")
    unit = str(speed.get("unit", "mps")).strip().lower()
    if unit not in {"mps", "kmh"}:
        raise ValueError("inputs.speed.unit must be mps or kmh")
    return {
        "image": {
            "width": width,
            "height": height,
            "color": color,
            "mean": mean,
            "std": std,
        },
        "speed": {"enabled": enabled, "unit": unit},
    }


def normalize_python_factory_inputs(raw: Any) -> dict[str, Any]:
    """Normalize the existing driving-policy Python factory input contract."""

    inputs = _object(raw, "inputs")
    _exact_keys(inputs, {"kind", "options"}, "inputs")
    if inputs.get("kind") != MODEL_OBSERVATION_INPUT_KIND:
        raise ValueError(f"inputs.kind must be {MODEL_OBSERVATION_INPUT_KIND!r}")
    options = inputs.get("options", {})
    if not isinstance(options, Mapping):
        raise TypeError("inputs.options must be an object")
    return {
        "kind": MODEL_OBSERVATION_INPUT_KIND,
        "options": {str(key): value for key, value in options.items()},
    }


def normalize_scene_perception_inputs(raw: Any) -> dict[str, Any]:
    """Normalize the exact-frame observation contract for scene perception.

    ``history`` declares how many consecutive RGB observations the adapter may
    require. The first runtime slice supports trusted Python factories, which
    own model-specific resizing, normalization and temporal buffering.
    """

    inputs = _object(raw, "inputs")
    _exact_keys(inputs, {"kind", "history", "options"}, "inputs")
    if inputs.get("kind") != SCENE_PERCEPTION_INPUT_KIND:
        raise ValueError(f"inputs.kind must be {SCENE_PERCEPTION_INPUT_KIND!r}")
    history = _integer(inputs.get("history", 1), "inputs.history", minimum=1, maximum=32)
    options = inputs.get("options", {})
    if not isinstance(options, Mapping):
        raise TypeError("inputs.options must be an object")
    return {
        "kind": SCENE_PERCEPTION_INPUT_KIND,
        "history": history,
        "options": {str(key): value for key, value in options.items()},
    }


def normalize_scene_perception_outputs(raw: Any) -> dict[str, Any]:
    """Normalize factual semantic capabilities advertised by one model.

    Downstream code uses this list to distinguish an empty prediction from a
    semantic the model never produces. In particular, traffic-light state is a
    separate capability and cannot be inferred merely because a light box is
    present.
    """

    outputs = _object(raw, "outputs")
    _exact_keys(outputs, {"kind", "capabilities"}, "outputs")
    if outputs.get("kind") != SCENE_PERCEPTION_OUTPUT_KIND:
        raise ValueError(f"outputs.kind must be {SCENE_PERCEPTION_OUTPUT_KIND!r}")
    capabilities = _string_sequence(outputs.get("capabilities"), "outputs.capabilities")
    unsupported = sorted(set(capabilities) - SCENE_PERCEPTION_CAPABILITIES)
    if unsupported:
        raise ValueError(
            "outputs.capabilities contains unsupported values: " + ", ".join(unsupported)
        )
    if (
        CAPABILITY_TRAFFIC_LIGHT_STATE in capabilities
        and CAPABILITY_TRAFFIC_LIGHTS not in capabilities
    ):
        raise ValueError("traffic_light_state capability requires traffic_lights")
    return {
        "kind": SCENE_PERCEPTION_OUTPUT_KIND,
        "capabilities": capabilities,
    }


def normalize_runtime_contract(
    *,
    runtime: str,
    role: str,
    inputs: Any,
    outputs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and normalize the contract used by one runtime package."""

    if role == "driving_policy":
        normalized_outputs = normalize_control_outputs(outputs)
        if runtime == "torchscript_control_v1":
            normalized_inputs = normalize_torchscript_control_inputs(inputs)
        elif runtime == "python_factory":
            normalized_inputs = normalize_python_factory_inputs(inputs)
        else:
            raise ValueError(f"runtime {runtime!r} does not support role='driving_policy'")
        return normalized_inputs, normalized_outputs

    if role == "scene_perception":
        if runtime != "python_factory":
            raise ValueError(
                f"runtime {runtime!r} does not support role='scene_perception'; "
                "use a trusted python_factory adapter"
            )
        return normalize_scene_perception_inputs(inputs), normalize_scene_perception_outputs(outputs)

    return _object(inputs, "inputs"), _object(outputs, "outputs")


__all__ = [
    "CAPABILITY_DRIVABLE_AREA",
    "CAPABILITY_LANE_MARKINGS",
    "CAPABILITY_OBJECTS",
    "CAPABILITY_TRAFFIC_LIGHTS",
    "CAPABILITY_TRAFFIC_LIGHT_STATE",
    "CAPABILITY_TRAFFIC_SIGNS",
    "CONTROL_OUTPUT_KIND",
    "MODEL_OBSERVATION_INPUT_KIND",
    "SCENE_PERCEPTION_CAPABILITIES",
    "SCENE_PERCEPTION_INPUT_KIND",
    "SCENE_PERCEPTION_OUTPUT_KIND",
    "normalize_control_outputs",
    "normalize_python_factory_inputs",
    "normalize_runtime_contract",
    "normalize_scene_perception_inputs",
    "normalize_scene_perception_outputs",
    "normalize_torchscript_control_inputs",
]
