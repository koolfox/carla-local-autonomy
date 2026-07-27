"""Strict pre-registration contract for promoted detector-model packages."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

MODEL_RELEASE_CONFIG_SCHEMA_VERSION = "1.0"
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]{0,127}$")
_PLACEHOLDER_PATTERN = re.compile(r"(?i)\b(?:unknown|tbd|todo|replace(?:-me)?|unlicensed)\b")


def _strict_keys(raw: Mapping[str, Any], required: set[str], name: str) -> None:
    keys = {str(key) for key in raw}
    missing = sorted(required - keys)
    unknown = sorted(keys - required)
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(unknown)}")


def _nonempty(value: Any, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    result = tuple(dict.fromkeys(str(item).strip() for item in value))
    if not result or any(not item for item in result):
        raise ValueError(f"{name} must contain non-empty unique strings")
    return result


@dataclass(frozen=True)
class ModelInputContract:
    color_order: str
    image_size: int
    resize_method: str
    normalization: str
    temporal_context_frames: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        required = {
            "color_order",
            "image_size",
            "resize_method",
            "normalization",
            "temporal_context_frames",
        }
        _strict_keys(raw, required, "model input contract")
        color_order = str(raw["color_order"]).strip().upper()
        if color_order not in {"BGR", "RGB"}:
            raise ValueError("input.color_order must be BGR or RGB")
        image_size = raw["image_size"]
        if (
            isinstance(image_size, bool)
            or not isinstance(image_size, int)
            or not 32 <= image_size <= 8192
        ):
            raise ValueError("input.image_size must be an integer in [32, 8192]")
        temporal = raw["temporal_context_frames"]
        if isinstance(temporal, bool) or temporal != 1:
            raise ValueError(
                "input.temporal_context_frames must be 1 for the current detector contract"
            )
        return cls(
            color_order=color_order,
            image_size=image_size,
            resize_method=_nonempty(raw["resize_method"], "input.resize_method"),
            normalization=_nonempty(raw["normalization"], "input.normalization"),
            temporal_context_frames=temporal,
        )


@dataclass(frozen=True)
class ModelSelectionContract:
    metric: str
    mode: str
    partitions: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        _strict_keys(raw, {"metric", "mode", "partitions"}, "model selection")
        mode = str(raw["mode"]).strip().lower()
        if mode not in {"min", "max"}:
            raise ValueError("selection.mode must be 'min' or 'max'")
        partitions = _strings(raw["partitions"], "selection.partitions")
        if any(partition.startswith("test") for partition in partitions):
            raise ValueError("model selection cannot use locked test partitions")
        return cls(
            metric=_nonempty(raw["metric"], "selection.metric"),
            mode=mode,
            partitions=partitions,
        )


@dataclass(frozen=True)
class ModelInferenceContract:
    detector_backend: str
    detector_factory: str | None
    options: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        _strict_keys(
            raw,
            {"detector_backend", "detector_factory", "options"},
            "model inference",
        )
        backend = str(raw["detector_backend"]).strip().lower().replace("_", "-")
        if backend not in {"rtdetr", "yolo", "custom"}:
            raise ValueError("inference.detector_backend must be rtdetr, yolo, or custom")
        factory_raw = raw["detector_factory"]
        factory = None if factory_raw is None else str(factory_raw).strip()
        if backend == "custom" and not factory:
            raise ValueError("custom inference backend requires detector_factory")
        if backend != "custom" and factory:
            raise ValueError("detector_factory is only valid for custom inference")
        options = raw["options"]
        if not isinstance(options, Mapping):
            raise TypeError("inference.options must be an object")
        return cls(
            detector_backend=backend,
            detector_factory=factory,
            options=dict(options),
        )


@dataclass(frozen=True)
class ModelLicenseContract:
    weights: str
    code: str
    upstream: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        _strict_keys(raw, {"weights", "code", "upstream"}, "model license")
        values = {
            name: _nonempty(raw[name], f"license.{name}")
            for name in ("weights", "code", "upstream")
        }
        placeholders = [
            name for name, value in values.items() if _PLACEHOLDER_PATTERN.search(value)
        ]
        if placeholders:
            raise ValueError(
                "license fields cannot contain placeholders: " + ", ".join(placeholders)
            )
        return cls(**values)


@dataclass(frozen=True)
class ModelReleaseConfig:
    schema_version: str
    model_id: str
    architecture: str
    backend: str
    checkpoint_role: str
    input: ModelInputContract
    inference: ModelInferenceContract
    output_contract: str
    selection: ModelSelectionContract
    intended_use: str
    limitations: tuple[str, ...]
    license: ModelLicenseContract
    upstream_source: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        required = {
            "schema_version",
            "model_id",
            "architecture",
            "backend",
            "checkpoint_role",
            "input",
            "inference",
            "output_contract",
            "selection",
            "intended_use",
            "limitations",
            "license",
            "upstream_source",
        }
        _strict_keys(raw, required, "model release config")
        schema_version = str(raw["schema_version"])
        if schema_version != MODEL_RELEASE_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported model release config schema {schema_version!r}; "
                f"expected {MODEL_RELEASE_CONFIG_SCHEMA_VERSION!r}"
            )
        model_id = str(raw["model_id"]).strip()
        if not _ID_PATTERN.fullmatch(model_id):
            raise ValueError("model_id must be lowercase and use only letters, digits, '.' or '-'")
        checkpoint_role = str(raw["checkpoint_role"]).strip().lower()
        if checkpoint_role not in {"best", "last"}:
            raise ValueError("checkpoint_role must be 'best' or 'last'")
        input_raw = raw["input"]
        inference_raw = raw["inference"]
        selection_raw = raw["selection"]
        license_raw = raw["license"]
        if not isinstance(input_raw, Mapping):
            raise TypeError("input must be an object")
        if not isinstance(inference_raw, Mapping):
            raise TypeError("inference must be an object")
        if not isinstance(selection_raw, Mapping):
            raise TypeError("selection must be an object")
        if not isinstance(license_raw, Mapping):
            raise TypeError("license must be an object")
        output_contract = str(raw["output_contract"]).strip()
        if output_contract != "carla-vision-detection-v1":
            raise ValueError(
                "output_contract must be 'carla-vision-detection-v1' for this framework"
            )
        return cls(
            schema_version=schema_version,
            model_id=model_id,
            architecture=_nonempty(raw["architecture"], "architecture"),
            backend=_nonempty(raw["backend"], "backend").lower().replace("_", "-"),
            checkpoint_role=checkpoint_role,
            input=ModelInputContract.from_mapping(input_raw),
            inference=ModelInferenceContract.from_mapping(inference_raw),
            output_contract=output_contract,
            selection=ModelSelectionContract.from_mapping(selection_raw),
            intended_use=_nonempty(raw["intended_use"], "intended_use"),
            limitations=_strings(raw["limitations"], "limitations"),
            license=ModelLicenseContract.from_mapping(license_raw),
            upstream_source=_nonempty(raw["upstream_source"], "upstream_source"),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_model_release_config(path: str | Path) -> ModelReleaseConfig:
    resolved = Path(path).expanduser().resolve(strict=True)
    with resolved.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("model release config must contain a JSON object")
    return ModelReleaseConfig.from_mapping(payload)


__all__ = [
    "MODEL_RELEASE_CONFIG_SCHEMA_VERSION",
    "ModelInputContract",
    "ModelInferenceContract",
    "ModelLicenseContract",
    "ModelReleaseConfig",
    "ModelSelectionContract",
    "load_model_release_config",
]
