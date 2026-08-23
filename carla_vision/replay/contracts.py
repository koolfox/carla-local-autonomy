"""Strict pre-registration contract for paired detector replay studies."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from ..contracts import DetectorConfig

REPLAY_CONFIG_SCHEMA_VERSION = "1.0"
REPLAY_RUN_SCHEMA_VERSION = "1.0"
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]{0,127}$")
_BUILTIN_BACKENDS = frozenset({"rtdetr", "yolo"})


def _strict_keys(
    raw: Mapping[str, Any],
    required: set[str],
    name: str,
) -> None:
    keys = {str(key) for key in raw}
    missing = sorted(required - keys)
    unknown = sorted(keys - required)
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(unknown)}")


def _identifier(value: Any, name: str) -> str:
    result = str(value).strip()
    if not _ID_PATTERN.fullmatch(result):
        raise ValueError(f"{name} must be lowercase and use only letters, digits, '.' or '-'")
    return result


def _nonempty(value: Any, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _positive_integer(value: Any, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be in [1, {maximum}]")
    return value


def _nonnegative_integer(value: Any, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} must be in [0, {maximum}]")
    return value


def _confidence(value: Any) -> float:
    result = float(value)
    if not 0.0 < result < 1.0:
        raise ValueError("bootstrap_confidence must be in (0, 1)")
    return result


def _resolve_optional_path(
    value: Any,
    *,
    base: Path,
    name: str,
) -> Path | None:
    if value is None:
        return None
    raw = _nonempty(value, name)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=True)


@dataclass(frozen=True)
class ReplayDetectorSpec:
    backend: str
    weights: Path | None
    image_size: int
    factory: str | None
    options: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, base: Path) -> Self:
        _strict_keys(
            raw,
            {"backend", "weights", "image_size", "factory", "options"},
            "replay detector",
        )
        backend = str(raw["backend"]).strip().lower().replace("_", "-")
        aliases = {
            "rt-detr": "rtdetr",
            "ultralytics-rtdetr": "rtdetr",
            "ultralytics-yolo": "yolo",
        }
        backend = aliases.get(backend, backend)
        if backend not in {*_BUILTIN_BACKENDS, "custom"}:
            raise ValueError(f"unsupported replay detector backend {backend!r}")
        weights = _resolve_optional_path(
            raw["weights"],
            base=base,
            name="detector.weights",
        )
        factory_raw = raw["factory"]
        factory = (
            None
            if factory_raw is None
            else _nonempty(
                factory_raw,
                "detector.factory",
            )
        )
        if backend in _BUILTIN_BACKENDS and weights is None:
            raise ValueError(f"{backend} detector requires weights")
        if backend == "custom" and factory is None:
            raise ValueError("custom detector requires factory")
        if backend != "custom" and factory is not None:
            raise ValueError("built-in detector cannot declare a custom factory")
        options = raw["options"]
        if not isinstance(options, Mapping):
            raise TypeError("detector.options must be an object")
        normalized_options = {str(key): value for key, value in options.items()}
        try:
            json.dumps(normalized_options, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("detector.options must contain finite JSON values") from error
        return cls(
            backend=backend,
            weights=weights,
            image_size=_positive_integer(
                raw["image_size"],
                "detector.image_size",
                16384,
            ),
            factory=factory,
            options=normalized_options,
        )

    def detector_config(self, *, device: str) -> DetectorConfig:
        return DetectorConfig(
            backend=self.backend,
            weights=self.weights,
            device=device,
            image_size=self.image_size,
            confidence=0.0,
            factory=self.factory,
            options=dict(self.options),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "weights": str(self.weights) if self.weights is not None else None,
            "image_size": self.image_size,
            "factory": self.factory,
            "options": dict(self.options),
        }


@dataclass(frozen=True)
class ReplayModelSpec:
    model_id: str
    label: str
    device: str
    model_package: Path | None
    detector: ReplayDetectorSpec | None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, base: Path) -> Self:
        _strict_keys(
            raw,
            {"model_id", "label", "device", "model_package", "detector"},
            "replay model",
        )
        package = _resolve_optional_path(
            raw["model_package"],
            base=base,
            name="model.model_package",
        )
        detector_raw = raw["detector"]
        if detector_raw is not None and not isinstance(detector_raw, Mapping):
            raise TypeError("model.detector must be an object or null")
        detector = (
            ReplayDetectorSpec.from_mapping(detector_raw, base=base)
            if isinstance(detector_raw, Mapping)
            else None
        )
        if (package is None) == (detector is None):
            raise ValueError("model must declare exactly one of model_package or detector")
        return cls(
            model_id=_identifier(raw["model_id"], "model.model_id"),
            label=_nonempty(raw["label"], "model.label"),
            device=_nonempty(raw["device"], "model.device"),
            model_package=package,
            detector=detector,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "label": self.label,
            "device": self.device,
            "model_package": (str(self.model_package) if self.model_package is not None else None),
            "detector": self.detector.as_dict() if self.detector is not None else None,
        }


@dataclass(frozen=True)
class ReplayConfig:
    schema_version: str
    replay_id: str
    title: str
    purpose: str
    master_seed: int
    reference_model_id: str
    bootstrap_replicates: int
    bootstrap_confidence: float
    latency_exclude_first_images: int
    montage_count: int
    models: tuple[ReplayModelSpec, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, base: Path) -> Self:
        required = {
            "schema_version",
            "replay_id",
            "title",
            "purpose",
            "master_seed",
            "reference_model_id",
            "bootstrap_replicates",
            "bootstrap_confidence",
            "latency_exclude_first_images",
            "montage_count",
            "models",
        }
        _strict_keys(raw, required, "replay config")
        schema_version = str(raw["schema_version"])
        if schema_version != REPLAY_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported replay config schema {schema_version!r}; "
                f"expected {REPLAY_CONFIG_SCHEMA_VERSION!r}"
            )
        purpose = str(raw["purpose"]).strip().lower()
        if purpose not in {"development", "confirmatory"}:
            raise ValueError("purpose must be development or confirmatory")
        master_seed = raw["master_seed"]
        if (
            isinstance(master_seed, bool)
            or not isinstance(master_seed, int)
            or not 0 <= master_seed < 2**63
        ):
            raise ValueError("master_seed must be an integer in [0, 2^63)")
        models_raw = raw["models"]
        if isinstance(models_raw, (str, bytes)) or not isinstance(
            models_raw,
            Sequence,
        ):
            raise TypeError("models must be an array")
        models = tuple(
            ReplayModelSpec.from_mapping(model, base=base)
            if isinstance(model, Mapping)
            else (_ for _ in ()).throw(TypeError("replay model must be an object"))
            for model in models_raw
        )
        if len(models) < 2:
            raise ValueError("replay requires at least two models")
        model_ids = [model.model_id for model in models]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("replay model IDs must be unique")
        reference_model_id = _identifier(
            raw["reference_model_id"],
            "reference_model_id",
        )
        if reference_model_id not in set(model_ids):
            raise ValueError("reference_model_id is absent from models")
        if purpose == "confirmatory" and any(model.model_package is None for model in models):
            raise ValueError("confirmatory replay requires verified model packages")
        return cls(
            schema_version=schema_version,
            replay_id=_identifier(raw["replay_id"], "replay_id"),
            title=_nonempty(raw["title"], "title"),
            purpose=purpose,
            master_seed=master_seed,
            reference_model_id=reference_model_id,
            bootstrap_replicates=_positive_integer(
                raw["bootstrap_replicates"],
                "bootstrap_replicates",
                1_000_000,
            ),
            bootstrap_confidence=_confidence(raw["bootstrap_confidence"]),
            latency_exclude_first_images=_nonnegative_integer(
                raw["latency_exclude_first_images"],
                "latency_exclude_first_images",
                100_000,
            ),
            montage_count=_positive_integer(
                raw["montage_count"],
                "montage_count",
                1000,
            ),
            models=models,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "replay_id": self.replay_id,
            "title": self.title,
            "purpose": self.purpose,
            "master_seed": self.master_seed,
            "reference_model_id": self.reference_model_id,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_confidence": self.bootstrap_confidence,
            "latency_exclude_first_images": self.latency_exclude_first_images,
            "montage_count": self.montage_count,
            "models": [model.as_dict() for model in self.models],
        }


def load_replay_config(path: str | Path) -> ReplayConfig:
    resolved = Path(path).expanduser().resolve(strict=True)
    with resolved.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("replay config must contain a JSON object")
    return ReplayConfig.from_mapping(payload, base=resolved.parent)


__all__ = [
    "REPLAY_CONFIG_SCHEMA_VERSION",
    "ReplayConfig",
    "ReplayDetectorSpec",
    "ReplayModelSpec",
    "load_replay_config",
]
