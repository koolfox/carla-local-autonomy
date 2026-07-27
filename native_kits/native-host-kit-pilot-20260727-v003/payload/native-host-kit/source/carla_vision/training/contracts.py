"""Strict contracts shared by training launchers and backend adapters."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, Self, runtime_checkable

TRAINING_CONFIG_SCHEMA_VERSION = "1.0"
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PROTECTED_BACKEND_OPTIONS = frozenset(
    {
        "data",
        "project",
        "name",
        "exist_ok",
        "resume",
        "seed",
        "deterministic",
        "epochs",
        "imgsz",
        "batch",
        "device",
        "workers",
    }
)


def _identifier(value: Any, name: str) -> str:
    result = str(value)
    if not _ID_PATTERN.fullmatch(result):
        raise ValueError(f"{name} must be 1-128 characters using letters, digits, '.', '_' or '-'")
    return result


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return result


def _positive_number(value: Any, name: str, maximum: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= maximum:
        raise ValueError(f"{name} must be finite and in (0, {maximum}]")
    return result


def _nonempty_string(value: Any, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _batch_size(value: Any) -> int:
    result = _integer(value, "batch_size", -1, 1_000_000)
    if result == 0:
        raise ValueError("batch_size must be -1 for automatic sizing or a positive integer")
    return result


def _string_sequence(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    result = tuple(dict.fromkeys(str(item).strip() for item in value))
    if not result or any(not item for item in result):
        raise ValueError(f"{name} must contain non-empty unique strings")
    return result


@dataclass(frozen=True)
class TrainingConfig:
    experiment_id: str
    schema_version: str
    backend: str
    custom_factory: str | None
    master_seed: int
    epochs: int
    image_size: int
    batch_size: int
    device: str
    workers: int
    patience: int
    optimizer: str
    initial_learning_rate: float
    final_learning_rate_fraction: float
    weight_decay: float
    warmup_epochs: float
    amp: bool
    cache: bool | str
    deterministic: bool
    save_period: int
    training_partitions: tuple[str, ...]
    validation_partitions: tuple[str, ...]
    extra_options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        required = {
            "experiment_id",
            "schema_version",
            "backend",
            "custom_factory",
            "master_seed",
            "epochs",
            "image_size",
            "batch_size",
            "device",
            "workers",
            "patience",
            "optimizer",
            "initial_learning_rate",
            "final_learning_rate_fraction",
            "weight_decay",
            "warmup_epochs",
            "amp",
            "cache",
            "deterministic",
            "save_period",
            "training_partitions",
            "validation_partitions",
            "extra_options",
        }
        keys = {str(key) for key in raw}
        missing = sorted(required - keys)
        unknown = sorted(keys - required)
        if missing:
            raise ValueError(f"training config is missing fields: {', '.join(missing)}")
        if unknown:
            raise ValueError(f"training config has unknown fields: {', '.join(unknown)}")
        schema_version = str(raw["schema_version"])
        if schema_version != TRAINING_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported training config schema {schema_version!r}; "
                f"expected {TRAINING_CONFIG_SCHEMA_VERSION!r}"
            )
        backend = str(raw["backend"]).strip().lower().replace("_", "-")
        if backend not in {
            "rtdetr",
            "ultralytics-rtdetr",
            "yolo",
            "ultralytics-yolo",
            "custom",
        }:
            raise ValueError(f"unsupported training backend {backend!r}")
        custom_factory_raw = raw["custom_factory"]
        custom_factory = None if custom_factory_raw is None else str(custom_factory_raw).strip()
        if backend == "custom" and not custom_factory:
            raise ValueError("custom training backend requires custom_factory")
        if backend != "custom" and custom_factory:
            raise ValueError("custom_factory is only valid for the custom backend")
        master_seed = _integer(raw["master_seed"], "master_seed", 0, (2**63) - 1)
        for boolean_field in ("amp", "deterministic"):
            if not isinstance(raw[boolean_field], bool):
                raise TypeError(f"{boolean_field} must be boolean")
        cache = raw["cache"]
        if not isinstance(cache, (bool, str)):
            raise TypeError("cache must be boolean or a backend-supported string")
        if isinstance(cache, str) and not cache.strip():
            raise ValueError("cache string must not be empty")
        training_partitions = _string_sequence(raw["training_partitions"], "training_partitions")
        validation_partitions = _string_sequence(
            raw["validation_partitions"], "validation_partitions"
        )
        if any(partition.startswith("test") for partition in training_partitions):
            raise ValueError("locked test partitions cannot be used for training")
        if any(partition.startswith("test") for partition in validation_partitions):
            raise ValueError("locked test partitions cannot be used for model selection")
        if set(training_partitions) & set(validation_partitions):
            raise ValueError("training and validation partitions must be disjoint")
        extra_options = raw["extra_options"]
        if not isinstance(extra_options, Mapping):
            raise TypeError("extra_options must be an object")
        protected = sorted(_PROTECTED_BACKEND_OPTIONS & set(extra_options))
        if protected:
            raise ValueError(
                "extra_options cannot override controlled fields: " + ", ".join(protected)
            )
        return cls(
            experiment_id=_identifier(raw["experiment_id"], "experiment_id"),
            schema_version=schema_version,
            backend=backend,
            custom_factory=custom_factory,
            master_seed=master_seed,
            epochs=_integer(raw["epochs"], "epochs", 1, 100_000),
            image_size=_integer(raw["image_size"], "image_size", 32, 8192),
            batch_size=_batch_size(raw["batch_size"]),
            device=_nonempty_string(raw["device"], "device"),
            workers=_integer(raw["workers"], "workers", 0, 1024),
            patience=_integer(raw["patience"], "patience", 0, 100_000),
            optimizer=_nonempty_string(raw["optimizer"], "optimizer"),
            initial_learning_rate=_positive_number(
                raw["initial_learning_rate"],
                "initial_learning_rate",
                10.0,
            ),
            final_learning_rate_fraction=_number(
                raw["final_learning_rate_fraction"],
                "final_learning_rate_fraction",
                0.0,
                10.0,
            ),
            weight_decay=_number(raw["weight_decay"], "weight_decay", 0.0, 10.0),
            warmup_epochs=_number(raw["warmup_epochs"], "warmup_epochs", 0.0, 10_000.0),
            amp=raw["amp"],
            cache=cache,
            deterministic=raw["deterministic"],
            save_period=_integer(raw["save_period"], "save_period", -1, 100_000),
            training_partitions=training_partitions,
            validation_partitions=validation_partitions,
            extra_options=dict(extra_options),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingRequest:
    config: TrainingConfig
    pretrained_weights: Path
    dataset_root: Path
    data_config: Path
    output_root: Path
    backend_seed: int


@dataclass(frozen=True)
class TrainingResult:
    backend_name: str
    output_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    metrics: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class TrainerBackend(Protocol):
    @property
    def name(self) -> str:
        """Stable backend implementation name."""

    def train(self, request: TrainingRequest) -> TrainingResult:
        """Execute training and return paths inside ``request.output_root``."""


def load_training_config(path: str | Path) -> TrainingConfig:
    resolved = Path(path).expanduser().resolve(strict=True)
    with resolved.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("training config must contain a JSON object")
    return TrainingConfig.from_mapping(payload)


__all__ = [
    "TRAINING_CONFIG_SCHEMA_VERSION",
    "TrainerBackend",
    "TrainingConfig",
    "TrainingRequest",
    "TrainingResult",
    "load_training_config",
]
