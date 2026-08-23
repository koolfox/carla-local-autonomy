"""Frozen configuration contract for canonical detector evaluation."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

EVALUATION_CONFIG_SCHEMA_VERSION = "1.0"
EVALUATION_RUN_SCHEMA_VERSION = "1.0"
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _probability(value: Any, name: str, *, allow_zero: bool) -> float:
    result = float(value)
    minimum = 0.0 if allow_zero else 1e-9
    if not minimum <= result <= 1.0:
        raise ValueError(f"{name} must be in [{minimum}, 1]")
    return result


def _positive_integer(value: Any, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be in [1, {maximum}]")
    return value


def _partitions(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("partitions must be an array")
    result = tuple(dict.fromkeys(str(item).strip() for item in value))
    if not result or any(not item for item in result):
        raise ValueError("partitions must contain non-empty unique strings")
    return result


@dataclass(frozen=True)
class EvaluationConfig:
    evaluation_id: str
    schema_version: str
    purpose: str
    master_seed: int
    partitions: tuple[str, ...]
    minimum_prediction_confidence: float
    operating_confidence: float
    matching_iou: float
    calibration_bins: int
    bootstrap_replicates: int
    bootstrap_confidence: float
    montage_count: int
    class_aliases: Mapping[str, str]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        required = {
            "evaluation_id",
            "schema_version",
            "purpose",
            "master_seed",
            "partitions",
            "minimum_prediction_confidence",
            "operating_confidence",
            "matching_iou",
            "calibration_bins",
            "bootstrap_replicates",
            "bootstrap_confidence",
            "montage_count",
            "class_aliases",
        }
        keys = {str(key) for key in raw}
        missing = sorted(required - keys)
        unknown = sorted(keys - required)
        if missing:
            raise ValueError(f"evaluation config is missing fields: {', '.join(missing)}")
        if unknown:
            raise ValueError(f"evaluation config has unknown fields: {', '.join(unknown)}")
        schema_version = str(raw["schema_version"])
        if schema_version != EVALUATION_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported evaluation config schema {schema_version!r}; "
                f"expected {EVALUATION_CONFIG_SCHEMA_VERSION!r}"
            )
        evaluation_id = str(raw["evaluation_id"])
        if not _ID_PATTERN.fullmatch(evaluation_id):
            raise ValueError("evaluation_id contains invalid characters")
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
        minimum_confidence = _probability(
            raw["minimum_prediction_confidence"],
            "minimum_prediction_confidence",
            allow_zero=True,
        )
        operating_confidence = _probability(
            raw["operating_confidence"],
            "operating_confidence",
            allow_zero=True,
        )
        if minimum_confidence > operating_confidence:
            raise ValueError("minimum_prediction_confidence cannot exceed operating_confidence")
        aliases = raw["class_aliases"]
        if not isinstance(aliases, Mapping):
            raise TypeError("class_aliases must be an object")
        normalized_aliases = {
            str(source).strip(): str(target).strip() for source, target in aliases.items()
        }
        if any(not source or not target for source, target in normalized_aliases.items()):
            raise ValueError("class_aliases cannot contain empty labels")
        return cls(
            evaluation_id=evaluation_id,
            schema_version=schema_version,
            purpose=purpose,
            master_seed=master_seed,
            partitions=_partitions(raw["partitions"]),
            minimum_prediction_confidence=minimum_confidence,
            operating_confidence=operating_confidence,
            matching_iou=_probability(
                raw["matching_iou"],
                "matching_iou",
                allow_zero=False,
            ),
            calibration_bins=_positive_integer(raw["calibration_bins"], "calibration_bins", 1000),
            bootstrap_replicates=_positive_integer(
                raw["bootstrap_replicates"],
                "bootstrap_replicates",
                1_000_000,
            ),
            bootstrap_confidence=_probability(
                raw["bootstrap_confidence"],
                "bootstrap_confidence",
                allow_zero=False,
            ),
            montage_count=_positive_integer(raw["montage_count"], "montage_count", 1000),
            class_aliases=normalized_aliases,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    resolved = Path(path).expanduser().resolve(strict=True)
    with resolved.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("evaluation config must contain a JSON object")
    return EvaluationConfig.from_mapping(payload)


__all__ = [
    "EVALUATION_CONFIG_SCHEMA_VERSION",
    "EvaluationConfig",
    "load_evaluation_config",
]
