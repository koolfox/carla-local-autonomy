"""Strict preregistration for validation-only operating-threshold selection."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

THRESHOLD_CONFIG_SCHEMA_VERSION = "1.0"
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]{0,127}$")
_OBJECTIVES = frozenset(
    {
        "maximize_f1",
        "minimum_recall",
        "minimum_precision",
        "weighted_error",
    }
)


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


def _probability_or_none(value: Any, name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be null or in [0, 1]")
    return result


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not 0.0 < result < float("inf"):
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_integer(value: Any, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be in [1, {maximum}]")
    return value


@dataclass(frozen=True)
class ThresholdGrid:
    minimum: float
    maximum: float
    steps: int
    scale: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        _strict_keys(
            raw,
            {"minimum", "maximum", "steps", "scale"},
            "threshold grid",
        )
        minimum = float(raw["minimum"])
        maximum = float(raw["maximum"])
        if not 0.0 <= minimum < maximum <= 1.0:
            raise ValueError("grid minimum and maximum must satisfy 0 <= minimum < maximum <= 1")
        scale = str(raw["scale"]).strip().lower()
        if scale not in {"linear", "log"}:
            raise ValueError("grid.scale must be linear or log")
        if scale == "log" and minimum <= 0.0:
            raise ValueError("log threshold grid requires a positive minimum")
        return cls(
            minimum=minimum,
            maximum=maximum,
            steps=_positive_integer(raw["steps"], "grid.steps", 100_000),
            scale=scale,
        )


@dataclass(frozen=True)
class ThresholdSelectionConfig:
    schema_version: str
    selection_id: str
    title: str
    purpose: str
    master_seed: int
    objective: str
    target_recall: float | None
    target_precision: float | None
    false_positive_cost: float
    false_negative_cost: float
    tie_breaker: str
    grid: ThresholdGrid
    bootstrap_replicates: int
    bootstrap_confidence: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        required = {
            "schema_version",
            "selection_id",
            "title",
            "purpose",
            "master_seed",
            "objective",
            "target_recall",
            "target_precision",
            "false_positive_cost",
            "false_negative_cost",
            "tie_breaker",
            "grid",
            "bootstrap_replicates",
            "bootstrap_confidence",
        }
        _strict_keys(raw, required, "threshold selection config")
        schema_version = str(raw["schema_version"])
        if schema_version != THRESHOLD_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported threshold config schema {schema_version!r}; "
                f"expected {THRESHOLD_CONFIG_SCHEMA_VERSION!r}"
            )
        selection_id = str(raw["selection_id"]).strip()
        if not _ID_PATTERN.fullmatch(selection_id):
            raise ValueError("selection_id must be lowercase and use letters, digits, '.' or '-'")
        title = str(raw["title"]).strip()
        if not title:
            raise ValueError("title must not be empty")
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
        objective = str(raw["objective"]).strip().lower()
        if objective not in _OBJECTIVES:
            raise ValueError(f"unsupported threshold objective {objective!r}")
        target_recall = _probability_or_none(
            raw["target_recall"],
            "target_recall",
        )
        target_precision = _probability_or_none(
            raw["target_precision"],
            "target_precision",
        )
        if objective == "minimum_recall" and target_recall is None:
            raise ValueError("minimum_recall objective requires target_recall")
        if objective != "minimum_recall" and target_recall is not None:
            raise ValueError("target_recall is only valid for minimum_recall objective")
        if objective == "minimum_precision" and target_precision is None:
            raise ValueError("minimum_precision objective requires target_precision")
        if objective != "minimum_precision" and target_precision is not None:
            raise ValueError("target_precision is only valid for minimum_precision objective")
        tie_breaker = str(raw["tie_breaker"]).strip().lower()
        if tie_breaker not in {"highest_threshold", "lowest_threshold"}:
            raise ValueError("tie_breaker must be highest_threshold or lowest_threshold")
        grid_raw = raw["grid"]
        if not isinstance(grid_raw, Mapping):
            raise TypeError("grid must be an object")
        confidence = float(raw["bootstrap_confidence"])
        if not 0.0 < confidence < 1.0:
            raise ValueError("bootstrap_confidence must be in (0, 1)")
        return cls(
            schema_version=schema_version,
            selection_id=selection_id,
            title=title,
            purpose=purpose,
            master_seed=master_seed,
            objective=objective,
            target_recall=target_recall,
            target_precision=target_precision,
            false_positive_cost=_positive(
                raw["false_positive_cost"],
                "false_positive_cost",
            ),
            false_negative_cost=_positive(
                raw["false_negative_cost"],
                "false_negative_cost",
            ),
            tie_breaker=tie_breaker,
            grid=ThresholdGrid.from_mapping(grid_raw),
            bootstrap_replicates=_positive_integer(
                raw["bootstrap_replicates"],
                "bootstrap_replicates",
                1_000_000,
            ),
            bootstrap_confidence=confidence,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_threshold_selection_config(
    path: str | Path,
) -> ThresholdSelectionConfig:
    resolved = Path(path).expanduser().resolve(strict=True)
    with resolved.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("threshold selection config must contain a JSON object")
    return ThresholdSelectionConfig.from_mapping(payload)


__all__ = [
    "THRESHOLD_CONFIG_SCHEMA_VERSION",
    "ThresholdGrid",
    "ThresholdSelectionConfig",
    "load_threshold_selection_config",
]
