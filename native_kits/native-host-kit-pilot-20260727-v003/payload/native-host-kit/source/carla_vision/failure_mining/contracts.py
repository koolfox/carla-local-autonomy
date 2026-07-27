"""Strict preregistration contract for validation-only failure mining."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

FAILURE_MINING_CONFIG_SCHEMA_VERSION = "1.0"
FAILURE_TYPES = (
    "false_negative",
    "false_positive",
    "misclassification",
    "localization",
)
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]{0,127}$")


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


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    result = float(value)
    minimum_ok = result >= 0.0 if allow_zero else result > 0.0
    if not math.isfinite(result) or not minimum_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _positive_integer(value: Any, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be in [1, {maximum}]")
    return value


def _strings(
    value: Any,
    name: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    result = tuple(dict.fromkeys(str(item).strip() for item in value))
    if any(not item for item in result) or (not allow_empty and not result):
        raise ValueError(f"{name} contains empty values or is empty")
    return result


@dataclass(frozen=True)
class FailureMiningConfig:
    schema_version: str
    mining_id: str
    title: str
    purpose: str
    master_seed: int
    threshold_source: str
    failure_types: tuple[str, ...]
    localization_iou_minimum: float
    max_candidates: int
    per_image_limit: int
    per_episode_limit: int
    montage_count: int
    severity_weights: Mapping[str, float]
    critical_categories: tuple[str, ...]
    critical_multiplier: float
    rarity_weight: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        required = {
            "schema_version",
            "mining_id",
            "title",
            "purpose",
            "master_seed",
            "threshold_source",
            "failure_types",
            "localization_iou_minimum",
            "max_candidates",
            "per_image_limit",
            "per_episode_limit",
            "montage_count",
            "severity_weights",
            "critical_categories",
            "critical_multiplier",
            "rarity_weight",
        }
        _strict_keys(raw, required, "failure mining config")
        schema_version = str(raw["schema_version"])
        if schema_version != FAILURE_MINING_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported failure mining schema {schema_version!r}; "
                f"expected {FAILURE_MINING_CONFIG_SCHEMA_VERSION!r}"
            )
        mining_id = str(raw["mining_id"]).strip()
        if not _ID_PATTERN.fullmatch(mining_id):
            raise ValueError("mining_id must be lowercase and use letters, digits, '.' or '-'")
        title = str(raw["title"]).strip()
        if not title:
            raise ValueError("title must not be empty")
        purpose = str(raw["purpose"]).strip().lower()
        if purpose != "development":
            raise ValueError(
                "failure mining is a development workflow; purpose must be development"
            )
        master_seed = raw["master_seed"]
        if (
            isinstance(master_seed, bool)
            or not isinstance(master_seed, int)
            or not 0 <= master_seed < 2**63
        ):
            raise ValueError("master_seed must be an integer in [0, 2^63)")
        threshold_source = str(raw["threshold_source"]).strip().lower()
        if threshold_source not in {
            "evaluation_config",
            "selection_artifact",
        }:
            raise ValueError("threshold_source must be evaluation_config or selection_artifact")
        failure_types = _strings(
            raw["failure_types"],
            "failure_types",
            allow_empty=False,
        )
        unsupported = sorted(set(failure_types) - set(FAILURE_TYPES))
        if unsupported:
            raise ValueError("unsupported failure types: " + ", ".join(unsupported))
        localization_iou = float(raw["localization_iou_minimum"])
        if not 0.0 < localization_iou < 1.0:
            raise ValueError("localization_iou_minimum must be in (0, 1)")
        weights_raw = raw["severity_weights"]
        if not isinstance(weights_raw, Mapping):
            raise TypeError("severity_weights must be an object")
        if set(weights_raw) != set(FAILURE_TYPES):
            raise ValueError("severity_weights must contain exactly: " + ", ".join(FAILURE_TYPES))
        weights = {
            failure_type: _positive(
                weights_raw[failure_type],
                f"severity_weights.{failure_type}",
            )
            for failure_type in FAILURE_TYPES
        }
        return cls(
            schema_version=schema_version,
            mining_id=mining_id,
            title=title,
            purpose=purpose,
            master_seed=master_seed,
            threshold_source=threshold_source,
            failure_types=failure_types,
            localization_iou_minimum=localization_iou,
            max_candidates=_positive_integer(
                raw["max_candidates"],
                "max_candidates",
                1_000_000,
            ),
            per_image_limit=_positive_integer(
                raw["per_image_limit"],
                "per_image_limit",
                100_000,
            ),
            per_episode_limit=_positive_integer(
                raw["per_episode_limit"],
                "per_episode_limit",
                1_000_000,
            ),
            montage_count=_positive_integer(
                raw["montage_count"],
                "montage_count",
                1000,
            ),
            severity_weights=weights,
            critical_categories=_strings(
                raw["critical_categories"],
                "critical_categories",
                allow_empty=True,
            ),
            critical_multiplier=_positive(
                raw["critical_multiplier"],
                "critical_multiplier",
            ),
            rarity_weight=_positive(
                raw["rarity_weight"],
                "rarity_weight",
                allow_zero=True,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_failure_mining_config(path: str | Path) -> FailureMiningConfig:
    resolved = Path(path).expanduser().resolve(strict=True)
    with resolved.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("failure mining config must contain a JSON object")
    return FailureMiningConfig.from_mapping(payload)


__all__ = [
    "FAILURE_MINING_CONFIG_SCHEMA_VERSION",
    "FAILURE_TYPES",
    "FailureMiningConfig",
    "load_failure_mining_config",
]
