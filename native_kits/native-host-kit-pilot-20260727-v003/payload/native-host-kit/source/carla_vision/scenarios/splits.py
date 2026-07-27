"""Leakage-safe episode-group split assignment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

SPLIT_PLAN_SCHEMA_VERSION = "1.0"
GROUP_KEY_VERSION = "carla-vision-group-v1"


def canonical_map_family(map_name: str) -> str:
    """Collapse CARLA map paths and ``_Opt`` variants into one family."""

    normalized = str(map_name).strip().replace("\\", "/").rstrip("/")
    if not normalized:
        raise ValueError("map_name must not be empty")
    basename = normalized.rsplit("/", 1)[-1]
    if basename.endswith(".umap"):
        basename = basename[:-5]
    if basename.casefold().endswith("_opt"):
        basename = basename[:-4]
    if not basename:
        raise ValueError("map_name does not contain a valid map family")
    return basename


@dataclass(frozen=True)
class GroupIdentity:
    simulator_build: str
    map_family: str
    route_or_road_region_id: str
    scenario_recipe_id: str
    static_layout_seed: int
    episode_id: str

    def canonical_key(self) -> str:
        values = (
            GROUP_KEY_VERSION,
            self.simulator_build,
            canonical_map_family(self.map_family),
            self.route_or_road_region_id,
            self.scenario_recipe_id,
            str(self.static_layout_seed),
            self.episode_id,
        )
        if any("\0" in value for value in values):
            raise ValueError("group identity fields must not contain NUL")
        return "\0".join(values)

    def bucket(self) -> int:
        digest = hashlib.sha256(self.canonical_key().encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big", signed=False) % 100

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "map_family": canonical_map_family(self.map_family),
            "group_key_version": GROUP_KEY_VERSION,
            "canonical_key_sha256": hashlib.sha256(
                self.canonical_key().encode("utf-8")
            ).hexdigest(),
            "bucket": self.bucket(),
        }


@dataclass(frozen=True)
class SplitAssignment:
    partition: str
    reason: str
    bucket: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SplitPlan:
    plan_id: str
    schema_version: str
    validation_map_families: frozenset[str]
    test_map_families: frozenset[str]
    validation_weather_ids: frozenset[str]
    test_weather_ids: frozenset[str]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        required = {
            "plan_id",
            "schema_version",
            "validation_map_families",
            "test_map_families",
            "validation_weather_ids",
            "test_weather_ids",
        }
        optional = {
            "seen_group_hash_buckets",
            "assignment_priority",
            "group_key_version",
        }
        keys = {str(key) for key in raw}
        missing = sorted(required - keys)
        unknown = sorted(keys - required - optional)
        if missing:
            raise ValueError(f"split plan is missing fields: {', '.join(missing)}")
        if unknown:
            raise ValueError(f"split plan has unknown fields: {', '.join(unknown)}")
        schema_version = str(raw["schema_version"])
        if schema_version != SPLIT_PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported split-plan schema {schema_version!r}; "
                f"expected {SPLIT_PLAN_SCHEMA_VERSION!r}"
            )

        def string_set(field: str, *, maps: bool = False) -> frozenset[str]:
            value = raw[field]
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TypeError(f"split plan {field} must be an array")
            normalized = frozenset(
                canonical_map_family(str(item)) if maps else str(item).strip() for item in value
            )
            if any(not item for item in normalized):
                raise ValueError(f"split plan {field} cannot contain empty values")
            return normalized

        validation_maps = string_set("validation_map_families", maps=True)
        test_maps = string_set("test_map_families", maps=True)
        validation_weather = string_set("validation_weather_ids")
        test_weather = string_set("test_weather_ids")
        map_overlap = validation_maps & test_maps
        weather_overlap = validation_weather & test_weather
        if map_overlap:
            raise ValueError(
                "validation and test map families overlap: " + ", ".join(sorted(map_overlap))
            )
        if weather_overlap:
            raise ValueError(
                "validation and test weather IDs overlap: " + ", ".join(sorted(weather_overlap))
            )
        if raw.get("group_key_version", GROUP_KEY_VERSION) != GROUP_KEY_VERSION:
            raise ValueError("split plan group_key_version is inconsistent")
        expected_buckets = {
            "train": [0, 84],
            "val_seen": [85, 92],
            "test_seen": [93, 99],
        }
        if raw.get("seen_group_hash_buckets", expected_buckets) != expected_buckets:
            raise ValueError("split plan seen_group_hash_buckets are inconsistent")
        expected_priority = [
            "test_map_ood",
            "val_map_ood",
            "test_weather_ood",
            "val_weather_ood",
            "seen_group_hash",
        ]
        if raw.get("assignment_priority", expected_priority) != expected_priority:
            raise ValueError("split plan assignment_priority is inconsistent")
        return cls(
            plan_id=str(raw["plan_id"]),
            schema_version=schema_version,
            validation_map_families=validation_maps,
            test_map_families=test_maps,
            validation_weather_ids=validation_weather,
            test_weather_ids=test_weather,
        )

    def assign(self, group: GroupIdentity, weather_id: str) -> SplitAssignment:
        family = canonical_map_family(group.map_family)
        bucket = group.bucket()
        if family in self.test_map_families:
            return SplitAssignment("test_map_ood", "held_out_map_family", bucket)
        if family in self.validation_map_families:
            return SplitAssignment("val_map_ood", "held_out_map_family", bucket)
        if weather_id in self.test_weather_ids:
            return SplitAssignment("test_weather_ood", "held_out_weather", bucket)
        if weather_id in self.validation_weather_ids:
            return SplitAssignment("val_weather_ood", "held_out_weather", bucket)
        if bucket <= 84:
            return SplitAssignment("train", "deterministic_group_hash", bucket)
        if bucket <= 92:
            return SplitAssignment("val_seen", "deterministic_group_hash", bucket)
        return SplitAssignment("test_seen", "deterministic_group_hash", bucket)

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "schema_version": self.schema_version,
            "validation_map_families": sorted(self.validation_map_families),
            "test_map_families": sorted(self.test_map_families),
            "validation_weather_ids": sorted(self.validation_weather_ids),
            "test_weather_ids": sorted(self.test_weather_ids),
            "seen_group_hash_buckets": {
                "train": [0, 84],
                "val_seen": [85, 92],
                "test_seen": [93, 99],
            },
            "assignment_priority": [
                "test_map_ood",
                "val_map_ood",
                "test_weather_ood",
                "val_weather_ood",
                "seen_group_hash",
            ],
            "group_key_version": GROUP_KEY_VERSION,
        }


def load_split_plan(path: str | Path) -> SplitPlan:
    resolved = Path(path).expanduser().resolve(strict=True)
    with resolved.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("split plan must be a JSON object")
    return SplitPlan.from_mapping(payload)


__all__ = [
    "GROUP_KEY_VERSION",
    "SPLIT_PLAN_SCHEMA_VERSION",
    "GroupIdentity",
    "SplitAssignment",
    "SplitPlan",
    "canonical_map_family",
    "load_split_plan",
]
