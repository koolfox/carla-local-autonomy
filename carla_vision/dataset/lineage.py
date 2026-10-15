"""Versioned split and ancestry contracts. No simulator or ML dependencies.

A change of episode, seed, filename or camera never authorizes repartitioning.
Route-endpoint grouping is conservative and is NOT proof of route-OOD geometry.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..scenarios.splits import canonical_map_family

LINEAGE_VERSION = "1.0"
PARTITIONS = frozenset(
    {
        "train",
        "val",
        "test",
        "val_seen",
        "test_seen",
        "val_map_ood",
        "test_map_ood",
        "val_weather_ood",
        "test_weather_ood",
        "unassigned",
        "acquisition_pool",
    }
)


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def partition_role(partition: str) -> str:
    if partition not in PARTITIONS:
        raise ValueError(f"unknown source partition {partition!r}")
    if partition.startswith("val"):
        return "val"
    if partition.startswith("test"):
        return "test"
    return partition


def require_train_parent(partition: str) -> None:
    if partition_role(partition) not in {"train", "acquisition_pool"}:
        raise ValueError("adaptive acquisition accepts only train/acquisition_pool parents")


def validate_derivation(parent_partition: str, child_partition: str, action: str) -> None:
    """Validate a *new* release; never modify a published parent in-place."""
    partition_role(parent_partition)
    partition_role(child_partition)
    if action not in {"RECAPTURE", "RELABEL", "NEW_SCENE", "FREEZE"}:
        raise ValueError("unknown derivation action")
    if parent_partition == "unassigned":
        raise ValueError("unassigned sources require a separately reviewed split release")
    if parent_partition == "acquisition_pool" and child_partition == "train":
        if action != "FREEZE":
            raise ValueError("only explicit FREEZE may promote acquisition_pool to train")
    elif child_partition != parent_partition:
        raise ValueError("a derived release must preserve the exact parent partition")


def scene_group_key(context: Mapping[str, Any], *, episode_id: str) -> str:
    """Return the strongest available grouping, without inventing geometry.

    An explicit scene root covers re-captures/multiview/relabels. Otherwise
    endpoint groups are deliberately broader than a seeded route identity.
    Legacy records without route evidence remain episode-scoped.
    """
    lineage = context.get("lineage")
    if isinstance(lineage, Mapping) and lineage.get("scene_root_id"):
        # A root is already a canonical identity. Rehashing it at every
        # derivation would split parents, children and grandchildren apart.
        root = lineage["scene_root_id"]
        if not isinstance(root, str) or not root.strip():
            raise ValueError("scene_root_id must be a non-empty string")
        return root
    else:
        route = context.get("route")
        route = route if isinstance(route, Mapping) else {}
        family = context.get("map_family") or context.get("map")
        if family and route.get("route_geometry_sha256"):
            payload = {
                "kind": "geometry",
                "map": canonical_map_family(str(family)),
                "opendrive": context.get("opendrive_sha256"),
                "route": route["route_geometry_sha256"],
            }
        elif family and all(
            route.get(key) is not None for key in ("start_spawn_index", "destination_spawn_index")
        ):
            payload = {
                "kind": "endpoints_not_route_ood",
                "map": canonical_map_family(str(family)),
                "start": route["start_spawn_index"],
                "destination": route["destination_spawn_index"],
            }
        else:
            payload = {"kind": "legacy_episode", "map": family, "episode": episode_id}
    return canonical_digest({"schema_version": LINEAGE_VERSION, **payload})


def route_geometry_digest(points: Sequence[Sequence[float]], *, precision: int = 3) -> str:
    """Direction-sensitive quantized polyline identity, not a proximity metric."""
    if isinstance(precision, bool) or not isinstance(precision, int) or not 0 <= precision <= 9:
        raise ValueError("precision must be an integer in [0, 9]")
    if len(points) < 2:
        raise ValueError("a route needs at least two points")
    normalized = []
    for point in points:
        if len(point) not in {2, 3} or not all(math.isfinite(float(x)) for x in point):
            raise ValueError("route coordinates must be finite 2D or 3D points")
        normalized.append([round(float(x), precision) for x in point])
    if len({len(p) for p in normalized}) != 1:
        raise ValueError("route dimensions must agree")
    return canonical_digest({"precision": precision, "points": normalized})


def validate_group_splits(rows: Sequence[Mapping[str, Any]]) -> str:
    """Validate explicit scene roots and return an immutable split identity."""
    assignments: dict[str, str] = {}
    identities: set[tuple[str, str]] = set()
    frozen = []
    for row in rows:
        group, partition = str(row["group_id"]), str(row["partition"])
        identity = (str(row["dataset_id"]), str(row["sample_id"]))
        partition_role(partition)
        if not group or identity in identities:
            raise ValueError("missing group or duplicate sample in split manifest")
        identities.add(identity)
        if assignments.setdefault(group, partition) != partition:
            raise ValueError("one scene/route group crosses source partitions")
        frozen.append(
            {
                "dataset_id": identity[0],
                "sample_id": identity[1],
                "group_id": group,
                "partition": partition,
            }
        )
    return canonical_digest(sorted(frozen, key=lambda r: (r["dataset_id"], r["sample_id"])))
