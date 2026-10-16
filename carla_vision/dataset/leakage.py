"""Deterministic 3-D route-corridor connected components for split auditing.

A geometric claim needs version-identified map bytes and actual polylines. Names,
seeds and endpoint hashes are not substitutes. Complexity is bounded explicitly;
exhausting the comparison budget raises instead of declaring routes disjoint.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from ..reproduction.review_contracts import digest, fields, integer, number, text
from .lineage import canonical_digest, partition_role


def segment_distance(a, b, c, d) -> float:
    """Exact minimum of the squared distance quadratic on [0,1]^2."""
    a, b, c, d = (np.asarray(x, dtype=np.float64) for x in (a, b, c, d))
    u, v, w = b - a, d - c, a - c
    uu, vv, uv, uw, vw = (float(u @ u), float(v @ v), float(u @ v), float(u @ w), float(v @ w))
    candidates = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    if uu > 0:
        candidates.extend((float(np.clip((t * uv - uw) / uu, 0, 1)), t) for t in (0.0, 1.0))
    if vv > 0:
        candidates.extend((s, float(np.clip((s * uv + vw) / vv, 0, 1))) for s in (0.0, 1.0))
    cross = np.cross(u, v)
    determinant = float(cross @ cross)
    if determinant > 0:
        s = float(np.cross(v, w) @ cross) / determinant
        t = float(np.cross(u, w) @ cross) / determinant
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append((s, t))
    return min(float(np.linalg.norm(w + s * u - t * v)) for s, t in candidates)


def audit_corridors(
    routes: list[dict[str, Any]], *, clearance_m: float, maximum_segment_checks: int = 1_000_000
) -> dict[str, Any]:
    clearance = number(clearance_m, "clearance_m", minimum=0)
    budget = integer(maximum_segment_checks, "comparison budget", minimum=1)
    if not isinstance(routes, list) or not routes:
        raise ValueError("a corridor audit requires actual route records")
    normalized = []
    for r in routes:
        fields(r, {"route_id", "scene_root_id", "partition", "map_sha256", "points_xyz"})
        for name in ("route_id", "scene_root_id", "partition"):
            text(r[name], name)
        partition_role(r["partition"])
        if r["partition"] == "unassigned":
            raise ValueError("unassigned routes cannot establish held-out geometry")
        digest(r["map_sha256"])
        points = np.asarray(r["points_xyz"], dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or len(points) < 2
            or not np.isfinite(points).all()
            or np.abs(points).max() > 1e9
        ):
            raise ValueError("route geometry must be finite metric XYZ points")
        if not np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) > 0):
            raise ValueError("route must contain a nonzero segment")
        normalized.append(dict(r, points_xyz=points.tolist()))
    normalized.sort(key=lambda r: r["route_id"])
    if len({r["route_id"] for r in normalized}) != len(normalized):
        raise ValueError("duplicate route identity")
    parents = list(range(len(normalized)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        parents[max(a, b)] = min(a, b)

    arrays = [np.asarray(r["points_xyz"]) for r in normalized]
    boxes = [(p.min(axis=0), p.max(axis=0)) for p in arrays]
    checks, edges = 0, []
    for i, j in combinations(range(len(normalized)), 2):
        left, right = normalized[i], normalized[j]
        reason, separation = None, None
        if left["scene_root_id"] == right["scene_root_id"]:
            reason = "shared_scene_root"
        elif left["map_sha256"] == right["map_sha256"]:
            low_a, high_a = boxes[i]
            low_b, high_b = boxes[j]
            gap = np.maximum(0, np.maximum(low_a - high_b, low_b - high_a))
            if float(np.linalg.norm(gap)) <= clearance:
                minimum = float("inf")
                for a, b in zip(arrays[i][:-1], arrays[i][1:], strict=True):
                    for c, d in zip(arrays[j][:-1], arrays[j][1:], strict=True):
                        checks += 1
                        if checks > budget:
                            raise ValueError(
                                "corridor comparison budget exhausted; audit incomplete"
                            )
                        minimum = min(minimum, segment_distance(a, b, c, d))
                if minimum <= clearance:
                    reason, separation = "corridor_proximity", minimum
        if reason:
            union(i, j)
            edges.append(
                {
                    "left": left["route_id"],
                    "right": right["route_id"],
                    "reason": reason,
                    "separation_m": separation,
                }
            )
    groups: dict[int, list[dict[str, Any]]] = {}
    for i, r in enumerate(normalized):
        groups.setdefault(find(i), []).append(r)
    components = []
    for members in groups.values():
        roots = sorted({r["scene_root_id"] for r in members})
        partitions = sorted({r["partition"] for r in members})
        # Strict physical route-OOD: even distinct validation/test names may not mix.
        components.append(
            {
                "component_id": canonical_digest(roots),
                "scene_roots": roots,
                "route_ids": [r["route_id"] for r in members],
                "partitions": partitions,
                "crosses_partitions": len(partitions) > 1,
            }
        )
    return {
        "schema_version": "1.0",
        "kind": "route_corridor_audit",
        "status": "failed" if any(c["crosses_partitions"] for c in components) else "passed",
        "geometry_scope": "supplied_3d_polylines_on_versioned_maps",
        "clearance_m": clearance,
        "source_sha256": canonical_digest(normalized),
        "segment_checks": checks,
        "components": components,
        "edges": edges,
        "live_capture_verified": False,
    }
