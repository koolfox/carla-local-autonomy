"""Aggregate read-only CARLA closed-loop evaluation summaries."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CLOSED_LOOP_REPORT_SCHEMA_VERSION = "1.0"


def _read_summary(path: Path) -> dict[str, Any]:
    target = path / "summary.json" if path.is_dir() else path
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evaluation summary must be an object: {target}")
    if payload.get("read_only_vehicle_control") is not True:
        raise ValueError(f"summary is not declared read-only: {target}")
    if payload.get("control_calls") != 0:
        raise ValueError(f"summary reports vehicle control calls: {target}")
    payload["_source"] = str(target.resolve())
    return payload


def _finite_number(payload: Mapping[str, Any], name: str) -> float:
    value = payload.get(name)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"summary field {name} must be finite")
    return float(value)


def aggregate_closed_loop_summaries(paths: Sequence[str | Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one evaluation summary is required")
    summaries = [_read_summary(Path(path).expanduser().resolve(strict=True)) for path in paths]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        label = summary.get("driver_label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"driver_label is missing: {summary['_source']}")
        grouped[label].append(summary)

    drivers: dict[str, Any] = {}
    for label, runs in sorted(grouped.items()):
        total_distance_m = sum(_finite_number(run, "distance_traveled_m") for run in runs)
        total_elapsed_s = sum(_finite_number(run, "elapsed_wall_seconds") for run in runs)
        total_collisions = sum(int(run.get("collision_count", 0)) for run in runs)
        total_lane_invasions = sum(int(run.get("lane_invasion_count", 0)) for run in runs)
        total_red_lights = sum(int(run.get("red_light_violation_count", 0)) for run in runs)
        total_full_brakes = sum(
            int(run.get("full_brake_intervention_count", 0)) for run in runs
        )
        completions = [_finite_number(run, "route_completion") for run in runs]
        reached = sum(bool(run.get("destination_reached")) for run in runs)
        distance_km = total_distance_m / 1000.0
        drivers[label] = {
            "run_count": len(runs),
            "seeds": sorted({int(run.get("seed", 0)) for run in runs}),
            "run_labels": [str(run.get("run_label", "")) for run in runs],
            "sources": [str(run["_source"]) for run in runs],
            "destination_reach_rate": reached / len(runs),
            "mean_route_completion": sum(completions) / len(completions),
            "minimum_route_completion": min(completions),
            "total_distance_m": total_distance_m,
            "total_elapsed_seconds": total_elapsed_s,
            "aggregate_mean_speed_mps": total_distance_m / max(total_elapsed_s, 1e-9),
            "collision_count": total_collisions,
            "collisions_per_km": total_collisions / max(distance_km, 1e-6),
            "lane_invasion_count": total_lane_invasions,
            "lane_invasions_per_km": total_lane_invasions / max(distance_km, 1e-6),
            "red_light_violation_count": total_red_lights,
            "red_light_violations_per_run": total_red_lights / len(runs),
            "full_brake_intervention_count": total_full_brakes,
            "full_brake_interventions_per_run": total_full_brakes / len(runs),
        }

    return {
        "schema_version": CLOSED_LOOP_REPORT_SCHEMA_VERSION,
        "status": "complete",
        "run_count": len(summaries),
        "driver_count": len(drivers),
        "drivers": drivers,
        "metric_notes": {
            "collisions_per_km": "aggregate collision events divided by aggregate distance",
            "lane_invasions_per_km": "aggregate lane events divided by aggregate distance",
            "red_light": "inherits the approximate trigger-volume heuristic from each run",
            "full_brake": "observed high-brake transitions; not guaranteed to be safety interventions",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate multiple read-only CARLA closed-loop evaluation summaries."
    )
    parser.add_argument("summary", nargs="+")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = aggregate_closed_loop_summaries(args.summary)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


__all__ = [
    "CLOSED_LOOP_REPORT_SCHEMA_VERSION",
    "aggregate_closed_loop_summaries",
    "build_parser",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
