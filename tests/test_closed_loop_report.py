from __future__ import annotations

import json
from pathlib import Path

from carla_vision.closed_loop_report import aggregate_closed_loop_summaries


def _summary(
    path: Path,
    *,
    driver: str,
    seed: int,
    distance_m: float,
    collisions: int,
    lane_invasions: int,
    completion: float,
    reached: bool,
) -> Path:
    payload = {
        "run_label": f"{driver}-{seed}",
        "driver_label": driver,
        "seed": seed,
        "distance_traveled_m": distance_m,
        "elapsed_wall_seconds": 100.0,
        "route_completion": completion,
        "destination_reached": reached,
        "collision_count": collisions,
        "lane_invasion_count": lane_invasions,
        "red_light_violation_count": 1 if collisions else 0,
        "full_brake_intervention_count": 2,
        "read_only_vehicle_control": True,
        "control_calls": 0,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_report_aggregates_by_driver_and_distance(tmp_path: Path) -> None:
    paths = [
        _summary(
            tmp_path / "model-a.json",
            driver="imitation",
            seed=1,
            distance_m=500.0,
            collisions=1,
            lane_invasions=2,
            completion=0.8,
            reached=False,
        ),
        _summary(
            tmp_path / "model-b.json",
            driver="imitation",
            seed=2,
            distance_m=500.0,
            collisions=0,
            lane_invasions=0,
            completion=1.0,
            reached=True,
        ),
        _summary(
            tmp_path / "teacher.json",
            driver="behavior",
            seed=1,
            distance_m=1000.0,
            collisions=0,
            lane_invasions=1,
            completion=1.0,
            reached=True,
        ),
    ]
    report = aggregate_closed_loop_summaries(paths)
    imitation = report["drivers"]["imitation"]
    assert report["run_count"] == 3
    assert imitation["run_count"] == 2
    assert imitation["destination_reach_rate"] == 0.5
    assert imitation["mean_route_completion"] == 0.9
    assert imitation["collisions_per_km"] == 1.0
    assert imitation["lane_invasions_per_km"] == 2.0
    assert report["drivers"]["behavior"]["collisions_per_km"] == 0.0


def test_report_rejects_summary_that_wrote_vehicle_controls(tmp_path: Path) -> None:
    path = _summary(
        tmp_path / "bad.json",
        driver="model",
        seed=1,
        distance_m=100.0,
        collisions=0,
        lane_invasions=0,
        completion=0.2,
        reached=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["control_calls"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        aggregate_closed_loop_summaries([path])
    except ValueError as error:
        assert "control calls" in str(error)
    else:
        raise AssertionError("control-writing summary was accepted")
