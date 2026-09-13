"""Native task adapter around the existing calibrated BehaviorAgent collector."""

from __future__ import annotations

import importlib
import math
import zipfile
from pathlib import Path
from typing import Any

from ..research_jobs import digest, identifier, write_json


def execute(request: dict[str, Any], output: Path) -> dict[str, Any]:
    # Imports are inside execution: dependency failures are reported before any
    # world mutation and do not take the bridge process down.
    clean = True
    result: dict[str, Any] = {}
    try:
        parameters = request["parameters"]
        allowed = {
            "scenario_suite",
            "split_plan",
            "camera_rig",
            "camera_rig_config",
            "behavior",
            "target_speed_kmh",
            "minimum_route_distance_m",
            "max_episodes",
            "dry_run",
        }
        if not isinstance(parameters, dict) or set(parameters) - allowed:
            raise ValueError("unsupported teacher-capture parameter")
        endpoint = request["endpoint"]
        suite, split = parameters["scenario_suite"], parameters["split_plan"]
        if not isinstance(suite, dict) or not isinstance(split, dict):
            raise ValueError("scenario_suite and split_plan must be JSON objects")
        if suite.get("carla_version") != endpoint["expected_carla_version"]:
            raise ValueError("scenario suite must match the host's CARLA version")
        if suite.get("traffic_manager_port") != endpoint["traffic_manager_port"]:
            raise ValueError("scenario suite must use the host's Traffic Manager port")
        count = parameters.get("max_episodes", 1)
        if type(count) is not int or not 1 <= count <= 32:
            raise ValueError("max_episodes must be in [1, 32]")
        for name, default, low, high in (
            ("target_speed_kmh", 25, 1, 80),
            ("minimum_route_distance_m", 40, 0, 1000),
        ):
            value = parameters.get(name, default)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(f"{name} must be finite and in [{low}, {high}]")
        if "dry_run" in parameters and type(parameters["dry_run"]) is not bool:
            raise ValueError("dry_run must be a boolean")
        from ...scenarios.planner import plan_scenarios
        from ..behavior_teacher import collect_behavior_teacher, parse_args

        inputs = output / "inputs"
        inputs.mkdir()
        print("Planning teacher capture", flush=True)
        write_json(
            inputs / "native_task.json",
            {
                "schema_version": "1.0",
                "job_id": request["job_id"],
                "task": request.get("task", {}),
                "endpoint": endpoint,
            },
        )
        write_json(inputs / "suite.json", suite)
        write_json(inputs / "splits.json", split)
        plan = plan_scenarios(
            suite_path=inputs / "suite.json",
            split_plan_path=inputs / "splits.json",
            runs_root=output / "runs",
            run_id="capture-plan",
        )
        argv = [
            "--scenario-plan",
            plan["run_dir"],
            "--dataset-id",
            identifier(request["job_id"]),
            "--datasets-root",
            str(output / "datasets"),
            "--host",
            endpoint["host"],
            "--port",
            str(endpoint["port"]),
            "--max-episodes",
            str(count),
            "--behavior",
            parameters.get("behavior", "cautious"),
            "--target-speed-kmh",
            str(parameters.get("target_speed_kmh", 25)),
            "--minimum-route-distance-m",
            str(parameters.get("minimum_route_distance_m", 40)),
            "--acknowledge-exclusive-tick-owner",
        ]
        if "camera_rig_config" in parameters:
            if "camera_rig" in parameters:
                raise ValueError("choose a rig preset or custom rig, not both")
            write_json(inputs / "rig.json", parameters["camera_rig_config"])
            argv += ["--camera-rig-config", str(inputs / "rig.json")]
        else:
            argv += ["--camera-rig", parameters.get("camera_rig", "front-three")]
        args = parse_args([*argv, "--dry-run"])
        # Validate the complete rig/episode selection before importing native code.
        preview = collect_behavior_teacher(args)
        print("Camera rig and scenario plan validated", flush=True)
        if parameters.get("dry_run", False):
            result = {"dry_run": True, "plan": preview}
        else:
            importlib.import_module("carla")
            from ..agent_support import load_navigation_module

            load_navigation_module(
                "behavior_agent", carla_version=endpoint["expected_carla_version"]
            )
            cancel_path = Path(request["cancel_file"])
            if cancel_path.exists():
                raise InterruptedError("native collection cancelled before capture")

            def cleanup_report(confirmed: bool) -> None:
                nonlocal clean
                clean = confirmed

            clean = False
            args.dry_run = False
            print(
                "Collecting teacher episodes; CARLA world is exclusively owned by this job",
                flush=True,
            )
            result = collect_behavior_teacher(
                args, cancel_check=cancel_path.exists, cleanup_report=cleanup_report
            )
        print(
            "Packaging verified outputs"
            if not parameters.get("dry_run", False)
            else "Packaging dry-run plan (no images captured)",
            flush=True,
        )
        archive = output / "artifacts.zip"
        with zipfile.ZipFile(
            archive, "x", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as zipped:
            for folder in ("inputs", "runs", "datasets"):
                for path in sorted((output / folder).rglob("*")):
                    if path.is_symlink():
                        raise ValueError("task artifacts must not contain symlinks")
                    if path.is_file():
                        zipped.write(path, path.relative_to(output).as_posix())
        return {
            "schema_version": "1.0",
            "status": "succeeded",
            "cleanup_confirmed": clean,
            "summary": result,
            "archive": {
                "path": "artifacts.zip",
                "sha256": digest(archive),
                "size_bytes": archive.stat().st_size,
            },
        }
    except BaseException as error:
        return {
            "schema_version": "1.0",
            "status": "cancelled" if isinstance(error, InterruptedError) else "failed",
            "cleanup_confirmed": clean,
            "error": f"{type(error).__name__}: {error}",
        }
