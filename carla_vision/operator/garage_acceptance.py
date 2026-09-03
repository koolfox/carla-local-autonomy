"""Repeatable live-CARLA acceptance gate for the Garage/Drive lifecycle.

This runner deliberately reuses the production Garage drive manager and the
Windows World Worker.  It does not create a second CARLA client abstraction,
spawn actors directly, or tick the simulator.  A passing report is live
evidence; unit tests only validate orchestration and failure accounting.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .garage_server import GarageOperatorDriveManager
from .world_worker_client import WorldWorkerClient, WorldWorkerScene

SCHEMA_VERSION = "1.0"
EXPECTED_CARLA_VERSION = "0.9.16"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_error(error: BaseException) -> dict[str, str]:
    return {
        "type": type(error).__qualname__,
        "module": type(error).__module__,
        "message": str(error)[:2000],
    }


def _check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    *,
    observed: Any = None,
    expected: Any = True,
    note: str = "",
    required: bool = True,
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "required": bool(required),
            "status": "pass" if passed else "fail",
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
            "note": note,
        }
    )


def _nested(mapping: Mapping[str, Any], *path: str) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _catalog_identifier(item: Any) -> str:
    if isinstance(item, Mapping):
        value = item.get("id", item.get("name", ""))
        return str(value)
    return str(item)


def _wait_for(
    manager: Any,
    predicate: Callable[[Mapping[str, Any]], bool],
    *,
    timeout: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    deadline = clock() + timeout
    last: dict[str, Any] = {}
    while clock() < deadline:
        last = dict(manager.state())
        if predicate(last):
            return last
        if str(last.get("status")) == "failed":
            raise RuntimeError(str(last.get("error") or "Garage drive failed"))
        sleep(0.05)
    raise TimeoutError(f"Garage state condition did not arrive within {timeout:.1f}s: {last}")


def _drive_payload(
    *,
    run_id: str,
    host: str,
    port: int,
    vehicle: str,
    map_name: str,
    seed: int,
    traffic_count: int,
    walker_count: int,
    mode: str,
    camera_fps: float,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "host": host,
        "port": port,
        "vehicle_blueprint": vehicle,
        "color": "",
        "seed": seed,
        "weather_preset": "clear-day",
        "prop_preset": "none",
        "detector_enabled": False,
        "voxel_enabled": False,
        "detector": "rtdetr",
        "weights": "",
        "device": "cpu",
        "image_size": 640,
        "confidence": 0.35,
        "resolution": "1280x720",
        "camera_fps": camera_fps,
        "camera_fov": 90.0,
        "record_video": False,
        "spectator_follow": False,
        "experiment_preset": "free_drive",
        "map_name": map_name,
        "traffic_count": traffic_count,
        "walker_count": walker_count,
        "route_mode": "free",
        "initial_control_mode": "manual",
        "pedestrian_crossing_factor": 0.2,
        "speed_difference_percent": 12.0,
        "following_distance_metres": 2.0,
        "control_mode": mode,
        "behavior": "normal",
        "acknowledge_autonomy": mode != "manual",
        # Population ownership stays in the World Worker.  These legacy Garage
        # PythonAPI population fields remain zero on purpose.
        "traffic_vehicles": 0,
        "walkers": 0,
        "tm_port": 8000,
        "target_speed_kmh": 35.0,
        "policy_checkpoint": "",
        "policy_device": "cpu",
        "voxel_readiness_report": "",
    }


class GarageAcceptanceRunner:
    """Run live sessions through the same managers used by the product API."""

    def __init__(
        self,
        manager: Any,
        worker: Any,
        *,
        host: str,
        port: int,
        map_name: str,
        vehicle: str,
        seed: int,
        traffic_count: int,
        walker_count: int,
        camera_fps: float,
        repeat: int,
        expected_version: str = EXPECTED_CARLA_VERSION,
        state_timeout: float = 45.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.manager = manager
        self.worker = worker
        self.host = host
        self.port = port
        self.map_name = map_name
        self.vehicle = vehicle
        self.seed = seed
        self.traffic_count = traffic_count
        self.walker_count = walker_count
        self.camera_fps = camera_fps
        self.repeat = repeat
        self.expected_version = expected_version
        self.state_timeout = state_timeout
        self.clock = clock
        self.sleep = sleep

    def run(self) -> dict[str, Any]:
        started = self.clock()
        report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "object_type": "garage_live_carla_acceptance",
            "live_carla_required": True,
            "started_at": _utc_now(),
            "expected_carla_version": self.expected_version,
            "target": {
                "host": self.host,
                "port": self.port,
                "map": self.map_name,
                "vehicle": self.vehicle,
                "seed": self.seed,
                "traffic_count": self.traffic_count,
                "walker_count": self.walker_count,
                "camera": {"resolution": [1280, 720], "fps": self.camera_fps},
                "repeat": self.repeat,
            },
            "checks": [],
            "sessions": [],
        }
        checks: list[dict[str, Any]] = report["checks"]
        try:
            self._preflight(report, checks)
            if any(item["required"] and not item["passed"] for item in checks):
                raise RuntimeError("live Garage preflight failed")
            for repetition in range(1, self.repeat + 1):
                report["sessions"].append(self._run_session("manual", repetition))
                report["sessions"].append(self._run_session("behavior", repetition))
        except BaseException as error:
            report["runner_error"] = _safe_error(error)
        finally:
            try:
                self.manager.shutdown()
            except BaseException as error:
                report["shutdown_error"] = _safe_error(error)

        session_checks = [
            check
            for session in report["sessions"]
            for check in session.get("checks", [])
            if check.get("required")
        ]
        required_checks = [item for item in checks if item.get("required")]
        passed = (
            "runner_error" not in report
            and "shutdown_error" not in report
            and bool(report["sessions"])
            and all(item.get("passed") is True for item in [*required_checks, *session_checks])
            and len(report["sessions"]) == self.repeat * 2
        )
        report["status"] = "pass" if passed else "fail"
        report["duration_seconds"] = max(0.0, self.clock() - started)
        report["completed_at"] = _utc_now()
        return report

    def _preflight(self, report: dict[str, Any], checks: list[dict[str, Any]]) -> None:
        catalog = dict(self.manager.catalog())
        health = dict(self.worker.health())
        report["preflight"] = {"operator_catalog": catalog, "worker_health": health}

        _check(
            checks,
            "bridge_connected",
            catalog.get("connected") is True,
            observed=catalog.get("connected"),
            note="DriveSessionManager.catalog uses the existing read-only CARLA bridge probe.",
        )
        worker_connected = _nested(catalog, "world_worker", "connected") is True
        _check(
            checks,
            "world_worker_connected",
            worker_connected,
            observed=catalog.get("world_worker"),
            note="The configured Windows Worker must be reachable before scene mutation.",
        )
        server_versions = {
            "bridge": catalog.get("server_version"),
            "worker": _nested(health, "carla", "server_version"),
        }
        _check(
            checks,
            "carla_server_version",
            all(value == self.expected_version for value in server_versions.values()),
            observed=server_versions,
            expected=self.expected_version,
        )
        client_version = _nested(health, "carla", "client_version")
        _check(
            checks,
            "worker_pythonapi_version",
            client_version == self.expected_version,
            observed=client_version,
            expected=self.expected_version,
            note="This is the official PythonAPI loaded by the Windows World Worker.",
        )
        camera_fault_capability = _nested(health, "capabilities", "camera_pause_resume") is True
        _check(
            checks,
            "camera_pause_resume_capability",
            camera_fault_capability,
            observed=_nested(health, "capabilities", "camera_pause_resume"),
            expected=True,
            note=(
                "The gate pauses the existing CARLA sensor with Sensor.stop() and resumes it "
                "with Sensor.listen(); no timestamp or fake-frame mutation is accepted."
            ),
        )
        current = dict(self.worker.current_scene())
        clean_start = current.get("status") == "idle" and current.get("scene") is None
        report["preflight"]["worker_scene"] = {
            "status": current.get("status"),
            "scene_id": _nested(current, "scene", "scene_id"),
        }
        _check(
            checks,
            "worker_clean_start",
            clean_start,
            observed=report["preflight"]["worker_scene"],
            expected={"status": "idle", "scene_id": None},
            note="A previous run must not require manual actor cleanup.",
        )
        maps = catalog.get("maps", [])
        map_available = self.map_name == "current" or any(
            _catalog_identifier(item) == self.map_name for item in maps
        )
        _check(
            checks,
            "target_map_available",
            map_available,
            observed=self.map_name,
            expected="map advertised by the live Worker catalog",
        )
        vehicles = catalog.get("vehicles", [])
        vehicle_available = any(
            _catalog_identifier(item) == self.vehicle for item in vehicles
        )
        _check(
            checks,
            "vehicle_available",
            vehicle_available,
            observed=self.vehicle,
            expected="vehicle advertised by the live catalog",
        )

    def _run_session(self, mode: str, repetition: int) -> dict[str, Any]:
        started = self.clock()
        session: dict[str, Any] = {
            "mode": mode,
            "repetition": repetition,
            "checks": [],
        }
        checks: list[dict[str, Any]] = session["checks"]
        session_id: str | None = None
        try:
            run_id = (
                f"garage-accept-{mode}-r{repetition}-"
                f"{int(time.time())}-{self.seed}"
            )
            payload = _drive_payload(
                run_id=run_id,
                host=self.host,
                port=self.port,
                vehicle=self.vehicle,
                map_name=self.map_name,
                seed=self.seed + repetition - 1,
                traffic_count=self.traffic_count,
                walker_count=self.walker_count,
                mode=mode,
                camera_fps=self.camera_fps,
            )
            session["requested"] = payload
            initial = dict(self.manager.start(payload))
            session_id = str(initial.get("session_id") or "")
            if not session_id:
                raise RuntimeError("Garage start did not return a session_id")
            running = _wait_for(
                self.manager,
                lambda state: str(state.get("status")) == "running",
                timeout=self.state_timeout,
                clock=self.clock,
                sleep=self.sleep,
            )
            session["running"] = running
            _check(checks, "session_running", True, observed=running.get("status"))
            _check(
                checks,
                "ego_spawned",
                isinstance(running.get("vehicle_id"), int) and int(running["vehicle_id"]) > 0,
                observed=running.get("vehicle_id"),
                expected="positive actor id",
            )
            _check(
                checks,
                "camera_spawned",
                isinstance(running.get("camera_id"), int) and int(running["camera_id"]) > 0,
                observed=running.get("camera_id"),
                expected="positive sensor id",
            )
            sequence, jpeg = self.manager.wait_for_frame("raw", -1, timeout=10.0)
            _check(
                checks,
                "camera_frame_received",
                sequence >= 0 and isinstance(jpeg, (bytes, bytearray)) and len(jpeg) > 100,
                observed={"sequence": sequence, "jpeg_bytes": len(jpeg)},
                expected="non-empty live JPEG",
            )
            _check(
                checks,
                "traffic_population_exact",
                running.get("traffic_count_actual") == self.traffic_count,
                observed=running.get("traffic_count_actual"),
                expected=self.traffic_count,
            )
            _check(
                checks,
                "walker_population_exact",
                running.get("walker_count_actual") == self.walker_count,
                observed=running.get("walker_count_actual"),
                expected=self.walker_count,
            )

            if mode == "manual":
                self._exercise_manual(session_id, running, checks)
            else:
                self._exercise_behavior(checks)

            stopped = dict(self.manager.stop({"session_id": session_id}))
            session["stopped"] = stopped
            _check(
                checks,
                "session_stopped_cleanly",
                stopped.get("status") == "success" and not stopped.get("cleanup_errors"),
                observed={
                    "status": stopped.get("status"),
                    "cleanup_errors": stopped.get("cleanup_errors"),
                },
                expected={"status": "success", "cleanup_errors": []},
            )
            current = dict(self.worker.current_scene())
            session["worker_after_stop"] = current
            _check(
                checks,
                "worker_scene_released",
                current.get("status") == "idle" and current.get("scene") is None,
                observed={"status": current.get("status"), "scene": current.get("scene")},
                expected={"status": "idle", "scene": None},
            )
        except BaseException as error:
            session["error"] = _safe_error(error)
            _check(checks, "session_exception_free", False, observed=session["error"])
            if session_id:
                try:
                    session["forced_stop"] = dict(self.manager.stop({"session_id": session_id}))
                except BaseException as stop_error:
                    session["forced_stop_error"] = _safe_error(stop_error)
        session["duration_seconds"] = max(0.0, self.clock() - started)
        session["status"] = (
            "pass"
            if "error" not in session
            and all(item.get("passed") for item in checks if item.get("required"))
            else "fail"
        )
        return session

    def _exercise_manual(
        self,
        session_id: str,
        running: Mapping[str, Any],
        checks: list[dict[str, Any]],
    ) -> None:
        baseline = abs(float(_nested(running, "telemetry", "speed") or 0.0))
        sequence = 0
        for _ in range(12):
            sequence += 1
            self.manager.control(
                {
                    "session_id": session_id,
                    "sequence": sequence,
                    "throttle": 0.35,
                    "steer": 0.0,
                    "brake": 0.0,
                    "hand_brake": False,
                    "reverse": False,
                }
            )
            self.sleep(0.10)
        moving = _wait_for(
            self.manager,
            lambda state: (
                abs(float(_nested(state, "telemetry", "speed") or 0.0)) > baseline + 0.15
                and int(state.get("controls_written") or 0) > 0
            ),
            timeout=5.0,
            clock=self.clock,
            sleep=self.sleep,
        )
        _check(
            checks,
            "manual_control_effect",
            True,
            observed={
                "baseline_speed_mps": baseline,
                "speed_mps": _nested(moving, "telemetry", "speed"),
                "controls_written": moving.get("controls_written"),
            },
            expected="vehicle speed increases after repeated browser throttle commands",
        )

        stale = _wait_for(
            self.manager,
            lambda state: state.get("control_source") == "browser_deadman"
            and state.get("deadman_active") is True,
            timeout=3.0,
            clock=self.clock,
            sleep=self.sleep,
        )
        _check(
            checks,
            "control_stale_fail_safe",
            True,
            observed={
                "control_source": stale.get("control_source"),
                "deadman_active": stale.get("deadman_active"),
            },
            expected="browser_deadman applies service brake after input lease expires",
        )

        before_pause_sequence, _ = self.manager.frame("raw")
        active_scene = WorldWorkerScene.from_response(self.worker.current_scene())
        self.worker.pause_camera(active_scene)
        paused = True
        try:
            camera_deadman = _wait_for(
                self.manager,
                lambda state: state.get("control_source") == "camera_deadman"
                and state.get("deadman_active") is True,
                timeout=4.0,
                clock=self.clock,
                sleep=self.sleep,
            )
            _check(
                checks,
                "camera_stale_fail_safe",
                _nested(camera_deadman, "stream", "stale") is True,
                observed={
                    "stream_stale": _nested(camera_deadman, "stream", "stale"),
                    "control_source": camera_deadman.get("control_source"),
                    "deadman_active": camera_deadman.get("deadman_active"),
                },
                expected="real sensor pause produces camera_deadman before the session fails",
            )
        finally:
            if paused:
                self.worker.resume_camera(active_scene)

        recovered_sequence, recovered_jpeg = self.manager.wait_for_frame(
            "raw", before_pause_sequence, timeout=10.0
        )
        recovered = _wait_for(
            self.manager,
            lambda state: _nested(state, "stream", "stale") is False,
            timeout=3.0,
            clock=self.clock,
            sleep=self.sleep,
        )
        _check(
            checks,
            "camera_recovered",
            recovered_sequence > before_pause_sequence
            and len(recovered_jpeg) > 100
            and _nested(recovered, "stream", "stale") is False,
            observed={
                "before_sequence": before_pause_sequence,
                "after_sequence": recovered_sequence,
                "jpeg_bytes": len(recovered_jpeg),
                "stream_stale": _nested(recovered, "stream", "stale"),
            },
            expected="same CARLA sensor resumes and publishes a newer frame",
        )

        self.manager.emergency_stop({"session_id": session_id})
        emergency = _wait_for(
            self.manager,
            lambda state: state.get("control_source") == "emergency_stop",
            timeout=2.0,
            clock=self.clock,
            sleep=self.sleep,
        )
        _check(
            checks,
            "emergency_brake",
            emergency.get("emergency_stop") is True
            and emergency.get("deadman_active") is True,
            observed={
                "emergency_stop": emergency.get("emergency_stop"),
                "deadman_active": emergency.get("deadman_active"),
                "control_source": emergency.get("control_source"),
            },
            expected="latched emergency_stop with service braking",
        )

    def _exercise_behavior(self, checks: list[dict[str, Any]]) -> None:
        state = _wait_for(
            self.manager,
            lambda value: bool(_nested(value, "autonomy", "policy_ready"))
            and int(_nested(value, "autonomy", "commands") or 0) > 0,
            timeout=8.0,
            clock=self.clock,
            sleep=self.sleep,
        )
        _check(
            checks,
            "behavior_agent_actuated",
            True,
            observed={
                "control_source": state.get("control_source"),
                "commands": _nested(state, "autonomy", "commands"),
            },
            expected="official BehaviorAgent produces applied commands",
        )
        intent = _nested(state, "autonomy", "detail", "navigation_intent")
        valid_intent = (
            isinstance(intent, Mapping)
            and _nested(intent, "source_frame", "exact") is True
            and isinstance(_nested(intent, "source_frame", "id"), int)
            and intent.get("privileged") is True
        )
        _check(
            checks,
            "behavior_navigation_intent",
            valid_intent,
            observed=intent,
            expected="exact-frame privileged NavigationIntent from CARLA LocalPlanner",
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeatable live CARLA Garage acceptance and retain JSON evidence."
    )
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--carla-host", default=os.environ.get("CARLA_HOST", "127.0.0.1"))
    parser.add_argument("--carla-port", type=int, default=int(os.environ.get("CARLA_PORT", "2000")))
    parser.add_argument(
        "--world-worker-url",
        default=os.environ.get("CARLA_WORLD_WORKER_URL", ""),
    )
    parser.add_argument(
        "--world-worker-token-env",
        default="CARLA_WORLD_WORKER_TOKEN",
        help="environment variable containing the Worker bearer token; the token is never a CLI value",
    )
    parser.add_argument("--map", dest="map_name", default="Town10HD_Opt")
    parser.add_argument("--vehicle", default="vehicle.tesla.model3")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--traffic", type=int, default=5)
    parser.add_argument("--walkers", type=int, default=5)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--expected-version", default=EXPECTED_CARLA_VERSION)
    parser.add_argument("--state-timeout", type=float, default=45.0)
    parser.add_argument("--output", default="")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not args.world_worker_url:
        raise ValueError("--world-worker-url or CARLA_WORLD_WORKER_URL is required")
    if not 1 <= args.carla_port <= 65535:
        raise ValueError("--carla-port must be in [1, 65535]")
    if not 0 <= args.traffic <= 250 or not 0 <= args.walkers <= 250:
        raise ValueError("--traffic and --walkers must be in [0, 250]")
    if not 1.0 <= args.camera_fps <= 60.0 or not math.isfinite(args.camera_fps):
        raise ValueError("--camera-fps must be finite and in [1, 60]")
    if not 1 <= args.repeat <= 10:
        raise ValueError("--repeat must be in [1, 10]")
    if not 5.0 <= args.state_timeout <= 300.0 or not math.isfinite(args.state_timeout):
        raise ValueError("--state-timeout must be finite and in [5, 300]")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _validate_args(args)
        workspace = Path(args.workspace).expanduser().resolve(strict=True)
        token = os.environ.get(args.world_worker_token_env, "")
        if not token:
            raise ValueError(
                f"environment variable {args.world_worker_token_env!r} must contain the Worker token"
            )
        worker = WorldWorkerClient(args.world_worker_url, token, timeout=5.0)
        manager = GarageOperatorDriveManager(
            workspace=workspace,
            carla_host=args.carla_host,
            carla_port=args.carla_port,
            world_worker=worker,
            experimental_enabled=True,
        )
        runner = GarageAcceptanceRunner(
            manager,
            worker,
            host=args.carla_host,
            port=args.carla_port,
            map_name=args.map_name,
            vehicle=args.vehicle,
            seed=args.seed,
            traffic_count=args.traffic,
            walker_count=args.walkers,
            camera_fps=args.camera_fps,
            repeat=args.repeat,
            expected_version=args.expected_version,
            state_timeout=args.state_timeout,
        )
        report = runner.run()
    except BaseException as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "object_type": "garage_live_carla_acceptance",
            "live_carla_required": True,
            "status": "fail",
            "started_at": _utc_now(),
            "completed_at": _utc_now(),
            "runner_error": _safe_error(error),
        }

    if args.output:
        output = Path(args.output).expanduser().resolve()
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = Path(args.workspace).expanduser().resolve() / "runs" / f"garage-acceptance-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report.get("status"), "output": str(output)}, sort_keys=True))
    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
