"""Dense Garage preparation acceptance matrix using the production CARLA path.

This command does not create actors or a second CARLA client.  It drives the
existing :class:`GaragePreviewManager` and authenticated World Worker, measures
control-plane responsiveness while the real prepare call is in flight, samples
the existing preview camera telemetry, and verifies cleanup between tiers.

The report deliberately distinguishes a successful tier from a bounded map
capacity rejection.  A 422 ``scene_population_capacity`` response with an
explicit maximum is useful evidence for #56; transport failures and population
shortfalls are not.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .garage_preview import GaragePreviewManager
from .world_worker_client import WorldWorkerClient

SCHEMA_VERSION = "1.0"
EXPECTED_CARLA_VERSION = "0.9.16"
_CAPACITY = re.compile(r"maximum of\s+(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class DensityTier:
    traffic: int
    walkers: int

    @classmethod
    def parse(cls, raw: str) -> "DensityTier":
        parts = raw.strip().split(":")
        if len(parts) != 2:
            raise ValueError("density tiers must use TRAFFIC:WALKERS")
        try:
            traffic, walkers = (int(value) for value in parts)
        except ValueError as error:
            raise ValueError("density tier counts must be integers") from error
        if not 0 <= traffic <= 250 or not 0 <= walkers <= 250:
            raise ValueError("density tier counts must be in [0, 250]")
        return cls(traffic=traffic, walkers=walkers)

    def as_dict(self) -> dict[str, int]:
        return {"traffic": self.traffic, "walkers": self.walkers}


def parse_tiers(raw: str) -> tuple[DensityTier, ...]:
    values = tuple(DensityTier.parse(item) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("at least one density tier is required")
    if len(values) > 20:
        raise ValueError("at most 20 density tiers are allowed")
    return values


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_error(error: BaseException) -> dict[str, Any]:
    status = getattr(error, "status", None)
    code = getattr(error, "code", None)
    return {
        "type": type(error).__qualname__,
        "module": type(error).__module__,
        "status": None if status is None else int(status),
        "code": None if code is None else str(code),
        "message": str(error)[:2000],
    }


def _timing(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"samples": 0, "minimum_ms": None, "median_ms": None, "maximum_ms": None}
    milliseconds = [value * 1000.0 for value in values]
    return {
        "samples": len(milliseconds),
        "minimum_ms": round(min(milliseconds), 3),
        "median_ms": round(statistics.median(milliseconds), 3),
        "maximum_ms": round(max(milliseconds), 3),
    }


def _semantic_progress(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": raw.get("status"),
        "stage": raw.get("stage"),
        "requested": dict(raw.get("requested", {}))
        if isinstance(raw.get("requested"), Mapping)
        else {},
        "actual": dict(raw.get("actual", {})) if isinstance(raw.get("actual"), Mapping) else {},
        "elapsed_seconds": raw.get("elapsed_seconds"),
        "error": raw.get("error"),
    }


class GarageDensityAcceptanceRunner:
    """Exercise independent clean Garage preparations over bounded density tiers."""

    def __init__(
        self,
        manager: Any,
        worker: Any,
        *,
        tiers: Sequence[DensityTier],
        map_name: str,
        vehicle: str,
        seed: int,
        pedestrian_crossing_factor: float,
        speed_difference_percent: float,
        following_distance_metres: float,
        camera_profile: str,
        expected_version: str = EXPECTED_CARLA_VERSION,
        probe_interval: float = 0.25,
        max_control_plane_latency: float = 1.0,
        prepare_timeout: float = 180.0,
        cancel_grace: float = 15.0,
        camera_sample_seconds: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.manager = manager
        self.worker = worker
        self.tiers = tuple(tiers)
        self.map_name = str(map_name)
        self.vehicle = str(vehicle)
        self.seed = int(seed)
        self.pedestrian_crossing_factor = float(pedestrian_crossing_factor)
        self.speed_difference_percent = float(speed_difference_percent)
        self.following_distance_metres = float(following_distance_metres)
        self.camera_profile = str(camera_profile)
        self.expected_version = str(expected_version)
        self.probe_interval = float(probe_interval)
        self.max_control_plane_latency = float(max_control_plane_latency)
        self.prepare_timeout = float(prepare_timeout)
        self.cancel_grace = float(cancel_grace)
        self.camera_sample_seconds = float(camera_sample_seconds)
        self.clock = clock
        self.sleep = sleep

    def run(self) -> dict[str, Any]:
        started = self.clock()
        report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "object_type": "garage_dense_scene_acceptance",
            "live_carla_required_for_empirical_claim": True,
            "started_at": _utc_now(),
            "expected_carla_version": self.expected_version,
            "target": {
                "map": self.map_name,
                "vehicle": self.vehicle,
                "seed": self.seed,
                "tiers": [tier.as_dict() for tier in self.tiers],
                "pedestrian_crossing_factor": self.pedestrian_crossing_factor,
                "speed_difference_percent": self.speed_difference_percent,
                "following_distance_metres": self.following_distance_metres,
                "camera_profile": self.camera_profile,
                "max_control_plane_latency_seconds": self.max_control_plane_latency,
            },
            "preflight": {},
            "tiers": [],
        }
        fatal = False
        try:
            fatal = not self._preflight(report)
            if not fatal:
                for index, tier in enumerate(self.tiers):
                    result = self._run_tier(tier, index)
                    report["tiers"].append(result)
                    if result.get("fatal"):
                        fatal = True
                        break
        except BaseException as error:
            report["runner_error"] = _safe_error(error)
            fatal = True
        finally:
            shutdown_error = self._bounded_shutdown()
            if shutdown_error is not None:
                report["shutdown_error"] = shutdown_error
                fatal = True

        tier_results = report["tiers"]
        passed = (
            not fatal
            and "runner_error" not in report
            and "shutdown_error" not in report
            and len(tier_results) == len(self.tiers)
            and all(bool(item.get("passed")) for item in tier_results)
        )
        report["status"] = "pass" if passed else "fail"
        report["duration_seconds"] = round(max(0.0, self.clock() - started), 3)
        report["completed_at"] = _utc_now()
        return report

    def _preflight(self, report: dict[str, Any]) -> bool:
        health_started = self.clock()
        health = dict(self.worker.health())
        health_latency = self.clock() - health_started
        scene_started = self.clock()
        scene = dict(self.worker.current_scene())
        scene_latency = self.clock() - scene_started
        capabilities = health.get("capabilities", {})
        if not isinstance(capabilities, Mapping):
            capabilities = {}
        carla = health.get("carla", {})
        if not isinstance(carla, Mapping):
            carla = {}
        clean = scene.get("status") == "idle" and scene.get("scene") is None
        checks = {
            "worker_ready": health.get("ready") is True,
            "client_version": carla.get("client_version") == self.expected_version,
            "server_version": carla.get("server_version") == self.expected_version,
            "observable_scene_preparation": capabilities.get("observable_scene_preparation") is True,
            "responsive_prepare_health": capabilities.get("responsive_prepare_health") is True,
            "prepare_cancellation": capabilities.get("prepare_cancellation") is True,
            "clean_worker_scene": clean,
            "health_latency_bounded": health_latency <= self.max_control_plane_latency,
            "scene_latency_bounded": scene_latency <= self.max_control_plane_latency,
        }
        report["preflight"] = {
            "checks": checks,
            "health_latency_ms": round(health_latency * 1000.0, 3),
            "current_scene_latency_ms": round(scene_latency * 1000.0, 3),
            "worker_status": health.get("status"),
            "carla": dict(carla),
            "capabilities": {
                "observable_scene_preparation": capabilities.get("observable_scene_preparation"),
                "responsive_prepare_health": capabilities.get("responsive_prepare_health"),
                "prepare_cancellation": capabilities.get("prepare_cancellation"),
            },
            "worker_scene": {
                "status": scene.get("status"),
                "scene_id": (
                    scene.get("scene", {}).get("scene_id")
                    if isinstance(scene.get("scene"), Mapping)
                    else None
                ),
            },
        }
        return all(checks.values())

    def _payload(self, tier: DensityTier, index: int) -> dict[str, Any]:
        return {
            "map_name": self.map_name,
            "weather_preset": "clear-day",
            "vehicle_blueprint": self.vehicle,
            "color": "",
            "seed": self.seed + index,
            "traffic_count": tier.traffic,
            "walker_count": tier.walkers,
            "prop_preset": "none",
            "route_mode": "free",
            "pedestrian_crossing_factor": self.pedestrian_crossing_factor,
            "speed_difference_percent": self.speed_difference_percent,
            "following_distance_metres": self.following_distance_metres,
            "spectator_mirror": False,
            "profile": self.camera_profile,
            "fov": 65.0,
        }

    def _run_tier(self, tier: DensityTier, index: int) -> dict[str, Any]:
        started = self.clock()
        result: dict[str, Any] = {
            "tier": tier.as_dict(),
            "requested": self._payload(tier, index),
            "progress": [],
            "control_plane": {},
            "passed": False,
        }
        before = dict(self.worker.current_scene())
        if before.get("status") != "idle" or before.get("scene") is not None:
            result.update(
                status="dirty_start",
                fatal=True,
                error={"code": "worker_not_idle", "message": "previous tier left a Worker scene"},
            )
            return result

        response_box: dict[str, Any] = {}
        error_box: list[BaseException] = []

        def configure() -> None:
            try:
                response_box.update(self.manager.configure(result["requested"]))
            except BaseException as error:
                error_box.append(error)

        thread = threading.Thread(
            target=configure,
            name=f"garage-density-{tier.traffic}-{tier.walkers}",
            daemon=True,
        )
        thread.start()

        health_latencies: list[float] = []
        scene_latencies: list[float] = []
        probe_errors: list[dict[str, Any]] = []
        progress: list[dict[str, Any]] = []
        last_progress: dict[str, Any] | None = None
        during_prepare_probes = 0
        deadline = started + self.prepare_timeout
        timed_out = False
        cancel_result: Any = None

        while thread.is_alive() or not health_latencies:
            try:
                health_started = self.clock()
                health = dict(self.worker.health())
                health_latencies.append(self.clock() - health_started)
                if health.get("status") == "preparing":
                    during_prepare_probes += 1
            except BaseException as error:
                probe_errors.append({"endpoint": "health", "error": _safe_error(error)})
            try:
                scene_started = self.clock()
                scene = dict(self.worker.current_scene())
                scene_latencies.append(self.clock() - scene_started)
                preparation = scene.get("preparation")
                if isinstance(preparation, Mapping):
                    semantic = _semantic_progress(preparation)
                    comparable = dict(semantic)
                    comparable.pop("elapsed_seconds", None)
                    if last_progress != comparable:
                        progress.append(semantic)
                        last_progress = comparable
            except BaseException as error:
                probe_errors.append({"endpoint": "current_scene", "error": _safe_error(error)})

            if not thread.is_alive():
                break
            if self.clock() >= deadline:
                timed_out = True
                try:
                    cancel_result = self.manager.cancel_preparation()
                except BaseException as error:
                    cancel_result = {"error": _safe_error(error)}
                thread.join(timeout=self.cancel_grace)
                break
            self.sleep(self.probe_interval)

        prepare_seconds = max(0.0, self.clock() - started)
        result["prepare_seconds"] = round(prepare_seconds, 3)
        result["progress"] = progress
        result["control_plane"] = {
            "health": _timing(health_latencies),
            "current_scene": _timing(scene_latencies),
            "during_prepare_probes": during_prepare_probes,
            "probe_errors": probe_errors,
        }
        maximum_health = max(health_latencies, default=math.inf)
        maximum_scene = max(scene_latencies, default=math.inf)
        responsive = (
            not probe_errors
            and maximum_health <= self.max_control_plane_latency
            and maximum_scene <= self.max_control_plane_latency
            and (during_prepare_probes > 0 or prepare_seconds <= self.max_control_plane_latency)
        )
        result["control_plane"]["responsive"] = responsive

        if timed_out:
            result["status"] = "prepare_timeout"
            result["cancel"] = cancel_result
            result["fatal"] = thread.is_alive()
            if error_box:
                result["error"] = _safe_error(error_box[0])
            elif thread.is_alive():
                result["error"] = {
                    "code": "prepare_cancel_timeout",
                    "message": "prepare did not finish within the cancellation grace interval",
                }
            else:
                result["error"] = {
                    "code": "prepare_timeout",
                    "message": "prepare exceeded the configured acceptance timeout and was cancelled",
                }
            result["cleanup"] = self._cleanup_after_failed_prepare()
            return result

        if error_box:
            error = _safe_error(error_box[0])
            result["error"] = error
            cleanup = self._cleanup_after_failed_prepare()
            result["cleanup"] = cleanup
            maximum = None
            match = _CAPACITY.search(error["message"])
            if match is not None:
                maximum = int(match.group(1))
            capacity_boundary = (
                error.get("status") == 422
                and error.get("code") == "scene_population_capacity"
                and maximum is not None
                and tier.traffic > maximum
                and cleanup["worker_idle"]
                and responsive
            )
            result["capacity_maximum_traffic"] = maximum
            result["status"] = "capacity_boundary" if capacity_boundary else "prepare_failed"
            result["passed"] = capacity_boundary
            return result

        snapshot = dict(response_box)
        result["prepared"] = {
            "status": snapshot.get("status"),
            "map": snapshot.get("map"),
            "traffic_requested": snapshot.get("traffic_count_requested"),
            "traffic_actual": snapshot.get("traffic_count"),
            "walkers_requested": snapshot.get("walker_count_requested"),
            "walkers_actual": snapshot.get("walker_count"),
            "pedestrian_crossing_factor": snapshot.get("pedestrian_crossing_factor"),
            "speed_difference_percent": snapshot.get("speed_difference_percent"),
            "following_distance_metres": snapshot.get("following_distance_metres"),
            "camera_source_fps": snapshot.get("camera_source_fps"),
            "camera_frame_age_seconds": snapshot.get("camera_frame_age_seconds"),
            "camera_stale": snapshot.get("camera_stale"),
        }

        if self.camera_sample_seconds > 0.0:
            self.sleep(self.camera_sample_seconds)
        camera = dict(self.manager.state())
        result["camera_sample"] = {
            "source_fps": camera.get("camera_source_fps"),
            "frame_age_seconds": camera.get("camera_frame_age_seconds"),
            "stale": camera.get("camera_stale"),
            "frame_sequence": camera.get("frame_sequence"),
            "transport": camera.get("camera_transport"),
        }
        cleanup = self._stop_and_verify()
        result["cleanup"] = cleanup

        crossing = snapshot.get("pedestrian_crossing_factor")
        exact_population = (
            snapshot.get("traffic_count") == tier.traffic
            and snapshot.get("walker_count") == tier.walkers
        )
        dynamics_match = (
            isinstance(crossing, (int, float))
            and math.isclose(float(crossing), self.pedestrian_crossing_factor, abs_tol=1e-9)
            and math.isclose(
                float(snapshot.get("speed_difference_percent")),
                self.speed_difference_percent,
                abs_tol=1e-9,
            )
            and math.isclose(
                float(snapshot.get("following_distance_metres")),
                self.following_distance_metres,
                abs_tol=1e-9,
            )
        )
        source_fps = camera.get("camera_source_fps")
        camera_ok = (
            isinstance(source_fps, (int, float))
            and float(source_fps) > 0.0
            and camera.get("camera_stale") is False
            and isinstance(camera.get("frame_sequence"), int)
            and int(camera["frame_sequence"]) >= 0
        )
        passed = (
            snapshot.get("status") == "running"
            and exact_population
            and dynamics_match
            and camera_ok
            and responsive
            and cleanup["worker_idle"]
            and cleanup["manager_error"] is None
        )
        result["checks"] = {
            "preview_running": snapshot.get("status") == "running",
            "exact_population": exact_population,
            "dynamics_match": dynamics_match,
            "camera_stream_healthy": camera_ok,
            "control_plane_responsive": responsive,
            "cleanup_confirmed": cleanup["worker_idle"] and cleanup["manager_error"] is None,
        }
        result["status"] = "pass" if passed else "failed_checks"
        result["passed"] = passed
        return result

    def _cleanup_after_failed_prepare(self) -> dict[str, Any]:
        try:
            stopped = dict(self.manager.stop({}))
            manager_error = stopped.get("error")
        except BaseException as error:
            manager_error = _safe_error(error)
        try:
            current = dict(self.worker.current_scene())
            worker_idle = current.get("status") == "idle" and current.get("scene") is None
        except BaseException as error:
            current = {"error": _safe_error(error)}
            worker_idle = False
        return {
            "manager_error": manager_error,
            "worker_idle": worker_idle,
            "worker_status": current.get("status"),
        }

    def _stop_and_verify(self) -> dict[str, Any]:
        try:
            stopped = dict(self.manager.stop({}))
            manager_error = stopped.get("error")
        except BaseException as error:
            stopped = {}
            manager_error = _safe_error(error)
        try:
            current = dict(self.worker.current_scene())
            worker_idle = current.get("status") == "idle" and current.get("scene") is None
        except BaseException as error:
            current = {"error": _safe_error(error)}
            worker_idle = False
        return {
            "manager_status": stopped.get("status"),
            "manager_error": manager_error,
            "worker_idle": worker_idle,
            "worker_status": current.get("status"),
        }

    def _bounded_shutdown(self) -> dict[str, Any] | None:
        errors: list[BaseException] = []

        def shutdown() -> None:
            try:
                self.manager.shutdown()
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=shutdown, name="garage-density-shutdown", daemon=True)
        thread.start()
        thread.join(timeout=self.cancel_grace)
        if thread.is_alive():
            return {
                "code": "shutdown_timeout",
                "message": "Garage manager shutdown exceeded the cancellation grace interval",
            }
        if errors:
            return _safe_error(errors[0])
        return None


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure dense Garage preparation, responsiveness, camera health and cleanup."
    )
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--carla-host", default=os.environ.get("CARLA_HOST", "127.0.0.1"))
    parser.add_argument("--carla-port", type=int, default=int(os.environ.get("CARLA_PORT", "2000")))
    parser.add_argument(
        "--world-worker-url", default=os.environ.get("CARLA_WORLD_WORKER_URL", "")
    )
    parser.add_argument("--world-worker-token-env", default="CARLA_WORLD_WORKER_TOKEN")
    parser.add_argument("--map", dest="map_name", default="Town10HD_Opt")
    parser.add_argument("--vehicle", default="vehicle.tesla.model3")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--tiers",
        default="0:0,25:25,75:75,250:0,0:250,250:250",
        help="comma-separated TRAFFIC:WALKERS pairs",
    )
    parser.add_argument("--crossing-factor", type=float, default=0.5)
    parser.add_argument("--speed-difference-percent", type=float, default=12.0)
    parser.add_argument("--following-distance-metres", type=float, default=2.0)
    parser.add_argument("--camera-profile", default="balanced")
    parser.add_argument("--probe-interval", type=float, default=0.25)
    parser.add_argument("--max-control-plane-latency", type=float, default=1.0)
    parser.add_argument("--prepare-timeout", type=float, default=180.0)
    parser.add_argument("--cancel-grace", type=float, default=15.0)
    parser.add_argument("--camera-sample-seconds", type=float, default=2.0)
    parser.add_argument("--expected-version", default=EXPECTED_CARLA_VERSION)
    parser.add_argument("--output", default="")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> tuple[DensityTier, ...]:
    tiers = parse_tiers(args.tiers)
    if not 1 <= args.carla_port <= 65535:
        raise ValueError("--carla-port must be in [1, 65535]")
    if not args.world_worker_url:
        raise ValueError("--world-worker-url or CARLA_WORLD_WORKER_URL is required")
    if not 0.0 <= args.crossing_factor <= 1.0 or not math.isfinite(args.crossing_factor):
        raise ValueError("--crossing-factor must be finite and in [0, 1]")
    if not -100.0 <= args.speed_difference_percent <= 100.0:
        raise ValueError("--speed-difference-percent must be in [-100, 100]")
    if not 0.1 <= args.following_distance_metres <= 20.0:
        raise ValueError("--following-distance-metres must be in [0.1, 20]")
    if not 0.05 <= args.probe_interval <= 5.0:
        raise ValueError("--probe-interval must be in [0.05, 5]")
    if not 0.1 <= args.max_control_plane_latency <= 30.0:
        raise ValueError("--max-control-plane-latency must be in [0.1, 30]")
    if not 5.0 <= args.prepare_timeout <= 900.0:
        raise ValueError("--prepare-timeout must be in [5, 900]")
    if not 1.0 <= args.cancel_grace <= 120.0:
        raise ValueError("--cancel-grace must be in [1, 120]")
    if not 0.0 <= args.camera_sample_seconds <= 30.0:
        raise ValueError("--camera-sample-seconds must be in [0, 30]")
    return tiers


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        tiers = _validate_args(args)
        token = os.environ.get(args.world_worker_token_env, "").strip()
        if not token:
            raise ValueError(
                f"environment variable {args.world_worker_token_env!r} must contain the Worker token"
            )
        Path(args.workspace).expanduser().resolve(strict=True)
        worker = WorldWorkerClient(args.world_worker_url, token, timeout=5.0)
        manager = GaragePreviewManager(
            carla_host=args.carla_host,
            carla_port=args.carla_port,
            world_worker=worker,
            drive_state=lambda: {"status": "idle"},
            world_mode_lock=threading.RLock(),
        )
        runner = GarageDensityAcceptanceRunner(
            manager,
            worker,
            tiers=tiers,
            map_name=args.map_name,
            vehicle=args.vehicle,
            seed=args.seed,
            pedestrian_crossing_factor=args.crossing_factor,
            speed_difference_percent=args.speed_difference_percent,
            following_distance_metres=args.following_distance_metres,
            camera_profile=args.camera_profile,
            expected_version=args.expected_version,
            probe_interval=args.probe_interval,
            max_control_plane_latency=args.max_control_plane_latency,
            prepare_timeout=args.prepare_timeout,
            cancel_grace=args.cancel_grace,
            camera_sample_seconds=args.camera_sample_seconds,
        )
        report = runner.run()
    except BaseException as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "object_type": "garage_dense_scene_acceptance",
            "live_carla_required_for_empirical_claim": True,
            "status": "fail",
            "started_at": _utc_now(),
            "completed_at": _utc_now(),
            "runner_error": _safe_error(error),
        }

    if args.output:
        output = Path(args.output).expanduser().resolve()
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = Path(args.workspace).expanduser().resolve() / "runs" / f"garage-density-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report.get("status"), "output": str(output)}, sort_keys=True))
    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
