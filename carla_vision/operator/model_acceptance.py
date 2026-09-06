"""Live CARLA acceptance for registered external driving-model packages.

The runner deliberately reuses the production Garage session manager, registered
model-package resolver, external-model session starter, and Windows World Worker.
It never imports CARLA, deserializes model artifacts itself, or creates actors.
Unit tests validate orchestration; only a retained report from a real CARLA
0.9.16 session is acceptance evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..model_registry import resolve_model_package
from . import garage_acceptance
from .external_model_drive import start_registered_model_session
from .garage_server import GarageOperatorDriveManager
from .world_worker_client import WorldWorkerClient

SCHEMA_VERSION = "1.0"
EXPECTED_CARLA_VERSION = garage_acceptance.EXPECTED_CARLA_VERSION
_SUPPORTED_DRIVING_RUNTIMES = frozenset({"torchscript_control_v1", "python_factory"})
_RUN_TOKEN = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_run_token(value: str) -> str:
    token = _RUN_TOKEN.sub("-", value.strip()).strip("-")
    return token[:80] or "model"


def _finite_nonnegative(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and float(value) >= 0.0


class ModelAcceptanceRunner:
    """Run registered driving policies through the production Garage model path."""

    def __init__(
        self,
        manager: Any,
        worker: Any,
        *,
        workspace: Path,
        host: str,
        port: int,
        map_name: str,
        vehicle: str,
        package_ids: Sequence[str],
        device: str,
        seed: int,
        traffic_count: int,
        walker_count: int,
        camera_fps: float,
        repeat: int,
        weather_presets: Sequence[str] = ("clear-day",),
        record_video: bool = False,
        trusted_code_acknowledged: bool = False,
        expected_version: str = EXPECTED_CARLA_VERSION,
        state_timeout: float = 45.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        package_resolver: Callable[..., Any] = resolve_model_package,
        session_starter: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]] = (
            start_registered_model_session
        ),
    ) -> None:
        self.manager = manager
        self.worker = worker
        self.workspace = workspace
        self.host = host
        self.port = port
        self.map_name = map_name
        self.vehicle = vehicle
        self.package_ids = tuple(package_ids)
        self.device = device
        self.seed = seed
        self.traffic_count = traffic_count
        self.walker_count = walker_count
        self.camera_fps = camera_fps
        self.repeat = repeat
        self.weather_presets = tuple(weather_presets)
        self.record_video = bool(record_video)
        self.trusted_code_acknowledged = bool(trusted_code_acknowledged)
        self.expected_version = expected_version
        self.state_timeout = state_timeout
        self.clock = clock
        self.sleep = sleep
        self.package_resolver = package_resolver
        self.session_starter = session_starter
        self._packages: dict[str, dict[str, Any]] = {}

    def run(self) -> dict[str, Any]:
        started = self.clock()
        report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "object_type": "external_model_live_carla_acceptance",
            "live_carla_required": True,
            "started_at": garage_acceptance._utc_now(),
            "expected_carla_version": self.expected_version,
            "target": {
                "host": self.host,
                "port": self.port,
                "map": self.map_name,
                "vehicle": self.vehicle,
                "device": self.device,
                "seed": self.seed,
                "traffic_count": self.traffic_count,
                "walker_count": self.walker_count,
                "camera": {"resolution": [1280, 720], "fps": self.camera_fps},
                "weather_presets": list(self.weather_presets),
                "repeat": self.repeat,
                "record_video": self.record_video,
            },
            "checks": [],
            "packages": [],
            "sessions": [],
        }
        checks: list[dict[str, Any]] = report["checks"]
        try:
            self._preflight(report, checks)
            if any(item["required"] and not item["passed"] for item in checks):
                raise RuntimeError("registered-model live acceptance preflight failed")
            for package_id in self.package_ids:
                package = self._packages[package_id]
                for weather in self.weather_presets:
                    for repetition in range(1, self.repeat + 1):
                        report["sessions"].append(
                            self._run_session(package, weather, repetition)
                        )
        except BaseException as error:
            report["runner_error"] = garage_acceptance._safe_error(error)
        finally:
            try:
                self.manager.shutdown()
            except BaseException as error:
                report["shutdown_error"] = garage_acceptance._safe_error(error)

        required_checks = [item for item in checks if item.get("required")]
        session_checks = [
            check
            for session in report["sessions"]
            for check in session.get("checks", [])
            if check.get("required")
        ]
        expected_sessions = len(self.package_ids) * len(self.weather_presets) * self.repeat
        passed = (
            "runner_error" not in report
            and "shutdown_error" not in report
            and expected_sessions > 0
            and len(report["sessions"]) == expected_sessions
            and all(item.get("passed") is True for item in [*required_checks, *session_checks])
            and all(session.get("status") == "pass" for session in report["sessions"])
        )
        report["status"] = "pass" if passed else "fail"
        report["duration_seconds"] = max(0.0, self.clock() - started)
        report["completed_at"] = garage_acceptance._utc_now()
        return report

    def _preflight(
        self,
        report: dict[str, Any],
        checks: list[dict[str, Any]],
    ) -> None:
        catalog = dict(self.manager.catalog())
        health = dict(self.worker.health())
        current = dict(self.worker.current_scene())
        report["preflight"] = {
            "operator_catalog": catalog,
            "worker_health": health,
            "worker_scene": {
                "status": current.get("status"),
                "scene_id": garage_acceptance._nested(current, "scene", "scene_id"),
            },
        }

        garage_acceptance._check(
            checks,
            "bridge_connected",
            catalog.get("connected") is True,
            observed=catalog.get("connected"),
        )
        garage_acceptance._check(
            checks,
            "world_worker_connected",
            garage_acceptance._nested(catalog, "world_worker", "connected") is True,
            observed=catalog.get("world_worker"),
        )
        versions = {
            "bridge_server": catalog.get("server_version"),
            "worker_server": garage_acceptance._nested(health, "carla", "server_version"),
            "worker_client": garage_acceptance._nested(health, "carla", "client_version"),
        }
        garage_acceptance._check(
            checks,
            "carla_version_exact",
            all(value == self.expected_version for value in versions.values()),
            observed=versions,
            expected=self.expected_version,
        )
        garage_acceptance._check(
            checks,
            "worker_clean_start",
            current.get("status") == "idle" and current.get("scene") is None,
            observed=report["preflight"]["worker_scene"],
            expected={"status": "idle", "scene_id": None},
        )
        maps = catalog.get("maps", [])
        garage_acceptance._check(
            checks,
            "target_map_available",
            self.map_name == "current"
            or any(
                garage_acceptance._catalog_identifier(item) == self.map_name
                for item in maps
            ),
            observed=self.map_name,
            expected="map advertised by the live catalog",
        )
        vehicles = catalog.get("vehicles", [])
        garage_acceptance._check(
            checks,
            "vehicle_available",
            any(
                garage_acceptance._catalog_identifier(item) == self.vehicle
                for item in vehicles
            ),
            observed=self.vehicle,
            expected="vehicle advertised by the live catalog",
        )
        garage_acceptance._check(
            checks,
            "trusted_model_code_acknowledged",
            self.trusted_code_acknowledged,
            observed=self.trusted_code_acknowledged,
            expected=True,
            note=(
                "Registered TorchScript and Python-factory packages are executable "
                "trusted content; acceptance never acknowledges them implicitly."
            ),
        )

        seen: set[str] = set()
        for package_id in self.package_ids:
            if package_id in seen:
                garage_acceptance._check(
                    checks,
                    f"package_unique:{package_id}",
                    False,
                    observed=package_id,
                    expected="each model package listed once",
                )
                continue
            seen.add(package_id)
            try:
                resolved = self.package_resolver(
                    self.workspace,
                    package_id,
                    required_role="driving_policy",
                )
                package = resolved.package
                artifact = dict(resolved.verify_artifact_reference())
                manifest = dict(resolved.verify_manifest_reference())
                runtime = str(package.runtime)
                runtime_supported = runtime in _SUPPORTED_DRIVING_RUNTIMES
                device_supported = self.device in tuple(package.devices)
                garage_acceptance._check(
                    checks,
                    f"package_runtime_supported:{package_id}",
                    runtime_supported,
                    observed=runtime,
                    expected=sorted(_SUPPORTED_DRIVING_RUNTIMES),
                )
                garage_acceptance._check(
                    checks,
                    f"package_device_supported:{package_id}",
                    device_supported,
                    observed={"selected": self.device, "advertised": list(package.devices)},
                    expected="selected device advertised by package",
                )
                record = {
                    "id": package.package_id,
                    "name": package.name,
                    "version": package.version,
                    "runtime": runtime,
                    "factory": package.factory,
                    "devices": list(package.devices),
                    "requires_trusted_code": bool(package.requires_trusted_code),
                    "artifact": artifact,
                    "manifest": manifest,
                }
                self._packages[package_id] = record
                report["packages"].append(record)
                garage_acceptance._check(
                    checks,
                    f"package_verified:{package_id}",
                    runtime_supported and device_supported,
                    observed={
                        "runtime": runtime,
                        "artifact_sha256": artifact.get("sha256"),
                        "manifest_sha256": manifest.get("sha256"),
                    },
                    expected="verified driving package compatible with selected runtime device",
                )
            except BaseException as error:
                garage_acceptance._check(
                    checks,
                    f"package_verified:{package_id}",
                    False,
                    observed=garage_acceptance._safe_error(error),
                    expected="resolvable driving package with stable manifest and artifact",
                )

    def _run_session(
        self,
        package: Mapping[str, Any],
        weather: str,
        repetition: int,
    ) -> dict[str, Any]:
        started = self.clock()
        package_id = str(package["id"])
        runtime = str(package["runtime"])
        artifact_sha256 = str(package["artifact"]["sha256"])
        session: dict[str, Any] = {
            "package_id": package_id,
            "runtime": runtime,
            "weather_preset": weather,
            "repetition": repetition,
            "checks": [],
        }
        checks: list[dict[str, Any]] = session["checks"]
        session_id: str | None = None
        try:
            run_id = (
                f"model-accept-{_safe_run_token(package_id)}-"
                f"{_safe_run_token(weather)}-r{repetition}-"
                f"{int(time.time())}-{self.seed}"
            )
            payload = garage_acceptance._drive_payload(
                run_id=run_id,
                host=self.host,
                port=self.port,
                vehicle=self.vehicle,
                map_name=self.map_name,
                seed=self.seed + repetition - 1,
                traffic_count=self.traffic_count,
                walker_count=self.walker_count,
                mode="model",
                camera_fps=self.camera_fps,
                start_spawn_index=None,
                destination_spawn_index=None,
            )
            payload.update(
                {
                    "weather_preset": weather,
                    "record_video": self.record_video,
                    "policy_device": self.device,
                    "model_package_id": package_id,
                    "model_trusted_code_acknowledged": self.trusted_code_acknowledged,
                    "max_policy_errors": 3,
                    "max_model_speed_kmh": 45.0,
                    "max_steer_rate": 2.5,
                }
            )
            session["requested"] = {
                "run_id": run_id,
                "model_package_id": package_id,
                "model_runtime": runtime,
                "model_artifact_sha256": artifact_sha256,
                "device": self.device,
                "weather_preset": weather,
                "traffic_count": self.traffic_count,
                "walker_count": self.walker_count,
                "record_video": self.record_video,
            }

            initial = dict(self.session_starter(self.manager, payload))
            session_id = str(initial.get("session_id") or "")
            if not session_id:
                raise RuntimeError("registered model start did not return a session_id")

            running = garage_acceptance._wait_for(
                self.manager,
                lambda state: str(state.get("status")) == "running",
                timeout=self.state_timeout,
                clock=self.clock,
                sleep=self.sleep,
            )
            session["running"] = running
            garage_acceptance._check(
                checks,
                "session_running",
                True,
                observed=running.get("status"),
            )
            garage_acceptance._check(
                checks,
                "ego_spawned",
                isinstance(running.get("vehicle_id"), int)
                and not isinstance(running.get("vehicle_id"), bool)
                and int(running["vehicle_id"]) > 0,
                observed=running.get("vehicle_id"),
                expected="positive actor id",
            )
            garage_acceptance._check(
                checks,
                "camera_spawned",
                isinstance(running.get("camera_id"), int)
                and not isinstance(running.get("camera_id"), bool)
                and int(running["camera_id"]) > 0,
                observed=running.get("camera_id"),
                expected="positive sensor id",
            )
            sequence, jpeg = self.manager.wait_for_frame("raw", -1, timeout=10.0)
            garage_acceptance._check(
                checks,
                "camera_frame_received",
                sequence >= 0
                and isinstance(jpeg, (bytes, bytearray))
                and len(jpeg) > 100,
                observed={"sequence": sequence, "jpeg_bytes": len(jpeg)},
                expected="non-empty live JPEG",
            )
            garage_acceptance._check(
                checks,
                "traffic_population_exact",
                running.get("traffic_count_actual") == self.traffic_count,
                observed=running.get("traffic_count_actual"),
                expected=self.traffic_count,
            )
            garage_acceptance._check(
                checks,
                "walker_population_exact",
                running.get("walker_count_actual") == self.walker_count,
                observed=running.get("walker_count_actual"),
                expected=self.walker_count,
            )

            actuated = garage_acceptance._wait_for(
                self.manager,
                lambda state: (
                    state.get("control_source") == "external_model"
                    and state.get("deadman_active") is False
                    and garage_acceptance._nested(state, "autonomy", "policy_ready") is True
                    and int(
                        garage_acceptance._nested(state, "autonomy", "commands") or 0
                    )
                    > 0
                ),
                timeout=self.state_timeout,
                clock=self.clock,
                sleep=self.sleep,
            )
            session["actuated"] = actuated
            autonomy = actuated.get("autonomy")
            if not isinstance(autonomy, Mapping):
                raise RuntimeError("running model session omitted autonomy evidence")
            detail = autonomy.get("detail")
            if not isinstance(detail, Mapping):
                raise RuntimeError("running model session omitted model detail evidence")

            garage_acceptance._check(
                checks,
                "registered_model_owns_control",
                actuated.get("garage_mode") == "model"
                and actuated.get("control_source") == "external_model"
                and actuated.get("deadman_active") is False
                and autonomy.get("model_output_actuated") is True,
                observed={
                    "garage_mode": actuated.get("garage_mode"),
                    "control_source": actuated.get("control_source"),
                    "deadman_active": actuated.get("deadman_active"),
                    "model_output_actuated": autonomy.get("model_output_actuated"),
                    "commands": autonomy.get("commands"),
                    "failsafes": autonomy.get("failsafes"),
                },
                expected="registered model policy is the non-failsafe actuation owner",
            )
            identity = {
                "model_package_id": detail.get("model_package_id"),
                "model_runtime": detail.get("model_runtime"),
                "model_artifact_sha256": detail.get("model_artifact_sha256"),
            }
            garage_acceptance._check(
                checks,
                "model_identity_matches_verified_package",
                identity
                == {
                    "model_package_id": package_id,
                    "model_runtime": runtime,
                    "model_artifact_sha256": artifact_sha256,
                },
                observed=identity,
                expected={
                    "model_package_id": package_id,
                    "model_runtime": runtime,
                    "model_artifact_sha256": artifact_sha256,
                },
            )
            timing = {
                "inference_latency_seconds": detail.get("inference_latency_seconds"),
                "frame_age_seconds": detail.get("frame_age_seconds"),
            }
            garage_acceptance._check(
                checks,
                "model_timing_evidence_finite",
                all(_finite_nonnegative(value) for value in timing.values()),
                observed=timing,
                expected="finite non-negative inference latency and frame age",
            )

            worker_running = dict(self.worker.current_scene())
            scene_payload = worker_running.get("scene")
            expected_map = (
                garage_acceptance._nested(self.worker.health(), "carla", "current_map")
                if self.map_name == "current"
                else self.map_name
            )
            garage_acceptance._check(
                checks,
                "worker_scene_matches_model_run",
                isinstance(scene_payload, Mapping)
                and scene_payload.get("map_name") == expected_map,
                observed={
                    "status": worker_running.get("status"),
                    "map_name": (
                        scene_payload.get("map_name")
                        if isinstance(scene_payload, Mapping)
                        else None
                    ),
                },
                expected={"status": "running", "map_name": expected_map},
            )

            stopped = dict(self.manager.stop({"session_id": session_id}))
            session["stopped"] = stopped
            garage_acceptance._check(
                checks,
                "session_stopped_cleanly",
                stopped.get("status") == "success"
                and not stopped.get("cleanup_errors"),
                observed={
                    "status": stopped.get("status"),
                    "cleanup_errors": stopped.get("cleanup_errors"),
                },
                expected={"status": "success", "cleanup_errors": []},
            )
            current = dict(self.worker.current_scene())
            session["worker_after_stop"] = current
            garage_acceptance._check(
                checks,
                "worker_scene_released",
                current.get("status") == "idle" and current.get("scene") is None,
                observed={"status": current.get("status"), "scene": current.get("scene")},
                expected={"status": "idle", "scene": None},
            )
        except BaseException as error:
            session["error"] = garage_acceptance._safe_error(error)
            garage_acceptance._check(
                checks,
                "session_exception_free",
                False,
                observed=session["error"],
            )
            if session_id:
                try:
                    session["forced_stop"] = dict(
                        self.manager.stop({"session_id": session_id})
                    )
                except BaseException as stop_error:
                    session["forced_stop_error"] = garage_acceptance._safe_error(
                        stop_error
                    )

        session["duration_seconds"] = max(0.0, self.clock() - started)
        session["status"] = (
            "pass"
            if "error" not in session
            and all(item.get("passed") for item in checks if item.get("required"))
            else "fail"
        )
        return session


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run registered driving-model packages through live CARLA and retain "
            "machine-readable actuation evidence."
        )
    )
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--carla-host", default=os.environ.get("CARLA_HOST", "127.0.0.1"))
    parser.add_argument(
        "--carla-port",
        type=int,
        default=int(os.environ.get("CARLA_PORT", "2000")),
    )
    parser.add_argument(
        "--world-worker-url",
        default=os.environ.get("CARLA_WORLD_WORKER_URL", ""),
    )
    parser.add_argument(
        "--world-worker-token-env",
        default="CARLA_WORLD_WORKER_TOKEN",
        help="environment variable containing the Worker bearer token",
    )
    parser.add_argument(
        "--model-package",
        action="append",
        dest="model_packages",
        default=[],
        help="registered driving model package id; repeat for a matrix",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--map", dest="map_name", default="Town10HD_Opt")
    parser.add_argument("--vehicle", default="vehicle.tesla.model3")
    parser.add_argument("--weather", action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--traffic", type=int, default=5)
    parser.add_argument("--walkers", type=int, default=5)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--acknowledge-trusted-model-code", action="store_true")
    parser.add_argument("--expected-version", default=EXPECTED_CARLA_VERSION)
    parser.add_argument("--state-timeout", type=float, default=45.0)
    parser.add_argument("--output", default="")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not args.world_worker_url:
        raise ValueError("--world-worker-url or CARLA_WORLD_WORKER_URL is required")
    if not args.model_packages:
        raise ValueError("at least one --model-package is required")
    if len(set(args.model_packages)) != len(args.model_packages):
        raise ValueError("--model-package values must be unique")
    if not args.acknowledge_trusted_model_code:
        raise ValueError(
            "--acknowledge-trusted-model-code is required for executable model acceptance"
        )
    if not args.device.strip():
        raise ValueError("--device must not be empty")
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
    weather_presets = args.weather or ["clear-day"]
    if any(not str(value).strip() for value in weather_presets):
        raise ValueError("--weather values must not be empty")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report: dict[str, Any]
    try:
        _validate_args(args)
        workspace = Path(args.workspace).expanduser().resolve(strict=True)
        token = os.environ.get(args.world_worker_token_env, "")
        if not token:
            raise ValueError(
                "environment variable "
                f"{args.world_worker_token_env!r} must contain the Worker token"
            )
        worker = WorldWorkerClient(args.world_worker_url, token, timeout=5.0)
        manager = GarageOperatorDriveManager(
            workspace=workspace,
            carla_host=args.carla_host,
            carla_port=args.carla_port,
            world_worker=worker,
            experimental_enabled=True,
        )
        runner = ModelAcceptanceRunner(
            manager,
            worker,
            workspace=workspace,
            host=args.carla_host,
            port=args.carla_port,
            map_name=args.map_name,
            vehicle=args.vehicle,
            package_ids=args.model_packages,
            device=args.device,
            seed=args.seed,
            traffic_count=args.traffic,
            walker_count=args.walkers,
            camera_fps=args.camera_fps,
            repeat=args.repeat,
            weather_presets=args.weather or ("clear-day",),
            record_video=args.record_video,
            trusted_code_acknowledged=args.acknowledge_trusted_model_code,
            expected_version=args.expected_version,
            state_timeout=args.state_timeout,
        )
        report = runner.run()
    except BaseException as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "object_type": "external_model_live_carla_acceptance",
            "live_carla_required": True,
            "status": "fail",
            "started_at": garage_acceptance._utc_now(),
            "completed_at": garage_acceptance._utc_now(),
            "runner_error": garage_acceptance._safe_error(error),
        }

    workspace_path = Path(args.workspace).expanduser().resolve()
    if args.output:
        output = Path(args.output).expanduser().resolve()
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = workspace_path / "runs" / f"model-acceptance-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "report": str(output)}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
