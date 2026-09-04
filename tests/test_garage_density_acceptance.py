from __future__ import annotations

import time
from typing import Any

import pytest

from carla_vision.operator.garage_density_acceptance import (
    DensityTier,
    GarageDensityAcceptanceRunner,
    _parse_args,
    _validate_args,
    parse_tiers,
)
from carla_vision.operator.world_worker_client import WorldWorkerError


class FakeWorker:
    def __init__(self) -> None:
        self.preparing = False
        self.active = False
        self.cancel_requested = False
        self.probes = 0
        self.traffic = 0
        self.walkers = 0
        self.crossing = 0.0
        self.prepare_cancellation = True

    def health(self) -> dict[str, Any]:
        self.probes += 1
        return {
            "status": "preparing" if self.preparing else "ready",
            "ready": True,
            "carla": {
                "connected": True,
                "client_version": "0.9.16",
                "server_version": "0.9.16",
                "current_map": "Town10HD_Opt",
            },
            "capabilities": {
                "observable_scene_preparation": True,
                "responsive_prepare_health": True,
                "prepare_cancellation": self.prepare_cancellation,
            },
        }

    def current_scene(self) -> dict[str, Any]:
        if self.preparing:
            actual_traffic = min(self.traffic, max(0, self.probes - 1) * 16)
            return {
                "status": "preparing",
                "scene": None,
                "preparation": {
                    "status": "preparing",
                    "stage": "traffic" if actual_traffic < self.traffic else "walkers",
                    "requested": {
                        "traffic": self.traffic,
                        "walkers": self.walkers,
                        "pedestrian_crossing_factor": self.crossing,
                    },
                    "actual": {
                        "traffic": actual_traffic,
                        "walkers": 0,
                        "pedestrian_crossing_factor": self.crossing,
                    },
                    "elapsed_seconds": self.probes / 100.0,
                    "error": None,
                },
            }
        if self.active:
            return {"status": "prepared", "scene": {"scene_id": "scene-test"}}
        return {"status": "idle", "scene": None}


class FakeManager:
    def __init__(
        self,
        worker: FakeWorker,
        *,
        maximum_traffic: int = 80,
        hang_until_cancel: bool = False,
        capacity_message: str | None = None,
    ) -> None:
        self.worker = worker
        self.maximum_traffic = maximum_traffic
        self.hang_until_cancel = hang_until_cancel
        self.capacity_message = capacity_message
        self.active = False
        self.cancel_calls = 0
        self.stop_calls = 0
        self.shutdown_calls = 0
        self.last_payload: dict[str, Any] = {}

    def configure(self, raw: dict[str, Any]) -> dict[str, Any]:
        self.last_payload = dict(raw)
        self.worker.preparing = True
        self.worker.traffic = int(raw["traffic_count"])
        self.worker.walkers = int(raw["walker_count"])
        self.worker.crossing = float(raw["pedestrian_crossing_factor"])
        if self.hang_until_cancel:
            while not self.worker.cancel_requested:
                time.sleep(0.001)
            self.worker.preparing = False
            raise WorldWorkerError(
                "scene preparation was cancelled",
                status=409,
                code="scene_prepare_cancelled",
            )
        while self.worker.probes < 3:
            time.sleep(0.001)
        if self.worker.traffic > self.maximum_traffic:
            self.worker.preparing = False
            message = self.capacity_message or (
                f"requested traffic_count {self.worker.traffic} exceeds this map's maximum of "
                f"{self.maximum_traffic} after reserving one ego spawn point"
            )
            raise WorldWorkerError(
                message,
                status=422,
                code="scene_population_capacity",
            )
        self.worker.preparing = False
        self.worker.active = True
        self.active = True
        return self._snapshot()

    def _snapshot(self) -> dict[str, Any]:
        return {
            "status": "running" if self.active else "idle",
            "active": self.active,
            "map": "Town10HD_Opt" if self.active else None,
            "traffic_count": self.worker.traffic,
            "walker_count": self.worker.walkers,
            "traffic_count_requested": self.worker.traffic,
            "walker_count_requested": self.worker.walkers,
            "pedestrian_crossing_factor": self.last_payload.get("pedestrian_crossing_factor"),
            "speed_difference_percent": self.last_payload.get("speed_difference_percent"),
            "following_distance_metres": self.last_payload.get("following_distance_metres"),
            "camera_source_fps": 29.8 if self.active else 0.0,
            "camera_frame_age_seconds": 0.02 if self.active else None,
            "camera_stale": False if self.active else True,
            "frame_sequence": 12 if self.active else -1,
            "camera_transport": "worker_mjpeg" if self.active else "pending",
            "error": None,
        }

    def state(self) -> dict[str, Any]:
        return self._snapshot()

    def cancel_preparation(self) -> dict[str, Any]:
        self.cancel_calls += 1
        self.worker.cancel_requested = True
        return {"status": "cancelling"}

    def stop(self, raw: dict[str, Any]) -> dict[str, Any]:
        assert raw == {}
        self.stop_calls += 1
        self.active = False
        self.worker.active = False
        self.worker.preparing = False
        self.worker.cancel_requested = False
        return self._snapshot()

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.active = False
        self.worker.active = False
        self.worker.preparing = False


def make_runner(
    manager: FakeManager,
    worker: FakeWorker,
    *,
    tiers: tuple[DensityTier, ...],
    prepare_timeout: float = 1.0,
) -> GarageDensityAcceptanceRunner:
    return GarageDensityAcceptanceRunner(
        manager,
        worker,
        tiers=tiers,
        map_name="Town10HD_Opt",
        vehicle="vehicle.tesla.model3",
        seed=42,
        pedestrian_crossing_factor=0.55,
        speed_difference_percent=12.0,
        following_distance_metres=2.0,
        camera_profile="balanced",
        probe_interval=0.005,
        max_control_plane_latency=0.2,
        prepare_timeout=prepare_timeout,
        cancel_grace=0.2,
        camera_sample_seconds=0.0,
    )


def test_preflight_requires_prepare_cancellation_capability() -> None:
    worker = FakeWorker()
    worker.prepare_cancellation = False
    manager = FakeManager(worker)
    report = make_runner(manager, worker, tiers=(DensityTier(25, 20),)).run()

    assert report["status"] == "fail"
    assert report["preflight"]["checks"]["prepare_cancellation"] is False
    assert report["tiers"] == []


def test_density_matrix_records_progress_dynamics_camera_cleanup_and_capacity() -> None:
    worker = FakeWorker()
    manager = FakeManager(worker, maximum_traffic=80)
    runner = make_runner(
        manager,
        worker,
        tiers=(DensityTier(25, 20), DensityTier(250, 0)),
    )

    report = runner.run()

    assert report["status"] == "pass"
    assert len(report["tiers"]) == 2
    first, capacity = report["tiers"]
    assert first["status"] == "pass"
    assert first["checks"] == {
        "preview_running": True,
        "exact_population": True,
        "dynamics_match": True,
        "camera_stream_healthy": True,
        "control_plane_responsive": True,
        "cleanup_confirmed": True,
    }
    assert first["prepared"]["pedestrian_crossing_factor"] == 0.55
    assert first["camera_sample"]["source_fps"] == 29.8
    assert first["cleanup"]["worker_idle"] is True
    assert first["control_plane"]["during_prepare_probes"] > 0
    assert first["progress"]
    assert first["progress"][0]["requested"]["traffic"] == 25

    assert capacity["status"] == "capacity_boundary"
    assert capacity["passed"] is True
    assert capacity["capacity_maximum_traffic"] == 80
    assert capacity["cleanup"]["worker_idle"] is True
    assert manager.stop_calls == 2
    assert manager.shutdown_calls == 1


def test_capacity_rejection_without_explicit_maximum_is_not_accepted() -> None:
    worker = FakeWorker()
    manager = FakeManager(
        worker,
        maximum_traffic=10,
        capacity_message="map cannot hold this population",
    )
    report = make_runner(manager, worker, tiers=(DensityTier(250, 0),)).run()
    assert report["status"] == "fail"
    assert report["tiers"][0]["status"] == "prepare_failed"
    assert report["tiers"][0]["capacity_maximum_traffic"] is None


def test_prepare_timeout_requests_cooperative_cancel_and_confirms_cleanup() -> None:
    worker = FakeWorker()
    manager = FakeManager(worker, hang_until_cancel=True)
    report = make_runner(
        manager,
        worker,
        tiers=(DensityTier(75, 75),),
        prepare_timeout=0.03,
    ).run()

    assert report["status"] == "fail"
    tier = report["tiers"][0]
    assert tier["status"] == "prepare_timeout"
    assert tier["fatal"] is False
    assert tier["cleanup"]["worker_idle"] is True
    assert manager.cancel_calls == 1


def test_parse_tiers_covers_independent_vehicle_and_walker_maxima() -> None:
    assert parse_tiers("0:0,250:0,0:250") == (
        DensityTier(0, 0),
        DensityTier(250, 0),
        DensityTier(0, 250),
    )
    with pytest.raises(ValueError, match=r"\[0, 250\]"):
        parse_tiers("251:0")
    with pytest.raises(ValueError, match="TRAFFIC:WALKERS"):
        parse_tiers("25")


def test_cli_keeps_worker_secret_in_environment_and_validates_bounds() -> None:
    args = _parse_args(
        [
            "--world-worker-url",
            "http://192.168.1.20:8766",
            "--world-worker-token-env",
            "DENSITY_WORKER_TOKEN",
            "--tiers",
            "0:0,25:25",
        ]
    )
    tiers = _validate_args(args)
    assert tiers == (DensityTier(0, 0), DensityTier(25, 25))
    assert args.world_worker_token_env == "DENSITY_WORKER_TOKEN"
    assert not hasattr(args, "world_worker_token")


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--crossing-factor", "1.1"),
        ("--probe-interval", "0"),
        ("--max-control-plane-latency", "0"),
        ("--prepare-timeout", "1"),
        ("--cancel-grace", "0"),
        ("--camera-sample-seconds", "31"),
    ],
)
def test_cli_rejects_unbounded_acceptance_values(flag: str, value: str) -> None:
    args = _parse_args(["--world-worker-url", "http://127.0.0.1:8766", flag, value])
    with pytest.raises(ValueError):
        _validate_args(args)
