from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from carla_vision.native import observable_world_worker as observable


def _health_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "worker_api_revision": 2,
        "status": "ready",
        "ready": True,
        "error_code": None,
        "carla": {
            "connected": True,
            "host": "127.0.0.1",
            "port": 2000,
            "client_version": "0.9.16",
            "server_version": "0.9.16",
            "current_map": "Town10HD_Opt",
        },
        "active_scene": None,
        "capabilities": {"traffic_manager": True},
    }


def _scene_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "worker_api_revision": 2,
        "status": "idle",
        "scene": None,
    }


def test_health_and_current_scene_do_not_wait_behind_dense_prepare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    monkeypatch.setattr(observable.base.WorldWorker, "health", lambda self: _health_payload())
    monkeypatch.setattr(
        observable.base.WorldWorker,
        "current_scene",
        lambda self: _scene_payload(),
    )

    def slow_prepare(self: observable.ObservableWorldWorker, raw: dict[str, Any]) -> dict[str, Any]:
        del raw
        with self._lock:
            self._set_preparation_stage("traffic")
            started.set()
            assert release.wait(2.0)
            return {"schema_version": "1.0", "status": "prepared", "scene_id": "scene-test"}

    monkeypatch.setattr(observable.base.WorldWorker, "prepare", slow_prepare)
    worker = observable.ObservableWorldWorker(start_monitor=False)
    result: dict[str, Any] = {}
    errors: list[BaseException] = []

    def run_prepare() -> None:
        try:
            result.update(worker.prepare({"traffic_count": 40, "walker_count": 30}))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run_prepare, daemon=True)
    thread.start()
    assert started.wait(1.0)

    before = time.monotonic()
    health = worker.health()
    scene = worker.current_scene()
    elapsed = time.monotonic() - before

    assert elapsed < 0.2
    assert health["status"] == "preparing"
    assert health["ready"] is True
    assert health["busy"] is True
    assert health["preparation"]["stage"] == "traffic"
    assert health["preparation"]["requested"]["traffic"] == 40
    assert health["preparation"]["requested"]["walkers"] == 30
    assert health["preparation"]["requested"]["pedestrian_crossing_factor"] == 0.2
    assert health["preparation"]["requested"]["speed_difference_percent"] == 12.0
    assert health["preparation"]["requested"]["following_distance_metres"] == 2.0
    assert scene["status"] == "preparing"
    assert scene["preparation"]["stage"] == "traffic"

    with pytest.raises(observable.WorkerError) as duplicate:
        worker.prepare({"traffic_count": 1})
    assert duplicate.value.status == 409
    assert duplicate.value.code == "scene_preparing"

    release.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert not errors
    assert result["status"] == "prepared"

    final = worker.health()["preparation"]
    assert final["status"] == "ready"
    assert final["stage"] == "ready"
    assert [item["stage"] for item in final["history"]][-2:] == ["route", "ready"]


def test_prepare_failure_is_observable_without_changing_base_cleanup_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(observable.base.WorldWorker, "health", lambda self: _health_payload())
    monkeypatch.setattr(
        observable.base.WorldWorker,
        "current_scene",
        lambda self: _scene_payload(),
    )

    def failing_prepare(self: observable.ObservableWorldWorker, raw: dict[str, Any]) -> dict[str, Any]:
        del raw
        with self._lock:
            self._set_preparation_stage("walkers", walkers=17)
            raise observable.WorkerError(
                422,
                "scene_population_shortfall",
                "CARLA could not create the exact requested population",
            )

    monkeypatch.setattr(observable.base.WorldWorker, "prepare", failing_prepare)
    worker = observable.ObservableWorldWorker(start_monitor=False)

    with pytest.raises(observable.WorkerError, match="exact requested population"):
        worker.prepare({"traffic_count": 20, "walker_count": 20})

    preparation = worker.current_scene()["preparation"]
    assert preparation["status"] == "failed"
    assert preparation["stage"] == "walkers"
    assert preparation["actual"]["walkers"] == 17
    assert preparation["error"] == {
        "code": "scene_population_shortfall",
        "message": "CARLA could not create the exact requested population",
        "stage": "walkers",
    }
    assert preparation["history"][-1]["stage"] == "failed"


def test_population_hooks_report_actual_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = observable.ObservableWorldWorker(start_monitor=False)
    worker._begin_preparation(observable.SceneConfig(traffic_count=3, walker_count=2))

    traffic_actors = [object(), object()]
    walkers = [object()]
    controllers = [object()]
    monkeypatch.setattr(
        observable.base.WorldWorker,
        "_spawn_traffic",
        lambda self, *args, **kwargs: traffic_actors,
    )
    monkeypatch.setattr(
        observable.base.WorldWorker,
        "_spawn_walkers",
        lambda self, *args, **kwargs: (walkers, controllers),
    )

    assert worker._spawn_traffic("scene", None, None, [], 0, 3, None, []) is traffic_actors
    assert worker._preparation_snapshot()["actual"]["traffic"] == 2
    actual_walkers, actual_controllers = worker._spawn_walkers("scene", None, 2, None, [])
    assert actual_walkers is walkers
    assert actual_controllers is controllers
    snapshot = worker._preparation_snapshot()
    assert snapshot["stage"] == "walkers"
    assert snapshot["actual"]["walkers"] == 1


def test_spawn_batch_hook_reports_incremental_population() -> None:
    worker = observable.ObservableWorldWorker(start_monitor=False)
    worker._begin_preparation(observable.SceneConfig(traffic_count=64, walker_count=40))
    worker._set_preparation_stage("traffic")

    class Item:
        def __init__(self, kind: str) -> None:
            self.kind = kind

    owned = [Item("traffic") for _ in range(32)]
    original = observable.base.WorldWorker._spawn_owned_batch
    try:
        observable.base.WorldWorker._spawn_owned_batch = lambda self, world, requests, owned: []
        worker._spawn_owned_batch(None, [], owned)
    finally:
        observable.base.WorldWorker._spawn_owned_batch = original

    assert worker._preparation_snapshot()["actual"]["traffic"] == 32


def test_ego_stage_confirms_applied_traffic_dynamics(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = observable.ObservableWorldWorker(start_monitor=False)
    config = observable.SceneConfig(
        pedestrian_crossing_factor=0.65,
        speed_difference_percent=-15.0,
        following_distance_metres=3.5,
    )
    worker._begin_preparation(config)
    monkeypatch.setattr(
        observable.base.WorldWorker,
        "_spawn_ego",
        lambda self, *args, **kwargs: (object(), 4),
    )
    worker._spawn_ego(None, config, "scene", [], None, [])
    actual = worker._preparation_snapshot()["actual"]
    assert actual["pedestrian_crossing_factor"] == 0.65
    assert actual["speed_difference_percent"] == -15.0
    assert actual["following_distance_metres"] == 3.5


def test_direct_file_entrypoint_keeps_single_host_script_workflow() -> None:
    path = Path(observable.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, str(path), "--help"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10.0,
        check=False,
    )

    assert completed.returncode == 0
    assert "authenticated CARLA 0.9.16 LAN World Worker" in completed.stdout
