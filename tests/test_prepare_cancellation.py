from __future__ import annotations

import json
import threading
import urllib.request
from typing import Any
from unittest import mock

import pytest

from carla_vision.native import observable_world_worker as observable
from carla_vision.native import world_worker as base_worker
from carla_vision.operator.garage_preview import GaragePreviewManager
from carla_vision.operator.world_worker_client import WorldWorkerClient


def test_observable_worker_cancel_interrupts_next_owned_spawn_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = observable.ObservableWorldWorker(start_monitor=False)
    worker._begin_preparation(observable.SceneConfig(traffic_count=64, walker_count=40))
    worker._set_preparation_stage("traffic")
    response = worker.cancel_preparation({})
    assert response["status"] == "cancelling"
    assert response["preparation"]["cancel_requested"] is True

    monkeypatch.setattr(
        observable.base.WorldWorker,
        "_spawn_owned_batch",
        lambda self, world, requests, owned: [],
    )
    with pytest.raises(observable.WorkerError) as caught:
        worker._spawn_owned_batch(None, [], [])
    assert caught.value.status == 409
    assert caught.value.code == "scene_prepare_cancelled"


def test_cancel_preparation_requires_active_prepare() -> None:
    worker = observable.ObservableWorldWorker(start_monitor=False)
    with pytest.raises(observable.WorkerError) as caught:
        worker.cancel_preparation({})
    assert caught.value.status == 409
    assert caught.value.code == "scene_not_preparing"


def test_world_worker_http_exposes_authenticated_cancel_route() -> None:
    class Worker:
        def __init__(self) -> None:
            self.calls = 0

        def cancel_preparation(self, raw: dict[str, Any]) -> dict[str, Any]:
            assert raw == {}
            self.calls += 1
            return {"schema_version": "1.0", "status": "cancelling"}

    worker = Worker()
    server = base_worker.create_server(
        bind="127.0.0.1",
        port=0,
        token="test-worker-token",
        worker=worker,  # type: ignore[arg-type]
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        request = urllib.request.Request(
            f"http://{host}:{port}/v1/scenes/preparation/cancel",
            data=b"{}",
            method="POST",
            headers={
                "Authorization": "Bearer test-worker-token",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert response.status == 202
        assert payload["status"] == "cancelling"
        assert worker.calls == 1
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()


def test_world_worker_client_cancel_uses_dedicated_non_scene_route() -> None:
    client = WorldWorkerClient("http://127.0.0.1:8766", "test-worker-token")
    with mock.patch.object(
        client,
        "_request",
        return_value={"schema_version": "1.0", "status": "cancelling"},
    ) as request:
        result = client.cancel_preparation()
    assert result["status"] == "cancelling"
    request.assert_called_once_with("POST", "/v1/scenes/preparation/cancel", {})


def test_preview_manager_cancel_is_outside_scene_mutation_lock() -> None:
    class Worker:
        def __init__(self) -> None:
            self.calls = 0

        def cancel_preparation(self) -> dict[str, Any]:
            self.calls += 1
            return {"status": "cancelling"}

    worker = Worker()
    world_mode_lock = threading.RLock()
    manager = GaragePreviewManager(
        carla_host="127.0.0.1",
        carla_port=2000,
        world_worker=worker,  # type: ignore[arg-type]
        drive_state=lambda: {"status": "idle"},
        world_mode_lock=world_mode_lock,
    )
    # Holding the scene mutation lock in this thread simulates configure owning
    # it. Cancellation must still go straight to the Worker's responsive plane.
    with world_mode_lock:
        result = manager.cancel_preparation()
    assert result == {"status": "cancelling"}
    assert worker.calls == 1
