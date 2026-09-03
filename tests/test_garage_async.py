from __future__ import annotations

import json
import threading
import time
import urllib.request
from collections.abc import Mapping
from typing import Any

import pytest

from carla_vision.operator.garage_async import GaragePreviewAsyncFacade
from carla_vision.operator.garage_server import create_server


def _preview_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "map_name": "Town10HD_Opt",
        "weather_preset": "clear-day",
        "vehicle_blueprint": "vehicle.tesla.model3",
        "color": "255,0,0",
        "seed": 42,
        "traffic_count": 15,
        "walker_count": 10,
        "prop_preset": "construction",
    }
    payload.update(overrides)
    return payload


class _BlockingPreviewManager:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls: list[dict[str, Any]] = []

    def configure(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(raw))
        self.entered.set()
        if not self.release.wait(2.0):
            raise TimeoutError("test did not release configure")
        return {"status": "running", "active": True, "applied_config": dict(raw)}

    def shutdown(self) -> None:
        self.release.set()


def test_async_facade_returns_before_slow_configure_finishes() -> None:
    manager = _BlockingPreviewManager()
    facade = GaragePreviewAsyncFacade(manager)

    started = time.monotonic()
    accepted = facade.start({"traffic_count": 25})
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert accepted["status"] == "starting"
    assert accepted["stage"] == "accepted"
    assert accepted["revision"] == 1
    assert manager.entered.wait(1.0)

    configuring = facade.wait_for_update(
        accepted["operation_id"],
        accepted["revision"],
        timeout=1.0,
    )
    assert configuring is not None
    assert configuring["stage"] == "configuring"
    assert configuring["status"] == "starting"

    manager.release.set()
    completed = facade.wait_for_update(
        accepted["operation_id"],
        configuring["revision"],
        timeout=1.0,
    )
    assert completed is not None
    assert completed["status"] == "running"
    assert completed["stage"] == "running"
    assert completed["result"]["active"] is True
    assert manager.calls == [{"traffic_count": 25}]


def test_async_facade_rejects_second_active_start_without_queueing() -> None:
    manager = _BlockingPreviewManager()
    facade = GaragePreviewAsyncFacade(manager)

    first = facade.start({"traffic_count": 10})
    assert manager.entered.wait(1.0)

    with pytest.raises(RuntimeError, match="already in progress"):
        facade.start({"traffic_count": 20})

    assert manager.calls == [{"traffic_count": 10}]
    manager.release.set()
    terminal = facade.wait_for_update(
        first["operation_id"],
        first["revision"],
        timeout=1.0,
    )
    while terminal is not None and terminal["status"] == "starting":
        terminal = facade.wait_for_update(
            first["operation_id"],
            terminal["revision"],
            timeout=1.0,
        )
    assert terminal is not None
    assert terminal["status"] == "running"


def test_async_facade_publishes_failure_as_terminal_event() -> None:
    class FailingPreviewManager:
        def configure(self, raw: Mapping[str, Any]) -> dict[str, Any]:
            del raw
            raise TimeoutError("CARLA scene preparation failed")

    facade = GaragePreviewAsyncFacade(FailingPreviewManager())
    accepted = facade.start({"traffic_count": 10})

    revision = accepted["revision"]
    terminal: dict[str, Any] | None = None
    for _ in range(3):
        update = facade.wait_for_update(
            accepted["operation_id"],
            revision,
            timeout=1.0,
        )
        assert update is not None
        revision = int(update["revision"])
        if update["status"] == "failed":
            terminal = update
            break

    assert terminal is not None
    assert terminal["stage"] == "failed"
    assert terminal["error"] == {
        "type": "TimeoutError",
        "message": "CARLA scene preparation failed",
    }


def test_async_facade_allows_new_start_after_terminal_operation() -> None:
    class ImmediatePreviewManager:
        def __init__(self) -> None:
            self.calls = 0

        def configure(self, raw: Mapping[str, Any]) -> dict[str, Any]:
            self.calls += 1
            return {"status": "running", "active": True, "applied_config": dict(raw)}

    manager = ImmediatePreviewManager()
    facade = GaragePreviewAsyncFacade(manager)
    first = facade.start({"seed": 1})

    first_update = facade.wait_for_update(first["operation_id"], 1, timeout=1.0)
    assert first_update is not None
    if first_update["status"] == "starting":
        first_update = facade.wait_for_update(
            first["operation_id"],
            first_update["revision"],
            timeout=1.0,
        )
    assert first_update is not None
    assert first_update["status"] == "running"

    second = facade.start({"seed": 2})
    assert second["operation_id"] != first["operation_id"]
    assert manager.calls >= 1


def test_async_garage_http_start_returns_202_and_sse_pushes_completion(tmp_path) -> None:
    server = create_server(
        workspace=tmp_path,
        bind="127.0.0.1",
        port=0,
        sessions_root=tmp_path / "sessions",
        carla_host="127.0.0.1",
        carla_port=65534,
    )
    original_preview = server.application.preview
    original_preview.shutdown()
    preview = _BlockingPreviewManager()
    server.application.preview = preview  # type: ignore[assignment]
    server.application.preview_async = GaragePreviewAsyncFacade(preview)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        request = urllib.request.Request(
            f"http://{host}:{port}/api/garage/preview/configure/start",
            data=json.dumps(_preview_payload()).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Operator-Token": server.application.token,
            },
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=1.0) as response:
            accepted = json.loads(response.read().decode("utf-8"))
            assert response.status == 202
        assert time.monotonic() - started < 0.5
        assert accepted["status"] == "starting"
        assert accepted["stage"] == "accepted"
        assert accepted["resolved_config"]["traffic_count"] == 15
        assert preview.entered.wait(1.0)

        operation_id = accepted["operation_id"]
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/garage/preview/operations/{operation_id}",
            timeout=1.0,
        ) as response:
            current = json.loads(response.read().decode("utf-8"))
        assert current["status"] == "starting"
        assert current["stage"] == "configuring"

        preview.release.set()
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/garage/preview/operations/{operation_id}/events",
            timeout=2.0,
        ) as response:
            events = response.read().decode("utf-8")
            assert response.headers.get_content_type() == "text/event-stream"
        assert "event: garage.preview.lifecycle" in events
        assert '"status":"running"' in events
        assert '"stage":"running"' in events
    finally:
        preview.release.set()
        server.shutdown()
        thread.join(timeout=3.0)
        server.server_close()
        server.application.jobs.shutdown()
