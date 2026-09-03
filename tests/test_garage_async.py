from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from typing import Any

import pytest

from carla_vision.operator.garage_async import GaragePreviewAsyncFacade


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
