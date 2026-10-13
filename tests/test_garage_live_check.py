from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from carla_vision.operator import garage_acceptance
from carla_vision.operator.garage_live_check import GarageLiveCheck


class FakePreview:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def cancel_preparation(self) -> None:
        self.calls.append("cancel_preparation")

    def stop_for_drive(self) -> None:
        self.calls.append("stop_for_drive")


WORKER = object()


def make_application(tmp_path: Path, *, drive_status: str = "idle", worker: Any = WORKER) -> Any:
    sessions = tmp_path / "operator_sessions"
    sessions.mkdir(exist_ok=True)
    return SimpleNamespace(
        workspace=tmp_path,
        carla_host="127.0.0.1",
        carla_port=2000,
        world_worker=worker,
        jobs=SimpleNamespace(sessions_root=sessions),
        preview=FakePreview(),
        drive=SimpleNamespace(state=lambda: {"status": drive_status}),
    )


def check(check_id: str, passed: bool) -> dict[str, Any]:
    return {"check_id": check_id, "required": True, "passed": passed}


class FakeRunner:
    report: dict[str, Any] = {}
    gate: threading.Event | None = None
    instances: list[FakeRunner] = []

    def __init__(self, manager: Any, worker: Any, **kwargs: Any) -> None:
        self.manager, self.worker, self.kwargs = manager, worker, kwargs
        FakeRunner.instances.append(self)

    def run(self) -> dict[str, Any]:
        if FakeRunner.gate is not None:
            assert FakeRunner.gate.wait(5)
        if isinstance(FakeRunner.report, Exception):
            raise FakeRunner.report
        return FakeRunner.report


@pytest.fixture(autouse=True)
def fake_runner(monkeypatch: pytest.MonkeyPatch) -> type[FakeRunner]:
    FakeRunner.report = {"status": "pass", "checks": [check("bridge_connected", True)], "sessions": []}
    FakeRunner.gate = None
    FakeRunner.instances = []
    monkeypatch.setattr(garage_acceptance, "GarageAcceptanceRunner", FakeRunner)
    return FakeRunner


def run_check(live: GarageLiveCheck) -> dict[str, Any]:
    live.start({"acknowledge": True})
    assert live._thread is not None
    live._thread.join(5)
    return live.state()


def test_start_requires_exact_acknowledgement(tmp_path: Path) -> None:
    live = GarageLiveCheck(make_application(tmp_path), threading.RLock(), manager_factory=lambda _: None)
    for body in ({}, {"acknowledge": False}, {"acknowledge": True, "map": "Town01"}, []):
        with pytest.raises(ValueError):
            live.start(body)
    assert live.state()["active"] is False


def test_start_refuses_without_worker_or_during_drive(tmp_path: Path) -> None:
    no_worker = GarageLiveCheck(
        make_application(tmp_path, worker=None), threading.RLock(), manager_factory=lambda _: None
    )
    with pytest.raises(ValueError, match="World Worker"):
        no_worker.start({"acknowledge": True})

    driving = GarageLiveCheck(
        make_application(tmp_path, drive_status="running"),
        threading.RLock(),
        manager_factory=lambda _: None,
    )
    with pytest.raises(RuntimeError, match="Stop & Save"):
        driving.start({"acknowledge": True})
    assert driving.state()["holds_world"] is False


def test_check_holds_world_pauses_preview_and_saves_report(tmp_path: Path) -> None:
    application = make_application(tmp_path)
    managers: list[bool] = []
    FakeRunner.gate = threading.Event()
    live = GarageLiveCheck(application, threading.RLock(), manager_factory=lambda behavior: managers.append(behavior) or "manager")

    started = live.start({"acknowledge": True})
    assert started["holds_world"] is True and started["active"] is True
    with pytest.raises(RuntimeError, match="owns the world"):
        live.require_world_available()

    FakeRunner.gate.set()
    assert live._thread is not None
    live._thread.join(5)
    state = live.state()

    assert application.preview.calls == ["cancel_preparation", "stop_for_drive"]
    assert state["phase"] == "passed"
    assert state["holds_world"] is False and state["active"] is False
    assert state["summary"]["passed_checks"] == 1 and state["summary"]["failed_checks"] == []
    report = tmp_path / state["report"]
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "pass"
    runner = FakeRunner.instances[0]
    assert runner.manager == "manager" and runner.kwargs["repeat"] == 1
    assert runner.kwargs["modes"] == (("manual", "behavior") if managers[0] else ("manual",))
    live.require_world_available()


def test_failed_checks_are_named_and_world_is_released(tmp_path: Path) -> None:
    FakeRunner.report = {
        "status": "fail",
        "checks": [check("bridge_connected", True)],
        "sessions": [{"checks": [check("session_running", True), check("frames_advance", False)]}],
    }
    live = GarageLiveCheck(make_application(tmp_path), threading.RLock(), manager_factory=lambda _: None)
    state = run_check(live)
    assert state["phase"] == "failed"
    assert state["summary"]["failed_checks"] == ["frames_advance"]
    assert state["summary"]["passed_checks"] == 2 and state["summary"]["total_checks"] == 3
    assert state["holds_world"] is False


def test_runner_crash_releases_world(tmp_path: Path) -> None:
    FakeRunner.report = RuntimeError("worker unreachable")
    live = GarageLiveCheck(make_application(tmp_path), threading.RLock(), manager_factory=lambda _: None)
    state = run_check(live)
    assert state["phase"] == "failed"
    assert state["error"] == "worker unreachable"
    assert state["holds_world"] is False and state["active"] is False


def test_operator_restart_mid_check_reports_failure_and_frees_world(tmp_path: Path) -> None:
    application = make_application(tmp_path)
    path = application.jobs.sessions_root / "live_check_state.json"
    path.write_text(json.dumps({"phase": "running", "active": True, "holds_world": True}), encoding="utf-8")
    state = GarageLiveCheck(application, threading.RLock(), manager_factory=lambda _: None).state()
    assert state["phase"] == "failed" and state["active"] is False and state["holds_world"] is False
