from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from carla_vision.launcher.config import LauncherError
from carla_vision.launcher.live_check import (
    describe_live_check,
    live_check_state,
    start_live_check,
)
from carla_vision.operator.garage_server import create_server


@pytest.fixture
def operator_url(tmp_path: Path) -> Iterator[str]:
    server = create_server(workspace=tmp_path, port=0, carla_host="127.0.0.1", carla_port=1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}/"
    finally:
        server.shutdown()
        server.server_close()
        server.application.jobs.shutdown()
        thread.join(timeout=5)


def test_launcher_reads_the_operator_live_check_state(operator_url: str) -> None:
    state = live_check_state(operator_url)
    assert state["phase"] == "idle"
    assert state["available"] is False
    assert describe_live_check(state) == "Live check has not run yet."


def test_launcher_start_reports_the_operator_reason(operator_url: str) -> None:
    with pytest.raises(LauncherError, match="connect the World Worker"):
        start_live_check(operator_url)


def test_unreachable_web_app_is_a_launcher_error() -> None:
    with pytest.raises(LauncherError, match="not reachable"):
        live_check_state("http://127.0.0.1:9/")


def test_description_names_failed_checks_and_report() -> None:
    text = describe_live_check(
        {
            "phase": "failed",
            "active": False,
            "report": "runs/garage-acceptance-1.json",
            "summary": {"passed_checks": 7, "total_checks": 9, "failed_checks": ["frames_advance", "cleanup"]},
        }
    )
    assert text == (
        "Live check failed: 7 of 9 checks passed. Failed: frames_advance, cleanup. "
        "Report: runs/garage-acceptance-1.json"
    )
    assert "running" in describe_live_check({"phase": "running", "active": True})
