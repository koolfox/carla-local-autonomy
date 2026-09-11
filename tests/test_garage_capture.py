"""Normal Garage capture uses one Worker connection and the shared Scene config."""

import json
import threading
import zipfile
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from carla_vision.operator.configuration import session_defaults
from carla_vision.operator.garage_capture import GarageCapture, unpack_capture
from carla_vision.operator.garage_server import create_server
from carla_vision.operator.world_worker_client import WorldWorkerError
from carla_vision.scenarios.planner import plan_scenarios


@pytest.fixture
def capture(tmp_path):
    client = Mock()
    client.tasks.return_value = {"tasks": [{"id": "teacher_capture", "manifest_sha256": "a" * 64}]}
    client.status.return_value = {"status": "succeeded", "cleanup_confirmed": True}
    app = SimpleNamespace(
        workspace=tmp_path,
        jobs=SimpleNamespace(sessions_root=tmp_path / "operator_sessions"),
        world_worker=SimpleNamespace(research=lambda: client),
        preview=Mock(),
        drive=Mock(),
    )
    app.preview.state.return_value = {"map": "Town03"}
    app.drive.state.return_value = {"status": "idle"}
    owner = GarageCapture(app, threading.RLock())
    owner._receive = Mock(return_value=["runs/capture-fixture"])
    yield owner, client
    owner.close()


def request():
    session = session_defaults(detector_enabled=False)
    session["identity"]["runId"] = "fixture"
    session["scene"].update(
        mapName="Town03", weatherPreset="clear-day", trafficCount=8, walkerCount=4
    )
    session["vehicle"]["blueprint"] = "vehicle.audi.tt"
    return {
        "session": session,
        "situation": {
            "situationId": "scene",
            "egoSpawnIndex": 2,
            "durationSeconds": 5,
            "captureFps": 5,
            "repetitions": 1,
        },
        "camera_rig": "front-three",
        "acknowledge": True,
    }


def finish(owner):
    owner._thread.join(timeout=5)
    assert not owner._thread.is_alive()
    return owner.state()


def test_capture_hands_off_preview_submits_shared_scene_and_imports_results(capture, tmp_path):
    owner, client = capture
    owner.start(request())
    state = finish(owner)
    assert state["phase"] == "saved", state
    assert not state["holds_world"] and not state["active"]
    owner.application.preview.stop_for_drive.assert_called_once()
    owner.application.preview.cancel_preparation.assert_called_once()
    submitted = client.submit.call_args.kwargs
    assert submitted["acknowledge_world_reload"] is True
    assert submitted["parameters"]["camera_rig"] == "front-three"
    owner._receive.assert_called_once_with(client, state["job_id"])
    # Use the real planner to validate the actual built-in submission contract.
    suite = tmp_path / "suite.json"
    splits = tmp_path / "splits.json"
    suite.write_text(json.dumps(submitted["parameters"]["scenario_suite"]))
    splits.write_text(json.dumps(submitted["parameters"]["split_plan"]))
    planned = plan_scenarios(
        suite_path=suite, split_plan_path=splits, runs_root=tmp_path / "runs", run_id="plan"
    )
    assert planned["run_dir"]


def test_capture_requires_acknowledgement_and_idle_drive(capture):
    owner, client = capture
    with pytest.raises(ValueError, match="acknowledgement"):
        owner.start({**request(), "acknowledge": False})
    owner.application.drive.state.return_value = {"status": "running"}
    with pytest.raises(RuntimeError, match="Stop & Save"):
        owner.start(request())
    client.submit.assert_not_called()


def test_failed_submission_unblocks_preview(capture):
    owner, client = capture
    client.submit.side_effect = WorldWorkerError("old bridge", status=404)
    owner.start(request())
    state = finish(owner)
    assert state["phase"] == "failed" and not state["holds_world"]
    owner._receive.assert_not_called()


def test_ambiguous_submit_response_observes_existing_job_without_resubmitting(capture):
    owner, client = capture
    client.submit.side_effect = WorldWorkerError("lost response", status=500)
    owner.start(request())
    assert finish(owner)["phase"] == "saved"
    client.submit.assert_called_once()


def test_cancel_during_preparation_never_submits_and_does_not_block_request(capture):
    owner, client = capture
    entered, release = threading.Event(), threading.Event()
    catalog = client.tasks.return_value

    def tasks():
        entered.set()
        assert release.wait(3)
        return catalog

    client.tasks.side_effect = tasks
    owner.start(request())
    try:
        assert entered.wait(2)
        assert owner.cancel()["cancel_requested"]
        with pytest.raises(RuntimeError, match="owns CARLA"):
            owner.require_world_available()
        with pytest.raises(RuntimeError):
            owner.start(request())
    finally:
        release.set()
    assert finish(owner)["phase"] == "cancelled"
    client.submit.assert_not_called()


def test_failed_cleanup_keeps_world_guard_and_does_not_import(capture):
    owner, client = capture
    client.status.return_value = {"status": "failed", "cleanup_confirmed": False}
    owner.start(request())
    state = finish(owner)
    assert state["phase"] == "failed" and state["holds_world"]
    owner._receive.assert_not_called()


@pytest.mark.parametrize(
    "name",
    ["../bad", "/bad", "datasets/../../bad", "datasets/./bad", "datasets\\bad", "datasets/C:bad"],
)
def test_archive_cannot_escape_import_directory(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(name, b"bad")
    with pytest.raises(ValueError, match="unsafe"):
        unpack_capture(archive, tmp_path / "imported")
    assert not (tmp_path / "imported").exists()


def test_archive_limits_and_case_collisions(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("datasets/A", b"a")
        stream.writestr("datasets/a", b"b")
    with pytest.raises(ValueError, match="unsafe"):
        unpack_capture(archive, tmp_path / "imported")
    with pytest.raises(ValueError, match="limits"):
        unpack_capture(archive, tmp_path / "imported", max_bytes=1)


def test_drive_is_guarded_at_existing_manager_not_only_http_route(tmp_path):
    server = create_server(workspace=tmp_path, port=0)
    try:
        server.application.capture._update(holds_world=True)
        with pytest.raises(RuntimeError, match="owns CARLA"):
            server.application.drive.start({})
    finally:
        server.application.close()
        server.server_close()
