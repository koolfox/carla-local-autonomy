"""Real subprocess/HTTP acceptance for the native host without a CARLA server."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from carla_vision.native.research_jobs import ResearchJobError, ResearchJobs, digest, write_json
from carla_vision.native.task_package import build_teacher_task
from carla_vision.native.world_worker import WorkerError, WorldWorker, create_server
from carla_vision.operator.native_research import NativeResearchClient
from carla_vision.operator.world_worker_client import WorldWorkerError

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "native-research-test-token-with-enough-length"
SCRIPT = """import argparse, hashlib, json, os, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--request",type=Path);p.add_argument("--output",type=Path)
a=p.parse_args();r=json.loads(a.request.read_text());settings=r["parameters"]
print("task-started",flush=True)
if settings.get("crash"): os._exit(9)
if settings.get("wait"):
 while not Path(r["cancel_file"]).exists(): time.sleep(.01)
cancelled=Path(r["cancel_file"]).exists()
archive=a.output/"artifacts.zip";archive.write_bytes(b"fixture-archive")
result={"status":"cancelled" if cancelled else "succeeded","cleanup_confirmed":True,
"archive":{"sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),"size_bytes":archive.stat().st_size},
"secret_leaked":bool(os.environ.get("CARLA_WORLD_WORKER_TOKEN"))}
(a.output/"result.json").write_text(json.dumps(result))
if settings.get("ignore_cancel"):
 while True: time.sleep(.1)
"""


def install(root: Path, *, script: str = SCRIPT, seconds: int = 60) -> dict:
    folder = root / "fixture"
    folder.mkdir(parents=True)
    (folder / "run.py").write_text(script)
    write_json(
        folder / "task.json",
        {
            "schema_version": "1.0",
            "id": "fixture",
            "version": "1",
            "world_access": "exclusive",
            "max_seconds": seconds,
            "entrypoint": "run.py",
            "files": {"run.py": digest(folder / "run.py")},
        },
    )
    return {
        "job_id": "sample",
        "task_id": "fixture",
        "parameters": {},
        "task_sha256": digest(folder / "task.json"),
        "acknowledge_world_reload": True,
    }


def wait_done(jobs, job_id="sample", seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        status = jobs.get(job_id)
        if status["status"] not in {"queued", "running", "cancelling"}:
            return status
        time.sleep(0.02)
    pytest.fail("native job did not terminate")


@pytest.fixture
def host(tmp_path, monkeypatch):
    raw = install(tmp_path / "tasks")
    monkeypatch.setenv("CARLA_WORLD_WORKER_TOKEN", TOKEN)
    worker = WorldWorker(start_monitor=False)
    jobs = ResearchJobs(tmp_path / "tasks", tmp_path / "jobs", worker)
    server = create_server(bind="127.0.0.1", port=0, token=TOKEN, worker=worker, research_jobs=jobs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = NativeResearchClient(f"http://127.0.0.1:{server.server_port}", TOKEN, timeout=2)
    yield worker, jobs, client, raw
    jobs.close()
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)
    worker._scene = None
    worker.close()


def test_http_submit_status_idempotence_logs_and_verified_download(host, tmp_path):
    worker, jobs, client, raw = host
    task = client.tasks()["tasks"][0]
    assert task["id"] == "fixture"
    status = client.submit(
        task_id="fixture", job_id="sample", parameters={}, acknowledge_world_reload=True
    )
    assert status["status"] in {"queued", "running"}
    final = wait_done(jobs)
    assert final["status"] == "succeeded" and final["cleanup_confirmed"]
    assert not final["result"]["secret_leaked"]
    assert client.status("sample")["status"] == "succeeded"
    assert "task-started" in client.log("sample")["text"]
    assert client._request("POST", "/v1/research/jobs", raw)["pid"] == final["pid"]
    assert worker._research_owner is None and not worker.research_recovery_required
    path = client.fetch("sample", tmp_path / "received.zip")
    assert path.read_bytes() == b"fixture-archive"
    with pytest.raises(FileExistsError):
        client.fetch("sample", path)
    (jobs.root / "sample/output/artifacts.zip").write_bytes(b"tampered")
    with pytest.raises(WorldWorkerError, match="verification"):
        client.fetch("sample", tmp_path / "rejected.zip")
    assert not (tmp_path / "rejected.zip").exists()


def test_active_world_ownership_and_cooperative_cancel(host):
    worker, jobs, client, raw = host
    jobs.submit({**raw, "parameters": {"wait": True}})
    assert worker.health()["status"] == "research_running"
    with pytest.raises(WorkerError, match="Native research owns"):
        worker._ensure_client()
    with pytest.raises(ResearchJobError, match="another native job"):
        jobs.submit({**raw, "job_id": "second"})
    with pytest.raises(ResearchJobError, match="another request"):
        jobs.submit(raw)
    deadline = time.monotonic() + 3
    while "task-started" not in client.log("sample")["text"] and time.monotonic() < deadline:
        time.sleep(0.02)
    assert "task-started" in client.log("sample")["text"]
    assert client.status("sample")["status"] == "running"
    client.cancel("sample")
    status = wait_done(jobs)
    assert status["status"] == "cancelled" and status["cleanup_confirmed"]
    assert worker._research_owner is None


def test_existing_garage_scene_is_not_stolen(host):
    worker, jobs, _, raw = host
    worker._scene = SimpleNamespace(status="prepared")
    with pytest.raises(WorkerError, match="Stop Drive"):
        jobs.submit(raw)
    assert worker._scene.status == "prepared"
    assert not (jobs.root / "sample").exists()


def test_busy_world_lock_rejects_without_waiting(host):
    worker, jobs, _, raw = host
    entered, release = threading.Event(), threading.Event()

    def hold():
        with worker._lock:
            entered.set()
            release.wait(3)

    thread = threading.Thread(target=hold)
    thread.start()
    assert entered.wait(1)
    try:
        with pytest.raises(WorkerError, match="still active"):
            jobs.submit(raw)
    finally:
        release.set()
        thread.join()


def test_native_crash_leaves_listener_alive_and_requires_recovery(host):
    worker, jobs, client, raw = host
    jobs.submit({**raw, "parameters": {"crash": True}})
    status = wait_done(jobs)
    assert status["status"] == "failed" and not status["cleanup_confirmed"]
    assert client.health()["status"] == "recovery_required"
    assert client.tasks()["recovery_required"]
    with pytest.raises(WorkerError):
        jobs.submit({**raw, "job_id": "unsafe-retry"})


def test_timeout_forces_process_without_trusting_early_cleanup_receipt(host, monkeypatch):
    worker, jobs, _, raw = host
    monkeypatch.setattr("carla_vision.native.research_jobs._CANCEL_GRACE_SECONDS", 0.1)
    task_path = jobs.tasks / "fixture/task.json"
    task = json.loads(task_path.read_text())
    task["max_seconds"] = 1
    write_json(task_path, task)
    jobs.submit({**raw, "task_sha256": digest(task_path), "parameters": {"ignore_cancel": True}})
    status = wait_done(jobs)
    assert status["status"] == "timed_out" and not status["cleanup_confirmed"]
    assert worker.research_recovery_required


def test_registry_refresh_and_checksum_rejection(host):
    worker, jobs, client, raw = host
    path = jobs.tasks / "fixture/task.json"
    task = json.loads(path.read_text())
    task["version"] = "2"
    write_json(path, task)
    assert client.tasks()["tasks"][0]["version"] == "2"
    with pytest.raises(ResearchJobError, match="task changed"):
        jobs.submit(raw)
    (jobs.tasks / "fixture/run.py").write_text("raise RuntimeError('changed')")
    with pytest.raises(ResearchJobError, match="checksum mismatch"):
        jobs.submit({**raw, "task_sha256": digest(path)})
    assert worker._research_owner is None


@pytest.mark.parametrize("path", ["../outside", "C:/private", "a\\b", "/outside"])
def test_paths_and_executable_selection_are_rejected(host, path):
    _, jobs, _, raw = host
    with pytest.raises(ResearchJobError):
        jobs.submit({**raw, "job_id": path})
    with pytest.raises(ResearchJobError):
        jobs.submit({**raw, "executable": path})


def test_unauthenticated_and_remote_code_routes_are_rejected(host):
    _, _, client, _ = host
    wrong = NativeResearchClient(client.base_url, "wrong-token", timeout=2)
    with pytest.raises(WorldWorkerError) as denied:
        wrong.tasks()
    assert denied.value.status == 401
    with pytest.raises(WorldWorkerError) as missing:
        client._request("POST", "/v1/research/install", {"code": "print('no')"})
    assert missing.value.status == 404


def test_restart_marks_unfinished_job_and_fails_closed(tmp_path):
    install(tmp_path / "tasks")
    path = tmp_path / "jobs/lost"
    path.mkdir(parents=True)
    write_json(path / "status.json", {"job_id": "lost", "status": "running"})
    worker = WorldWorker(start_monitor=False)
    jobs = ResearchJobs(tmp_path / "tasks", tmp_path / "jobs", worker)
    assert jobs.get("lost")["status"] == "interrupted"
    assert worker.research_recovery_required
    assert (jobs.root / "recovery-required.json").is_file()
    jobs.close()
    worker.close()


def test_corrupt_status_keeps_host_running_but_requires_recovery(tmp_path):
    install(tmp_path / "tasks")
    path = tmp_path / "jobs/lost"
    path.mkdir(parents=True)
    (path / "status.json").write_text("{truncated")
    worker = WorldWorker(start_monitor=False)
    jobs = ResearchJobs(tmp_path / "tasks", tmp_path / "jobs", worker)
    try:
        assert worker.health()["status"] == "recovery_required"
        assert jobs.catalog()["tasks"][0]["id"] == "fixture"
    finally:
        jobs.close()
        worker.close()


def test_files_outside_result_allowlist_or_symlink_are_rejected(host, tmp_path):
    _, jobs, client, raw = host
    jobs.submit(raw)
    wait_done(jobs)
    with pytest.raises(ResearchJobError):
        jobs.output_file("sample", "../request.json")
    with pytest.raises(WorldWorkerError) as denied:
        client._request("GET", "/v1/research/jobs/sample/files/task.log")
    assert denied.value.status == 404
    archive = jobs.root / "sample/output/artifacts.zip"
    archive.unlink()
    private = tmp_path / "private.txt"
    private.write_text("private")
    archive.symlink_to(private)
    with pytest.raises(ResearchJobError, match="symlinks"):
        jobs.output_file("sample", "artifacts.zip")


def test_disabled_host_is_explicitly_opt_in(host):
    _, _, client, _ = host
    # The server already has authenticated transport; only the optional host is absent.
    server = create_server(
        bind="127.0.0.1", port=0, token=TOKEN, worker=WorldWorker(start_monitor=False)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = NativeResearchClient(f"http://127.0.0.1:{server.server_port}", TOKEN, timeout=2)
        with pytest.raises(WorldWorkerError) as disabled:
            client.tasks()
        assert disabled.value.status == 501
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        server.worker.close()


def test_missing_task_dependencies_report_clean_preflight_failure(tmp_path):
    task = build_teacher_task(tmp_path / "tasks/teacher_capture", version="fixture")
    output = tmp_path / "output"
    output.mkdir()
    request = tmp_path / "request.json"
    write_json(request, {})
    process = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(task / "run.py"),
            "--request",
            str(request),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert process.returncode == 1
    receipt = json.loads((output / "result.json").read_text())
    assert receipt["cleanup_confirmed"]
    assert "Preflight: ModuleNotFoundError" in receipt["error"]


def test_exported_teacher_task_runs_dry_without_native_api(tmp_path):
    task = build_teacher_task(tmp_path / "tasks/teacher_capture", version="fixture")
    worker = WorldWorker(start_monitor=False)
    jobs = ResearchJobs(task.parent, tmp_path / "jobs", worker)
    parameters = json.loads((ROOT / "configs/capture/remote_teacher_smoke.json").read_text())
    try:
        jobs.submit(
            {
                "job_id": "dry",
                "task_id": "teacher_capture",
                "parameters": parameters,
                "task_sha256": digest(task / "task.json"),
                "acknowledge_world_reload": True,
            }
        )
        status = wait_done(jobs, "dry")
        assert status["status"] == "succeeded", jobs.log("dry")
        assert status["result"]["summary"]["dry_run"]
        assert len(next(iter(status["result"]["summary"]["plan"]["camera_rigs"].values()))) == 3
        assert jobs.output_file("dry", "artifacts.zip").is_file()
        assert status["task_version"] == "fixture"
        assert (
            json.loads((jobs.root / "dry/output/inputs/native_task.json").read_text())["task"][
                "manifest_sha256"
            ]
            == status["task_sha256"]
        )
    finally:
        jobs.close()
        worker.close()


def test_optional_host_module_imports_only_stdlib_in_isolation():
    path = ROOT / "carla_vision/native/research_jobs.py"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import runpy,sys; runpy.run_path(sys.argv[1]); "
            "print(sorted(set(sys.modules)&{'carla','numpy','torch','cv2'}))",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert result.stdout.strip() == "[]"


def test_standalone_bridge_launches_tasks_without_project_dependencies(tmp_path, monkeypatch):
    raw = install(tmp_path / "tasks")
    monkeypatch.setenv("CARLA_WORLD_WORKER_TOKEN", TOKEN)
    log = tmp_path / "bridge.log"
    with log.open("w") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                str(ROOT / "carla_vision/native/world_worker.py"),
                "--internal-worker",
                "--bind",
                "127.0.0.1",
                "--port",
                "0",
                "--research-tasks-dir",
                str(tmp_path / "tasks"),
                "--research-jobs-root",
                str(tmp_path / "jobs"),
                "--research-python",
                sys.executable,
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
            cwd=tmp_path,
        )
        try:
            deadline = time.monotonic() + 10
            ready = None
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    ready = json.loads(log.read_text())
                except ValueError:
                    time.sleep(0.02)
                    continue
                break
            assert ready and ready["status"] == "listening", log.read_text()
            client = NativeResearchClient(f"http://127.0.0.1:{ready['port']}", TOKEN, timeout=2)
            client._request("POST", "/v1/research/jobs", raw)
            result = wait_done(SimpleNamespace(get=client.status))
            assert result["status"] == "succeeded", result
            assert client.tasks()["tasks"][0]["id"] == "fixture"
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
