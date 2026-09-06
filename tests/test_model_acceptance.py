from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from carla_vision.operator.model_acceptance import (
    EXPECTED_CARLA_VERSION,
    ModelAcceptanceRunner,
    _parse_args,
    _validate_args,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += max(0.0, seconds)


class FakeResolvedPackage:
    def __init__(
        self,
        package_id: str,
        runtime: str,
        sha256: str,
        *,
        devices: tuple[str, ...] = ("cpu",),
        fail_artifact: bool = False,
    ) -> None:
        self.package = SimpleNamespace(
            package_id=package_id,
            name=f"{package_id} name",
            version="1.0.0",
            runtime=runtime,
            factory=(
                "research_models.policy:create_driver"
                if runtime == "python_factory"
                else None
            ),
            devices=devices,
            requires_trusted_code=True,
        )
        self.sha256 = sha256
        self.fail_artifact = fail_artifact

    def verify_artifact_reference(self) -> dict[str, Any]:
        if self.fail_artifact:
            raise ValueError("artifact SHA-256 mismatch")
        return {
            "path": f"/models/{self.package.package_id}/model.pt",
            "sha256": self.sha256,
            "size_bytes": 123,
        }

    def verify_manifest_reference(self) -> dict[str, Any]:
        return {
            "path": f"/models/{self.package.package_id}/model.json",
            "sha256": (
                "a" if self.package.runtime == "torchscript_control_v1" else "b"
            )
            * 64,
            "size_bytes": 456,
        }


class FakeWorker:
    def __init__(self) -> None:
        self.active = False
        self.map_name = "Town10HD_Opt"

    def health(self) -> dict[str, Any]:
        return {
            "carla": {
                "server_version": EXPECTED_CARLA_VERSION,
                "client_version": EXPECTED_CARLA_VERSION,
                "current_map": self.map_name,
            }
        }

    def current_scene(self) -> dict[str, Any]:
        if not self.active:
            return {"status": "idle", "scene": None}
        return {
            "status": "running",
            "scene": {
                "scene_id": "scene-1",
                "map_name": self.map_name,
            },
        }


class FakeManager:
    def __init__(
        self,
        worker: FakeWorker,
        packages: dict[str, FakeResolvedPackage],
        *,
        actuates: bool = True,
        cleanup_errors: list[str] | None = None,
    ) -> None:
        self.worker = worker
        self.packages = packages
        self.actuates = actuates
        self.cleanup_errors = cleanup_errors or []
        self.payloads: list[dict[str, Any]] = []
        self.active_payload: dict[str, Any] | None = None
        self.session_counter = 0
        self.shutdown_called = False

    def catalog(self) -> dict[str, Any]:
        return {
            "connected": True,
            "server_version": EXPECTED_CARLA_VERSION,
            "world_worker": {"connected": True},
            "maps": [{"id": "Town10HD_Opt"}],
            "vehicles": [{"id": "vehicle.tesla.model3"}],
        }

    def start_model(self, raw: dict[str, Any]) -> dict[str, Any]:
        self.session_counter += 1
        self.active_payload = dict(raw)
        self.payloads.append(dict(raw))
        self.worker.active = True
        self.worker.map_name = str(raw["map_name"])
        return {
            "status": "starting",
            "session_id": f"session-{self.session_counter}",
        }

    def state(self) -> dict[str, Any]:
        if self.active_payload is None:
            return {"status": "idle"}
        raw = self.active_payload
        package = self.packages[str(raw["model_package_id"])]
        control_source = "external_model" if self.actuates else "model_error"
        commands = 3 if self.actuates else 0
        failsafes = 0 if self.actuates else 4
        return {
            "status": "running",
            "session_id": f"session-{self.session_counter}",
            "vehicle_id": 101,
            "camera_id": 202,
            "traffic_count_actual": int(raw["traffic_count"]),
            "walker_count_actual": int(raw["walker_count"]),
            "garage_mode": "model",
            "control_source": control_source,
            "deadman_active": not self.actuates,
            "autonomy": {
                "enabled": True,
                "operator_acknowledged": True,
                "model_output_actuated": True,
                "policy_ready": True,
                "commands": commands,
                "failsafes": failsafes,
                "detail": {
                    "model_package_id": package.package.package_id,
                    "model_runtime": package.package.runtime,
                    "model_artifact_sha256": package.sha256,
                    "inference_latency_seconds": 0.012,
                    "frame_age_seconds": 0.025,
                },
            },
        }

    def wait_for_frame(
        self,
        stream: str,
        sequence: int,
        *,
        timeout: float,
    ) -> tuple[int, bytes]:
        assert stream == "raw"
        assert sequence == -1
        assert timeout > 0
        return 7, b"j" * 512

    def stop(self, raw: dict[str, Any]) -> dict[str, Any]:
        assert str(raw["session_id"]).startswith("session-")
        self.active_payload = None
        self.worker.active = False
        return {
            "status": "success",
            "cleanup_errors": list(self.cleanup_errors),
        }

    def shutdown(self) -> None:
        self.shutdown_called = True
        self.active_payload = None
        self.worker.active = False


def _runner(
    tmp_path: Path,
    packages: dict[str, FakeResolvedPackage],
    *,
    manager: FakeManager | None = None,
    device: str = "cpu",
    package_ids: tuple[str, ...] | None = None,
    weather_presets: tuple[str, ...] = ("clear-day",),
    repeat: int = 1,
    state_timeout: float = 1.0,
) -> tuple[ModelAcceptanceRunner, FakeManager, FakeClock]:
    worker = manager.worker if manager is not None else FakeWorker()
    manager = manager or FakeManager(worker, packages)
    clock = FakeClock()

    def resolver(
        _workspace: Path,
        package_id: str,
        *,
        required_role: str | None = None,
    ) -> FakeResolvedPackage:
        assert required_role == "driving_policy"
        return packages[package_id]

    def starter(fake_manager: FakeManager, raw: dict[str, Any]) -> dict[str, Any]:
        return fake_manager.start_model(raw)

    runner = ModelAcceptanceRunner(
        manager,
        worker,
        workspace=tmp_path,
        host="127.0.0.1",
        port=2000,
        map_name="Town10HD_Opt",
        vehicle="vehicle.tesla.model3",
        package_ids=package_ids or tuple(packages),
        device=device,
        seed=17,
        traffic_count=4,
        walker_count=3,
        camera_fps=30.0,
        repeat=repeat,
        weather_presets=weather_presets,
        record_video=True,
        trusted_code_acknowledged=True,
        state_timeout=state_timeout,
        clock=clock,
        sleep=clock.sleep,
        package_resolver=resolver,
        session_starter=starter,
    )
    return runner, manager, clock


def test_runner_exercises_torchscript_and_python_factory_matrix(tmp_path: Path) -> None:
    packages = {
        "torch-policy": FakeResolvedPackage(
            "torch-policy",
            "torchscript_control_v1",
            "1" * 64,
        ),
        "python-policy": FakeResolvedPackage(
            "python-policy",
            "python_factory",
            "2" * 64,
        ),
    }
    runner, manager, _clock = _runner(
        tmp_path,
        packages,
        weather_presets=("clear-day", "wet-sunset"),
        repeat=2,
    )

    report = runner.run()

    assert report["status"] == "pass"
    assert len(report["packages"]) == 2
    assert len(report["sessions"]) == 8
    assert {row["runtime"] for row in report["packages"]} == {
        "torchscript_control_v1",
        "python_factory",
    }
    assert all(session["status"] == "pass" for session in report["sessions"])
    assert all(
        session["requested"]["model_artifact_sha256"] in {"1" * 64, "2" * 64}
        for session in report["sessions"]
    )
    assert all(
        payload["control_mode"] == "model"
        and payload["model_trusted_code_acknowledged"] is True
        and payload["record_video"] is True
        for payload in manager.payloads
    )
    assert {payload["weather_preset"] for payload in manager.payloads} == {
        "clear-day",
        "wet-sunset",
    }
    assert manager.shutdown_called is True
    assert manager.worker.current_scene() == {"status": "idle", "scene": None}


def test_device_mismatch_blocks_before_session_start(tmp_path: Path) -> None:
    packages = {
        "cuda-only": FakeResolvedPackage(
            "cuda-only",
            "python_factory",
            "3" * 64,
            devices=("cuda",),
        )
    }
    runner, manager, _clock = _runner(tmp_path, packages, device="cpu")

    report = runner.run()

    assert report["status"] == "fail"
    assert report["sessions"] == []
    assert manager.payloads == []
    failed = {row["check_id"] for row in report["checks"] if not row["passed"]}
    assert "package_device_supported:cuda-only" in failed
    assert "runner_error" in report


def test_artifact_verification_failure_blocks_before_model_load(tmp_path: Path) -> None:
    packages = {
        "changed": FakeResolvedPackage(
            "changed",
            "torchscript_control_v1",
            "4" * 64,
            fail_artifact=True,
        )
    }
    runner, manager, _clock = _runner(tmp_path, packages)

    report = runner.run()

    assert report["status"] == "fail"
    assert report["sessions"] == []
    assert manager.payloads == []
    check = next(
        row for row in report["checks"] if row["check_id"] == "package_verified:changed"
    )
    assert check["passed"] is False
    assert "SHA-256" in check["observed"]["message"]


def test_model_that_never_actuates_fails_and_is_forced_stopped(tmp_path: Path) -> None:
    packages = {
        "failing-policy": FakeResolvedPackage(
            "failing-policy",
            "python_factory",
            "5" * 64,
        )
    }
    worker = FakeWorker()
    manager = FakeManager(worker, packages, actuates=False)
    runner, manager, _clock = _runner(
        tmp_path,
        packages,
        manager=manager,
        state_timeout=0.2,
    )

    report = runner.run()

    assert report["status"] == "fail"
    assert len(report["sessions"]) == 1
    session = report["sessions"][0]
    assert session["status"] == "fail"
    assert session["error"]["type"] == "TimeoutError"
    assert session["forced_stop"]["status"] == "success"
    assert worker.current_scene() == {"status": "idle", "scene": None}


def test_cleanup_error_is_retained_as_acceptance_failure(tmp_path: Path) -> None:
    packages = {
        "policy": FakeResolvedPackage(
            "policy",
            "torchscript_control_v1",
            "6" * 64,
        )
    }
    worker = FakeWorker()
    manager = FakeManager(worker, packages, cleanup_errors=["camera destroy failed"])
    runner, _manager, _clock = _runner(tmp_path, packages, manager=manager)

    report = runner.run()

    assert report["status"] == "fail"
    session = report["sessions"][0]
    cleanup = next(
        row for row in session["checks"] if row["check_id"] == "session_stopped_cleanly"
    )
    assert cleanup["passed"] is False
    assert cleanup["observed"]["cleanup_errors"] == ["camera destroy failed"]


def test_cli_requires_explicit_trusted_code_acknowledgement() -> None:
    args = _parse_args(
        [
            "--world-worker-url",
            "http://127.0.0.1:8766",
            "--model-package",
            "road-policy",
        ]
    )
    with pytest.raises(ValueError, match="acknowledge-trusted-model-code"):
        _validate_args(args)

    accepted = _parse_args(
        [
            "--world-worker-url",
            "http://127.0.0.1:8766",
            "--model-package",
            "road-policy",
            "--acknowledge-trusted-model-code",
        ]
    )
    _validate_args(accepted)


def test_cli_rejects_duplicate_model_packages() -> None:
    args = _parse_args(
        [
            "--world-worker-url",
            "http://127.0.0.1:8766",
            "--model-package",
            "road-policy",
            "--model-package",
            "road-policy",
            "--acknowledge-trusted-model-code",
        ]
    )
    with pytest.raises(ValueError, match="unique"):
        _validate_args(args)
