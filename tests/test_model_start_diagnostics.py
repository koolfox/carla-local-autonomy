from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import carla_vision.operator.model_start_diagnostics as diagnostics
from carla_vision.operator.model_start_diagnostics import ModelInitializationError


class _FakeConfig:
    model_package_id = "road-policy"
    model_runtime = "python_factory"
    model_factory = "research_adapter.policy:create_driver"
    policy_device = "cpu"
    policy_checkpoint = Path("/workspace/models/road-policy/policy.pth")
    model_artifact_sha256 = "a" * 64
    model_manifest_path = "models/road-policy/model.json"
    model_manifest_sha256 = "b" * 64
    model_adapter_reference = {"sha256": "c" * 64}
    traffic_vehicles = 0
    walkers = 0


class _FakeSession:
    def __init__(self, config: object, *, workspace: Path, world_worker: object = None) -> None:
        self.config = config
        self.workspace = workspace
        self.world_worker = world_worker
        self._policy = None
        self._policy_detail: dict[str, object] = {}
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def snapshot(self) -> dict[str, object]:
        return {
            "status": "starting" if self.started else "idle",
            "autonomy": {
                "policy_ready": self._policy is not None,
                "detail": dict(self._policy_detail),
            },
        }

    def _close_garage_extensions(self) -> None:
        self.closed = True


class _FakeConfigFactory:
    @classmethod
    def from_mapping(cls, *_args: object, **_kwargs: object) -> _FakeConfig:
        return _FakeConfig()


def _manager(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        workspace=tmp_path,
        carla_host="127.0.0.1",
        carla_port=2000,
        _world_worker=None,
        experimental_enabled=True,
        _lock=threading.RLock(),
        _session=None,
    )


def test_failed_model_initialization_is_synchronous_and_never_publishes_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path)
    sessions: list[_FakeSession] = []

    def make_session(*args: object, **kwargs: object) -> _FakeSession:
        session = _FakeSession(*args, **kwargs)
        sessions.append(session)
        return session

    def fail_policy(*_args: object, **_kwargs: object) -> object:
        try:
            raise ValueError("size mismatch for encoder.weight")
        except ValueError as cause:
            raise RuntimeError("broken checkpoint") from cause

    monkeypatch.setattr(diagnostics, "ExternalModelDriveStartConfig", _FakeConfigFactory)
    monkeypatch.setattr(diagnostics, "ExternalModelDriveSession", make_session)
    monkeypatch.setattr(diagnostics, "_ExternalModelPolicy", fail_policy)

    with pytest.raises(ModelInitializationError) as captured:
        diagnostics.start_registered_model_session(manager, {"model_package_id": "road-policy"})

    error = captured.value
    payload = error.as_dict()
    assert manager._session is None
    assert len(sessions) == 1
    assert sessions[0].started is False
    assert sessions[0].closed is True
    assert payload["phase"] == "model_initialization"
    assert payload["status"] == "failed"
    assert payload["package_id"] == "road-policy"
    assert payload["runtime"] == "python_factory"
    assert payload["factory"] == "research_adapter.policy:create_driver"
    assert payload["device"] == "cpu"
    assert payload["artifact_sha256"] == "a" * 64
    assert payload["manifest_sha256"] == "b" * 64
    assert payload["adapter_sha256"] == "c" * 64
    assert payload["exception"] == {"type": "RuntimeError", "message": "broken checkpoint"}
    assert payload["exception_chain"] == [
        {"type": "RuntimeError", "message": "broken checkpoint"},
        {"type": "ValueError", "message": "size mismatch for encoder.weight"},
    ]
    assert "fail_policy" in payload["traceback"]
    assert "size mismatch for encoder.weight" in payload["traceback"]


def test_successful_model_initialization_is_reused_by_started_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path)
    policy = object()
    calls = 0

    def build_policy(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return policy

    monkeypatch.setattr(diagnostics, "ExternalModelDriveStartConfig", _FakeConfigFactory)
    monkeypatch.setattr(diagnostics, "ExternalModelDriveSession", _FakeSession)
    monkeypatch.setattr(diagnostics, "_ExternalModelPolicy", build_policy)

    snapshot = diagnostics.start_registered_model_session(
        manager,
        {"model_package_id": "road-policy"},
    )

    assert calls == 1
    assert manager._session is not None
    assert manager._session.started is True
    assert manager._session._policy is policy
    assert snapshot["autonomy"]["policy_ready"] is True
    initialization = snapshot["autonomy"]["detail"]["model_initialization"]
    assert initialization["status"] == "ready"
    assert initialization["package_id"] == "road-policy"
    assert initialization["exception"] if "exception" in initialization else None is None


def test_existing_active_session_blocks_model_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path)
    manager._session = SimpleNamespace(snapshot=lambda: {"status": "running"})
    calls = 0

    def build_policy(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return object()

    monkeypatch.setattr(diagnostics, "ExternalModelDriveStartConfig", _FakeConfigFactory)
    monkeypatch.setattr(diagnostics, "ExternalModelDriveSession", _FakeSession)
    monkeypatch.setattr(diagnostics, "_ExternalModelPolicy", build_policy)

    with pytest.raises(RuntimeError, match="another interactive drive session"):
        diagnostics.start_registered_model_session(manager, {})

    assert calls == 0
