from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import carla_vision.operator.local_entrypoint as local_entrypoint
import carla_vision.operator.model_preflight as model_preflight
from carla_vision.operator.model_preflight import (
    ModelPreflightDiagnostic,
    ModelPreflightError,
    preflight_registered_model_request,
)

HOST = "127.0.0.1"
PORT = 65534


def _request(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": "model-preflight-test",
        "host": HOST,
        "port": PORT,
        "vehicle_blueprint": "vehicle.tesla.model3",
        "color": "",
        "seed": 11,
        "weather_preset": "clear-day",
        "prop_preset": "none",
        "detector_enabled": False,
        "detector": "rtdetr",
        "weights": "",
        "device": "cpu",
        "image_size": 640,
        "confidence": 0.35,
        "resolution": "640x384",
        "camera_fps": 10.0,
        "camera_fov": 90.0,
        "record_video": False,
        "spectator_follow": False,
        "control_mode": "model",
        "behavior": "normal",
        "acknowledge_autonomy": True,
        "traffic_vehicles": 0,
        "walkers": 0,
        "tm_port": 8000,
        "target_speed_kmh": 35.0,
        "policy_checkpoint": "",
        "policy_device": "cpu",
        "voxel_readiness_report": "",
        "max_policy_errors": 3,
        "max_model_speed_kmh": 45.0,
        "max_steer_rate": 2.5,
        "model_package_id": "road-policy",
        "model_trusted_code_acknowledged": True,
    }
    payload.update(overrides)
    return payload


def _package(root: Path) -> None:
    directory = root / "models" / "road-policy"
    directory.mkdir(parents=True)
    artifact = b"state-dict-placeholder"
    (directory / "policy.pth").write_bytes(artifact)
    manifest = {
        "schema_version": "1.0",
        "object_type": "runtime_model_package",
        "id": "road-policy",
        "name": "Road Policy",
        "version": "1.0",
        "role": "driving_policy",
        "runtime": "python_factory",
        "artifact": "policy.pth",
        "sha256": hashlib.sha256(artifact).hexdigest(),
        "factory": "carla_vision.model_examples.lane_center:create_driver",
        "devices": ["cpu"],
        "inputs": {"kind": "model_observation_v1", "options": {}},
        "outputs": {"kind": "vehicle_control_v1"},
    }
    (directory / "model.json").write_text(json.dumps(manifest), encoding="utf-8")


def _manager(root: Path) -> Any:
    return SimpleNamespace(
        workspace=root.resolve(),
        carla_host=HOST,
        carla_port=PORT,
        _world_worker=None,
        experimental_enabled=True,
    )


class _FakeModel:
    def __init__(
        self,
        *,
        reset_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.reset_error = reset_error
        self.close_error = close_error
        self.reset_calls = 0
        self.close_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1
        if self.reset_error is not None:
            raise self.reset_error

    def predict(self, _observation: object) -> object:
        raise AssertionError("preflight must not invent a synthetic observation")

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def _install_loader(
    monkeypatch: pytest.MonkeyPatch,
    model: _FakeModel,
    *,
    module_name: str = "carla_vision._preflight_test_adapter",
) -> None:
    sys.modules[module_name] = SimpleNamespace()
    monkeypatch.setattr(
        model_preflight,
        "_load_verified_factory",
        lambda *_args: (lambda _config: model, module_name),
    )
    monkeypatch.setattr(
        model_preflight,
        "create_driving_model_from_factory",
        lambda factory, config: factory(config),
    )


def _preflight_error() -> ModelPreflightError:
    return ModelPreflightError(
        ModelPreflightDiagnostic(
            phase="model_initialization",
            package_id="road-policy",
            runtime="python_factory",
            factory="road_policy:create_driver",
            device="cpu",
            artifact_path="models/road-policy/policy.pth",
            artifact_sha256="a" * 64,
            exception_type="RuntimeError",
            message="state_dict shape mismatch",
        )
    )


def test_preflight_loads_resets_closes_and_returns_sanitized_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _package(tmp_path)
    model = _FakeModel()
    module_name = "carla_vision._preflight_success_adapter"
    _install_loader(monkeypatch, model, module_name=module_name)

    result = preflight_registered_model_request(_manager(tmp_path), _request())

    assert result["status"] == "ready"
    assert result["phase"] == "complete"
    assert result["package_id"] == "road-policy"
    assert result["runtime"] == "python_factory"
    assert result["device"] == "cpu"
    assert result["artifact"]["path"] == "models/road-policy/policy.pth"
    assert result["artifact"]["sha256"]
    assert model.reset_calls == 1
    assert model.close_calls == 1
    assert module_name not in sys.modules


def test_model_initialization_failure_has_phase_and_full_traceback_only_in_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _package(tmp_path)
    module_name = "carla_vision._preflight_failure_adapter"
    sys.modules[module_name] = SimpleNamespace()
    monkeypatch.setattr(
        model_preflight,
        "_load_verified_factory",
        lambda *_args: (lambda _config: object(), module_name),
    )

    def fail_initialization(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("state_dict shape mismatch")

    monkeypatch.setattr(
        model_preflight,
        "create_driving_model_from_factory",
        fail_initialization,
    )

    with caplog.at_level(logging.ERROR), pytest.raises(ModelPreflightError) as caught:
        preflight_registered_model_request(_manager(tmp_path), _request())

    error = caught.value
    payload = error.response_payload()
    diagnostic = payload["error"]["diagnostic"]
    assert diagnostic["phase"] == "model_initialization"
    assert diagnostic["package_id"] == "road-policy"
    assert diagnostic["runtime"] == "python_factory"
    assert diagnostic["device"] == "cpu"
    assert diagnostic["exception"] == {
        "type": "RuntimeError",
        "message": "state_dict shape mismatch",
    }
    assert "traceback" not in json.dumps(payload).lower()
    assert "Traceback" in caplog.text
    assert "state_dict shape mismatch" in caplog.text
    assert module_name not in sys.modules


def test_reset_failure_closes_constructed_model_and_reports_reset_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _package(tmp_path)
    model = _FakeModel(reset_error=ValueError("bad recurrent state"))
    _install_loader(monkeypatch, model)

    with pytest.raises(ModelPreflightError) as caught:
        preflight_registered_model_request(_manager(tmp_path), _request())

    assert caught.value.diagnostic.phase == "model_reset"
    assert caught.value.diagnostic.exception_type == "ValueError"
    assert model.reset_calls == 1
    assert model.close_calls == 1


def test_cleanup_failure_is_not_retried_and_reports_cleanup_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _package(tmp_path)
    model = _FakeModel(close_error=RuntimeError("GPU teardown failed"))
    _install_loader(monkeypatch, model)

    with pytest.raises(ModelPreflightError) as caught:
        preflight_registered_model_request(_manager(tmp_path), _request())

    assert caught.value.diagnostic.phase == "model_cleanup"
    assert caught.value.diagnostic.message == "GPU teardown failed"
    assert model.close_calls == 1


def test_package_resolution_failure_is_structured_before_adapter_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _package(tmp_path)
    called = False

    def should_not_load(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("adapter must not load when package validation fails")

    monkeypatch.setattr(model_preflight, "_load_verified_factory", should_not_load)

    with pytest.raises(ModelPreflightError) as caught:
        preflight_registered_model_request(
            _manager(tmp_path),
            _request(policy_device="cuda"),
        )

    diagnostic = caught.value.diagnostic
    assert diagnostic.phase == "package_resolution"
    assert diagnostic.package_id == "road-policy"
    assert diagnostic.device == "cuda"
    assert "does not advertise device" in diagnostic.message
    assert called is False


def test_start_canonical_session_does_not_start_drive_when_preflight_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = False
    application = SimpleNamespace(drive=object())

    def fail_preflight(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise _preflight_error()

    def should_not_start(*_args: object, **_kwargs: object) -> dict[str, Any]:
        nonlocal started
        started = True
        return {"status": "starting"}

    monkeypatch.setattr(
        local_entrypoint,
        "preflight_registered_model_request",
        fail_preflight,
    )
    monkeypatch.setattr(
        local_entrypoint,
        "start_registered_model_session",
        should_not_start,
    )

    with pytest.raises(ModelPreflightError):
        local_entrypoint._start_canonical_session(application, {"control_mode": "model"})

    assert started is False


def test_start_canonical_session_attaches_preflight_evidence_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = SimpleNamespace(drive=object())
    evidence = {
        "status": "ready",
        "phase": "complete",
        "package_id": "road-policy",
    }
    monkeypatch.setattr(
        local_entrypoint,
        "preflight_registered_model_request",
        lambda *_args, **_kwargs: evidence,
    )
    monkeypatch.setattr(
        local_entrypoint,
        "start_registered_model_session",
        lambda *_args, **_kwargs: {"status": "starting"},
    )

    result = local_entrypoint._start_canonical_session(
        application,
        {"control_mode": "model"},
    )

    assert result == {
        "status": "starting",
        "model_preflight": evidence,
    }


def test_start_canonical_session_leaves_non_model_modes_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class Drive:
        def start(self, request: dict[str, Any]) -> dict[str, Any]:
            calls.append(request)
            return {"status": "starting", "garage_mode": "manual"}

    application = SimpleNamespace(drive=Drive())
    monkeypatch.setattr(
        local_entrypoint,
        "preflight_registered_model_request",
        lambda *_args, **_kwargs: pytest.fail("manual mode must not run model preflight"),
    )

    request = {"control_mode": "manual"}
    result = local_entrypoint._start_canonical_session(application, request)

    assert result["garage_mode"] == "manual"
    assert calls == [request]
