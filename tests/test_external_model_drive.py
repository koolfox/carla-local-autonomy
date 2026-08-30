from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import carla_vision.operator.external_model_drive as external_model_drive
from carla_vision.artifacts import RunArtifactTracker, fingerprint_file
from carla_vision.model_driver import ModelControl
from carla_vision.operator.external_model_drive import (
    ExternalModelDriveSession,
    ExternalModelDriveStartConfig,
    _ExternalModelPolicy,
    _load_verified_factory,
)
from carla_vision.verification import verify_research_object

HOST = "127.0.0.1"
PORT = 65534


def _base(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": "external-model-test",
        "host": HOST,
        "port": PORT,
        "vehicle_blueprint": "vehicle.tesla.model3",
        "color": "",
        "seed": 7,
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


def _package(
    root: Path,
    *,
    runtime: str = "torchscript_control_v1",
    artifact_name: str = "policy.pt",
    factory: str | None = None,
    devices: list[str] | None = None,
    artifact: bytes = b"model",
    expected_hash: str | None = None,
    include_hash: bool = True,
) -> None:
    directory = root / "models" / "road-policy"
    directory.mkdir(parents=True)
    (directory / artifact_name).write_bytes(artifact)
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "object_type": "runtime_model_package",
        "id": "road-policy",
        "name": "Road Policy",
        "version": "1.0",
        "role": "driving_policy",
        "runtime": runtime,
        "artifact": artifact_name,
        "devices": devices or ["cpu"],
        "inputs": (
            {"kind": "model_observation_v1", "options": {}}
            if runtime == "python_factory"
            else {
                "image": {
                    "width": 320,
                    "height": 180,
                    "color": "rgb",
                    "mean": [0.0, 0.0, 0.0],
                    "std": [1.0, 1.0, 1.0],
                },
                "speed": {"enabled": True, "unit": "mps"},
            }
        ),
        "outputs": {"kind": "vehicle_control_v1"},
    }
    if factory is not None:
        manifest["factory"] = factory
    if include_hash:
        manifest["sha256"] = expected_hash or hashlib.sha256(artifact).hexdigest()
    (directory / "model.json").write_text(json.dumps(manifest), encoding="utf-8")


def _config(root: Path, **overrides: object) -> ExternalModelDriveStartConfig:
    return ExternalModelDriveStartConfig.from_mapping(
        _base(**overrides),
        workspace=root,
        expected_host=HOST,
        expected_port=PORT,
        world_worker_configured=False,
        experimental_enabled=True,
    )


def test_torchscript_package_maps_to_builtin_model_driver_and_manifest_identity(tmp_path: Path) -> None:
    artifact = b"torchscript-placeholder"
    _package(tmp_path, artifact=artifact)

    config = _config(tmp_path)

    assert config.control_mode == "model"
    assert config.autonomous is True
    assert config.model_output_actuated is True
    assert config.model_factory == "carla_vision.torchscript_driver:create_driver"
    assert config.policy_checkpoint == (tmp_path / "models" / "road-policy" / "policy.pt").resolve()
    assert config.model_artifact_sha256 == hashlib.sha256(artifact).hexdigest()
    assert config.model_artifact_size_bytes == len(artifact)
    assert config.model_manifest_sha256
    assert config.model_adapter_reference["factory"] == (
        "carla_vision.torchscript_driver:create_driver"
    )
    manifest = config.manifest_config()
    assert manifest["garage_mode"] == "model"
    assert manifest["control_owner"] == "external_model"
    assert manifest["model_package_id"] == "road-policy"
    assert manifest["model_runtime"] == "torchscript_control_v1"


def test_every_executable_runtime_requires_separate_trusted_code_acknowledgement(
    tmp_path: Path,
) -> None:
    _package(tmp_path)

    with pytest.raises(ValueError, match="trusted-code acknowledgement"):
        _config(tmp_path, model_trusted_code_acknowledged=False)


def test_python_factory_requires_importable_fingerprinted_adapter(tmp_path: Path) -> None:
    _package(
        tmp_path,
        runtime="python_factory",
        artifact_name="policy.pth",
        factory="carla_vision.model_examples.lane_center:create_driver",
    )

    with pytest.raises(ValueError, match="trusted-code acknowledgement"):
        _config(tmp_path, model_trusted_code_acknowledged=False)

    config = _config(tmp_path)
    assert config.model_factory == "carla_vision.model_examples.lane_center:create_driver"
    assert config.model_requires_trusted_code is True
    assert config.model_adapter_reference["sha256"]


def test_package_device_must_match_selected_runtime_device(tmp_path: Path) -> None:
    _package(tmp_path, devices=["cpu"])

    with pytest.raises(ValueError, match="does not advertise device 'cuda'"):
        _config(tmp_path, policy_device="cuda")


def test_manifest_hash_mismatch_blocks_actuation_before_model_load(tmp_path: Path) -> None:
    _package(tmp_path, artifact=b"actual", expected_hash="0" * 64)

    with pytest.raises(ValueError, match="SHA-256"):
        _config(tmp_path)


def test_registered_model_mode_requires_experimental_opt_in(tmp_path: Path) -> None:
    _package(tmp_path)

    with pytest.raises(PermissionError, match="--enable-experimental"):
        ExternalModelDriveStartConfig.from_mapping(
            _base(),
            workspace=tmp_path,
            expected_host=HOST,
            expected_port=PORT,
            world_worker_configured=False,
            experimental_enabled=False,
        )


class _FakeModel:
    def __init__(self, *, on_predict: object | None = None) -> None:
        self.on_predict = on_predict
        self.closed = False

    def reset(self) -> None:
        return None

    def predict(self, _observation: object) -> ModelControl:
        if callable(self.on_predict):
            self.on_predict()
        return ModelControl(throttle=0.4, steer=0.1, brake=0.0)

    def close(self) -> None:
        self.closed = True


class _ClosingFailureModel(_FakeModel):
    def close(self) -> None:
        self.closed = True
        raise RuntimeError("close failed")


def _policy_session(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: _FakeModel,
) -> tuple[ExternalModelDriveSession, _ExternalModelPolicy]:
    _package(root)
    config = _config(root)
    session = ExternalModelDriveSession(config, workspace=root)
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    success, jpeg = cv2.imencode(".jpg", image)
    assert success
    session._raw_jpeg = jpeg.tobytes()
    session._raw_frame_sequence = 7
    session._frame_received_monotonic["raw"] = 10.0
    monkeypatch.setattr(
        external_model_drive,
        "create_driving_model_from_factory",
        lambda *_args: model,
    )
    return session, _ExternalModelPolicy(session, config)


def test_inference_that_finishes_after_frame_expiry_is_never_actuated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, policy = _policy_session(tmp_path, monkeypatch, model=_FakeModel())
    clock = iter((10.0, 10.8))
    monkeypatch.setattr(external_model_drive.time, "monotonic", lambda: next(clock))

    command, source, failsafe, detail = policy.step(now=10.0, speed_mps=2.0, dt_s=0.05)

    assert source == "model_inference_stale"
    assert failsafe is True
    assert command.throttle == 0.0
    assert command.brake == 1.0
    assert detail["inference_latency_seconds"] == pytest.approx(0.8)
    assert detail["frame_age_seconds"] == pytest.approx(0.8)
    policy.close()
    session._close_garage_extensions()


def test_emergency_latched_during_predict_discards_model_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder: dict[str, ExternalModelDriveSession] = {}

    def latch_emergency() -> None:
        with holder["session"]._lock:
            holder["session"]._emergency = True

    model = _FakeModel(on_predict=latch_emergency)
    session, policy = _policy_session(tmp_path, monkeypatch, model=model)
    holder["session"] = session
    clock = iter((10.0, 10.1))
    monkeypatch.setattr(external_model_drive.time, "monotonic", lambda: next(clock))

    command, source, failsafe, _detail = policy.step(now=10.0, speed_mps=2.0, dt_s=0.05)

    assert source == "emergency_stop"
    assert failsafe is True
    assert command.throttle == 0.0
    assert command.brake == 1.0
    policy.close()
    session._close_garage_extensions()


def test_adapter_module_is_released_even_when_model_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ClosingFailureModel()
    session, policy = _policy_session(tmp_path, monkeypatch, model=model)
    isolated_name = policy._adapter_module_name
    assert isolated_name is not None and isolated_name in sys.modules

    with pytest.raises(RuntimeError, match="close failed"):
        policy.close()

    assert policy.model is None
    assert policy._adapter_module_name is None
    assert isolated_name not in sys.modules
    session._close_garage_extensions()


def test_model_setup_failure_is_attempted_once_and_latched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _package(tmp_path)
    session = ExternalModelDriveSession(_config(tmp_path), workspace=tmp_path)
    calls = 0

    def fail_setup(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise RuntimeError("broken checkpoint")

    monkeypatch.setattr(external_model_drive, "_ExternalModelPolicy", fail_setup)
    with pytest.raises(RuntimeError, match=r"setup failed \(latched\)"):
        session._ensure_extensions()
    with pytest.raises(RuntimeError, match=r"setup failed \(latched\)"):
        session._ensure_extensions()

    assert calls == 1
    session._close_garage_extensions()


def test_artifact_change_during_model_load_closes_model_and_fails_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _package(tmp_path)
    config = _config(tmp_path)
    session = ExternalModelDriveSession(config, workspace=tmp_path)
    model = _FakeModel()

    def mutate_artifact(*_args: object, **_kwargs: object) -> _FakeModel:
        assert config.policy_checkpoint is not None
        config.policy_checkpoint.write_bytes(b"changed-during-load")
        return model

    monkeypatch.setattr(
        external_model_drive,
        "create_driving_model_from_factory",
        mutate_artifact,
    )

    with pytest.raises(ValueError, match="model artifact changed"):
        _ExternalModelPolicy(session, config)

    assert model.closed is True
    session._close_garage_extensions()


def test_actuating_model_manifest_artifact_and_adapter_are_in_run_lineage(
    tmp_path: Path,
) -> None:
    _package(tmp_path)
    session = ExternalModelDriveSession(_config(tmp_path), workspace=tmp_path)

    reference = session._model_references()[0]

    assert reference["kind"] == "actuating_model_package"
    assert reference["actuation_authorized"] is True
    assert reference["manifest"]["sha256"]
    assert reference["artifact"]["sha256"]
    assert reference["adapter"]["factory"] == "carla_vision.torchscript_driver:create_driver"
    tracker = RunArtifactTracker(
        tmp_path / "runs",
        run_id="model-lineage",
        repository_root=tmp_path,
        model_refs=session._model_references(),
        package_names=(),
    )
    with tracker:
        pass
    assert verify_research_object(tracker.run_dir).external_reference_count == 3
    session._close_garage_extensions()


def test_verified_factory_executes_fingerprinted_source_not_cached_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "research_adapter"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    adapter = package / "policy.py"
    adapter.write_text(
        "VALUE = 'cached-old'\n\ndef create_driver(config):\n    return VALUE\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    cached = importlib.import_module("research_adapter.policy")
    assert cached.create_driver(None) == "cached-old"
    adapter.write_text(
        "VALUE = 'verified-new'\n\ndef create_driver(config):\n    return VALUE\n",
        encoding="utf-8",
    )
    reference = {
        "kind": "driving_model_adapter",
        "runtime": "python_factory",
        "factory": "research_adapter.policy:create_driver",
        **fingerprint_file(adapter),
    }

    factory, isolated_name = _load_verified_factory(
        "research_adapter.policy:create_driver",
        reference,
    )
    try:
        assert factory(None) == "verified-new"
        assert cached.create_driver(None) == "cached-old"
    finally:
        sys.modules.pop(isolated_name, None)
        sys.modules.pop("research_adapter.policy", None)
        sys.modules.pop("research_adapter", None)
