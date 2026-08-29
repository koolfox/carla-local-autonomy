from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from carla_vision.operator.external_model_drive import ExternalModelDriveStartConfig

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
        "model_trusted_code_acknowledged": False,
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
) -> None:
    directory = root / "models" / "road-policy"
    directory.mkdir(parents=True)
    (directory / artifact_name).write_bytes(artifact)
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "id": "road-policy",
        "name": "Road Policy",
        "version": "1.0",
        "role": "driving_policy",
        "runtime": runtime,
        "artifact": artifact_name,
        "devices": devices or ["cpu"],
        "inputs": {
            "image": {
                "width": 320,
                "height": 180,
                "color": "rgb",
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
            },
            "speed": {"enabled": True, "unit": "mps"},
        },
        "outputs": {"kind": "vehicle_control_v1"},
    }
    if factory is not None:
        manifest["factory"] = factory
    if expected_hash is not None:
        manifest["sha256"] = expected_hash
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
    manifest = config.manifest_config()
    assert manifest["garage_mode"] == "model"
    assert manifest["control_owner"] == "external_model"
    assert manifest["model_package_id"] == "road-policy"
    assert manifest["model_runtime"] == "torchscript_control_v1"


def test_python_factory_requires_separate_trusted_code_acknowledgement(tmp_path: Path) -> None:
    _package(
        tmp_path,
        runtime="python_factory",
        artifact_name="policy.pth",
        factory="research_models.policy:create_driver",
    )

    with pytest.raises(ValueError, match="trusted-code acknowledgement"):
        _config(tmp_path)

    config = _config(tmp_path, model_trusted_code_acknowledged=True)
    assert config.model_factory == "research_models.policy:create_driver"
    assert config.model_requires_trusted_code is True


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
