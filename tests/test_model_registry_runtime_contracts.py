from __future__ import annotations

import json

import pytest

from carla_vision.model_registry import discover_model_packages, resolve_model_package


def _write_package(root, *, runtime="torchscript_control_v1", role="driving_policy"):
    package = root / "models" / "demo_policy"
    package.mkdir(parents=True)
    artifact = package / "model.pt"
    artifact.write_bytes(b"test-checkpoint")
    import hashlib

    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (package / "model.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "id": "demo-policy",
                "name": "Demo Policy",
                "version": "0.1.0",
                "role": role,
                "runtime": runtime,
                "artifact": "model.pt",
                "sha256": digest,
                "inputs": {"camera": "rgb"},
                "outputs": {"control": "throttle-steer-brake"},
            }
        ),
        encoding="utf-8",
    )


def test_registry_resolves_valid_runtime_package(tmp_path):
    _write_package(tmp_path)

    registry = discover_model_packages(tmp_path)
    assert registry["packages"][0]["id"] == "demo-policy"
    assert registry["packages"][0]["requiresTrustedCode"] is True

    resolved = resolve_model_package(tmp_path, "demo-policy", required_role="driving_policy")
    assert resolved.verify_artifact() == resolved.package.expected_sha256


def test_registry_rejects_wrong_runtime_role_pair(tmp_path):
    _write_package(tmp_path, runtime="torchscript_control_v1", role="detector")

    registry = discover_model_packages(tmp_path)

    assert registry["packages"] == []
    assert registry["invalid"][0]["errorType"] == "ValueError"
    with pytest.raises(KeyError):
        resolve_model_package(tmp_path, "demo-policy")
