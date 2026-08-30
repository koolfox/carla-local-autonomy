from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import carla_vision.model_registry as model_registry
from carla_vision.model_registry import discover_model_packages, resolve_model_package


def _write_package(root: Path, name: str, payload: dict[str, object], artifact: bytes = b"model") -> Path:
    directory = root / "models" / name
    directory.mkdir(parents=True)
    artifact_path = directory / str(payload.get("artifact", "model.pt"))
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(artifact)
    (directory / "model.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return directory


def _torchscript_payload(**overrides: object) -> dict[str, object]:
    artifact = overrides.pop("artifact_bytes", b"model")
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "object_type": "runtime_model_package",
        "id": "road-policy",
        "name": "Road Policy",
        "version": "1.0.0",
        "role": "driving_policy",
        "runtime": "torchscript_control_v1",
        "artifact": "policy.pt",
        "sha256": hashlib.sha256(bytes(artifact)).hexdigest(),
        "devices": ["cpu"],
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
    payload.update(overrides)
    return payload


def test_registry_discovers_explicit_torchscript_driving_package_without_loading_it(
    tmp_path: Path,
) -> None:
    artifact = b"not-a-real-torch-model-and-that-is-intentional"
    payload = _torchscript_payload(
        artifact_bytes=artifact,
        devices=["cpu", "cuda", "mps"],
        source="local test package",
    )
    _write_package(
        tmp_path,
        "road-policy",
        payload,
        artifact=artifact,
    )

    registry = discover_model_packages(tmp_path)

    assert registry["invalid"] == []
    assert len(registry["packages"]) == 1
    package = registry["packages"][0]
    assert package["id"] == "road-policy"
    assert package["role"] == "driving_policy"
    assert package["runtime"] == "torchscript_control_v1"
    assert package["artifact"] == "models/road-policy/policy.pt"
    assert package["requiresTrustedCode"] is True
    assert package["devices"] == ["cpu", "cuda", "mps"]
    assert package["inputs"]["image"]["width"] == 320


def test_unmanifested_checkpoints_are_not_advertised_as_models(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "legacy.pt").write_bytes(b"legacy")
    legacy_dir = models / "old-experiment"
    legacy_dir.mkdir()
    (legacy_dir / "checkpoint.pt").write_bytes(b"legacy")

    registry = discover_model_packages(tmp_path)

    assert registry == {"schema_version": "1.0", "packages": [], "invalid": []}


def test_python_factory_is_explicitly_marked_as_trusted_code(tmp_path: Path) -> None:
    artifact = b"python-model"
    _write_package(
        tmp_path,
        "external-policy",
        {
            "schema_version": "1.0",
            "id": "external-policy",
            "name": "External Policy",
            "version": "2026.08",
            "role": "driving_policy",
            "runtime": "python_factory",
            "artifact": "checkpoint.pth",
            "factory": "research_models.external:create_driver",
            "devices": ["cpu"],
            "sha256": hashlib.sha256(artifact).hexdigest(),
            "inputs": {"kind": "model_observation_v1", "options": {"gain": 0.4}},
            "outputs": {"kind": "vehicle_control_v1"},
        },
        artifact=artifact,
    )

    package = discover_model_packages(tmp_path)["packages"][0]

    assert package["factory"] == "research_models.external:create_driver"
    assert package["requiresTrustedCode"] is True


def test_registry_reports_invalid_package_instead_of_exposing_unsafe_artifact_path(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "models" / "unsafe"
    directory.mkdir(parents=True)
    (tmp_path / "outside.pt").write_bytes(b"outside")
    (directory / "model.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "id": "unsafe",
                "name": "Unsafe",
                "version": "1",
                "role": "driving_policy",
                "runtime": "torchscript_control_v1",
                "artifact": "../../outside.pt",
            }
        ),
        encoding="utf-8",
    )

    registry = discover_model_packages(tmp_path)

    assert registry["packages"] == []
    assert registry["invalid"][0]["path"] == "models/unsafe"
    assert "safe relative path" in registry["invalid"][0]["message"]


def test_duplicate_model_ids_are_not_runnable(tmp_path: Path) -> None:
    payload = _torchscript_payload(id="same-policy", name="Policy", version="1")
    _write_package(tmp_path, "first", payload)
    _write_package(tmp_path, "second", {**payload, "name": "Policy Two"})

    registry = discover_model_packages(tmp_path)

    assert registry["packages"] == []
    assert len(registry["invalid"]) == 2
    assert all("duplicate model package id" in row["message"] for row in registry["invalid"])


def test_resolved_package_verifies_required_manifest_sha256(tmp_path: Path) -> None:
    artifact = b"deterministic model bytes"
    digest = hashlib.sha256(artifact).hexdigest()
    _write_package(
        tmp_path,
        "verified",
        _torchscript_payload(
            id="verified",
            name="Verified Policy",
            version="1",
            sha256=digest,
        ),
        artifact=artifact,
    )

    resolved = resolve_model_package(tmp_path, "verified", required_role="driving_policy")
    assert resolved.verify_artifact() == digest

    resolved.artifact_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        resolved.verify_artifact()


def test_role_mismatch_is_rejected_when_resolving(tmp_path: Path) -> None:
    _write_package(
        tmp_path,
        "detector",
        {
            "schema_version": "1.0",
            "id": "detector",
            "name": "Detector",
            "version": "1",
            "role": "detector",
            "runtime": "ultralytics",
            "artifact": "weights.pt",
        },
    )

    with pytest.raises(ValueError, match="expected 'driving_policy'"):
        resolve_model_package(tmp_path, "detector", required_role="driving_policy")


def test_runtime_incompatible_torchscript_contract_is_not_advertised(tmp_path: Path) -> None:
    artifact = b"invalid-contract"
    payload = _torchscript_payload(artifact_bytes=artifact)
    payload["inputs"] = {"image": "front_rgb", "speed": True}
    _write_package(tmp_path, "invalid-contract", payload, artifact=artifact)

    registry = discover_model_packages(tmp_path)

    assert registry["packages"] == []
    assert "inputs.image must be an object" in registry["invalid"][0]["message"]


def test_driving_package_without_declared_artifact_digest_is_not_advertised(
    tmp_path: Path,
) -> None:
    payload = _torchscript_payload()
    payload.pop("sha256")
    _write_package(tmp_path, "missing-digest", payload)

    registry = discover_model_packages(tmp_path)

    assert registry["packages"] == []
    assert "require an artifact sha256" in registry["invalid"][0]["message"]


def test_verified_detector_release_descriptor_is_not_a_runtime_registry_error(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "models" / "released-detector"
    directory.mkdir(parents=True)
    (directory / "model.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "object_type": "detector_model_release",
                "status": "complete",
                "model_id": "released-detector",
                "weights": {"path": "weights/model.pt"},
            }
        ),
        encoding="utf-8",
    )

    assert discover_model_packages(tmp_path) == {
        "schema_version": "1.0",
        "packages": [],
        "invalid": [],
    }


def test_stable_fingerprint_failure_blocks_artifact_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = b"stable"
    _write_package(
        tmp_path,
        "stable",
        _torchscript_payload(
            artifact_bytes=artifact,
            id="stable",
            sha256=hashlib.sha256(artifact).hexdigest(),
        ),
        artifact=artifact,
    )
    resolved = resolve_model_package(tmp_path, "stable")

    def changed_while_hashing(_path: object) -> dict[str, object]:
        raise RuntimeError("artifact changed while hashing")

    monkeypatch.setattr(model_registry, "fingerprint_file", changed_while_hashing)
    with pytest.raises(RuntimeError, match="changed while hashing"):
        resolved.verify_artifact_reference()
