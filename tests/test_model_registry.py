from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

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


def test_registry_discovers_explicit_torchscript_driving_package_without_loading_it(
    tmp_path: Path,
) -> None:
    _write_package(
        tmp_path,
        "road-policy",
        {
            "schema_version": "1.0",
            "id": "road-policy",
            "name": "Road Policy",
            "version": "1.0.0",
            "role": "driving_policy",
            "runtime": "torchscript_control_v1",
            "artifact": "policy.pt",
            "devices": ["cpu", "cuda", "mps"],
            "inputs": {"image": "front_rgb", "speed": True},
            "outputs": {"kind": "vehicle_control_v1"},
            "source": "local test package",
        },
        artifact=b"not-a-real-torch-model-and-that-is-intentional",
    )

    registry = discover_model_packages(tmp_path)

    assert registry["invalid"] == []
    assert len(registry["packages"]) == 1
    package = registry["packages"][0]
    assert package["id"] == "road-policy"
    assert package["role"] == "driving_policy"
    assert package["runtime"] == "torchscript_control_v1"
    assert package["artifact"] == "models/road-policy/policy.pt"
    assert package["requiresTrustedCode"] is False
    assert package["devices"] == ["cpu", "cuda", "mps"]


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
        },
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
    payload = {
        "schema_version": "1.0",
        "id": "same-policy",
        "name": "Policy",
        "version": "1",
        "role": "driving_policy",
        "runtime": "torchscript_control_v1",
        "artifact": "policy.pt",
    }
    _write_package(tmp_path, "first", payload)
    _write_package(tmp_path, "second", {**payload, "name": "Policy Two"})

    registry = discover_model_packages(tmp_path)

    assert registry["packages"] == []
    assert len(registry["invalid"]) == 2
    assert all("duplicate model package id" in row["message"] for row in registry["invalid"])


def test_resolved_package_verifies_optional_manifest_sha256(tmp_path: Path) -> None:
    artifact = b"deterministic model bytes"
    digest = hashlib.sha256(artifact).hexdigest()
    _write_package(
        tmp_path,
        "verified",
        {
            "schema_version": "1.0",
            "id": "verified",
            "name": "Verified Policy",
            "version": "1",
            "role": "driving_policy",
            "runtime": "torchscript_control_v1",
            "artifact": "policy.pt",
            "sha256": digest,
        },
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
