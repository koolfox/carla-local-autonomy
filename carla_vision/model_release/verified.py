"""Consumer-side verification for immutable detector-model packages."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import fingerprint_file
from ..contracts import DetectorConfig
from ..verification import ArtifactIntegrityError, verify_research_object
from .contracts import MODEL_RELEASE_SCHEMA_VERSION, ModelReleaseConfig


class ModelIntegrityError(RuntimeError):
    """Raised when a model package cannot be trusted by a consumer."""


@dataclass(frozen=True)
class VerifiedModel:
    root: Path
    model_id: str
    backend: str
    architecture: str
    image_size: int
    weights_path: Path
    descriptor: Mapping[str, Any]
    release_config: ModelReleaseConfig
    reference: Mapping[str, Any]

    def detector_config(
        self,
        *,
        device: str,
        confidence: float,
    ) -> DetectorConfig:
        inference = self.release_config.inference
        return DetectorConfig(
            backend=inference.detector_backend,
            weights=self.weights_path,
            device=device,
            image_size=self.image_size,
            confidence=confidence,
            factory=inference.detector_factory,
            options=dict(inference.options),
        )


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ModelIntegrityError(f"could not read {name}: {error}") from error
    if not isinstance(value, Mapping):
        raise ModelIntegrityError(f"{name} must contain a JSON object")
    return value


def _artifact_path(
    root: Path,
    manifest: Mapping[str, Any],
    role: str,
) -> Path:
    matches = [
        entry
        for entry in manifest["artifacts"]
        if isinstance(entry, Mapping) and entry.get("role") == role
    ]
    if len(matches) != 1:
        raise ModelIntegrityError(f"model package must contain exactly one {role!r} artifact")
    return (root / str(matches[0]["path"])).resolve(strict=True)


def load_verified_model(path: str | Path) -> VerifiedModel:
    try:
        verification = verify_research_object(
            path,
            verify_references=True,
            deep=False,
            reject_unregistered=True,
        )
    except ArtifactIntegrityError as error:
        raise ModelIntegrityError(str(error)) from error
    root = Path(verification.root)
    manifest = _load_json(root / "manifest.json", "model tracker manifest")
    descriptor_path = _artifact_path(root, manifest, "model_release_manifest")
    config_path = _artifact_path(root, manifest, "model_release_configuration")
    weights_path = _artifact_path(root, manifest, "packaged_model_weights")
    descriptor = _load_json(descriptor_path, "model release manifest")
    config = ModelReleaseConfig.from_mapping(_load_json(config_path, "model release configuration"))
    if (
        descriptor.get("schema_version") != MODEL_RELEASE_SCHEMA_VERSION
        or descriptor.get("object_type") != "detector_model_release"
        or descriptor.get("status") != "complete"
    ):
        raise ModelIntegrityError("model release descriptor envelope is invalid")
    model_id = str(descriptor.get("model_id", ""))
    if model_id != verification.run_id or model_id != config.model_id:
        raise ModelIntegrityError("model identity disagrees across package files")
    if descriptor.get("runtime_sensor_contract") != "front_monocular_rgb_only":
        raise ModelIntegrityError("model runtime sensor contract is not monocular RGB")
    if descriptor.get("backend") != config.backend:
        raise ModelIntegrityError("model backend disagrees with release configuration")
    if descriptor.get("architecture") != config.architecture:
        raise ModelIntegrityError("model architecture disagrees with release configuration")
    if descriptor.get("inference") != config.as_dict()["inference"]:
        raise ModelIntegrityError("model inference adapter disagrees with release configuration")
    weights = descriptor.get("weights")
    if not isinstance(weights, Mapping):
        raise ModelIntegrityError("model weights descriptor is missing")
    weights_reference = fingerprint_file(weights_path)
    if (
        weights.get("path") != weights_path.relative_to(root).as_posix()
        or weights.get("sha256") != weights_reference["sha256"]
        or weights.get("size_bytes") != weights_reference["size_bytes"]
        or weights.get("checkpoint_role") != config.checkpoint_role
    ):
        raise ModelIntegrityError("packaged weights disagree with model descriptor")
    input_contract = descriptor.get("input")
    if not isinstance(input_contract, Mapping) or input_contract != config.as_dict()["input"]:
        raise ModelIntegrityError("model input contract disagrees with release configuration")
    ontology = descriptor.get("ontology")
    if not isinstance(ontology, Mapping) or not isinstance(
        ontology.get("categories"),
        list,
    ):
        raise ModelIntegrityError("model ontology mapping is missing")
    return VerifiedModel(
        root=root,
        model_id=model_id,
        backend=config.backend,
        architecture=config.architecture,
        image_size=config.input.image_size,
        weights_path=weights_path,
        descriptor=descriptor,
        release_config=config,
        reference={
            "kind": "verified_model_release",
            "model_id": model_id,
            "manifest": fingerprint_file(root / "manifest.json"),
            "model_manifest": fingerprint_file(descriptor_path),
            "weights": fingerprint_file(weights_path),
        },
    )


__all__ = [
    "ModelIntegrityError",
    "VerifiedModel",
    "load_verified_model",
]
