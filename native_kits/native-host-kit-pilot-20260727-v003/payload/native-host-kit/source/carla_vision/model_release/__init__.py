"""Immutable detector-model package contracts, promotion, and verification."""

from .contracts import (
    MODEL_RELEASE_CONFIG_SCHEMA_VERSION,
    ModelReleaseConfig,
    load_model_release_config,
)
from .package import MODEL_RELEASE_SCHEMA_VERSION, package_model
from .verified import ModelIntegrityError, VerifiedModel, load_verified_model

__all__ = [
    "MODEL_RELEASE_CONFIG_SCHEMA_VERSION",
    "MODEL_RELEASE_SCHEMA_VERSION",
    "ModelIntegrityError",
    "ModelReleaseConfig",
    "VerifiedModel",
    "load_model_release_config",
    "load_verified_model",
    "package_model",
]
