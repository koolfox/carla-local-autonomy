"""Immutable model contracts with packaging dependencies loaded on demand."""

from typing import Any

from .contracts import (
    MODEL_RELEASE_CONFIG_SCHEMA_VERSION,
    MODEL_RELEASE_SCHEMA_VERSION,
    ModelReleaseConfig,
    load_model_release_config,
)
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


def __getattr__(name: str) -> Any:
    if name == "package_model":
        from . import package

        return getattr(package, name)
    raise AttributeError(name)
