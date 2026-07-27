"""Paired, immutable-RGB replay comparisons for detector models."""

from .contracts import (
    REPLAY_CONFIG_SCHEMA_VERSION,
    ReplayConfig,
    ReplayDetectorSpec,
    ReplayModelSpec,
    load_replay_config,
)
from .runner import run_replay
from .verified import ReplayIntegrityError, VerifiedReplay, load_verified_replay

__all__ = [
    "REPLAY_CONFIG_SCHEMA_VERSION",
    "ReplayConfig",
    "ReplayDetectorSpec",
    "ReplayIntegrityError",
    "ReplayModelSpec",
    "VerifiedReplay",
    "load_replay_config",
    "load_verified_replay",
    "run_replay",
]
