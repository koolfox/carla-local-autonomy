"""Paired replay contracts with a lazily loaded plotting runner."""

from typing import Any

from .contracts import (
    REPLAY_CONFIG_SCHEMA_VERSION,
    ReplayConfig,
    ReplayDetectorSpec,
    ReplayModelSpec,
    load_replay_config,
)
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


def __getattr__(name: str) -> Any:
    """Avoid importing matplotlib and evaluation backends in the Operator UI."""

    if name == "run_replay":
        from .runner import run_replay

        return run_replay
    raise AttributeError(name)
