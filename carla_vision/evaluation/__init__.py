"""Canonical evaluation contracts with the runner loaded on demand."""

from typing import Any

from .contracts import EvaluationConfig, load_evaluation_config
from .verified import (
    EvaluationIntegrityError,
    VerifiedEvaluation,
    load_verified_evaluation,
)

__all__ = [
    "EvaluationConfig",
    "EvaluationIntegrityError",
    "VerifiedEvaluation",
    "load_evaluation_config",
    "load_verified_evaluation",
    "run_evaluation",
]


def __getattr__(name: str) -> Any:
    if name == "run_evaluation":
        from .runner import run_evaluation

        return run_evaluation
    raise AttributeError(name)
