"""Canonical offline detector evaluation and thesis artifacts."""

from .contracts import EvaluationConfig, load_evaluation_config
from .runner import run_evaluation
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
