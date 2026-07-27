"""Model-neutral, artifact-first detector training."""

from .contracts import (
    TrainerBackend,
    TrainingConfig,
    TrainingRequest,
    TrainingResult,
    load_training_config,
)
from .runner import run_training

__all__ = [
    "TrainerBackend",
    "TrainingConfig",
    "TrainingRequest",
    "TrainingResult",
    "load_training_config",
    "run_training",
]
