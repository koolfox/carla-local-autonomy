"""Process-level deterministic seed application with explicit disclosure."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from ..scenarios.seeds import SeedBundle


def apply_training_seeds(
    seeds: SeedBundle,
    *,
    deterministic: bool,
) -> dict[str, Any]:
    random.seed(seeds.values["python"])
    np.random.seed(seeds.values["numpy"])
    torch.manual_seed(seeds.values["torch"])
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        torch.cuda.manual_seed_all(seeds.values["torch"])
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False
    return {
        "python_seed": seeds.values["python"],
        "numpy_seed": seeds.values["numpy"],
        "torch_seed": seeds.values["torch"],
        "backend_seed": seeds.values["model_init"],
        "torch_deterministic_algorithms": deterministic,
        "cuda_available": cuda_available,
        "cudnn_deterministic": (
            bool(torch.backends.cudnn.deterministic) if hasattr(torch.backends, "cudnn") else None
        ),
        "cudnn_benchmark": (
            bool(torch.backends.cudnn.benchmark) if hasattr(torch.backends, "cudnn") else None
        ),
        "limitation": (
            "Ultralytics exposes one trainer seed; model initialization, data "
            "order, and augmentation sub-seeds are recorded independently but "
            "the backend may internally derive them from the backend seed."
        ),
    }


__all__ = ["apply_training_seeds"]
