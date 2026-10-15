"""Wall-clock inference measurements with explicit asynchronous-device barriers."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def synchronize_device(device: str) -> None:
    family = str(device).split(":", 1)[0].lower()
    if family == "cpu":
        return
    if family not in {"cuda", "mps"}:
        raise ValueError("latency measurement requires an explicit cpu/cuda/mps device")
    import torch

    if family == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA timing requested but CUDA is unavailable")
        torch.cuda.synchronize(device)
    else:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS timing requested but MPS is unavailable")
        torch.mps.synchronize()


def measured_inference(call: Callable[[], T], *, device: str) -> tuple[T, float]:
    synchronize_device(device)
    started = time.perf_counter()
    result = call()
    synchronize_device(device)
    return result, time.perf_counter() - started


def timing_contract(device: str) -> dict[str, Any]:
    return {
        "clock": "perf_counter",
        "device": device,
        "scope": "adapter_preprocess_forward_postprocess_device_barrier",
        "excludes": ["file_decode", "simulator_render", "network", "display", "model_load"],
        "device_barrier": str(device).split(":")[0] in {"cuda", "mps"},
    }
