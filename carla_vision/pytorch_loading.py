"""Conservative PyTorch checkpoint loading for trusted model adapters.

This module intentionally does not know about CARLA observations, model architectures,
or preprocessing. A ``python_factory`` adapter owns those model-specific decisions.
The shared boundary here is deliberately smaller:

* deserialize checkpoint data with ``weights_only=True`` and CPU map-location;
* require an explicit state-dict location instead of guessing checkpoint structure;
* require exact state-dict keys for the constructed architecture;
* validate the requested inference device before moving model state;
* move the module to the selected device and put it into evaluation mode.

There is deliberately no fallback to ``weights_only=False``. Whole-module pickle
checkpoints and checkpoints requiring custom pickle globals need a purpose-built,
explicitly trusted adapter rather than a generic loader.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class PyTorchCheckpointError(RuntimeError):
    """Raised when checkpoint bytes cannot be loaded through the weights-only boundary."""


class PyTorchStateDictError(ValueError):
    """Raised when checkpoint state does not match the constructed model contract."""


def _import_torch() -> Any:
    try:
        import torch
    except (ImportError, OSError) as error:  # pragma: no cover - profile dependent
        raise RuntimeError(
            "PyTorch model loading requires the experimental PyTorch profile"
        ) from error
    return torch


def _regular_checkpoint_path(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise PyTorchCheckpointError("checkpoint must not be a symbolic link")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as error:
        raise PyTorchCheckpointError(f"checkpoint does not exist: {raw}") from error
    if not resolved.is_file():
        raise PyTorchCheckpointError(f"checkpoint is not a regular file: {resolved}")
    return resolved


def resolve_torch_device(device: str) -> Any:
    """Resolve one supported inference device and fail before model mutation.

    The product currently advertises CPU, CUDA and Apple MPS for direct driving
    policies. CUDA indices are validated when explicitly supplied.
    """

    torch = _import_torch()
    requested = str(device).strip()
    if not requested:
        raise ValueError("PyTorch device must not be empty")
    try:
        resolved = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"invalid PyTorch device {requested!r}") from error

    if resolved.type == "cpu":
        return resolved

    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if resolved.index is not None:
            count = int(torch.cuda.device_count())
            if not 0 <= resolved.index < count:
                raise RuntimeError(
                    f"CUDA device index {resolved.index} is unavailable; device count is {count}"
                )
        return resolved

    if resolved.type == "mps":
        backend = getattr(torch.backends, "mps", None)
        if backend is None or not backend.is_available():
            raise RuntimeError("MPS was requested but the PyTorch MPS backend is unavailable")
        return resolved

    raise ValueError(
        f"unsupported PyTorch inference device type {resolved.type!r}; "
        "supported device types are cpu, cuda and mps"
    )


def load_weights_only_checkpoint(path: str | Path) -> Any:
    """Deserialize one checkpoint on CPU without enabling arbitrary pickle globals.

    This function never retries with ``weights_only=False``. Loading onto CPU
    first also avoids allocating checkpoint storage directly on accelerator
    memory before the model contract has been validated.
    """

    torch = _import_torch()
    checkpoint = _regular_checkpoint_path(path)
    try:
        return torch.load(
            checkpoint,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
    except Exception as error:
        raise PyTorchCheckpointError(
            "checkpoint could not be loaded with weights_only=True; "
            "whole-module pickle and custom pickle globals are not supported "
            "by the generic state-dict loader"
        ) from error


def extract_state_dict(
    payload: Any,
    *,
    state_dict_key: str | None = None,
) -> Mapping[str, Any]:
    """Return the explicitly selected state dictionary from checkpoint payload data.

    ``state_dict_key=None`` means the checkpoint itself must be the state dict.
    Envelope checkpoints must name their top-level key explicitly; this helper
    does not guess between names such as ``state_dict`` or ``model_state_dict``.
    """

    selected = payload
    if state_dict_key is not None:
        key = str(state_dict_key).strip()
        if not key:
            raise ValueError("state_dict_key must not be empty")
        if not isinstance(payload, Mapping):
            raise PyTorchStateDictError(
                f"checkpoint must be a mapping to select state_dict_key={key!r}"
            )
        if key not in payload:
            available = sorted(str(candidate) for candidate in payload.keys())
            preview = ", ".join(available[:8])
            suffix = "" if len(available) <= 8 else ", ..."
            raise PyTorchStateDictError(
                f"checkpoint does not contain state_dict_key={key!r}; "
                f"available top-level keys: {preview}{suffix}"
            )
        selected = payload[key]

    if not isinstance(selected, Mapping):
        location = "checkpoint" if state_dict_key is None else f"checkpoint[{state_dict_key!r}]"
        raise PyTorchStateDictError(f"{location} must contain a state-dict mapping")

    invalid_keys = [key for key in selected if not isinstance(key, str) or not key]
    if invalid_keys:
        raise PyTorchStateDictError("state-dict keys must be non-empty strings")
    return selected


def _describe_key_mismatch(
    *,
    expected: set[str],
    actual: set[str],
) -> str:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)

    def preview(values: list[str]) -> str:
        if not values:
            return "none"
        body = ", ".join(values[:8])
        return body if len(values) <= 8 else f"{body}, ... (+{len(values) - 8})"

    return (
        "state-dict keys do not match the constructed model; "
        f"missing: {preview(missing)}; unexpected: {preview(unexpected)}"
    )


def prepare_module_for_inference(
    module: Any,
    state_dict: Mapping[str, Any],
    *,
    device: str,
) -> Any:
    """Strictly restore one ``torch.nn.Module`` and prepare it for inference.

    The returned value is the resolved ``torch.device``. The caller keeps
    ownership of the module instance it constructed.
    """

    torch = _import_torch()
    if not isinstance(module, torch.nn.Module):
        raise TypeError("state-dict target must be a torch.nn.Module")

    target_device = resolve_torch_device(device)
    expected_keys = set(module.state_dict().keys())
    actual_keys = set(state_dict.keys())
    if expected_keys != actual_keys:
        raise PyTorchStateDictError(
            _describe_key_mismatch(expected=expected_keys, actual=actual_keys)
        )

    try:
        module.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise PyTorchStateDictError(
            f"state dict is incompatible with {type(module).__qualname__}: {error}"
        ) from error

    try:
        module.to(target_device)
    except (RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"could not move {type(module).__qualname__} to PyTorch device {target_device}"
        ) from error
    module.eval()
    return target_device


def load_state_dict_for_inference(
    module: Any,
    checkpoint: str | Path,
    *,
    device: str,
    state_dict_key: str | None = None,
) -> Any:
    """Load a weights-only checkpoint into an already-constructed module.

    Architecture construction remains the responsibility of the trusted
    ``python_factory`` adapter. This helper only restores exact model state.
    """

    # Resolve device availability before deserializing or mutating the model.
    resolve_torch_device(device)
    payload = load_weights_only_checkpoint(checkpoint)
    state_dict = extract_state_dict(payload, state_dict_key=state_dict_key)
    return prepare_module_for_inference(module, state_dict, device=device)


__all__ = [
    "PyTorchCheckpointError",
    "PyTorchStateDictError",
    "extract_state_dict",
    "load_state_dict_for_inference",
    "load_weights_only_checkpoint",
    "prepare_module_for_inference",
    "resolve_torch_device",
]
