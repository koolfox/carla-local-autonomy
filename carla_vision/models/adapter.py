from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelMetadata:
    """Identity information recorded with an inference run."""

    name: str
    version: str = "unknown"
    framework: str = "unknown"


class ModelAdapter(Protocol):
    """Stable boundary between CARLA observations and a model runtime.

    Implementations must not own CARLA clients, actors, sensors, or control.
    They only transform an observation into a model output.
    """

    @property
    def metadata(self) -> ModelMetadata: ...

    def infer(self, observation: Any) -> Any: ...

    def close(self) -> None: ...


class TorchModelAdapter:
    """Generic PyTorch checkpoint adapter.

    Any torch module exposing a normal callable interface can be loaded. The
    CARLA runtime does not need to know whether the model is YOLO, BEV, policy,
    or a custom research network.
    """

    def __init__(
        self,
        model: Any,
        *,
        metadata: ModelMetadata,
        device: str = "cpu",
    ) -> None:
        self._model = model
        self._device = device
        self._metadata = metadata
        self._closed = False

        if hasattr(self._model, "to"):
            self._model.to(device)
        if hasattr(self._model, "eval"):
            self._model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        factory: Any,
        *,
        metadata: ModelMetadata,
        device: str = "cpu",
    ) -> "TorchModelAdapter":
        import torch

        model = factory()
        state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if hasattr(model, "load_state_dict"):
            model.load_state_dict(state)
        return cls(model, metadata=metadata, device=device)

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def infer(self, observation: Any) -> Any:
        if self._closed:
            raise RuntimeError("model adapter is closed")
        if hasattr(self._model, "__call__"):
            return self._model(observation)
        raise TypeError("torch model is not callable")

    def close(self) -> None:
        self._closed = True
