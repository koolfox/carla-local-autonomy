"""Training backend factory and Ultralytics RT-DETR/YOLO implementation."""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from .contracts import TrainerBackend, TrainingConfig, TrainingRequest, TrainingResult


def _json_safe_metrics(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe_metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_metrics(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe_metrics(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


class UltralyticsTrainingBackend:
    def __init__(self, architecture: str) -> None:
        if architecture not in {"rtdetr", "yolo"}:
            raise ValueError(f"unsupported Ultralytics architecture {architecture!r}")
        self.architecture = architecture

    @property
    def name(self) -> str:
        return f"ultralytics-{self.architecture}"

    def train(self, request: TrainingRequest) -> TrainingResult:
        from ultralytics import RTDETR, YOLO

        model_class = RTDETR if self.architecture == "rtdetr" else YOLO
        model = model_class(str(request.pretrained_weights))
        config = request.config
        project = request.output_root
        run_name = "fit"
        arguments = {
            "data": str(request.data_config),
            "epochs": config.epochs,
            "imgsz": config.image_size,
            "batch": config.batch_size,
            "device": config.device,
            "workers": config.workers,
            "patience": config.patience,
            "optimizer": config.optimizer,
            "lr0": config.initial_learning_rate,
            "lrf": config.final_learning_rate_fraction,
            "weight_decay": config.weight_decay,
            "warmup_epochs": config.warmup_epochs,
            "amp": config.amp,
            "cache": config.cache,
            "deterministic": config.deterministic,
            "seed": request.backend_seed,
            "save": True,
            "save_period": config.save_period,
            "val": True,
            "plots": True,
            "project": str(project),
            "name": run_name,
            "exist_ok": False,
            "resume": False,
            "verbose": True,
            **dict(config.extra_options),
        }
        metrics_object = model.train(**arguments)
        output_dir = project / run_name
        best = output_dir / "weights" / "best.pt"
        last = output_dir / "weights" / "last.pt"
        if not best.is_file() or not last.is_file():
            raise RuntimeError("Ultralytics training completed without both best.pt and last.pt")
        metrics = getattr(metrics_object, "results_dict", None)
        if metrics is None and getattr(model, "trainer", None) is not None:
            validator = getattr(model.trainer, "validator", None)
            metrics = getattr(getattr(validator, "metrics", None), "results_dict", {})
        return TrainingResult(
            backend_name=self.name,
            output_dir=output_dir,
            best_checkpoint=best,
            last_checkpoint=last,
            metrics=cast(Mapping[str, Any], _json_safe_metrics(metrics or {})),
            metadata={
                "ultralytics_version": importlib.metadata.version("ultralytics"),
                "architecture": self.architecture,
                "arguments": _json_safe_metrics(arguments),
            },
        )


def _load_custom_factory(reference: str) -> Callable[[TrainingConfig], TrainerBackend]:
    module_name, separator, attribute_name = reference.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("custom trainer factory must use module:callable syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise TypeError(f"custom trainer factory {reference!r} is not callable")
    return cast(Callable[[TrainingConfig], TrainerBackend], factory)


def _validate_backend(candidate: Any) -> TrainerBackend:
    if not isinstance(candidate, TrainerBackend):
        raise TypeError("trainer backend must expose name and train(request)")
    return candidate


def create_training_backend(config: TrainingConfig) -> TrainerBackend:
    backend = config.backend
    if backend in {"rtdetr", "ultralytics-rtdetr"}:
        return UltralyticsTrainingBackend("rtdetr")
    if backend in {"yolo", "ultralytics-yolo"}:
        return UltralyticsTrainingBackend("yolo")
    if backend == "custom":
        if config.custom_factory is None:
            raise ValueError("custom training backend requires custom_factory")
        return _validate_backend(_load_custom_factory(config.custom_factory)(config))
    raise ValueError(f"unsupported training backend {backend!r}")


__all__ = [
    "UltralyticsTrainingBackend",
    "create_training_backend",
]
