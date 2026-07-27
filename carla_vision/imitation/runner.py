"""Train and evaluate an RGB-plus-speed BehaviorAgent imitation baseline."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import BehaviorImitationDataset, split_summary
from .model import ImitationControlNet, ImitationModelConfig
from .objectives import compute_imitation_losses, control_metrics

IMITATION_TRAINING_SCHEMA_VERSION = "1.0"


def _parse_image_size(value: str) -> tuple[int, int]:
    normalized = value.lower().replace("x", " ").split()
    if len(normalized) != 2:
        raise argparse.ArgumentTypeError("image size must use HEIGHTxWIDTH")
    try:
        height, width = (int(item) for item in normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError("image dimensions must be integers") from error
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("image dimensions must be positive")
    return height, width


def _resolve_device(value: str) -> torch.device:
    requested = value.strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and (
        getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"training output directory is not empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _loader(
    dataset: BehaviorImitationDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
    )


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        batch["image"].to(device=device, dtype=torch.float32, non_blocking=True),
        batch["speed"].to(device=device, dtype=torch.float32, non_blocking=True),
        batch["target"].to(device=device, dtype=torch.float32, non_blocking=True),
    )


def _run_epoch(
    model: ImitationControlNet,
    loader: DataLoader[dict[str, Any]],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    steer_weight: float,
    longitudinal_weight: float,
    high_steer_boost: float,
    brake_boost: float,
    max_batches: int | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    batches = 0
    samples = 0
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        image, speed, target = _move_batch(batch, device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(image, speed)
            losses = compute_imitation_losses(
                outputs,
                target,
                steer_weight=steer_weight,
                longitudinal_weight=longitudinal_weight,
                high_steer_boost=high_steer_boost,
                brake_boost=brake_boost,
            )
            if optimizer is not None:
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        metrics = losses.detached()
        metrics.update(control_metrics(outputs, target))
        batch_samples = int(image.shape[0])
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value) * batch_samples
        samples += batch_samples
        batches += 1
    if batches == 0 or samples == 0:
        raise RuntimeError("data loader produced no batches")
    elapsed = time.perf_counter() - started
    result = {name: value / samples for name, value in totals.items()}
    result.update(
        {
            "batches": float(batches),
            "samples": float(samples),
            "seconds": elapsed,
            "samples_per_second": samples / max(elapsed, 1e-9),
        }
    )
    return result


def _checkpoint_payload(
    model: ImitationControlNet,
    *,
    args: argparse.Namespace,
    epoch: int,
    metrics: Mapping[str, Any],
    source_datasets: Sequence[Path],
    split_information: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": IMITATION_TRAINING_SCHEMA_VERSION,
        "task": "behavior_agent_control_imitation",
        "runtime_input_contract": "front_rgb_bgr_uint8_plus_speed_mps",
        "teacher_source": "BehaviorAgent.run_step",
        "model_config": model.config.as_dict(),
        "state_dict": model.state_dict(),
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "source_datasets": [str(path) for path in source_datasets],
        "split_summary": dict(split_information),
        "training_seed": int(args.seed),
        "augmentation": {
            "seed": int(args.augmentation_seed),
            "brightness": args.brightness,
            "contrast": args.contrast,
            "horizontal_flip_probability": args.horizontal_flip_probability,
        },
    }


def _synthetic_smoke(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    config = ImitationModelConfig(
        image_size_hw=args.image_size,
        speed_scale_mps=args.speed_scale_mps,
        encoder_channels=min(args.encoder_channels, 16),
        hidden_dim=min(args.hidden_dim, 32),
        dropout=0.0,
    )
    model = ImitationControlNet(config).to(device)
    height, width = args.image_size
    image = torch.rand(2, 3, height, width, device=device)
    speed = torch.tensor([[0.15], [0.5]], dtype=torch.float32, device=device)
    target = torch.tensor([[0.2, 0.4], [-0.3, -0.6]], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    outputs = model(image, speed)
    losses = compute_imitation_losses(outputs, target)
    optimizer.zero_grad(set_to_none=True)
    losses.total.backward()
    optimizer.step()
    return {
        "dry_run": True,
        "synthetic": True,
        "device": str(device),
        "parameter_count": model.parameter_count,
        "output_shapes": {name: list(value.shape) for name, value in outputs.items()},
        **losses.detached(),
        **control_metrics(outputs, target),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an RGB-plus-speed BehaviorAgent imitation-driving baseline."
    )
    parser.add_argument("--dataset", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, default=Path("models/imitation-baseline"))
    parser.add_argument("--image-size", type=_parse_image_size, default=(192, 320))
    parser.add_argument("--speed-scale-mps", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--augmentation-seed", type=int, default=23)
    parser.add_argument("--brightness", type=float, default=0.12)
    parser.add_argument("--contrast", type=float, default=0.12)
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.0)
    parser.add_argument("--steer-weight", type=float, default=2.0)
    parser.add_argument("--longitudinal-weight", type=float, default=1.0)
    parser.add_argument("--high-steer-boost", type=float, default=1.5)
    parser.add_argument("--brake-boost", type=float, default=2.0)
    parser.add_argument("--encoder-channels", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--skip-dataset-verification", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("epochs", "batch_size", "encoder_channels", "hidden_dim"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if args.speed_scale_mps <= 0.0:
        raise ValueError("speed_scale_mps must be positive")
    if not 0.0 <= args.val_fraction < 1.0 or not 0.0 <= args.test_fraction < 1.0:
        raise ValueError("split fractions must be in [0, 1)")
    if args.val_fraction + args.test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be less than 1")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if not 0.0 <= args.horizontal_flip_probability <= 1.0:
        raise ValueError("horizontal_flip_probability must be in [0, 1]")
    for name in (
        "brightness",
        "contrast",
        "steer_weight",
        "longitudinal_weight",
        "high_steer_boost",
        "brake_boost",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("max_batches must be positive")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    _set_seed(args.seed)
    device = _resolve_device(args.device)
    if not args.dataset:
        if not args.dry_run:
            raise ValueError("at least one --dataset is required for training")
        summary = _synthetic_smoke(args, device)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary

    roots = tuple(path.expanduser().resolve(strict=True) for path in args.dataset)
    common = {
        "roots": roots,
        "image_size": args.image_size,
        "speed_scale_mps": args.speed_scale_mps,
        "split_seed": args.seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "augmentation_seed": args.augmentation_seed,
        "brightness": args.brightness,
        "contrast": args.contrast,
        "horizontal_flip_probability": args.horizontal_flip_probability,
        "verify": not args.skip_dataset_verification,
        "max_samples": args.max_samples,
    }
    datasets = {
        "train": BehaviorImitationDataset(partition="train", augment=True, **common),
        "val": BehaviorImitationDataset(partition="val", augment=False, **common),
        "test": BehaviorImitationDataset(partition="test", augment=False, **common),
    }
    split_information = split_summary(datasets)
    if split_information["route_group_overlap"]:
        raise RuntimeError("route/seed leakage detected across imitation splits")
    if len(datasets["train"]) == 0:
        raise RuntimeError(
            "training split has no samples; add route groups or adjust split fractions"
        )

    loaders = {
        name: (
            _loader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=name == "train",
                seed=args.seed + index,
            )
            if len(dataset)
            else None
        )
        for index, (name, dataset) in enumerate(datasets.items())
    }
    config = ImitationModelConfig(
        image_size_hw=args.image_size,
        speed_scale_mps=args.speed_scale_mps,
        encoder_channels=args.encoder_channels,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    model = ImitationControlNet(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    output = _ensure_output(args.output)
    history: list[dict[str, Any]] = []
    best_score = math.inf
    best_epoch = 0
    common_epoch = {
        "device": device,
        "steer_weight": args.steer_weight,
        "longitudinal_weight": args.longitudinal_weight,
        "high_steer_boost": args.high_steer_boost,
        "brake_boost": args.brake_boost,
        "max_batches": args.max_batches,
    }
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model,
            loaders["train"],
            optimizer=optimizer,
            **common_epoch,
        )
        val_loader = loaders["val"]
        val_metrics = (
            _run_epoch(model, val_loader, optimizer=None, **common_epoch)
            if val_loader is not None
            else None
        )
        score = float((val_metrics or train_metrics)["loss"])
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        payload = _checkpoint_payload(
            model,
            args=args,
            epoch=epoch,
            metrics=record,
            source_datasets=roots,
            split_information=split_information,
        )
        torch.save(payload, output / "checkpoint_last.pt")
        if score < best_score:
            best_score = score
            best_epoch = epoch
            torch.save(payload, output / "checkpoint_best.pt")
        print(
            f"epoch={epoch}/{args.epochs} train_loss={train_metrics['loss']:.6f} "
            f"val_loss={score:.6f}",
            flush=True,
        )

    best_payload = torch.load(
        output / "checkpoint_best.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(best_payload["state_dict"], strict=True)
    test_loader = loaders["test"]
    test_metrics = (
        _run_epoch(model, test_loader, optimizer=None, **common_epoch)
        if test_loader is not None
        else None
    )
    summary = {
        "schema_version": IMITATION_TRAINING_SCHEMA_VERSION,
        "task": "behavior_agent_control_imitation",
        "device": str(device),
        "parameter_count": model.parameter_count,
        "source_datasets": [str(path) for path in roots],
        "output": str(output),
        "best_epoch": best_epoch,
        "best_validation_loss": best_score,
        "test": test_metrics,
        "split_summary": split_information,
        "model_config": config.as_dict(),
        "runtime_factory": "carla_vision.imitation.predictor:create_driver",
        "closed_loop_evaluation_completed": False,
    }
    _write_json(output / "history.json", history)
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


__all__ = [
    "IMITATION_TRAINING_SCHEMA_VERSION",
    "build_parser",
    "main",
    "run",
]


if __name__ == "__main__":
    raise SystemExit(main())
