"""Train and evaluate a temporal RGB-only voxel occupancy baseline."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..contracts import VoxelGridSpec
from .dataset import (
    COMPACT_SEMANTIC_NAMES,
    TemporalVoxelDataset,
    split_summary,
)
from .losses import compute_voxel_losses
from .metrics import occupancy_metrics, semantic_mean_iou
from .model import TemporalVoxelModelConfig, TemporalVoxelNet

TRAINING_SCHEMA_VERSION = "1.0"


def _parse_horizons(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("horizons must be comma-separated numbers") from error
    if not values or any(not math.isfinite(item) or item < 0 for item in values):
        raise argparse.ArgumentTypeError("horizons must be finite and non-negative")
    if tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("horizons must be unique and increasing")
    return values


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
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _model_config(args: argparse.Namespace, spec: VoxelGridSpec) -> TemporalVoxelModelConfig:
    return TemporalVoxelModelConfig(
        horizons_s=args.horizons,
        grid_shape_zyx=spec.shape,
        semantic_classes=len(COMPACT_SEMANTIC_NAMES),
        encoder_channels=args.encoder_channels,
        recurrent_channels=args.recurrent_channels,
        seed_channels=args.seed_channels,
        decoder_channels=args.decoder_channels,
    )


def _loader(
    dataset: TemporalVoxelDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        batch["rgb_history"].to(device=device, dtype=torch.float32, non_blocking=True),
        batch["occupancy"].to(device=device, dtype=torch.int8, non_blocking=True),
        batch["semantics"].to(device=device, dtype=torch.uint8, non_blocking=True),
    )


def _run_epoch(
    model: TemporalVoxelNet,
    loader: DataLoader[dict[str, Any]],
    *,
    device: torch.device,
    horizons_s: tuple[float, ...],
    optimizer: torch.optim.Optimizer | None,
    semantic_weight: float,
    temporal_weight: float,
    focal_gamma: float,
    positive_weight: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    batches = 0
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        rgb, occupancy, semantics = _move_batch(batch, device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(rgb)
            losses = compute_voxel_losses(
                outputs,
                occupancy,
                semantics,
                semantic_weight=semantic_weight,
                temporal_weight=temporal_weight,
                focal_gamma=focal_gamma,
                positive_weight=positive_weight,
            )
            if optimizer is not None:
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        metrics = losses.detached()
        with torch.no_grad():
            metrics.update(occupancy_metrics(outputs["occupancy_logits"], occupancy, horizons_s))
            metrics["semantic_miou"] = semantic_mean_iou(
                outputs["semantic_logits"],
                semantics,
                occupancy,
                class_count=model.config.semantic_classes,
            )
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        batches += 1
    if batches == 0:
        raise RuntimeError("data loader produced no batches")
    elapsed = time.perf_counter() - started
    result = {name: value / batches for name, value in totals.items()}
    result["batches"] = float(batches)
    result["seconds"] = elapsed
    result["batches_per_second"] = batches / max(elapsed, 1e-9)
    return result


def _checkpoint_payload(
    model: TemporalVoxelNet,
    *,
    spec: VoxelGridSpec,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "model_config": model.config.as_dict(),
        "state_dict": model.state_dict(),
        "grid_spec": spec.as_dict(),
        "history_frames": args.history_frames,
        "image_size_hw": list(args.image_size),
        "horizons_s": list(args.horizons),
        "semantic_class_names": list(COMPACT_SEMANTIC_NAMES),
        "epoch": epoch,
        "metrics": metrics,
    }


def _synthetic_smoke(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    spec = VoxelGridSpec(
        x_min=0.0,
        x_max=8.0,
        y_min=-4.0,
        y_max=4.0,
        z_min=-1.0,
        z_max=3.0,
        resolution=1.0,
    )
    config = TemporalVoxelModelConfig(
        horizons_s=args.horizons,
        grid_shape_zyx=spec.shape,
        semantic_classes=len(COMPACT_SEMANTIC_NAMES),
        encoder_channels=min(args.encoder_channels, 32),
        recurrent_channels=min(args.recurrent_channels, 64),
        seed_channels=min(args.seed_channels, 32),
        decoder_channels=min(args.decoder_channels, 16),
    )
    model = TemporalVoxelNet(config).to(device)
    height, width = args.image_size
    rgb = torch.rand(1, args.history_frames, 3, height, width, device=device)
    occupancy = torch.zeros(1, len(args.horizons), *spec.shape, dtype=torch.int8, device=device)
    occupancy[..., 1:3, 3:5, 3:5] = 1
    occupancy[..., 0, :, :] = -1
    semantics = torch.full_like(occupancy, 255, dtype=torch.uint8)
    semantics[occupancy == 1] = 3
    outputs = model(rgb)
    losses = compute_voxel_losses(outputs, occupancy, semantics)
    metrics = occupancy_metrics(outputs["occupancy_logits"], occupancy, args.horizons)
    return {
        "dry_run": True,
        "synthetic": True,
        "device": str(device),
        "parameter_count": model.parameter_count,
        "occupancy_shape": list(outputs["occupancy_logits"].shape),
        "semantic_shape": list(outputs["semantic_logits"].shape),
        **losses.detached(),
        **metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a temporal RGB-only voxel occupancy baseline."
    )
    parser.add_argument("--dataset", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, default=Path("models/temporal-voxel-baseline"))
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument("--horizons", type=_parse_horizons, default=(0.0, 0.5, 1.0, 2.0))
    parser.add_argument("--image-size", type=_parse_image_size, default=(192, 320))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--semantic-weight", type=float, default=0.25)
    parser.add_argument("--temporal-weight", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--encoder-channels", type=int, default=64)
    parser.add_argument("--recurrent-channels", type=int, default=128)
    parser.add_argument("--seed-channels", type=int, default=64)
    parser.add_argument("--decoder-channels", type=int, default=24)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("history_frames", "epochs", "batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if args.semantic_weight < 0 or args.temporal_weight < 0:
        raise ValueError("loss weights must be non-negative")
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
        "history_frames": args.history_frames,
        "horizons_s": args.horizons,
        "image_size": args.image_size,
        "split_seed": args.seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "max_samples": args.max_samples,
    }
    train_dataset = TemporalVoxelDataset(partition="train", **common)
    val_dataset = TemporalVoxelDataset(partition="val", **common)
    test_dataset = TemporalVoxelDataset(partition="test", **common)
    available = {
        "train": len(train_dataset),
        "val": len(val_dataset),
        "test": len(test_dataset),
    }
    if len(train_dataset) == 0:
        raise RuntimeError(
            "training split has no samples; add more route/seed groups or adjust split fractions"
        )
    spec = train_dataset.spec
    model = TemporalVoxelNet(_model_config(args, spec)).to(device)
    train_loader = _loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = (
        _loader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        if len(val_dataset)
        else None
    )

    if args.dry_run:
        metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            horizons_s=args.horizons,
            optimizer=None,
            semantic_weight=args.semantic_weight,
            temporal_weight=args.temporal_weight,
            focal_gamma=args.focal_gamma,
            positive_weight=args.positive_weight,
            max_batches=1,
        )
        summary = {
            "dry_run": True,
            "synthetic": False,
            "device": str(device),
            "parameter_count": model.parameter_count,
            "grid": spec.as_dict(),
            "samples": available,
            "splits": split_summary(
                roots,
                split_seed=args.seed,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
            ),
            "metrics": metrics,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary

    output = _ensure_output(args.output)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            horizons_s=args.horizons,
            optimizer=optimizer,
            semantic_weight=args.semantic_weight,
            temporal_weight=args.temporal_weight,
            focal_gamma=args.focal_gamma,
            positive_weight=args.positive_weight,
            max_batches=args.max_batches,
        )
        val_metrics = None
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = _run_epoch(
                    model,
                    val_loader,
                    device=device,
                    horizons_s=args.horizons,
                    optimizer=None,
                    semantic_weight=args.semantic_weight,
                    temporal_weight=args.temporal_weight,
                    focal_gamma=args.focal_gamma,
                    positive_weight=args.positive_weight,
                    max_batches=args.max_batches,
                )
        score_metrics = val_metrics or train_metrics
        score = float(score_metrics["mean_occupied_iou"])
        epoch_record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(epoch_record)
        payload = _checkpoint_payload(
            model,
            spec=spec,
            args=args,
            epoch=epoch,
            metrics=score_metrics,
        )
        torch.save(payload, output / "checkpoint_last.pt")
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(payload, output / "checkpoint_best.pt")
        _write_json(output / "history.json", history)

    summary = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "device": str(device),
        "output": str(output),
        "parameter_count": model.parameter_count,
        "grid": spec.as_dict(),
        "samples": available,
        "splits": split_summary(
            roots,
            split_seed=args.seed,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
        ),
        "best_epoch": best_epoch,
        "best_mean_occupied_iou": best_score,
        "semantic_class_names": list(COMPACT_SEMANTIC_NAMES),
        "flow_head": False,
        "flow_note": "teacher artifacts do not yet provide stable voxel motion vectors",
    }
    _write_json(output / "summary.json", summary)
    _write_json(
        output / "config.json",
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
