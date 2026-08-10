"""Train the temporal RGB-only occupancy baseline with sparse voxel-flow supervision."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..contracts import VoxelGridSpec
from .dataset import COMPACT_SEMANTIC_NAMES, split_summary
from .flow_dataset import TemporalVoxelFlowDataset
from .flow_losses import compute_flow_voxel_losses
from .flow_metrics import dynamic_occupancy_iou, flow_endpoint_metrics
from .flow_model import TemporalVoxelFlowNet
from .metrics import occupancy_metrics, semantic_mean_iou
from .model import TemporalVoxelModelConfig
from .runner import (
    _ensure_output,
    _parse_horizons,
    _parse_image_size,
    _resolve_device,
    _set_seed,
    _validate_args as _validate_base_args,
    _write_json,
)

FLOW_TRAINING_SCHEMA_VERSION = "1.0"


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
    dataset: TemporalVoxelFlowDataset,
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
        batch["flow_mps"].to(device=device, dtype=torch.float32, non_blocking=True),
        batch["flow_valid"].to(device=device, dtype=torch.bool, non_blocking=True),
    )


def _run_epoch(
    model: TemporalVoxelFlowNet,
    loader: DataLoader[dict[str, Any]],
    *,
    device: torch.device,
    horizons_s: tuple[float, ...],
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
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
        rgb, occupancy, semantics, flow, flow_valid = _move_batch(batch, device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(rgb)
            losses = compute_flow_voxel_losses(
                outputs,
                occupancy,
                semantics,
                flow,
                flow_valid,
                flow_weight=args.flow_weight,
                flow_beta=args.flow_beta,
                semantic_weight=args.semantic_weight,
                temporal_weight=args.temporal_weight,
                focal_gamma=args.focal_gamma,
                positive_weight=args.positive_weight,
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
            metrics.update(
                dynamic_occupancy_iou(
                    outputs["occupancy_logits"],
                    outputs["semantic_logits"],
                    occupancy,
                    semantics,
                    horizons_s,
                )
            )
            metrics.update(flow_endpoint_metrics(outputs["flow_mps"], flow, flow_valid))
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        batches += 1
    if batches == 0:
        raise RuntimeError("data loader produced no flow-supervised batches")
    elapsed = time.perf_counter() - started
    result = {name: value / batches for name, value in totals.items()}
    result["batches"] = float(batches)
    result["seconds"] = elapsed
    result["batches_per_second"] = batches / max(elapsed, 1e-9)
    return result


def _inference_latency(
    model: TemporalVoxelFlowNet,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    *,
    iterations: int = 20,
) -> dict[str, float | None]:
    batch = next(iter(loader), None)
    if batch is None:
        return {"inference_mean_ms": None, "inference_p95_ms": None}
    rgb = batch["rgb_history"].to(device=device, dtype=torch.float32)
    model.eval()
    values: list[float] = []
    with torch.inference_mode():
        for index in range(iterations + 3):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            model(rgb)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if index >= 3:
                values.append(elapsed_ms)
    values.sort()
    p95_index = min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)
    return {
        "inference_mean_ms": sum(values) / len(values),
        "inference_p95_ms": values[p95_index],
    }


def _checkpoint_payload(
    model: TemporalVoxelFlowNet,
    *,
    spec: VoxelGridSpec,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "schema_version": FLOW_TRAINING_SCHEMA_VERSION,
        "model_type": "TemporalVoxelFlowNet",
        "flow_head": True,
        "flow_target": "instantaneous_dynamic_velocity_mps_xyz",
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
    spec = VoxelGridSpec(x_min=0, x_max=8, y_min=-4, y_max=4, z_min=-1, z_max=3, resolution=1)
    config = TemporalVoxelModelConfig(
        horizons_s=args.horizons,
        grid_shape_zyx=spec.shape,
        semantic_classes=len(COMPACT_SEMANTIC_NAMES),
        encoder_channels=min(args.encoder_channels, 32),
        recurrent_channels=min(args.recurrent_channels, 64),
        seed_channels=min(args.seed_channels, 32),
        decoder_channels=min(args.decoder_channels, 16),
    )
    model = TemporalVoxelFlowNet(config).to(device)
    height, width = args.image_size
    rgb = torch.rand(1, args.history_frames, 3, height, width, device=device)
    occupancy = torch.zeros(1, len(args.horizons), *spec.shape, dtype=torch.int8, device=device)
    semantics = torch.full_like(occupancy, 255, dtype=torch.uint8)
    occupancy[..., 1:3, 3:5, 3:5] = 1
    semantics[occupancy == 1] = 3
    flow = torch.zeros(1, 3, *spec.shape, dtype=torch.float32, device=device)
    flow[:, 0, 1:3, 3:5, 3:5] = 4.0
    valid = torch.zeros(1, *spec.shape, dtype=torch.bool, device=device)
    valid[:, 1:3, 3:5, 3:5] = True
    outputs = model(rgb)
    losses = compute_flow_voxel_losses(outputs, occupancy, semantics, flow, valid)
    result = {
        "dry_run": True,
        "synthetic": True,
        "device": str(device),
        "parameter_count": model.parameter_count,
        "occupancy_shape": list(outputs["occupancy_logits"].shape),
        "semantic_shape": list(outputs["semantic_logits"].shape),
        "flow_shape": list(outputs["flow_mps"].shape),
        **losses.detached(),
    }
    result.update(flow_endpoint_metrics(outputs["flow_mps"], flow, valid))
    result.update(
        dynamic_occupancy_iou(
            outputs["occupancy_logits"], outputs["semantic_logits"], occupancy, semantics, args.horizons
        )
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train temporal RGB-only occupancy with voxel flow.")
    parser.add_argument("--dataset", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, default=Path("models/temporal-voxel-flow"))
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
    parser.add_argument("--flow-weight", type=float, default=0.20)
    parser.add_argument("--flow-beta", type=float, default=1.0)
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
    _validate_base_args(args)
    if args.flow_weight < 0 or args.flow_beta <= 0:
        raise ValueError("flow_weight must be non-negative and flow_beta positive")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    _set_seed(args.seed)
    device = _resolve_device(args.device)
    if not args.dataset:
        if not args.dry_run:
            raise ValueError("at least one --dataset is required for flow training")
        summary = _synthetic_smoke(args, device)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary

    roots = tuple(path.expanduser().resolve(strict=True) for path in args.dataset)
    common = dict(
        roots=roots,
        history_frames=args.history_frames,
        horizons_s=args.horizons,
        image_size=args.image_size,
        split_seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        max_samples=args.max_samples,
    )
    train_dataset = TemporalVoxelFlowDataset(partition="train", **common)
    val_dataset = TemporalVoxelFlowDataset(partition="val", **common)
    test_dataset = TemporalVoxelFlowDataset(partition="test", **common)
    available = {"train": len(train_dataset), "val": len(val_dataset), "test": len(test_dataset)}
    if not len(train_dataset):
        raise RuntimeError("training split has no samples with teacher_flow labels")
    spec = train_dataset.spec
    model = TemporalVoxelFlowNet(_model_config(args, spec)).to(device)
    train_loader = _loader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = _loader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False) if len(val_dataset) else None
    test_loader = _loader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False) if len(test_dataset) else None

    if args.dry_run:
        metrics = _run_epoch(model, train_loader, device=device, horizons_s=args.horizons, optimizer=None, args=args, max_batches=1)
        summary = {"dry_run": True, "synthetic": False, "device": str(device), "parameter_count": model.parameter_count, "grid": spec.as_dict(), "samples": available, "metrics": metrics}
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary

    output = _ensure_output(args.output)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, device=device, horizons_s=args.horizons, optimizer=optimizer, args=args, max_batches=args.max_batches)
        val_metrics = None
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = _run_epoch(model, val_loader, device=device, horizons_s=args.horizons, optimizer=None, args=args, max_batches=args.max_batches)
        score_metrics = val_metrics or train_metrics
        score = float(score_metrics["mean_dynamic_iou"])
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        payload = _checkpoint_payload(model, spec=spec, args=args, epoch=epoch, metrics=score_metrics)
        torch.save(payload, output / "checkpoint_last.pt")
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(payload, output / "checkpoint_best.pt")
        _write_json(output / "history.json", history)

    eval_loader = test_loader or val_loader or train_loader
    with torch.no_grad():
        evaluation = _run_epoch(model, eval_loader, device=device, horizons_s=args.horizons, optimizer=None, args=args, max_batches=args.max_batches)
    latency = _inference_latency(model, eval_loader, device)
    peak_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else None
    )
    summary = {
        "schema_version": FLOW_TRAINING_SCHEMA_VERSION,
        "device": str(device),
        "output": str(output),
        "parameter_count": model.parameter_count,
        "grid": spec.as_dict(),
        "samples": available,
        "splits": split_summary(roots, split_seed=args.seed, val_fraction=args.val_fraction, test_fraction=args.test_fraction),
        "best_epoch": best_epoch,
        "best_mean_dynamic_iou": best_score,
        "flow_head": True,
        "flow_target": "instantaneous_dynamic_velocity_mps_xyz",
        "evaluation": evaluation,
        "latency": latency,
        "peak_memory_mb": peak_memory_mb,
        "rgb_only_inference": True,
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "config.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
