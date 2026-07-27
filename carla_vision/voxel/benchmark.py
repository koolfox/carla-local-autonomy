"""Benchmark present and future camera-only voxel predictions against teacher targets."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .training.dataset import (
    COMPACT_SEMANTIC_NAMES,
    compact_semantic_labels,
    load_voxel_episode,
)

VOXEL_BENCHMARK_SCHEMA_VERSION = "1.0"


def _read_prediction(path: Path) -> tuple[np.ndarray, tuple[float, ...], np.ndarray | None]:
    with np.load(path, allow_pickle=False) as payload:
        occupancy = np.asarray(payload["occupancy_probability"], dtype=np.float32)
        horizons = tuple(float(value) for value in np.asarray(payload["horizons_s"]).tolist())
        semantics = (
            np.asarray(payload["semantic_logits"], dtype=np.float32)
            if "semantic_logits" in payload and payload["semantic_logits"].size
            else None
        )
    return occupancy, horizons, semantics


def _read_teacher(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        occupancy = np.asarray(payload["occupancy"], dtype=np.int8)
        semantics = np.asarray(payload["semantics"], dtype=np.uint8)
    return occupancy, semantics


def _target_for_horizon(mapping: Mapping[float, int | None], horizon: float) -> int | None:
    for available, target in mapping.items():
        if abs(float(available) - horizon) <= 1e-4:
            return target
    return None


@dataclass(slots=True)
class BinaryOccupancyAccumulator:
    threshold: float
    uncertainty_band: float
    sample_count: int = 0
    known_voxel_count: int = 0
    occupied_intersection: int = 0
    occupied_union: int = 0
    free_true_positive: int = 0
    predicted_free_count: int = 0
    actual_free_count: int = 0
    brier_sum: float = 0.0
    uncertain_count: int = 0
    empty_intersection: int = 0
    empty_union: int = 0
    persistence_intersection: int = 0
    persistence_union: int = 0

    def update(
        self,
        probability: np.ndarray,
        teacher: np.ndarray,
        *,
        persistence: np.ndarray | None,
    ) -> None:
        probability = np.asarray(probability, dtype=np.float32)
        teacher = np.asarray(teacher, dtype=np.int8)
        if probability.shape != teacher.shape:
            raise ValueError("prediction and teacher occupancy shapes do not match")
        if not np.isfinite(probability).all() or np.any(
            (probability < 0.0) | (probability > 1.0)
        ):
            raise ValueError("prediction occupancy probabilities must be finite and in [0, 1]")
        known = teacher >= 0
        if not np.any(known):
            return
        target_occupied = teacher == 1
        predicted_occupied = probability >= self.threshold
        predicted_free = ~predicted_occupied
        target_free = teacher == 0
        self.sample_count += 1
        self.known_voxel_count += int(np.sum(known))
        self.occupied_intersection += int(np.sum(known & target_occupied & predicted_occupied))
        self.occupied_union += int(np.sum(known & (target_occupied | predicted_occupied)))
        self.free_true_positive += int(np.sum(known & target_free & predicted_free))
        self.predicted_free_count += int(np.sum(known & predicted_free))
        self.actual_free_count += int(np.sum(known & target_free))
        target_float = target_occupied.astype(np.float32)
        self.brier_sum += float(np.sum((probability[known] - target_float[known]) ** 2))
        uncertain = np.abs(probability - 0.5) <= self.uncertainty_band
        self.uncertain_count += int(np.sum(known & uncertain))

        empty_prediction = np.zeros_like(target_occupied, dtype=bool)
        self.empty_intersection += int(np.sum(known & target_occupied & empty_prediction))
        self.empty_union += int(np.sum(known & (target_occupied | empty_prediction)))
        if persistence is not None:
            persistence_array = np.asarray(persistence, dtype=np.int8)
            if persistence_array.shape != teacher.shape:
                raise ValueError("persistence occupancy shape does not match teacher")
            persistence_occupied = persistence_array == 1
            self.persistence_intersection += int(
                np.sum(known & target_occupied & persistence_occupied)
            )
            self.persistence_union += int(
                np.sum(known & (target_occupied | persistence_occupied))
            )

    @staticmethod
    def _ratio(numerator: int | float, denominator: int | float) -> float | None:
        if denominator <= 0:
            return None
        return float(numerator / denominator)

    def as_metrics(self) -> dict[str, Any]:
        occupied_iou = self._ratio(self.occupied_intersection, self.occupied_union)
        empty_iou = self._ratio(self.empty_intersection, self.empty_union)
        persistence_iou = self._ratio(
            self.persistence_intersection,
            self.persistence_union,
        )
        return {
            "sample_count": self.sample_count,
            "known_voxel_count": self.known_voxel_count,
            "occupied_iou": occupied_iou,
            "free_precision": self._ratio(
                self.free_true_positive,
                self.predicted_free_count,
            ),
            "free_recall": self._ratio(self.free_true_positive, self.actual_free_count),
            "brier_score": self._ratio(self.brier_sum, self.known_voxel_count),
            "uncertain_voxel_fraction": self._ratio(
                self.uncertain_count,
                self.known_voxel_count,
            ),
            "empty_space_occupied_iou": empty_iou,
            "persistence_occupied_iou": persistence_iou,
            "model_minus_empty_iou": (
                None if occupied_iou is None or empty_iou is None else occupied_iou - empty_iou
            ),
            "model_minus_persistence_iou": (
                None
                if occupied_iou is None or persistence_iou is None
                else occupied_iou - persistence_iou
            ),
            "raw_counts": asdict(self),
        }


@dataclass(slots=True)
class SemanticAccumulator:
    class_count: int
    confusion: np.ndarray

    @classmethod
    def create(cls, class_count: int) -> "SemanticAccumulator":
        if class_count != len(COMPACT_SEMANTIC_NAMES):
            raise ValueError(
                "semantic class count must match the configured compact semantic classes"
            )
        return cls(
            class_count=class_count,
            confusion=np.zeros((class_count, class_count), dtype=np.int64),
        )

    def update(self, logits: np.ndarray, raw_semantics: np.ndarray, occupancy: np.ndarray) -> None:
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim != 4 or logits.shape[0] != self.class_count:
            raise ValueError("semantic logits must have shape (C, Z, Y, X)")
        target = compact_semantic_labels(raw_semantics, occupancy)
        predicted = np.argmax(logits, axis=0).astype(np.int64)
        valid = target != 255
        if not np.any(valid):
            return
        target_values = target[valid].astype(np.int64)
        predicted_values = predicted[valid]
        if np.any((predicted_values < 0) | (predicted_values >= self.class_count)):
            raise ValueError("predicted semantic class is outside configured classes")
        bins = target_values * self.class_count + predicted_values
        self.confusion += np.bincount(
            bins,
            minlength=self.class_count * self.class_count,
        ).reshape(self.class_count, self.class_count)

    def as_metrics(self) -> dict[str, Any]:
        per_class: dict[str, float | None] = {}
        valid_ious: list[float] = []
        for index, name in enumerate(COMPACT_SEMANTIC_NAMES):
            true_positive = int(self.confusion[index, index])
            false_positive = int(np.sum(self.confusion[:, index])) - true_positive
            false_negative = int(np.sum(self.confusion[index, :])) - true_positive
            union = true_positive + false_positive + false_negative
            iou = None if union == 0 else true_positive / union
            per_class[name] = iou
            if iou is not None:
                valid_ious.append(float(iou))
        return {
            "semantic_miou": None if not valid_ious else float(np.mean(valid_ious)),
            "semantic_iou_by_class": per_class,
            "semantic_confusion": self.confusion.tolist(),
        }


def benchmark_voxel_runs(
    roots: Sequence[str | Path],
    *,
    threshold: float = 0.5,
    uncertainty_band: float = 0.08,
) -> dict[str, Any]:
    if not roots:
        raise ValueError("at least one voxel evaluation run is required")
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be finite and in (0, 1)")
    if not math.isfinite(uncertainty_band) or not 0.0 <= uncertainty_band < 0.5:
        raise ValueError("uncertainty_band must be finite and in [0, 0.5)")

    accumulators: dict[float, BinaryOccupancyAccumulator] = {}
    semantic_accumulators: dict[float, SemanticAccumulator] = {}
    run_summaries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for raw_root in roots:
        episode = load_voxel_episode(raw_root)
        by_frame = {int(record["frame"]): record for record in episode.records}
        evaluated_predictions = 0
        for record in episode.records:
            prediction_relative = record.get("prediction")
            if not prediction_relative:
                continue
            frame = int(record["frame"])
            prediction_path = episode.root / str(prediction_relative)
            occupancy, horizons, semantics = _read_prediction(prediction_path)
            if occupancy.ndim != 4 or occupancy.shape[1:] != episode.spec.shape:
                raise ValueError(f"prediction shape does not match grid: {prediction_path}")
            if occupancy.shape[0] != len(horizons):
                raise ValueError(f"prediction horizon count mismatch: {prediction_path}")
            if semantics is not None and (
                semantics.ndim != 5
                or semantics.shape[0] != len(horizons)
                or semantics.shape[2:] != episode.spec.shape
            ):
                raise ValueError(f"semantic prediction shape mismatch: {prediction_path}")
            future_mapping = episode.future_targets.get(frame)
            if future_mapping is None:
                skipped.append(
                    {"run": str(episode.root), "frame": frame, "reason": "missing_future_map"}
                )
                continue
            current_record = by_frame.get(frame)
            current_teacher = None
            if current_record and current_record.get("teacher_voxel"):
                current_teacher, _current_semantics = _read_teacher(
                    episode.root / str(current_record["teacher_voxel"])
                )
            prediction_used = False
            for horizon_index, horizon in enumerate(horizons):
                target_frame = _target_for_horizon(future_mapping, horizon)
                target_record = None if target_frame is None else by_frame.get(int(target_frame))
                if target_record is None or not target_record.get("teacher_voxel"):
                    skipped.append(
                        {
                            "run": str(episode.root),
                            "frame": frame,
                            "horizon_s": horizon,
                            "reason": "missing_teacher_target",
                        }
                    )
                    continue
                target_occupancy, target_semantics = _read_teacher(
                    episode.root / str(target_record["teacher_voxel"])
                )
                accumulator = accumulators.setdefault(
                    horizon,
                    BinaryOccupancyAccumulator(
                        threshold=threshold,
                        uncertainty_band=uncertainty_band,
                    ),
                )
                accumulator.update(
                    occupancy[horizon_index],
                    target_occupancy,
                    persistence=(current_teacher if horizon > 1e-6 else None),
                )
                if semantics is not None:
                    semantic = semantic_accumulators.setdefault(
                        horizon,
                        SemanticAccumulator.create(semantics.shape[1]),
                    )
                    semantic.update(
                        semantics[horizon_index],
                        target_semantics,
                        target_occupancy,
                    )
                prediction_used = True
            if prediction_used:
                evaluated_predictions += 1
        run_summaries.append(
            {
                "root": str(episode.root),
                "episode_id": episode.episode_id,
                "group_key": episode.group_key,
                "record_count": len(episode.records),
                "evaluated_prediction_frames": evaluated_predictions,
            }
        )

    horizon_metrics: dict[str, Any] = {}
    for horizon, accumulator in sorted(accumulators.items()):
        metrics = accumulator.as_metrics()
        if horizon in semantic_accumulators:
            metrics.update(semantic_accumulators[horizon].as_metrics())
        else:
            metrics.update(
                {
                    "semantic_miou": None,
                    "semantic_iou_by_class": None,
                    "semantic_confusion": None,
                }
            )
        horizon_metrics[f"{horizon:.3f}"] = metrics

    current = horizon_metrics.get("0.000", {})
    future_one_second = horizon_metrics.get("1.000", {})
    all_brier = [
        metrics["brier_score"]
        for metrics in horizon_metrics.values()
        if metrics.get("brier_score") is not None
    ]
    all_uncertainty = [
        metrics["uncertain_voxel_fraction"]
        for metrics in horizon_metrics.values()
        if metrics.get("uncertain_voxel_fraction") is not None
    ]
    return {
        "schema_version": VOXEL_BENCHMARK_SCHEMA_VERSION,
        "status": "complete" if horizon_metrics else "empty",
        "run_count": len(run_summaries),
        "runs": run_summaries,
        "threshold": threshold,
        "uncertainty_band": uncertainty_band,
        "horizons": horizon_metrics,
        "current_occupied_iou": current.get("occupied_iou"),
        "future_occupied_iou_1s": future_one_second.get("occupied_iou"),
        "future_persistence_iou_1s": future_one_second.get("persistence_occupied_iou"),
        "future_model_minus_persistence_iou_1s": future_one_second.get(
            "model_minus_persistence_iou"
        ),
        "mean_brier_score": None if not all_brier else float(np.mean(all_brier)),
        "maximum_uncertain_voxel_fraction": (
            None if not all_uncertainty else float(max(all_uncertainty))
        ),
        "evaluated_prediction_frame_count": sum(
            item["evaluated_prediction_frames"] for item in run_summaries
        ),
        "skipped_target_count": len(skipped),
        "skipped_targets": skipped,
        "flow_evaluation_available": False,
        "flow_note": "teacher artifacts still do not contain stable voxel motion vectors",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark current and future voxel predictions against CARLA teacher targets."
    )
    parser.add_argument("--run", action="append", type=Path, default=[])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--uncertainty-band", type=float, default=0.08)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = benchmark_voxel_runs(
        args.run,
        threshold=args.threshold,
        uncertainty_band=args.uncertainty_band,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["status"] == "complete" else 2


__all__ = [
    "BinaryOccupancyAccumulator",
    "SemanticAccumulator",
    "VOXEL_BENCHMARK_SCHEMA_VERSION",
    "benchmark_voxel_runs",
    "build_parser",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
