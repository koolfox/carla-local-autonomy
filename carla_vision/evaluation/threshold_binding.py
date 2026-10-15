"""Bind a frozen validation operating point to the exact evaluated model/task."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..artifacts import fingerprint_file
from ..contracts import DetectorConfig
from ..dataset.verified import VerifiedDataset
from .contracts import EvaluationConfig


def bind_threshold(
    path: str | Path,
    *,
    detector: DetectorConfig,
    evaluation: EvaluationConfig,
    dataset: VerifiedDataset,
) -> tuple[float, Mapping[str, Any]]:
    # Lazy imports avoid the selector/evaluator/consumer import cycle.
    from ..thresholds.verified import load_verified_threshold_selection
    from .verified import load_verified_evaluation

    selection = load_verified_threshold_selection(path)
    source = load_verified_evaluation(
        Path(selection.descriptor["source_evaluation"]["manifest"]["path"]).parent
    )
    if source.resolved.get("threshold_selection") is not None:
        raise ValueError(
            "threshold calibration must use an independent, unbound validation evaluation"
        )
    if evaluation.purpose == "confirmatory" and selection.config.purpose != "confirmatory":
        raise ValueError("confirmatory evaluation requires a confirmatory threshold selection")
    old = source.resolved["detector"]
    current = detector.as_dict()
    for key in ("backend", "factory", "image_size", "options", "device"):
        if old.get(key) != current.get(key):
            raise ValueError(f"threshold detector identity differs: {key}")
    old_weights = old.get("weights")
    if bool(old_weights) != bool(detector.weights):
        raise ValueError("threshold checkpoint presence differs")
    if (
        old_weights
        and fingerprint_file(old_weights)["sha256"] != fingerprint_file(detector.weights)["sha256"]
    ):
        raise ValueError("threshold checkpoint digest differs")
    if source.dataset.coco["categories"] != dataset.coco["categories"]:
        raise ValueError("threshold ontology differs")
    if source.config.matching_iou != evaluation.matching_iou or dict(
        source.config.class_aliases
    ) != dict(evaluation.class_aliases):
        raise ValueError("threshold matching/ontology mapping contract differs")
    if evaluation.minimum_prediction_confidence > selection.operating_confidence:
        raise ValueError("prediction retention cutoff exceeds the selected operating threshold")
    return selection.operating_confidence, dict(selection.reference)
