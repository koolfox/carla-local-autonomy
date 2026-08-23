"""Consumer-side semantic verification for canonical detector evaluations."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import fingerprint_file
from ..dataset.verified import VerifiedDataset, load_verified_dataset
from ..verification import ArtifactIntegrityError, verify_research_object
from .contracts import EVALUATION_RUN_SCHEMA_VERSION, EvaluationConfig
from .metrics import operating_metrics


class EvaluationIntegrityError(RuntimeError):
    """Raised when a completed evaluation cannot be trusted by a consumer."""


@dataclass(frozen=True)
class VerifiedEvaluation:
    root: Path
    run_id: str
    config: EvaluationConfig
    resolved: Mapping[str, Any]
    summary: Mapping[str, Any]
    predictions: tuple[Mapping[str, Any], ...]
    frame_rows: tuple[Mapping[str, Any], ...]
    dataset: VerifiedDataset
    reference: Mapping[str, Any]


def _load_json(path: Path, name: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationIntegrityError(f"could not read {name}: {error}") from error


def _load_jsonl(path: Path, name: str) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise EvaluationIntegrityError(f"{name} line {line_number} must be an object")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationIntegrityError(f"could not read {name}: {error}") from error
    return tuple(rows)


def _artifact_path(
    root: Path,
    manifest: Mapping[str, Any],
    role: str,
) -> Path:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise EvaluationIntegrityError("evaluation tracker artifacts must be an array")
    matches = [
        entry for entry in artifacts if isinstance(entry, Mapping) and entry.get("role") == role
    ]
    if len(matches) != 1:
        raise EvaluationIntegrityError(f"evaluation must contain exactly one {role!r} artifact")
    return (root / str(matches[0]["path"])).resolve(strict=True)


def _source_dataset(
    resolved: Mapping[str, Any],
) -> tuple[VerifiedDataset, Mapping[str, Any]]:
    reference = resolved.get("dataset")
    if not isinstance(reference, Mapping):
        raise EvaluationIntegrityError("evaluation dataset reference is missing")
    manifest_reference = reference.get("manifest")
    if not isinstance(manifest_reference, Mapping):
        raise EvaluationIntegrityError("evaluation dataset tracker reference is missing")
    path_value = manifest_reference.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise EvaluationIntegrityError("evaluation dataset manifest path is missing")
    try:
        dataset = load_verified_dataset(Path(path_value).resolve(strict=True).parent)
    except (OSError, RuntimeError, ValueError) as error:
        raise EvaluationIntegrityError(
            f"evaluation source dataset failed verification: {error}"
        ) from error
    if dataset.dataset_id != reference.get("dataset_id") or dataset.reference != reference:
        raise EvaluationIntegrityError("evaluation dataset identity or fingerprint changed")
    return dataset, reference


def _validate_prediction(
    prediction: Mapping[str, Any],
    *,
    image_ids: set[int],
    category_ids: set[int],
    minimum_confidence: float,
) -> None:
    try:
        image_id = int(prediction["image_id"])
        category_id = int(prediction["category_id"])
        score = float(prediction["score"])
        bbox = prediction["bbox"]
        values = tuple(float(value) for value in bbox)
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationIntegrityError("evaluation canonical prediction is malformed") from error
    if (
        image_id not in image_ids
        or category_id not in category_ids
        or not math.isfinite(score)
        or not minimum_confidence <= score <= 1.0
        or len(values) != 4
        or not all(math.isfinite(value) for value in values)
        or values[0] < 0.0
        or values[1] < 0.0
        or values[2] <= 0.0
        or values[3] <= 0.0
    ):
        raise EvaluationIntegrityError("evaluation canonical prediction violates its contract")


def load_verified_evaluation(path: str | Path) -> VerifiedEvaluation:
    try:
        verification = verify_research_object(
            path,
            verify_references=True,
            deep=False,
            reject_unregistered=True,
        )
    except ArtifactIntegrityError as error:
        raise EvaluationIntegrityError(str(error)) from error
    root = Path(verification.root)
    manifest = _load_json(root / "manifest.json", "evaluation tracker manifest")
    if not isinstance(manifest, Mapping):
        raise EvaluationIntegrityError("evaluation tracker manifest must be an object")
    resolved_path = _artifact_path(
        root,
        manifest,
        "resolved_evaluation_configuration",
    )
    summary_path = _artifact_path(root, manifest, "evaluation_summary")
    predictions_path = _artifact_path(
        root,
        manifest,
        "canonical_coco_predictions",
    )
    frames_path = _artifact_path(
        root,
        manifest,
        "per_image_prediction_log",
    )
    resolved = _load_json(resolved_path, "resolved evaluation configuration")
    summary = _load_json(summary_path, "evaluation summary")
    predictions_raw = _load_json(predictions_path, "canonical predictions")
    frame_rows = _load_jsonl(frames_path, "per-image prediction log")
    if (
        not isinstance(resolved, Mapping)
        or not isinstance(summary, Mapping)
        or not isinstance(predictions_raw, list)
        or any(not isinstance(item, Mapping) for item in predictions_raw)
    ):
        raise EvaluationIntegrityError("evaluation JSON artifact envelope is invalid")
    if (
        resolved.get("schema_version") != EVALUATION_RUN_SCHEMA_VERSION
        or resolved.get("object_type") != "offline_detector_evaluation"
        or summary.get("schema_version") != EVALUATION_RUN_SCHEMA_VERSION
        or summary.get("status") != "success"
    ):
        raise EvaluationIntegrityError("evaluation artifact envelope is invalid")
    config_raw = resolved.get("evaluation")
    if not isinstance(config_raw, Mapping):
        raise EvaluationIntegrityError("resolved evaluation contract is missing")
    try:
        config = EvaluationConfig.from_mapping(config_raw)
    except (TypeError, ValueError) as error:
        raise EvaluationIntegrityError(
            f"resolved evaluation contract is invalid: {error}"
        ) from error
    if (
        summary.get("evaluation_id") != config.evaluation_id
        or summary.get("purpose") != config.purpose
    ):
        raise EvaluationIntegrityError("evaluation identity or purpose is inconsistent")
    dataset, dataset_reference = _source_dataset(resolved)
    selected_samples = [
        sample for sample in dataset.dataset["samples"] if str(sample["split"]) in config.partitions
    ]
    if not selected_samples:
        raise EvaluationIntegrityError("evaluation selected no source samples")
    if resolved.get("selected_sample_count") != len(selected_samples) or len(frame_rows) != len(
        selected_samples
    ):
        raise EvaluationIntegrityError("evaluation sample count is inconsistent")

    image_ids: list[int] = []
    flattened_predictions: list[Mapping[str, Any]] = []
    for position, (sample, frame) in enumerate(
        zip(selected_samples, frame_rows, strict=True),
        start=1,
    ):
        expected = {
            "position": position,
            "sample_id": str(sample["sample_id"]),
            "image_id": int(sample["image_id"]),
            "partition": str(sample["split"]),
            "scenario_id": str(sample["scenario_id"]),
            "episode_id": str(sample["episode_id"]),
            "carla_frame": int(sample["carla_frame"]),
        }
        if any(frame.get(key) != value for key, value in expected.items()):
            raise EvaluationIntegrityError("evaluation frame log differs from dataset sample order")
        mapped = frame.get("mapped_predictions")
        if not isinstance(mapped, list) or any(not isinstance(item, Mapping) for item in mapped):
            raise EvaluationIntegrityError("evaluation frame mapped predictions are invalid")
        if frame.get("mapped_prediction_count") != len(mapped):
            raise EvaluationIntegrityError(
                "evaluation frame mapped prediction count is inconsistent"
            )
        flattened_predictions.extend(mapped)
        image_ids.append(int(sample["image_id"]))
    predictions = tuple(predictions_raw)
    if flattened_predictions != list(predictions):
        raise EvaluationIntegrityError("canonical predictions differ from per-image frame records")
    category_ids = {int(category["id"]) for category in dataset.coco["categories"]}
    for prediction in predictions:
        _validate_prediction(
            prediction,
            image_ids=set(image_ids),
            category_ids=category_ids,
            minimum_confidence=config.minimum_prediction_confidence,
        )

    selected_image_ids = set(image_ids)
    ground_truth = [
        annotation
        for annotation in dataset.coco["annotations"]
        if int(annotation["image_id"]) in selected_image_ids
    ]
    categories = {
        int(category["id"]): str(category["name"]) for category in dataset.coco["categories"]
    }
    operating = operating_metrics(
        ground_truth,
        predictions,
        image_ids=image_ids,
        categories=categories,
        confidence_threshold=config.operating_confidence,
        iou_threshold=config.matching_iou,
    )
    if dict(operating.overall) != summary.get("operating_point"):
        raise EvaluationIntegrityError("evaluation operating metrics do not reproduce")
    selection = summary.get("selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("partitions") != list(config.partitions)
        or selection.get("images") != len(image_ids)
        or selection.get("ground_truth_annotations") != len(ground_truth)
    ):
        raise EvaluationIntegrityError("evaluation selection summary is inconsistent")
    if summary.get("dataset") != dataset_reference:
        raise EvaluationIntegrityError("evaluation summary dataset reference changed")
    if summary.get("model") != resolved.get("model"):
        raise EvaluationIntegrityError("evaluation model reference changed between artifacts")
    return VerifiedEvaluation(
        root=root,
        run_id=verification.run_id,
        config=config,
        resolved=resolved,
        summary=summary,
        predictions=predictions,
        frame_rows=frame_rows,
        dataset=dataset,
        reference={
            "kind": "verified_detector_evaluation",
            "run_id": verification.run_id,
            "manifest": fingerprint_file(root / "manifest.json"),
            "resolved_configuration": fingerprint_file(resolved_path),
            "predictions": fingerprint_file(predictions_path),
            "summary": fingerprint_file(summary_path),
        },
    )


__all__ = [
    "EvaluationIntegrityError",
    "VerifiedEvaluation",
    "load_verified_evaluation",
]
