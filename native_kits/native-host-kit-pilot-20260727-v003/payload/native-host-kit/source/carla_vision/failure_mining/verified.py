"""Consumer-side semantic verification for failure-mining releases."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import fingerprint_file
from ..evaluation.verified import load_verified_evaluation
from ..thresholds.verified import load_verified_threshold_selection
from ..verification import ArtifactIntegrityError, verify_research_object
from .contracts import FailureMiningConfig
from .miner import FAILURE_MINING_RUN_SCHEMA_VERSION, _failure_id


class FailureMiningIntegrityError(RuntimeError):
    """Raised when a failure-mining release cannot be trusted."""


@dataclass(frozen=True)
class VerifiedFailureMining:
    root: Path
    mining_id: str
    config: FailureMiningConfig
    descriptor: Mapping[str, Any]
    all_failures: tuple[Mapping[str, Any], ...]
    review_queue: tuple[Mapping[str, Any], ...]
    reference: Mapping[str, Any]


def _load_json(path: Path, name: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FailureMiningIntegrityError(f"could not read {name}: {error}") from error


def _load_jsonl(path: Path, name: str) -> tuple[Mapping[str, Any], ...]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise FailureMiningIntegrityError(
                        f"{name} line {line_number} must be an object"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FailureMiningIntegrityError(f"could not read {name}: {error}") from error
    return tuple(rows)


def _artifact_path(
    root: Path,
    manifest: Mapping[str, Any],
    role: str,
) -> Path:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise FailureMiningIntegrityError("failure-mining tracker artifacts must be an array")
    matches = [
        entry for entry in artifacts if isinstance(entry, Mapping) and entry.get("role") == role
    ]
    if len(matches) != 1:
        raise FailureMiningIntegrityError(
            f"failure mining must contain exactly one {role!r} artifact"
        )
    return (root / str(matches[0]["path"])).resolve(strict=True)


def _validate_failure(
    row: Mapping[str, Any],
    *,
    source_run_id: str,
    dataset_id: str,
    operating_confidence: float,
    allowed_types: set[str],
) -> None:
    try:
        image_id = int(row["image_id"])
        ground_truth_id = row["ground_truth_id"]
        prediction_index = row["prediction_index"]
        priority = float(row["priority_score"])
        base = float(row["base_severity"])
        rarity = float(row["rarity_factor"])
    except (KeyError, TypeError, ValueError) as error:
        raise FailureMiningIntegrityError(
            "failure candidate numeric contract is invalid"
        ) from error
    failure_type = str(row.get("failure_type", ""))
    expected_id = _failure_id(
        evaluation_run_id=source_run_id,
        image_id=image_id,
        failure_type=failure_type,
        ground_truth_id=(int(ground_truth_id) if ground_truth_id is not None else None),
        prediction_index=(int(prediction_index) if prediction_index is not None else None),
        threshold=operating_confidence,
    )
    if (
        row.get("failure_id") != expected_id
        or row.get("source_evaluation_id") != source_run_id
        or row.get("dataset_id") != dataset_id
        or row.get("review_status") != "pending"
        or failure_type not in allowed_types
        or not all(math.isfinite(value) and value > 0.0 for value in (priority, base, rarity))
        or float(row.get("operating_confidence", -1.0)) != operating_confidence
    ):
        raise FailureMiningIntegrityError(
            f"failure candidate {row.get('failure_id')!r} is inconsistent"
        )
    gt_present = ground_truth_id is not None
    prediction_present = prediction_index is not None
    expected_presence = {
        "false_negative": (True, False),
        "false_positive": (False, True),
        "misclassification": (True, True),
        "localization": (True, True),
    }[failure_type]
    if (gt_present, prediction_present) != expected_presence:
        raise FailureMiningIntegrityError(
            f"failure candidate {expected_id!r} object presence is invalid"
        )


def _review_template_ids(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            expected = {
                "failure_id",
                "selected_rank",
                "failure_type",
                "object_category",
                "decision",
                "corrected_failure_type",
                "reviewer",
                "review_notes",
                "proposed_remedy",
            }
            if set(reader.fieldnames or ()) != expected:
                raise FailureMiningIntegrityError("failure review template schema is invalid")
            ids = []
            for row in reader:
                if any(
                    row[field]
                    for field in (
                        "decision",
                        "corrected_failure_type",
                        "reviewer",
                        "review_notes",
                        "proposed_remedy",
                    )
                ):
                    raise FailureMiningIntegrityError(
                        "failure review template was modified after release"
                    )
                ids.append(row["failure_id"])
            return ids
    except (OSError, UnicodeError, csv.Error) as error:
        raise FailureMiningIntegrityError(
            f"could not read failure review template: {error}"
        ) from error


def load_verified_failure_mining(
    path: str | Path,
) -> VerifiedFailureMining:
    try:
        verification = verify_research_object(
            path,
            verify_references=True,
            deep=False,
            reject_unregistered=True,
        )
    except ArtifactIntegrityError as error:
        raise FailureMiningIntegrityError(str(error)) from error
    root = Path(verification.root)
    manifest = _load_json(root / "manifest.json", "failure tracker manifest")
    if not isinstance(manifest, Mapping):
        raise FailureMiningIntegrityError("failure tracker manifest must be an object")
    resolved_path = _artifact_path(
        root,
        manifest,
        "resolved_failure_mining_configuration",
    )
    descriptor_path = _artifact_path(
        root,
        manifest,
        "failure_mining_release_manifest",
    )
    all_path = _artifact_path(root, manifest, "all_failure_candidates")
    queue_path = _artifact_path(root, manifest, "failure_review_queue")
    template_path = _artifact_path(root, manifest, "failure_review_template")
    summary_path = _artifact_path(root, manifest, "failure_mining_summary")
    resolved = _load_json(resolved_path, "resolved failure mining config")
    descriptor = _load_json(descriptor_path, "failure mining descriptor")
    summary = _load_json(summary_path, "failure mining summary")
    if not all(isinstance(value, Mapping) for value in (resolved, descriptor, summary)):
        raise FailureMiningIntegrityError("failure-mining JSON envelope is invalid")
    config_raw = resolved.get("mining")
    if not isinstance(config_raw, Mapping):
        raise FailureMiningIntegrityError("resolved failure-mining preregistration is missing")
    try:
        config = FailureMiningConfig.from_mapping(config_raw)
    except (TypeError, ValueError) as error:
        raise FailureMiningIntegrityError(
            f"resolved failure-mining config is invalid: {error}"
        ) from error
    if (
        descriptor.get("schema_version") != FAILURE_MINING_RUN_SCHEMA_VERSION
        or descriptor.get("object_type") != "validation_failure_mining_release"
        or descriptor.get("status") != "complete"
    ):
        raise FailureMiningIntegrityError("failure-mining descriptor envelope is invalid")
    mining_id = str(descriptor.get("mining_id", ""))
    if mining_id != verification.run_id or mining_id != config.mining_id:
        raise FailureMiningIntegrityError("failure-mining identity is inconsistent")
    if descriptor.get("purpose") != "development":
        raise FailureMiningIntegrityError("failure-mining release is not marked development")
    source_reference = descriptor.get("source_evaluation")
    if not isinstance(source_reference, Mapping) or source_reference != resolved.get(
        "source_evaluation"
    ):
        raise FailureMiningIntegrityError("failure-mining source reference is inconsistent")
    source_manifest = source_reference.get("manifest")
    if not isinstance(source_manifest, Mapping):
        raise FailureMiningIntegrityError("failure-mining source manifest reference is missing")
    try:
        source = load_verified_evaluation(
            Path(str(source_manifest["path"])).resolve(strict=True).parent
        )
    except (OSError, KeyError, RuntimeError, ValueError) as error:
        raise FailureMiningIntegrityError(
            f"failure-mining source evaluation failed verification: {error}"
        ) from error
    if source.reference != source_reference or any(
        not partition.startswith("val") for partition in source.config.partitions
    ):
        raise FailureMiningIntegrityError("failure-mining source changed or is not validation-only")
    if (
        descriptor.get("dataset") != source.dataset.reference
        or descriptor.get("model") != source.resolved.get("model")
        or descriptor.get("validation_partitions") != list(source.config.partitions)
    ):
        raise FailureMiningIntegrityError("failure-mining source lineage is inconsistent")
    threshold_reference = descriptor.get("threshold_selection")
    if threshold_reference != resolved.get("threshold_selection"):
        raise FailureMiningIntegrityError("failure-mining threshold reference is inconsistent")
    if config.threshold_source == "selection_artifact":
        if not isinstance(threshold_reference, Mapping):
            raise FailureMiningIntegrityError("failure-mining threshold selection is missing")
        threshold_manifest = threshold_reference.get("manifest")
        if not isinstance(threshold_manifest, Mapping):
            raise FailureMiningIntegrityError("failure-mining threshold manifest is missing")
        try:
            threshold = load_verified_threshold_selection(
                Path(str(threshold_manifest["path"])).resolve(strict=True).parent
            )
        except (OSError, KeyError, RuntimeError, ValueError) as error:
            raise FailureMiningIntegrityError(
                f"failure threshold selection failed verification: {error}"
            ) from error
        if (
            threshold.reference != threshold_reference
            or threshold.descriptor["source_evaluation"] != source.reference
        ):
            raise FailureMiningIntegrityError("failure threshold selection lineage changed")
        operating_confidence = threshold.operating_confidence
    else:
        if threshold_reference is not None:
            raise FailureMiningIntegrityError(
                "evaluation-config mining cannot reference threshold selection"
            )
        operating_confidence = source.config.operating_confidence
    if (
        float(descriptor.get("operating_confidence", -1.0)) != operating_confidence
        or float(resolved.get("operating_confidence", -1.0)) != operating_confidence
    ):
        raise FailureMiningIntegrityError("failure-mining operating threshold changed")

    all_failures = _load_jsonl(all_path, "all failure candidates")
    queue = _load_jsonl(queue_path, "failure review queue")
    if not all_failures or not queue:
        raise FailureMiningIntegrityError("failure-mining candidate or queue artifact is empty")
    all_ids: set[str] = set()
    for row in all_failures:
        _validate_failure(
            row,
            source_run_id=source.run_id,
            dataset_id=source.dataset.dataset_id,
            operating_confidence=operating_confidence,
            allowed_types=set(config.failure_types),
        )
        failure_id = str(row["failure_id"])
        if failure_id in all_ids:
            raise FailureMiningIntegrityError(f"duplicate failure candidate ID {failure_id!r}")
        all_ids.add(failure_id)
    queue_ids = [str(row.get("failure_id", "")) for row in queue]
    if (
        len(queue_ids) != len(set(queue_ids))
        or not set(queue_ids) <= all_ids
        or len(queue) > config.max_candidates
        or [int(row.get("selected_rank", -1)) for row in queue] != list(range(1, len(queue) + 1))
    ):
        raise FailureMiningIntegrityError("failure review queue identity or ranking is invalid")
    all_by_id = {str(row["failure_id"]): row for row in all_failures}
    for queue_row in queue:
        source_row = all_by_id[str(queue_row["failure_id"])]
        if queue_row != source_row or queue_row.get("selected") is not True:
            raise FailureMiningIntegrityError("failure queue row differs from all-candidate record")
    image_counts = Counter(int(row["image_id"]) for row in queue)
    episode_counts = Counter(str(row["episode_key"]) for row in queue)
    if (
        max(image_counts.values()) > config.per_image_limit
        or max(episode_counts.values()) > config.per_episode_limit
    ):
        raise FailureMiningIntegrityError("failure queue violates diversity caps")
    if _review_template_ids(template_path) != queue_ids:
        raise FailureMiningIntegrityError("failure review template does not match queue order")
    if (
        descriptor.get("all_failure_count") != len(all_failures)
        or descriptor.get("review_queue_count") != len(queue)
        or summary.get("all_failure_count") != len(all_failures)
        or summary.get("review_queue_count") != len(queue)
        or descriptor.get("review_status") != "pending"
        or summary.get("review_status") != "pending"
    ):
        raise FailureMiningIntegrityError("failure-mining counts or review status are inconsistent")
    return VerifiedFailureMining(
        root=root,
        mining_id=mining_id,
        config=config,
        descriptor=descriptor,
        all_failures=all_failures,
        review_queue=queue,
        reference={
            "kind": "verified_failure_mining_release",
            "mining_id": mining_id,
            "manifest": fingerprint_file(root / "manifest.json"),
            "mining_manifest": fingerprint_file(descriptor_path),
            "review_queue": fingerprint_file(queue_path),
        },
    )


__all__ = [
    "FailureMiningIntegrityError",
    "VerifiedFailureMining",
    "load_verified_failure_mining",
]
