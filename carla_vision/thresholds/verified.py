"""Consumer-side verification for sealed validation threshold selections."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import fingerprint_file
from ..evaluation.verified import load_verified_evaluation
from ..verification import ArtifactIntegrityError, verify_research_object
from .contracts import ThresholdSelectionConfig
from .selector import THRESHOLD_SELECTION_RUN_SCHEMA_VERSION, _select


class ThresholdSelectionIntegrityError(RuntimeError):
    """Raised when a threshold-selection release cannot be trusted."""


@dataclass(frozen=True)
class VerifiedThresholdSelection:
    root: Path
    selection_id: str
    config: ThresholdSelectionConfig
    descriptor: Mapping[str, Any]
    operating_confidence: float
    reference: Mapping[str, Any]


def _load_json(path: Path, name: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ThresholdSelectionIntegrityError(f"could not read {name}: {error}") from error


def _artifact_path(
    root: Path,
    manifest: Mapping[str, Any],
    role: str,
) -> Path:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ThresholdSelectionIntegrityError("threshold tracker artifacts must be an array")
    matches = [
        entry for entry in artifacts if isinstance(entry, Mapping) and entry.get("role") == role
    ]
    if len(matches) != 1:
        raise ThresholdSelectionIntegrityError(
            f"threshold release must contain exactly one {role!r} artifact"
        )
    return (root / str(matches[0]["path"])).resolve(strict=True)


def _sweep_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            expected = {
                "threshold",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "false_negative_rate",
                "weighted_error",
                "feasible",
            }
            if set(reader.fieldnames or ()) != expected:
                raise ThresholdSelectionIntegrityError("threshold sweep CSV schema is invalid")
            for raw in reader:
                row = {
                    "threshold": float(raw["threshold"]),
                    "tp": int(raw["tp"]),
                    "fp": int(raw["fp"]),
                    "fn": int(raw["fn"]),
                    "precision": float(raw["precision"]),
                    "recall": float(raw["recall"]),
                    "f1": float(raw["f1"]),
                    "false_negative_rate": float(raw["false_negative_rate"]),
                    "weighted_error": float(raw["weighted_error"]),
                    "feasible": raw["feasible"].casefold() == "true",
                }
                if (
                    not all(
                        math.isfinite(float(row[field]))
                        for field in (
                            "threshold",
                            "precision",
                            "recall",
                            "f1",
                            "false_negative_rate",
                            "weighted_error",
                        )
                    )
                    or min(row["tp"], row["fp"], row["fn"]) < 0
                ):
                    raise ThresholdSelectionIntegrityError(
                        "threshold sweep contains invalid numeric values"
                    )
                rows.append(row)
    except (
        OSError,
        UnicodeError,
        csv.Error,
        TypeError,
        ValueError,
    ) as error:
        if isinstance(error, ThresholdSelectionIntegrityError):
            raise
        raise ThresholdSelectionIntegrityError(
            f"could not read threshold sweep: {error}"
        ) from error
    if not rows:
        raise ThresholdSelectionIntegrityError("threshold sweep is empty")
    if [row["threshold"] for row in rows] != sorted(row["threshold"] for row in rows):
        raise ThresholdSelectionIntegrityError("threshold sweep order is not ascending")
    return rows


def load_verified_threshold_selection(
    path: str | Path,
) -> VerifiedThresholdSelection:
    try:
        verification = verify_research_object(
            path,
            verify_references=True,
            deep=False,
            reject_unregistered=True,
        )
    except ArtifactIntegrityError as error:
        raise ThresholdSelectionIntegrityError(str(error)) from error
    root = Path(verification.root)
    manifest = _load_json(
        root / "manifest.json",
        "threshold tracker manifest",
    )
    if not isinstance(manifest, Mapping):
        raise ThresholdSelectionIntegrityError("threshold tracker manifest must be an object")
    resolved_path = _artifact_path(
        root,
        manifest,
        "resolved_threshold_selection_configuration",
    )
    descriptor_path = _artifact_path(
        root,
        manifest,
        "threshold_selection_release_manifest",
    )
    selected_path = _artifact_path(
        root,
        manifest,
        "selected_operating_threshold",
    )
    sweep_path = _artifact_path(root, manifest, "threshold_sweep_table")
    bootstrap_path = _artifact_path(
        root,
        manifest,
        "threshold_selection_bootstrap",
    )
    resolved = _load_json(resolved_path, "resolved threshold configuration")
    descriptor = _load_json(descriptor_path, "threshold release descriptor")
    selected = _load_json(selected_path, "selected operating threshold")
    bootstrap = _load_json(bootstrap_path, "threshold selection bootstrap")
    if not all(isinstance(value, Mapping) for value in (resolved, descriptor, selected, bootstrap)):
        raise ThresholdSelectionIntegrityError("threshold JSON artifact envelope is invalid")
    config_raw = resolved.get("selection")
    if not isinstance(config_raw, Mapping):
        raise ThresholdSelectionIntegrityError("resolved threshold preregistration is missing")
    try:
        config = ThresholdSelectionConfig.from_mapping(config_raw)
    except (TypeError, ValueError) as error:
        raise ThresholdSelectionIntegrityError(
            f"resolved threshold preregistration is invalid: {error}"
        ) from error
    if (
        descriptor.get("schema_version") != THRESHOLD_SELECTION_RUN_SCHEMA_VERSION
        or descriptor.get("object_type") != "validation_threshold_selection_release"
        or descriptor.get("status") != "complete"
    ):
        raise ThresholdSelectionIntegrityError("threshold release descriptor envelope is invalid")
    selection_id = str(descriptor.get("selection_id", ""))
    if (
        selection_id != verification.run_id
        or selection_id != config.selection_id
        or selected.get("selection_id") != selection_id
    ):
        raise ThresholdSelectionIntegrityError("threshold selection identity is inconsistent")
    if config.purpose == "confirmatory":
        try:
            verify_research_object(
                root,
                verify_references=True,
                deep=False,
                reject_unregistered=True,
                require_clean_git=True,
            )
        except ArtifactIntegrityError as error:
            raise ThresholdSelectionIntegrityError(
                f"confirmatory threshold clean-git gate failed: {error}"
            ) from error

    source_reference = descriptor.get("source_evaluation")
    if (
        not isinstance(source_reference, Mapping)
        or source_reference != resolved.get("source_evaluation")
        or source_reference != selected.get("source_evaluation")
    ):
        raise ThresholdSelectionIntegrityError(
            "threshold source evaluation reference is inconsistent"
        )
    manifest_reference = source_reference.get("manifest")
    if not isinstance(manifest_reference, Mapping):
        raise ThresholdSelectionIntegrityError("threshold source evaluation manifest is missing")
    try:
        source = load_verified_evaluation(
            Path(str(manifest_reference["path"])).resolve(strict=True).parent
        )
    except (OSError, KeyError, RuntimeError, ValueError) as error:
        raise ThresholdSelectionIntegrityError(
            f"threshold source evaluation failed verification: {error}"
        ) from error
    if source.reference != source_reference:
        raise ThresholdSelectionIntegrityError("threshold source evaluation identity changed")
    if any(not partition.startswith("val") for partition in source.config.partitions):
        raise ThresholdSelectionIntegrityError("threshold source is not validation-only")
    if descriptor.get("validation_partitions") != list(source.config.partitions) or selected.get(
        "validation_partitions"
    ) != list(source.config.partitions):
        raise ThresholdSelectionIntegrityError("threshold validation partitions changed")
    if descriptor.get("model") != source.resolved.get("model"):
        raise ThresholdSelectionIntegrityError("threshold model reference changed")
    if descriptor.get("dataset") != source.dataset.reference:
        raise ThresholdSelectionIntegrityError("threshold dataset reference changed")

    sweep = _sweep_rows(sweep_path)
    if (
        len(sweep) != config.grid.steps
        or descriptor.get("sweep_count") != len(sweep)
        or not math.isclose(
            float(sweep[0]["threshold"]),
            config.grid.minimum,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or not math.isclose(
            float(sweep[-1]["threshold"]),
            config.grid.maximum,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ThresholdSelectionIntegrityError("threshold grid does not match preregistration")
    reproduced = _select(config, sweep)
    if reproduced is None:
        raise ThresholdSelectionIntegrityError(
            "threshold objective is infeasible on retained sweep"
        )
    operating_confidence = float(selected.get("operating_confidence", -1.0))
    if (
        selected.get("objective") != config.objective
        or selected.get("tie_breaker") != config.tie_breaker
        or selected.get("operating_point") != reproduced
        or descriptor.get("selected_operating_point") != reproduced
        or descriptor.get("selected_threshold") != operating_confidence
        or operating_confidence != float(reproduced["threshold"])
    ):
        raise ThresholdSelectionIntegrityError(
            "selected threshold does not reproduce from the sweep"
        )
    if (
        bootstrap.get("replicates") != config.bootstrap_replicates
        or bootstrap.get("confidence") != config.bootstrap_confidence
        or bootstrap.get("episode_count") != descriptor.get("episode_count")
        or bootstrap != descriptor.get("bootstrap")
    ):
        raise ThresholdSelectionIntegrityError("threshold bootstrap contract is inconsistent")
    return VerifiedThresholdSelection(
        root=root,
        selection_id=selection_id,
        config=config,
        descriptor=descriptor,
        operating_confidence=operating_confidence,
        reference={
            "kind": "verified_threshold_selection",
            "selection_id": selection_id,
            "manifest": fingerprint_file(root / "manifest.json"),
            "selection_manifest": fingerprint_file(descriptor_path),
            "selected_threshold": fingerprint_file(selected_path),
        },
    )


__all__ = [
    "ThresholdSelectionIntegrityError",
    "VerifiedThresholdSelection",
    "load_verified_threshold_selection",
]
