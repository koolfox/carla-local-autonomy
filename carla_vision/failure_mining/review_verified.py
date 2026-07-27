"""Consumer-side verification for finalized human failure-review catalogs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import fingerprint_file
from ..verification import ArtifactIntegrityError, verify_research_object
from .contracts import FAILURE_TYPES
from .review import FAILURE_REVIEW_RUN_SCHEMA_VERSION
from .review_contracts import (
    REVIEW_DECISIONS,
    FailureReviewConfig,
)
from .verified import load_verified_failure_mining


class FailureReviewIntegrityError(RuntimeError):
    """Raised when a reviewed failure catalog cannot be trusted."""


@dataclass(frozen=True)
class VerifiedFailureReview:
    root: Path
    review_id: str
    config: FailureReviewConfig
    descriptor: Mapping[str, Any]
    reviews: tuple[Mapping[str, Any], ...]
    confirmed: tuple[Mapping[str, Any], ...]
    label_issues: tuple[Mapping[str, Any], ...]
    reference: Mapping[str, Any]


def _load_json(path: Path, name: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FailureReviewIntegrityError(f"could not read {name}: {error}") from error


def _load_jsonl(path: Path, name: str) -> tuple[Mapping[str, Any], ...]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise FailureReviewIntegrityError(
                        f"{name} line {line_number} must be an object"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FailureReviewIntegrityError(f"could not read {name}: {error}") from error
    return tuple(rows)


def _artifact_path(
    root: Path,
    manifest: Mapping[str, Any],
    role: str,
) -> Path:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise FailureReviewIntegrityError("failure-review tracker artifacts must be an array")
    matches = [
        entry for entry in artifacts if isinstance(entry, Mapping) and entry.get("role") == role
    ]
    if len(matches) != 1:
        raise FailureReviewIntegrityError(
            f"failure review must contain exactly one {role!r} artifact"
        )
    return (root / str(matches[0]["path"])).resolve(strict=True)


def _csv_reviews(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise FailureReviewIntegrityError("normalized review CSV has no header")
            return list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise FailureReviewIntegrityError(
            f"could not read normalized review CSV: {error}"
        ) from error


def load_verified_failure_review(
    path: str | Path,
) -> VerifiedFailureReview:
    try:
        verification = verify_research_object(
            path,
            verify_references=True,
            deep=False,
            reject_unregistered=True,
        )
    except ArtifactIntegrityError as error:
        raise FailureReviewIntegrityError(str(error)) from error
    root = Path(verification.root)
    manifest = _load_json(
        root / "manifest.json",
        "failure-review tracker manifest",
    )
    if not isinstance(manifest, Mapping):
        raise FailureReviewIntegrityError("failure-review tracker manifest must be an object")
    resolved_path = _artifact_path(
        root,
        manifest,
        "resolved_failure_review_configuration",
    )
    descriptor_path = _artifact_path(
        root,
        manifest,
        "failure_review_release_manifest",
    )
    reviews_path = _artifact_path(root, manifest, "normalized_failure_reviews")
    source_review_path = _artifact_path(
        root,
        manifest,
        "completed_failure_review_source",
    )
    reviews_csv_path = _artifact_path(
        root,
        manifest,
        "normalized_failure_review_table",
    )
    confirmed_path = _artifact_path(
        root,
        manifest,
        "confirmed_failure_catalog",
    )
    label_issues_path = _artifact_path(
        root,
        manifest,
        "failure_label_issue_catalog",
    )
    summary_path = _artifact_path(root, manifest, "failure_review_summary")
    resolved = _load_json(resolved_path, "resolved failure-review config")
    descriptor = _load_json(descriptor_path, "failure-review descriptor")
    summary = _load_json(summary_path, "failure-review summary")
    if not all(isinstance(value, Mapping) for value in (resolved, descriptor, summary)):
        raise FailureReviewIntegrityError("failure-review JSON envelope is invalid")
    config_raw = resolved.get("review")
    if not isinstance(config_raw, Mapping):
        raise FailureReviewIntegrityError("resolved failure-review preregistration is missing")
    try:
        config = FailureReviewConfig.from_mapping(config_raw)
    except (TypeError, ValueError) as error:
        raise FailureReviewIntegrityError(
            f"resolved failure-review config is invalid: {error}"
        ) from error
    if (
        descriptor.get("schema_version") != FAILURE_REVIEW_RUN_SCHEMA_VERSION
        or descriptor.get("object_type") != "reviewed_failure_catalog_release"
        or descriptor.get("status") != "complete"
        or descriptor.get("review_status") != "complete"
    ):
        raise FailureReviewIntegrityError("failure-review descriptor envelope is invalid")
    review_id = str(descriptor.get("review_id", ""))
    if review_id != verification.run_id or review_id != config.review_id:
        raise FailureReviewIntegrityError("failure-review identity is inconsistent")
    source_reference = descriptor.get("source_mining")
    if not isinstance(source_reference, Mapping) or source_reference != resolved.get(
        "source_mining"
    ):
        raise FailureReviewIntegrityError("failure-review source mining reference is inconsistent")
    source_manifest = source_reference.get("manifest")
    if not isinstance(source_manifest, Mapping):
        raise FailureReviewIntegrityError("failure-review source mining manifest is missing")
    try:
        mining = load_verified_failure_mining(
            Path(str(source_manifest["path"])).resolve(strict=True).parent
        )
    except (OSError, KeyError, RuntimeError, ValueError) as error:
        raise FailureReviewIntegrityError(
            f"failure-review source mining failed verification: {error}"
        ) from error
    if source_reference != mining.reference:
        raise FailureReviewIntegrityError("failure-review source mining identity changed")
    completed_review = resolved.get("completed_review")
    if not isinstance(completed_review, Mapping):
        raise FailureReviewIntegrityError("completed review source reference is missing")
    archived_review = fingerprint_file(source_review_path)
    if (
        completed_review.get("sha256") != archived_review["sha256"]
        or completed_review.get("size_bytes") != archived_review["size_bytes"]
    ):
        raise FailureReviewIntegrityError(
            "archived completed review differs from its source fingerprint"
        )
    if (
        descriptor.get("dataset") != mining.descriptor["dataset"]
        or descriptor.get("model") != mining.descriptor["model"]
        or descriptor.get("source_evaluation") != mining.descriptor["source_evaluation"]
    ):
        raise FailureReviewIntegrityError("failure-review lineage is inconsistent")

    reviews = _load_jsonl(reviews_path, "normalized failure reviews")
    confirmed = _load_jsonl(confirmed_path, "confirmed failure catalog")
    label_issues = _load_jsonl(label_issues_path, "label-issue catalog")
    if len(reviews) != len(mining.review_queue):
        raise FailureReviewIntegrityError("failure-review count differs from source queue")
    queue_by_id = {str(row["failure_id"]): row for row in mining.review_queue}
    seen: set[str] = set()
    for position, review in enumerate(reviews, start=1):
        failure_id = str(review.get("failure_id", ""))
        source = queue_by_id.get(failure_id)
        if source is None or failure_id in seen:
            raise FailureReviewIntegrityError(
                "failure review contains unknown or duplicate failure ID"
            )
        if (
            review.get("schema_version") != FAILURE_REVIEW_RUN_SCHEMA_VERSION
            or review.get("selected_rank") != position
            or review.get("selected_rank") != source["selected_rank"]
            or review.get("original_failure_type") != source["failure_type"]
            or review.get("object_category") != source["object_category"]
            or review.get("decision") not in REVIEW_DECISIONS
            or review.get("reviewer") not in config.reviewers
            or not str(review.get("review_notes", "")).strip()
            or review.get("source_mining_id") != mining.mining_id
            or review.get("final_failure_type") not in FAILURE_TYPES
        ):
            raise FailureReviewIntegrityError(f"normalized review {failure_id!r} is invalid")
        if (
            review["decision"] in {"confirmed", "label_issue"}
            and not str(review.get("proposed_remedy", "")).strip()
        ):
            raise FailureReviewIntegrityError(f"normalized review {failure_id!r} lacks a remedy")
        seen.add(failure_id)
    if seen != set(queue_by_id):
        raise FailureReviewIntegrityError("failure review omits source queue items")

    csv_rows = _csv_reviews(reviews_csv_path)
    if [row.get("failure_id") for row in csv_rows] != [str(row["failure_id"]) for row in reviews]:
        raise FailureReviewIntegrityError("normalized review CSV order differs from JSONL")
    review_by_id = {str(row["failure_id"]): row for row in reviews}
    for catalog, decision in (
        (confirmed, "confirmed"),
        (label_issues, "label_issue"),
    ):
        expected_ids = [str(row["failure_id"]) for row in reviews if row["decision"] == decision]
        if [str(row.get("failure_id", "")) for row in catalog] != expected_ids:
            raise FailureReviewIntegrityError(
                f"{decision} catalog identity or order is inconsistent"
            )
        for row in catalog:
            failure_id = str(row["failure_id"])
            if (
                row.get("review") != review_by_id[failure_id]
                or row.get("final_failure_type") != review_by_id[failure_id]["final_failure_type"]
                or {
                    key: value
                    for key, value in row.items()
                    if key not in {"review", "final_failure_type"}
                }
                != dict(queue_by_id[failure_id])
            ):
                raise FailureReviewIntegrityError(f"{decision} catalog row {failure_id!r} changed")
    decision_counts = dict(sorted(Counter(row["decision"] for row in reviews).items()))
    if (
        descriptor.get("reviewed_count") != len(reviews)
        or descriptor.get("confirmed_count") != len(confirmed)
        or descriptor.get("label_issue_count") != len(label_issues)
        or descriptor.get("decision_counts") != decision_counts
        or summary.get("reviewed_count") != len(reviews)
        or summary.get("confirmed_count") != len(confirmed)
        or summary.get("label_issue_count") != len(label_issues)
        or summary.get("decision_counts") != decision_counts
        or summary.get("review_status") != "complete"
    ):
        raise FailureReviewIntegrityError("failure-review counts or decisions are inconsistent")
    return VerifiedFailureReview(
        root=root,
        review_id=review_id,
        config=config,
        descriptor=descriptor,
        reviews=reviews,
        confirmed=confirmed,
        label_issues=label_issues,
        reference={
            "kind": "verified_failure_review",
            "review_id": review_id,
            "manifest": fingerprint_file(root / "manifest.json"),
            "review_manifest": fingerprint_file(descriptor_path),
            "confirmed_catalog": fingerprint_file(confirmed_path),
        },
    )


__all__ = [
    "FailureReviewIntegrityError",
    "VerifiedFailureReview",
    "load_verified_failure_review",
]
