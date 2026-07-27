"""Finalize a completed human review into an immutable reviewed catalog."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402

from ..artifacts import RunArtifactTracker, fingerprint_file
from .contracts import FAILURE_TYPES
from .review_contracts import (
    REVIEW_DECISIONS,
    FailureReviewConfig,
    load_failure_review_config,
)
from .verified import VerifiedFailureMining, load_verified_failure_mining

FAILURE_REVIEW_RUN_SCHEMA_VERSION = "1.0"
_REVIEW_FIELDS = (
    "failure_id",
    "selected_rank",
    "failure_type",
    "object_category",
    "decision",
    "corrected_failure_type",
    "reviewer",
    "review_notes",
    "proposed_remedy",
)


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def _write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    _atomic_write_text(
        path,
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_reviews(
    path: Path,
    *,
    config: FailureReviewConfig,
    mining: VerifiedFailureMining,
) -> list[dict[str, Any]]:
    queue_by_id = {str(row["failure_id"]): row for row in mining.review_queue}
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != _REVIEW_FIELDS:
                raise ValueError(
                    "review CSV fields or field order differ from the released template"
                )
            raw_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise RuntimeError(f"could not read completed review CSV: {error}") from error
    if len(raw_rows) != len(queue_by_id):
        raise ValueError("completed review must contain exactly one row for every queue item")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for raw in raw_rows:
        failure_id = raw["failure_id"].strip()
        source = queue_by_id.get(failure_id)
        if source is None or failure_id in seen:
            raise ValueError(f"review contains unknown or duplicate failure ID {failure_id!r}")
        if (
            raw["selected_rank"].strip() != str(source["selected_rank"])
            or raw["failure_type"].strip() != source["failure_type"]
            or raw["object_category"].strip() != source["object_category"]
        ):
            raise ValueError(f"review changed immutable queue fields for {failure_id!r}")
        decision = raw["decision"].strip().lower()
        corrected = raw["corrected_failure_type"].strip().lower()
        reviewer = raw["reviewer"].strip()
        notes = raw["review_notes"].strip()
        remedy = raw["proposed_remedy"].strip()
        if decision not in REVIEW_DECISIONS:
            raise ValueError(f"review decision is invalid for {failure_id!r}: {decision!r}")
        if corrected and corrected not in FAILURE_TYPES:
            raise ValueError(f"corrected failure type is invalid for {failure_id!r}")
        if reviewer not in config.reviewers:
            raise ValueError(f"reviewer {reviewer!r} is not preregistered for {failure_id!r}")
        if not notes:
            raise ValueError(f"review notes are required for {failure_id!r}")
        if decision in {"confirmed", "label_issue"} and not remedy:
            raise ValueError(f"proposed remedy is required for {decision} {failure_id!r}")
        final_type = corrected or str(source["failure_type"])
        normalized.append(
            {
                "schema_version": FAILURE_REVIEW_RUN_SCHEMA_VERSION,
                "failure_id": failure_id,
                "selected_rank": int(source["selected_rank"]),
                "original_failure_type": str(source["failure_type"]),
                "final_failure_type": final_type,
                "object_category": str(source["object_category"]),
                "decision": decision,
                "reviewer": reviewer,
                "review_notes": notes,
                "proposed_remedy": remedy,
                "source_mining_id": mining.mining_id,
            }
        )
        seen.add(failure_id)
    if seen != set(queue_by_id):
        raise ValueError("completed review omits one or more queue items")
    return sorted(normalized, key=lambda row: int(row["selected_rank"]))


def _merged_records(
    mining: VerifiedFailureMining,
    reviews: Sequence[Mapping[str, Any]],
    decisions: set[str],
) -> list[dict[str, Any]]:
    queue_by_id = {str(row["failure_id"]): row for row in mining.review_queue}
    return [
        {
            **dict(queue_by_id[str(review["failure_id"])]),
            "review": dict(review),
            "final_failure_type": str(review["final_failure_type"]),
        }
        for review in reviews
        if str(review["decision"]) in decisions
    ]


def _save_figure(figure: Any, base_path: Path) -> tuple[Path, Path]:
    png = base_path.with_suffix(".png")
    svg = base_path.with_suffix(".svg")
    figure.tight_layout()
    figure.savefig(png, dpi=180)
    figure.savefig(svg)
    plt.close(figure)
    return png, svg


def _bar_plot(
    base_path: Path,
    counts: Mapping[str, int],
    *,
    title: str,
    color: str,
) -> tuple[Path, Path]:
    labels = sorted(counts)
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    x = np.arange(len(labels))
    axis.bar(x, [counts[label] for label in labels], color=color)
    axis.set_xticks(x, labels, rotation=20, ha="right")
    axis.set_ylabel("Reviewed failures")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    return _save_figure(figure, base_path)


def _write_checksum_index(
    root: Path,
    paths: Sequence[Path],
    output: Path,
) -> None:
    lines = []
    for path in sorted(
        paths,
        key=lambda value: value.relative_to(root).as_posix().encode("utf-8"),
    ):
        reference = fingerprint_file(path)
        lines.append(f"{reference['sha256']}  {path.relative_to(root).as_posix()}")
    _atomic_write_text(output, "\n".join(lines) + "\n")


def _copy_stable(source: Path, destination: Path) -> None:
    before = fingerprint_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    after = fingerprint_file(source)
    copied = fingerprint_file(destination)
    if (
        before["sha256"] != after["sha256"]
        or before["size_bytes"] != after["size_bytes"]
        or before["sha256"] != copied["sha256"]
        or before["size_bytes"] != copied["size_bytes"]
    ):
        destination.unlink(missing_ok=True)
        raise RuntimeError("completed review CSV changed while being archived")


def run_failure_review(
    *,
    config_path: str | Path,
    mining_run: str | Path,
    review_csv: str | Path,
    runs_root: str | Path = "runs",
    cli_args: Sequence[str] | Mapping[str, Any] = (),
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    config_path_resolved = Path(config_path).expanduser().resolve(strict=True)
    review_path = Path(review_csv).expanduser().resolve(strict=True)
    config = load_failure_review_config(config_path_resolved)
    mining = load_verified_failure_mining(mining_run)
    reviews = _load_reviews(
        review_path,
        config=config,
        mining=mining,
    )
    confirmed = _merged_records(mining, reviews, {"confirmed"})
    label_issues = _merged_records(mining, reviews, {"label_issue"})
    tracker = RunArtifactTracker(
        runs_root,
        run_id=config.review_id,
        cli_args=cli_args,
        config={
            "schema_version": FAILURE_REVIEW_RUN_SCHEMA_VERSION,
            "object_type": "reviewed_failure_catalog",
            "review": config.as_dict(),
        },
        repository_root=repository_root,
        model_refs=[mining.descriptor["model"]],
        dataset_refs=[mining.descriptor["dataset"]],
        input_refs=[
            {
                "kind": "failure_review_preregistration",
                **fingerprint_file(config_path_resolved),
            },
            mining.reference,
            {
                "kind": "completed_human_review",
                **fingerprint_file(review_path),
            },
        ],
    )
    descriptor: dict[str, Any]
    with tracker:
        resolved_path = tracker.artifact_path("resolved_failure_review_config.json")
        source_review_path = tracker.artifact_path("reviews/completed_review_source.csv")
        reviews_path = tracker.artifact_path("reviews/reviews.jsonl")
        reviews_csv_path = tracker.artifact_path("reviews/reviews.csv")
        confirmed_path = tracker.artifact_path("catalog/confirmed_failures.jsonl")
        label_issues_path = tracker.artifact_path("catalog/label_issues.jsonl")
        summary_path = tracker.artifact_path("review_summary.json")
        descriptor_path = tracker.artifact_path("failure_review.json")
        report_path = tracker.artifact_path("failure_review.md")
        checksum_path = tracker.artifact_path("checksums.sha256")
        _copy_stable(review_path, source_review_path)
        resolved = {
            "schema_version": FAILURE_REVIEW_RUN_SCHEMA_VERSION,
            "object_type": "reviewed_failure_catalog",
            "review": config.as_dict(),
            "source_config": fingerprint_file(config_path_resolved),
            "source_mining": dict(mining.reference),
            "completed_review": fingerprint_file(review_path),
            "dataset": mining.descriptor["dataset"],
            "model": mining.descriptor["model"],
            "runtime_sensor_contract": "front_monocular_rgb_only",
        }
        _write_json(resolved_path, resolved)
        _write_jsonl(reviews_path, reviews)
        _write_csv(
            reviews_csv_path,
            reviews,
            fields=(
                "failure_id",
                "selected_rank",
                "original_failure_type",
                "final_failure_type",
                "object_category",
                "decision",
                "reviewer",
                "review_notes",
                "proposed_remedy",
                "source_mining_id",
                "schema_version",
            ),
        )
        _write_jsonl(confirmed_path, confirmed)
        _write_jsonl(label_issues_path, label_issues)
        decision_counts = dict(sorted(Counter(row["decision"] for row in reviews).items()))
        confirmed_type_counts = dict(
            sorted(Counter(row["final_failure_type"] for row in confirmed).items())
        )
        reviewer_counts = dict(sorted(Counter(row["reviewer"] for row in reviews).items()))
        summary = {
            "schema_version": FAILURE_REVIEW_RUN_SCHEMA_VERSION,
            "review_id": config.review_id,
            "source_mining_id": mining.mining_id,
            "reviewed_count": len(reviews),
            "confirmed_count": len(confirmed),
            "label_issue_count": len(label_issues),
            "decision_counts": decision_counts,
            "confirmed_type_counts": confirmed_type_counts,
            "reviewer_counts": reviewer_counts,
            "review_status": "complete",
        }
        _write_json(summary_path, summary)
        plot_paths: list[Path] = []
        plot_paths.extend(
            _bar_plot(
                tracker.artifact_path(
                    "plots/review_decisions",
                    create_parent=True,
                ),
                decision_counts,
                title="Human review decisions",
                color="#3a86ff",
            )
        )
        plot_paths.extend(
            _bar_plot(
                tracker.artifact_path(
                    "plots/confirmed_failure_types",
                    create_parent=True,
                ),
                confirmed_type_counts or {"none": 0},
                title="Confirmed failures by final type",
                color="#e76f51",
            )
        )
        plot_paths.extend(
            _bar_plot(
                tracker.artifact_path(
                    "plots/reviewer_workload",
                    create_parent=True,
                ),
                reviewer_counts,
                title="Reviewed records by reviewer",
                color="#2a9d8f",
            )
        )
        descriptor = {
            "schema_version": FAILURE_REVIEW_RUN_SCHEMA_VERSION,
            "object_type": "reviewed_failure_catalog_release",
            "status": "complete",
            "review_id": config.review_id,
            "title": config.title,
            "purpose": config.purpose,
            "source_mining": dict(mining.reference),
            "source_evaluation": mining.descriptor["source_evaluation"],
            "dataset": mining.descriptor["dataset"],
            "model": mining.descriptor["model"],
            "reviewers": list(config.reviewers),
            "reviewed_count": len(reviews),
            "confirmed_count": len(confirmed),
            "label_issue_count": len(label_issues),
            "decision_counts": decision_counts,
            "review_status": "complete",
            "runtime_sensor_contract": "front_monocular_rgb_only",
            "training_boundary": (
                "catalog references validation images; it is not a training dataset"
            ),
            "limitations": [
                "Human review decisions may contain reviewer subjectivity.",
                "Confirmed validation images remain excluded from training.",
                "Proposed remedies require new scenarios and a new dataset release.",
                "Failure confirmation does not establish causal root cause.",
            ],
        }
        _write_json(descriptor_path, descriptor)
        _atomic_write_text(
            report_path,
            f"""# {config.title}

- Review ID: `{config.review_id}`
- Source mining release: `{mining.mining_id}`
- Reviewed queue records: `{len(reviews)}`
- Confirmed failures: `{len(confirmed)}`
- Label issues: `{len(label_issues)}`
- Reviewers: {", ".join(config.reviewers)}
- Status: `complete`

## Decisions

{json.dumps(decision_counts, sort_keys=True)}

## Training boundary

This catalog references validation evidence. It is not a training dataset and
does not copy validation RGB into a training partition. Proposed remedies must
be generated using new scenario/episode identities and released as a new
dataset version before training.
""",
        )
        payload_paths = [
            resolved_path,
            source_review_path,
            reviews_path,
            reviews_csv_path,
            confirmed_path,
            label_issues_path,
            summary_path,
            descriptor_path,
            report_path,
            *plot_paths,
        ]
        _write_checksum_index(tracker.run_dir, payload_paths, checksum_path)
        artifacts = {
            resolved_path: "resolved_failure_review_configuration",
            source_review_path: "completed_failure_review_source",
            reviews_path: "normalized_failure_reviews",
            reviews_csv_path: "normalized_failure_review_table",
            confirmed_path: "confirmed_failure_catalog",
            label_issues_path: "failure_label_issue_catalog",
            summary_path: "failure_review_summary",
            descriptor_path: "failure_review_release_manifest",
            report_path: "failure_review_report",
            checksum_path: "failure_review_checksum_index",
            **{path: "failure_review_plot" for path in plot_paths},
        }
        for path, role in artifacts.items():
            tracker.register_artifact(
                path,
                role=role,
                metadata={
                    "review_id": config.review_id,
                    "source_mining": mining.mining_id,
                },
            )
    return {
        "review_id": tracker.run_id,
        "run_dir": str(tracker.run_dir),
        "manifest": str(tracker.manifest_path),
        "descriptor": descriptor,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Finalize a completed failure-review CSV into an immutable reviewed catalog")
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--mining-run", required=True)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--runs-root", default="runs")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_failure_review(
        config_path=args.config,
        mining_run=args.mining_run,
        review_csv=args.review_csv,
        runs_root=args.runs_root,
        cli_args=vars(args),
        repository_root=Path.cwd(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


__all__ = [
    "FAILURE_REVIEW_RUN_SCHEMA_VERSION",
    "main",
    "parse_args",
    "run_failure_review",
]
