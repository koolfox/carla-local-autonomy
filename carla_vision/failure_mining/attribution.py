"""Evidence-based error triage. Decisions are rules, not calibrated probabilities.

No fault-injection oracle, test labels, or hidden simulator-state classifier is
accepted here. An uncertain/incorrect student is NOT evidence of a bad teacher.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

CHECKS = frozenset({"frame", "calibration", "files", "split", "teacher_timing"})


@dataclass(frozen=True)
class AttributionEvidence:
    capture_checks: Mapping[str, bool | None]
    model_error: float
    label_quality_lower: float | None
    independent_review: str = "unknown"
    capture_evidence_ref: str = ""
    quality_evidence_ref: str = ""
    review_evidence_ref: str = ""
    action_quality_lower: float | None = None
    task: str = "detection"

    def __post_init__(self) -> None:
        if set(self.capture_checks) != CHECKS:
            raise ValueError(
                "capture checks must explicitly report every required check (None is allowed)"
            )
        if any(
            value is not None and not isinstance(value, bool)
            for value in self.capture_checks.values()
        ):
            raise ValueError("capture check results must be bool or None")
        if not self.capture_evidence_ref:
            raise ValueError("capture evidence reference is required")
        if (
            isinstance(self.model_error, bool)
            or not math.isfinite(self.model_error)
            or self.model_error < 0
        ):
            raise ValueError("model error must be finite and non-negative")
        for value in (self.label_quality_lower, self.action_quality_lower):
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1
            ):
                raise ValueError("quality bounds must be finite probabilities or unknown")
        if self.independent_review not in {"valid", "invalid", "unknown"}:
            raise ValueError("unknown independent-review outcome")
        if self.independent_review != "unknown" and not self.review_evidence_ref:
            raise ValueError("independent review must cite its blinded audit")
        if (
            self.label_quality_lower is not None or self.action_quality_lower is not None
        ) and not self.quality_evidence_ref:
            raise ValueError("quality estimate must cite its audit evidence")
        if self.task not in {"detection", "imitation"}:
            raise ValueError("unknown task")


@dataclass(frozen=True)
class AttributionRecord:
    source_kind: str
    recommended_action: str
    evidence_refs: tuple[str, ...]
    unknown_reason: str | None = None
    confidence_type: str = "rule_based_not_calibrated_probability"
    schema_version: str = "1.0"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def attribute_error(
    evidence: AttributionEvidence, *, minimum_quality: float = 0.9, error_threshold: float = 0.0
) -> AttributionRecord:
    if not math.isfinite(minimum_quality) or not 0 <= minimum_quality <= 1:
        raise ValueError("minimum quality must be finite in [0, 1]")
    if not math.isfinite(error_threshold) or error_threshold < 0:
        raise ValueError("error threshold must be finite and non-negative")
    refs = tuple(
        ref
        for ref in (
            evidence.capture_evidence_ref,
            evidence.quality_evidence_ref,
            evidence.review_evidence_ref,
        )
        if ref
    )
    if any(value is False for value in evidence.capture_checks.values()):
        return AttributionRecord("capture_error", "RECAPTURE", refs)
    if evidence.independent_review == "invalid":
        return AttributionRecord("supervision_error", "RELABEL", refs)
    if any(value is None for value in evidence.capture_checks.values()):
        return AttributionRecord("unknown", "REVIEW", refs, "capture_check_unresolved")
    quality = (
        evidence.label_quality_lower
        if evidence.task == "detection"
        else evidence.action_quality_lower
    )
    reliable = evidence.independent_review == "valid" or (
        quality is not None and quality >= minimum_quality
    )
    if not reliable:
        return AttributionRecord(
            "unknown", "REVIEW", refs, "independent_quality_support_insufficient"
        )
    if evidence.model_error > error_threshold:
        return AttributionRecord("model_error", "NEW_SCENE", refs)
    return AttributionRecord("no_detected_error", "KEEP", refs)
