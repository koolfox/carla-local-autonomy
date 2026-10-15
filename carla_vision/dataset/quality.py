"""Independent, group-weighted supervision audit; perceptual difficulty is not a label error."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReviewRecord:
    sample_id: str
    group_id: str
    reviewer_id: str
    decision: str
    evidence_ref: str
    task: str = "label"
    blind_to_model: bool = True

    def __post_init__(self) -> None:
        if not all((self.sample_id, self.group_id, self.reviewer_id, self.evidence_ref)):
            raise ValueError("review identities and evidence reference must be non-empty")
        if self.decision not in {"valid", "invalid", "ambiguous"} or self.task not in {
            "label",
            "action",
        }:
            raise ValueError("invalid audit task or decision")
        if self.blind_to_model is not True:
            raise ValueError("independent quality estimation requires blinded review")


def estimate_quality(
    records: Sequence[ReviewRecord],
    *,
    task: str,
    seed: int = 17,
    replicates: int = 1000,
    confidence: float = 0.95,
    minimum_groups: int = 5,
) -> dict[str, Any]:
    if task not in {"label", "action"}:
        raise ValueError("task must distinguish label and action quality")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates <= 0
        or isinstance(minimum_groups, bool)
        or not isinstance(minimum_groups, int)
        or minimum_groups < 2
        or not math.isfinite(confidence)
        or not 0 < confidence < 1
    ):
        raise ValueError("invalid group-bootstrap settings")
    cases: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for record in records:
        if record.task != task:
            continue
        key = (record.group_id, record.sample_id)
        if record.reviewer_id in cases[key]:
            raise ValueError("duplicate review by the same reviewer")
        cases[key][record.reviewer_id] = record.decision
    groups: dict[str, list[float]] = defaultdict(list)
    unresolved = 0
    for (group, _), reviews in sorted(cases.items()):
        decisions = set(reviews.values())
        valid = decisions == {"valid"}
        unresolved += len(decisions) > 1 or "ambiguous" in decisions
        # Conservative: ambiguity is not silently removed from the denominator.
        groups[group].append(float(valid))
    means = np.asarray([np.mean(groups[group]) for group in sorted(groups)], dtype=float)
    result = {
        "schema_version": "1.0",
        "task": task,
        "sample_count": len(cases),
        "group_count": len(groups),
        "unresolved_count": unresolved,
        "mean_group_valid_fraction": float(means.mean()) if means.size else None,
        "quality_lower": None,
        "quality_upper": None,
        "confidence": confidence,
        "method": "independent_group_bootstrap_with_bounded_group_hoeffding_floor",
        "seed": seed,
        "replicates": replicates,
        "unknown_reason": "insufficient_independent_groups",
    }
    if len(groups) >= minimum_groups:
        rng = np.random.default_rng(seed)
        draws = [
            float(rng.choice(means, len(means), replace=True).mean()) for _ in range(replicates)
        ]
        alpha = (1 - confidence) / 2
        # Bootstrap alone gives [1,1] after a small all-valid audit. Retain a
        # distribution-free bounded-group bound so absence of observed errors
        # is not treated as certainty. Independent sampled groups are assumed.
        radius = math.sqrt(math.log(2 / (1 - confidence)) / (2 * len(groups)))
        result.update(
            quality_lower=max(
                0.0, min(float(np.quantile(draws, alpha)), float(means.mean()) - radius)
            ),
            quality_upper=min(
                1.0, max(float(np.quantile(draws, 1 - alpha)), float(means.mean()) + radius)
            ),
            bootstrap_interval=[
                float(np.quantile(draws, alpha)),
                float(np.quantile(draws, 1 - alpha)),
            ],
            unknown_reason=None,
        )
    return result
