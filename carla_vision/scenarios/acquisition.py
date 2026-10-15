"""Cost-bounded selection of research actions with common baseline capabilities.

This is an offline decision planner, not an autonomous CARLA controller. Scores
are preregistered heuristics, not a claim of optimality or empirical superiority.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from ..dataset.lineage import canonical_digest, require_train_parent

METHODS = {
    "random",
    "stratified",
    "uncertainty",
    "failure_only",
    "noise_aware",
    "capability_matched",
    "error_source_aware",
}
ACTIONS = {"RECAPTURE", "RELABEL", "NEW_SCENE", "REVIEW", "KEEP"}


@dataclass(frozen=True)
class AcquisitionCandidate:
    candidate_id: str
    group_id: str
    condition: str
    partition: str
    estimated_cost: float
    model_error: float
    uncertainty: float
    quality_lower: float | None
    action: str
    evidence_ref: str
    repair_value: float = 0.0

    def __post_init__(self) -> None:
        require_train_parent(self.partition)
        if (
            not all((self.candidate_id, self.group_id, self.condition, self.evidence_ref))
            or self.action not in ACTIONS
        ):
            raise ValueError("candidate identities, evidence and action are required")
        for name in ("estimated_cost", "model_error", "uncertainty", "repair_value"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.estimated_cost <= 0:
            raise ValueError("candidate cost must be strictly positive")
        if self.quality_lower is not None and (
            isinstance(self.quality_lower, bool)
            or not math.isfinite(self.quality_lower)
            or not 0 <= self.quality_lower <= 1
        ):
            raise ValueError("quality must be a lower bound in [0,1] or unknown")


@dataclass(frozen=True)
class AcquisitionConfig:
    method: str
    budget: float
    max_actions: int
    seed: int = 17
    exploration_fraction: float = 0.1
    per_group_limit: int = 1
    minimum_per_condition: Mapping[str, int] | None = None
    ablations: tuple[str, ...] = ()
    cost_unit: str = "preregistered_normalized_cost"

    def __post_init__(self) -> None:
        if self.method not in METHODS or not math.isfinite(self.budget) or self.budget <= 0:
            raise ValueError("invalid method or budget")
        for value in (self.max_actions, self.per_group_limit):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("selection counts must be positive integers")
        if not math.isfinite(self.exploration_fraction) or not 0 <= self.exploration_fraction <= 1:
            raise ValueError("exploration fraction must be in [0, 1]")
        if set(self.ablations) - {"quality", "attribution", "coverage", "exploration"}:
            raise ValueError("unknown ablation")
        for condition, count in (self.minimum_per_condition or {}).items():
            if not condition or isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("condition quotas must be non-negative integer counts")


def select_acquisition(
    candidates: Sequence[AcquisitionCandidate], config: AcquisitionConfig
) -> dict[str, Any]:
    candidates = tuple(sorted(candidates, key=lambda item: item.candidate_id))
    if len({item.candidate_id for item in candidates}) != len(candidates):
        raise ValueError("candidate IDs must be unique")
    rng = random.Random(config.seed)
    priorities = {item.candidate_id: rng.random() for item in candidates}
    selected: list[dict[str, Any]] = []
    used, groups, conditions, spent = set(), Counter(), Counter(), 0.0

    def feasible(item):
        return (
            item.candidate_id not in used
            and groups[item.group_id] < config.per_group_limit
            and spent + item.estimated_cost <= config.budget + 1e-9
            and len(selected) < config.max_actions
        )

    def add(item, reason):
        nonlocal spent
        selected.append({**asdict(item), "selection_reason": reason})
        used.add(item.candidate_id)
        groups[item.group_id] += 1
        conditions[item.condition] += 1
        spent += item.estimated_cost

    quotas = {} if "coverage" in config.ablations else dict(config.minimum_per_condition or {})
    for condition, count in sorted(quotas.items()):
        for _ in range(count):
            pool = [item for item in candidates if item.condition == condition and feasible(item)]
            if not pool:
                raise ValueError(
                    "greedy coverage construction failed; not a proof that no feasible plan exists"
                )
            add(
                min(
                    pool,
                    key=lambda item: (
                        item.estimated_cost,
                        priorities[item.candidate_id],
                        item.candidate_id,
                    ),
                ),
                "coverage",
            )
    fraction = 0.0 if "exploration" in config.ablations else config.exploration_fraction
    target_exploration = math.ceil(config.max_actions * fraction)
    for _ in range(target_exploration):
        pool = [item for item in candidates if feasible(item)]
        if not pool:
            break
        add(
            min(pool, key=lambda item: (priorities[item.candidate_id], item.candidate_id)),
            "exploration",
        )

    def utility(item):
        method = config.method
        if "attribution" in config.ablations and method == "error_source_aware":
            method = "capability_matched"
        quality = (
            1.0
            if "quality" in config.ablations
            else (item.quality_lower if item.quality_lower is not None else 0.0)
        )
        if method == "random":
            return priorities[item.candidate_id]
        if method == "stratified":
            return 1.0 / (1 + conditions[item.condition])
        if method == "uncertainty":
            return item.uncertainty
        if method == "failure_only":
            return item.model_error
        if method == "noise_aware":
            return quality * item.model_error + (1 - quality) * item.uncertainty
        if method == "capability_matched":
            return quality * item.model_error + item.repair_value + (1 - quality) * item.uncertainty
        if item.action in {"RECAPTURE", "RELABEL", "REVIEW"}:
            return (
                item.repair_value
            )  # expected repair value must be preregistered, never observed test gain
        return quality * item.model_error

    while len(selected) < config.max_actions:
        pool = [item for item in candidates if feasible(item)]
        if not pool:
            break
        if config.method == "random":
            item = min(pool, key=lambda item: (priorities[item.candidate_id], item.candidate_id))
        elif config.method == "stratified":
            item = min(
                pool,
                key=lambda item: (
                    conditions[item.condition],
                    priorities[item.candidate_id],
                    item.candidate_id,
                ),
            )
        else:
            item = min(
                pool,
                key=lambda item: (
                    -utility(item) / item.estimated_cost,
                    priorities[item.candidate_id],
                    item.candidate_id,
                ),
            )
        add(
            item,
            "random_order"
            if config.method == "random"
            else "condition_balance"
            if config.method == "stratified"
            else "utility_per_cost",
        )
    result = {
        "schema_version": "1.0",
        "object_type": "acquisition_action_plan",
        "config": asdict(config),
        "selected": selected,
        "estimated_cost": spent,
        "planned_condition_counts": dict(conditions),
        "candidate_count": len(candidates),
        "candidate_digest": canonical_digest([asdict(item) for item in candidates]),
        "exploration_target": target_exploration,
        "exploration_realized": sum(item["selection_reason"] == "exploration" for item in selected),
        "cost_scope": "selection_estimates_only_use_ledger_for_actual_total",
        "training_data_created": False,
        "empirical_superiority_established": False,
    }
    result["plan_sha256"] = canonical_digest(result)
    return result
