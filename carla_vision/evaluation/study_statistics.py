"""Paired study accounting and transparent statistical planning, without SciPy.

Seed intervals condition on one fixed holdout. They do not also estimate route
sampling uncertainty. COCO AP is never averaged over per-frame/per-route APs.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any

import numpy as np

from ..reproduction.review_contracts import digest, fields, integer, number, text


def paired_seed_analysis(
    left: Mapping[int, float],
    right: Mapping[int, float],
    *,
    replicates: int = 2000,
    seed: int = 17,
    confidence: float = 0.95,
) -> dict[str, Any]:
    if set(left) != set(right) or len(left) < 2:
        raise ValueError("at least two identically paired training seeds are required")
    integer(replicates, "replicates", minimum=2)
    integer(seed, "seed")
    number(confidence, "confidence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0,1)")
    ids = sorted(integer(k, "training seed") for k in left)
    delta = np.array([number(right[k], "metric") - number(left[k], "metric") for k in ids])
    rng = np.random.default_rng(seed)
    # Memory bounded by seeds rather than replicates x seeds.
    samples = np.array(
        [rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(replicates)]
    )
    lo, hi = np.quantile(samples, [(1 - confidence) / 2, (1 + confidence) / 2])
    return {
        "schema_version": "1.0",
        "seed_count": len(ids),
        "seeds": ids,
        "mean_difference": float(delta.mean()),
        "sample_std_difference": float(delta.std(ddof=1)),
        "lower": float(lo),
        "upper": float(hi),
        "confidence": confidence,
        "replicates": replicates,
        "resampling_seed": seed,
        "interval_scope": "training_seed_variation_fixed_evaluation_set",
        "small_sample_warning": len(ids) < 10,
    }


def plan_paired_sample_size(
    differences: Sequence[float],
    *,
    effect: float,
    source_role: str,
    power: float = 0.8,
    alpha: float = 0.05,
    comparisons: int = 1,
) -> dict[str, Any]:
    if source_role != "pilot":
        raise ValueError("sample-size planning must use an independent pilot, never final test")
    values = np.array([number(v, "pilot difference") for v in differences])
    if len(values) < 3:
        raise ValueError("at least three independent pilot pairs are required")
    number(effect, "effect", minimum=0)
    number(power, "power")
    number(alpha, "alpha")
    integer(comparisons, "comparisons", minimum=1)
    if effect <= 0 or not 0 < power < 1 or not 0 < alpha < 1:
        raise ValueError("effect must be positive; alpha/power must lie in (0,1)")
    sd = float(values.std(ddof=1))
    if not math.isfinite(sd) or sd <= 0:
        raise ValueError("a zero-variance pilot cannot justify sample-size certainty")
    normal = NormalDist()
    n = math.ceil(
        ((normal.inv_cdf(1 - alpha / (2 * comparisons)) + normal.inv_cdf(power)) * sd / effect) ** 2
    )
    return {
        "schema_version": "1.0",
        "pilot_pair_count": len(values),
        "pilot_std": sd,
        "minimum_effect": effect,
        "required_independent_pairs": max(3, n),
        "target_power": power,
        "alpha": alpha,
        "comparisons": comparisons,
        "method": "paired_normal_approximation_bonferroni_planning_only",
        "empirical_power_established": False,
        "limitations": "IID independent-pair normal approximation; validate with a separate pilot and sensitivity analysis.",
    }


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    p = [number(v, "p-value") for v in pvalues]
    if any(not 0 <= v <= 1 for v in p):
        raise ValueError("p-values must be in [0,1]")
    adjusted, highest = [0.0] * len(p), 0.0
    for rank, index in enumerate(sorted(range(len(p)), key=lambda k: p[k])):
        highest = max(highest, min(1.0, (len(p) - rank) * p[index]))
        adjusted[index] = highest
    return adjusted


def agreement(left: Sequence[str], right: Sequence[str]) -> dict[str, Any]:
    if not left or len(left) != len(right):
        raise ValueError("paired, nonempty reviewer decisions are required")
    for value in [*left, *right]:
        text(value, "review decision")
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    a, b = Counter(left), Counter(right)
    expected = sum(a[k] * b[k] for k in a.keys() | b.keys()) / len(left) ** 2
    return {
        "pairs": len(left),
        "agreement": observed,
        "expected_agreement": expected,
        "kappa": (observed - expected) / (1 - expected) if expected < 1 else None,
        "scope": "paired_categorical_agreement_not_annotation_correctness",
    }


def validate_run_matrix(protocol: Mapping, rows: Sequence[Mapping]) -> dict[str, Any]:
    fields(protocol, {"methods", "seeds", "holdout_sha256", "budget"}, optional={"cost_weights"})
    methods, seeds = protocol["methods"], protocol["seeds"]
    if (
        not isinstance(methods, list)
        or not methods
        or len(set(methods)) != len(methods)
        or not isinstance(seeds, list)
        or not seeds
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("methods/seeds must be unique nonempty arrays")
    for method in methods:
        text(method, "method")
    for seed in seeds:
        integer(seed, "seed")
    holdout = digest(protocol["holdout_sha256"])
    budget = number(protocol["budget"], "budget", minimum=0)
    expected = {(m, s) for m in methods for s in seeds}
    observed, run_ids, failed = set(), set(), []
    normalized = []
    for row in rows:
        fields(row, {"method", "seed", "run_id", "status", "cost", "holdout_sha256", "metric"})
        key = row["method"], integer(row["seed"], "seed")
        if key not in expected or key in observed or text(row["run_id"], "run_id") in run_ids:
            raise ValueError("unregistered or duplicate method/seed/run")
        if digest(row["holdout_sha256"]) != holdout:
            raise ValueError("holdout identity changed across comparisons")
        cost = number(row["cost"], "cost", minimum=0)
        if cost > budget:
            raise ValueError("run exceeds the preregistered common budget")
        if row["status"] not in {"success", "failed", "cancelled"}:
            raise ValueError("invalid run outcome")
        if row["status"] == "success":
            number(row["metric"], "metric")
        elif row["metric"] is not None:
            raise ValueError("failed/cancelled runs cannot claim a valid final metric")
        else:
            failed.append(row["run_id"])
        observed.add(key)
        run_ids.add(row["run_id"])
        normalized.append(dict(row))
    missing = sorted(expected - observed)
    return {
        "schema_version": "1.0",
        "complete": not missing and not failed,
        "expected_runs": len(expected),
        "observed_runs": len(observed),
        "missing": [{"method": m, "seed": s} for m, s in missing],
        "failed": sorted(failed),
        "total_cost_all_attempts": sum(r["cost"] for r in rows),
        "rows": sorted(normalized, key=lambda r: (r["method"], r["seed"])),
    }


def paired_crossed_analysis(left, right, *, replicates=2000, seed=17, confidence=0.95):
    """Crossed generation x training-seed bootstrap on one fixed holdout.

    Resample generation IDs and initialization IDs independently, preserving all
    method pairs. Flattening the Cartesian product would exaggerate effective N.
    This is an uncertainty estimate, not proof that two seeds produce independent
    simulations, and does not include resampling the held-out route population.
    """
    if set(left) != set(right) or not left:
        raise ValueError("identically paired generation/training keys are required")
    integer(replicates, "replicates", minimum=2)
    integer(seed, "seed")
    number(confidence, "confidence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    for key in left:
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("keys must be (generation_seed, training_seed)")
        for item in key:
            integer(item, "seed")
    generations = sorted({g for g, _ in left})
    seeds = sorted({s for _, s in left})
    if len(generations) < 2 or len(seeds) < 2:
        raise ValueError("two or more levels of each independent factor are required")
    if set(left) != {(g, s) for g in generations for s in seeds}:
        raise ValueError("incomplete Cartesian experiment; do not drop failed cells")
    delta = np.array(
        [
            [number(right[g, s], "metric") - number(left[g, s], "metric") for s in seeds]
            for g in generations
        ]
    )
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        gs = rng.integers(len(generations), size=len(generations))
        ss = rng.integers(len(seeds), size=len(seeds))
        values.append(float(delta[np.ix_(gs, ss)].mean()))
    lower, upper = np.quantile(values, [(1 - confidence) / 2, (1 + confidence) / 2])
    return {
        "schema_version": "1.0",
        "mean_difference": float(delta.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "confidence": confidence,
        "generation_seed_count": len(generations),
        "training_seed_count": len(seeds),
        "run_pairs": len(left),
        "independent_pair_count_claimed": False,
        "replicates": replicates,
        "seed": seed,
        "interval_scope": "crossed_generation_and_training_seeds_fixed_holdout",
        "small_factor_warning": min(len(generations), len(seeds)) < 10,
    }


def run_seed_analysis(
    results_path,
    *,
    baseline,
    source_commit,
    runs_root="runs",
    run_id=None,
    replicates=2000,
    seed=17,
    confidence=0.95,
):
    """Recompute paired differences from producer-bound result rows.

    A fixed holdout and one generation condition are assumed. Use the crossed
    API for replicated generation factors; never flatten those cells into seeds.
    """
    import csv
    import json
    import re
    from pathlib import Path

    from ..artifacts import RunArtifactTracker, fingerprint_file
    from ..reproduction.review_contracts import read_json
    from ..reproduction.reviewer import _check_role

    path = Path(results_path).resolve(strict=True)
    raw = read_json(path)
    if not isinstance(raw.get("synthetic"), bool):
        raise ValueError("results must declare synthetic status")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("explicit source commit required")
    checked = _check_role(
        "results", path, {"source_commit": source_commit}, empirical=not raw["synthetic"]
    )
    by_method = {}
    for row in checked["rows"]:
        by_method.setdefault(row["method"], {})[row["seed"]] = row["metric"]
    if baseline not in by_method or len(by_method) < 2:
        raise ValueError("registered baseline plus a comparison method are required")
    comparisons = {
        method: paired_seed_analysis(
            by_method[baseline], values, replicates=replicates, seed=seed, confidence=confidence
        )
        for method, values in sorted(by_method.items())
        if method != baseline
    }
    result = {
        "schema_version": "1.0",
        "synthetic": raw["synthetic"],
        "baseline": baseline,
        "source_commit": source_commit,
        "comparisons": comparisons,
        "total_cost_all_attempts": checked["total_cost_all_attempts"],
        "scope": "paired_training_seed_analysis_fixed_generation_and_holdout",
        "statistical_significance_claimed": False,
        "acceptance_guaranteed": False,
    }
    with RunArtifactTracker(
        runs_root,
        run_id=run_id,
        input_refs=[fingerprint_file(path)],
        config={"object_type": "paired_seed_analysis", "synthetic": raw["synthetic"]},
    ) as tracker:
        report = tracker.artifact_path("seed_analysis.json")
        report.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
        tracker.register_artifact(report, role="registered_seed_analysis")
        table = tracker.artifact_path("paired_effects.csv")
        with table.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["baseline", "method", "seeds", "mean_difference", "lower", "upper"],
            )
            writer.writeheader()
            for method, row in comparisons.items():
                writer.writerow(
                    {
                        "baseline": baseline,
                        "method": method,
                        "seeds": row["seed_count"],
                        **{key: row[key] for key in ("mean_difference", "lower", "upper")},
                    }
                )
        tracker.register_artifact(table, role="manuscript_paired_effect_table")
    return {"run_dir": str(tracker.run_dir), **result}


def main(argv=None):
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run_seed_analysis(
                args.results,
                baseline=args.baseline,
                source_commit=args.source_commit,
                replicates=args.replicates,
                seed=args.seed,
                runs_root=args.runs_root,
                run_id=args.run_id,
            ),
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
