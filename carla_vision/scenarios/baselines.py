"""Published geometric selection component, not a full paper-result reproduction.

Sener & Savarese, ICLR 2018, arXiv:1708.00489, Algorithm 1 (k-center greedy).
This does not implement their robust mixed-integer solver, nor QActor/NAS/ADAA.
Embeddings, their fit data, initialization and their computation cost must be
frozen and disclosed by the study. No oracle labels enter this component.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def kcenter_greedy(embeddings: np.ndarray, *, labeled: Sequence[int], count: int) -> list[int]:
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("finite nonempty (samples, features) embeddings required")
    if (
        not labeled
        or any(
            isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(x) for i in labeled
        )
        or len(set(labeled)) != len(labeled)
    ):
        raise ValueError("unique valid initial labeled indices are required")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 0 <= count <= len(x) - len(labeled)
    ):
        raise ValueError("invalid batch count")
    minimum = np.full(len(x), np.inf)
    available = np.ones(len(x), dtype=bool)
    available[list(labeled)] = False

    def update(index):
        distance = np.linalg.norm(x - x[index], axis=1)
        if not np.isfinite(distance).all():
            raise ValueError("embedding distance overflow")
        np.minimum(minimum, distance, out=minimum)

    for index in labeled:
        update(index)
    selected = []
    for _ in range(count):
        index = int(np.argmax(np.where(available, minimum, -np.inf)))
        selected.append(index)
        available[index] = False
        update(index)
    return selected


def select_feature_pool(protocol_path, *, runs_root="runs", run_id=None):
    """Execute the cited greedy algorithm on a frozen, training-side feature pool.

    Units are rows with equal acquisition cost. This does not silently adapt the
    paper's k-center objective to variable-cost scenario generation.
    """
    import json
    from pathlib import Path

    from ..artifacts import RunArtifactTracker, fingerprint_file
    from ..dataset.lineage import require_train_parent
    from ..reproduction.review_contracts import digest, fields, integer, read_json, text

    protocol_path = Path(protocol_path).resolve(strict=True)
    raw = read_json(protocol_path)
    fields(
        raw, {"schema_version", "features", "feature_model", "items", "count", "unit", "synthetic"}
    )
    if (
        raw["schema_version"] != "1.0"
        or raw["unit"] != "equal_cost_feature_row"
        or not isinstance(raw["synthetic"], bool)
    ):
        raise ValueError("unsupported core-set feature pool")
    for ref in (raw["features"], raw["feature_model"]):
        fields(ref, {"path", "sha256", "size_bytes"})
        digest(ref["sha256"])
        integer(ref["size_bytes"], "source size", minimum=1)
        if fingerprint_file(ref["path"]) != ref:
            raise ValueError("core-set input fingerprint changed")
    features = np.load(raw["features"]["path"], allow_pickle=False)
    if features.ndim != 2 or len(features) != len(raw["items"]):
        raise ValueError("feature rows do not match item inventory")
    labeled, ids = [], set()
    for i, item in enumerate(raw["items"]):
        fields(item, {"sample_id", "group_id", "partition", "labeled"})
        text(item["sample_id"], "sample ID")
        text(item["group_id"], "group ID")
        require_train_parent(item["partition"])
        if item["sample_id"] in ids or not isinstance(item["labeled"], bool):
            raise ValueError("duplicate item or invalid labeling status")
        ids.add(item["sample_id"])
        if item["labeled"]:
            labeled.append(i)
    indices = kcenter_greedy(features, labeled=labeled, count=raw["count"])
    result = {
        "schema_version": "1.0",
        "algorithm": "SenerSavarese2018_Algorithm1_greedy_k_center",
        "citation": "arXiv:1708.00489",
        "unit": raw["unit"],
        "selected": [raw["items"][i] for i in indices],
        "selected_indices": indices,
        "synthetic": raw["synthetic"],
        "empirical_results_available": False,
        "limitations": [
            "Not robust mixed-integer core-set optimization",
            "Not QActor, NAS or noisy-oracle detection",
            "Frozen embeddings and equal-cost rows only",
        ],
    }
    with RunArtifactTracker(
        runs_root,
        run_id=run_id,
        config={"object_type": "published_greedy_coreset", "synthetic": raw["synthetic"]},
        input_refs=[fingerprint_file(protocol_path), raw["features"], raw["feature_model"]],
    ) as tracker:
        path = tracker.artifact_path("selection.json")
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tracker.register_artifact(path, role="published_coreset_selection")
    return {"run_dir": str(tracker.run_dir), **result}


def main(argv=None):
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            select_feature_pool(args.protocol, runs_root=args.runs_root, run_id=args.run_id),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
