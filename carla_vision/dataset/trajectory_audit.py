"""Derive corridor audits from verified captured positions, not user-drawn paths.

The audit checks observed, piecewise-linear 3D trajectories at the saved sampling
rate. It cannot certify unobserved motion, simulator determinism, or all possible
roads between the recorded positions. A maximum inter-sample distance bounds the
accepted interpolation; long gaps fail rather than being declared independent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from ..artifacts import RunArtifactTracker, fingerprint_file
from ..reproduction.review_contracts import digest, fields, number, read_json, text
from .leakage import audit_corridors
from .lineage import canonical_digest, scene_group_key
from .shards import expand_dataset_roots
from .verified import load_verified_dataset


def captured_routes(dataset_roots, map_documents, *, maximum_step_m: float):
    number(maximum_step_m, "maximum_step_m", minimum=0)
    if maximum_step_m <= 0:
        raise ValueError("maximum interpolation step must be positive")
    maps, references = {}, []
    for path in map_documents:
        path = Path(path).resolve(strict=True)
        raw = read_json(path)
        fields(
            raw,
            {"schema_version", "map_name", "opendrive_sha256", "opendrive", "source", "synthetic"},
        )
        if raw["schema_version"] != "1.0" or raw["source"] != "world.get_map().to_opendrive":
            raise ValueError("map description has no captured-source contract")
        text(raw["map_name"], "map_name")
        document = text(raw["opendrive"], "OpenDRIVE")
        identity = digest(raw["opendrive_sha256"])
        if hashlib.sha256(document.encode("utf-8")).hexdigest() != identity:
            raise ValueError("map description hash differs from its actual bytes")
        if identity in maps:
            raise ValueError("duplicate map source description")
        maps[identity] = raw
        references.append(fingerprint_file(path))
    groups = defaultdict(list)
    sources = []
    source_ids = set()
    for original in dataset_roots:
        original = Path(original).resolve(strict=True)
        identity_path = original / (
            "shards.json" if (original / "shards.json").is_file() else "dataset.json"
        )
        source_id = text(read_json(identity_path)["dataset_id"], "dataset identity")
        if source_id in source_ids:
            raise ValueError("duplicate trajectory release identity")
        source_ids.add(source_id)
        sources.extend((source_id, root) for root in expand_dataset_roots([original]))
        if identity_path.name == "shards.json":
            references.append(fingerprint_file(identity_path))
    if len({root for _, root in sources}) != len(sources):
        raise ValueError("duplicate trajectory source/shard")
    for source_id, root in sources:
        dataset = load_verified_dataset(root)
        references.append(fingerprint_file(root / "manifest.json"))
        for sample in dataset.dataset["samples"]:
            metadata = read_json(root / sample["metadata"]["path"])
            context = metadata["context"]
            identity = context.get("opendrive_sha256")
            if context.get("map_evidence_available") is not True or identity not in maps:
                raise ValueError(
                    "recorded sample lacks matching OpenDRIVE evidence; no route-OOD claim"
                )
            if maps[identity]["map_name"] != context.get("map"):
                raise ValueError("map identity/name differs from recorded sample")
            pose = context["privileged_evaluation"]["world_transform"]
            point = [number(pose[axis], axis) for axis in ("x", "y", "z")]
            group = scene_group_key(
                context, episode_id=f"{sample['scenario_id']}/{sample['episode_id']}"
            )
            key = (
                source_id,
                sample["scenario_id"],
                sample["episode_id"],
                group,
                sample["split"],
                identity,
            )
            groups[key].append((sample["carla_frame"], point))
    routes = []
    import math

    for key, observations in sorted(groups.items()):
        observations.sort()
        frames = [frame for frame, _ in observations]
        if len(set(frames)) != len(frames):
            raise ValueError("duplicate captured frame in trajectory")
        points = []
        for _, point in observations:
            if not points or point != points[-1]:
                if points and math.dist(points[-1], point) > maximum_step_m:
                    raise ValueError("trajectory gap exceeds registered interpolation distance")
                points.append(point)
        if len(points) < 2:
            raise ValueError(
                "stationary/undersampled trajectory cannot establish corridor separation"
            )
        routes.append(
            {
                "route_id": canonical_digest(key),
                "scene_root_id": key[3],
                "partition": key[4],
                "map_sha256": key[5],
                "points_xyz": points,
            }
        )
    if not routes:
        raise ValueError("no recorded trajectory evidence")
    return routes, references


def run_trajectory_audit(
    dataset_roots,
    map_documents,
    *,
    clearance_m,
    maximum_step_m,
    maximum_segment_checks=1_000_000,
    runs_root="runs",
    run_id=None,
):
    routes, refs = captured_routes(dataset_roots, map_documents, maximum_step_m=maximum_step_m)
    result = audit_corridors(
        routes, clearance_m=clearance_m, maximum_segment_checks=maximum_segment_checks
    )
    payload = {
        "schema_version": "1.0",
        "source_kind": "verified_capture_trajectories",
        "routes": routes,
        "clearance_m": clearance_m,
        "maximum_step_m": maximum_step_m,
        "maximum_segment_checks": maximum_segment_checks,
        "source_artifacts": refs,
        "datasets": [str(Path(p).resolve()) for p in dataset_roots],
        "map_documents": [fingerprint_file(p) for p in map_documents],
        "result": result,
        "empirical_model_improvement": False,
    }
    with RunArtifactTracker(
        runs_root,
        run_id=run_id,
        input_refs=refs,
        config={"object_type": "captured_trajectory_audit"},
    ) as tracker:
        path = tracker.artifact_path("trajectory_audit.json")
        path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
        tracker.register_artifact(path, role="captured_trajectory_audit")
    return {"run_dir": str(tracker.run_dir), **result}


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--map-document", action="append", required=True)
    parser.add_argument("--clearance-m", type=float, required=True)
    parser.add_argument("--maximum-step-m", type=float, required=True)
    parser.add_argument("--maximum-segment-checks", type=int, default=1_000_000)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    result = run_trajectory_audit(
        args.dataset,
        args.map_document,
        clearance_m=args.clearance_m,
        maximum_step_m=args.maximum_step_m,
        maximum_segment_checks=args.maximum_segment_checks,
        runs_root=args.runs_root,
        run_id=args.run_id,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
