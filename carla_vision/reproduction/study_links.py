"""Bind reviewer roles to one frozen study; never certify empirical truth.

A split inventory is rebuilt from immutable verified releases, not from operator-
supplied group names. This module checks relationships *in addition* to the
individual role verifiers. It neither executes models nor reads a paper's prose.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..artifacts import fingerprint_file
from ..dataset.lineage import canonical_digest, partition_role, scene_group_key
from ..dataset.shards import expand_dataset_roots
from ..dataset.verified import load_verified_dataset
from ..scenarios.costs import verify_cost_ledger
from .review_contracts import (
    digest,
    fields,
    integer,
    parse_json,
    read_json,
    relative_path,
    text,
)

SPLIT_SCHEMA = "1.1"
MAXIMUM_SAMPLES = 1_000_000
LINK_ROLES = frozenset(
    {
        "protocol",
        "results",
        "geometry",
        "claim_registry",
        "manuscript",
        "label_audit",
        "baseline_registry",
        "costs",
    }
)


def _path(value: str | Path) -> Path:
    candidate = Path(value).absolute()
    if any(p.is_symlink() for p in (candidate, *candidate.parents)):
        raise ValueError("symlink evidence is not allowed")
    return candidate.resolve(strict=True)


def _reference(base: Path, value: Any) -> Path:
    fields(value, {"path", "sha256", "size_bytes"})
    digest(value["sha256"])
    integer(value["size_bytes"], "artifact size", minimum=1)
    raw = Path(text(value["path"], "artifact path"))
    path = _path(raw if raw.is_absolute() else relative_path(base, value["path"]))
    actual = fingerprint_file(path)
    if any(actual[k] != value[k] for k in ("sha256", "size_bytes")):
        raise ValueError("study source reference digest/size changed")
    return path


def _object(path: Path) -> Mapping[str, Any]:
    if path.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("study evidence exceeds JSON inspection budget")
    raw = read_json(path)
    if not isinstance(raw, Mapping):
        raise ValueError("study evidence must be a JSON object")
    return raw


def _inventory(roots: Sequence[str | Path], maximum_samples: int) -> dict[str, Any]:
    integer(maximum_samples, "sample budget", minimum=1)
    if maximum_samples > MAXIMUM_SAMPLES:
        raise ValueError("sample budget exceeds supported inventory capacity")
    if isinstance(roots, (str, bytes)) or not roots:
        raise ValueError("nonempty dataset root list required")
    originals = [_path(root) for root in roots]
    if len(set(originals)) != len(originals):
        raise ValueError("duplicate dataset roots")
    descriptors, manifests, samples = [], [], []
    groups, rgb_splits, seen_shards, identities = {}, {}, set(), set()
    for original in sorted(originals):
        descriptor = original / (
            "shards.json" if (original / "shards.json").is_file() else "manifest.json"
        )
        descriptor_ref = fingerprint_file(_path(descriptor))
        descriptors.append(descriptor_ref)
        for root in expand_dataset_roots([original]):
            if root in seen_shards:
                raise ValueError("duplicate canonical source/shard")
            seen_shards.add(root)
            dataset = load_verified_dataset(root)
            manifest_ref = fingerprint_file(root / "manifest.json")
            manifests.append(manifest_ref)
            if len(samples) + len(dataset.dataset["samples"]) > maximum_samples:
                raise ValueError("sample inventory budget exceeded; nothing published")
            for sample in dataset.dataset["samples"]:
                partition = sample["split"]
                if partition_role(partition) == "unassigned":
                    raise ValueError("unassigned data cannot define a frozen study split")
                context = read_json(root / sample["metadata"]["path"])["context"]
                group = scene_group_key(
                    context, episode_id=f"{sample['scenario_id']}/{sample['episode_id']}"
                )
                rgb = sample["rgb"]["sha256"]
                key = (manifest_ref["sha256"], sample["sample_id"])
                if key in identities:
                    raise ValueError("duplicate source sample identity")
                identities.add(key)
                if groups.setdefault(group, partition) != partition:
                    raise ValueError("one scene group crosses source partitions")
                if rgb_splits.setdefault(rgb, partition) != partition:
                    raise ValueError("exact RGB overlap crosses source partitions")
                samples.append(
                    {
                        "manifest_sha256": key[0],
                        "dataset_id": dataset.dataset_id,
                        "sample_id": key[1],
                        "group_id": group,
                        "partition": partition,
                        "rgb_sha256": rgb,
                    }
                )
            if fingerprint_file(root / "manifest.json") != manifest_ref:
                raise ValueError("source manifest changed during inventory")
        if fingerprint_file(descriptor) != descriptor_ref:
            raise ValueError("source release descriptor changed during inventory")
    if not samples:
        raise ValueError("empty split inventory")
    samples.sort(key=lambda row: (row["manifest_sha256"], row["sample_id"]))
    holdout = [s for s in samples if partition_role(s["partition"]) == "test"]
    return {
        "schema_version": SPLIT_SCHEMA,
        "groups": dict(sorted(groups.items())),
        "release_descriptors": sorted(descriptors, key=lambda r: (r["sha256"], r["path"])),
        "dataset_manifests": sorted(manifests, key=lambda r: (r["sha256"], r["path"])),
        "samples": samples,
        "holdout_sha256": canonical_digest(holdout),
        "holdout_count": len(holdout),
        "maximum_samples": maximum_samples,
    }


def freeze_split(
    dataset_roots: Sequence[str | Path],
    output: str | Path,
    *,
    maximum_samples: int = 100_000,
) -> dict[str, Any]:
    """Publish a new sample-level inventory, preserving all original assignments."""
    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    raw = _inventory(dataset_roots, maximum_samples)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _path(destination.parent)
    descriptor, name = tempfile.mkstemp(prefix=".split-", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(raw, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            if os.fstat(stream.fileno()).st_size > 256 * 1024 * 1024:
                raise ValueError("split exceeds JSON inspection budget; nothing published")
        # Link publication is atomic and fails if another writer won the name.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return raw


def verify_split(path: str | Path) -> dict[str, Any]:
    """Recompute membership and holdout identity from all verified source bytes."""
    path = _path(path)
    raw = _object(path)
    fields(
        raw,
        {
            "schema_version",
            "groups",
            "release_descriptors",
            "dataset_manifests",
            "samples",
            "holdout_sha256",
            "holdout_count",
            "maximum_samples",
        },
    )
    if raw["schema_version"] != SPLIT_SCHEMA:
        raise ValueError("empirical study requires source-bound split schema 1.1")
    refs = raw["release_descriptors"]
    if not isinstance(refs, list) or not refs:
        raise ValueError("split requires source release descriptors")
    roots = []
    for ref in refs:
        descriptor = _reference(path.parent, ref)
        if descriptor.name not in {"manifest.json", "shards.json"}:
            raise ValueError("split descriptor must be a canonical release manifest")
        roots.append(descriptor.parent)
    computed = _inventory(roots, raw["maximum_samples"])
    # JSON comparison is type-sensitive (True must not impersonate count 1).
    if canonical_digest(computed) != canonical_digest(raw):
        raise ValueError("split inventory differs from verified source membership")
    return computed


def _unique(value: Any, name: str, *, seeds: bool = False) -> set:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    result = []
    for item in value:
        result.append(integer(item, name) if seeds else text(item, name))
    if len(set(result)) != len(result):
        raise ValueError(f"duplicate {name}")
    return set(result)


def _cost_events(path: Path) -> dict[str, dict]:
    verify_cost_ledger(path)
    events = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = parse_json(line)
        events[row["event_id"]] = {
            key: row[key] for key in ("event_id", "kind", "outcome", "resources", "evidence_ref")
        }
    return events


def check_study_links(paths: Mapping[str, Path], *, study_id: str) -> dict[str, Any]:
    """Reject unrelated evidence roles after their independent semantic validation.

    This is a necessary consistency gate, not a replacement for role validators,
    held-out prediction replay, human review, or empirical hypothesis testing.
    """
    if not LINK_ROLES <= paths.keys():
        raise ValueError("cross-role study evidence is incomplete")
    paths = {key: _path(paths[key]) for key in LINK_ROLES}
    documents = {key: _object(p) for key, p in paths.items() if key != "costs"}
    protocol = documents["protocol"]
    if protocol["study_id"] != text(study_id, "study identity"):
        raise ValueError("protocol belongs to a different study")
    split = verify_split(_reference(paths["protocol"].parent, protocol["split_manifest"]))
    if not split["holdout_count"] or not any(s["partition"] == "train" for s in split["samples"]):
        raise ValueError("empirical study needs frozen training and test samples")
    methods = _unique(protocol["methods"], "methods")
    seeds = _unique(protocol["training_seeds"], "seeds", seeds=True)
    results = documents["results"]
    matrix = _object(_reference(paths["results"].parent, results["protocol"]))
    if _unique(matrix["methods"], "matrix methods") != methods:
        raise ValueError("result methods differ from the selected protocol")
    if _unique(matrix["seeds"], "matrix seeds", seeds=True) != seeds:
        raise ValueError("result seeds differ from the selected protocol")
    if digest(matrix["holdout_sha256"]) != split["holdout_sha256"]:
        raise ValueError("result holdout differs from the frozen sample inventory")
    if not methods <= _unique(documents["baseline_registry"]["methods"], "baseline methods"):
        raise ValueError("baseline registry omits a selected method")

    source_ids = {r["sha256"] for r in split["dataset_manifests"]}
    geometry_ids = set()
    for ref in documents["geometry"]["source_artifacts"]:
        _reference(paths["geometry"].parent, ref)
        geometry_ids.add(ref["sha256"])
    if not source_ids <= geometry_ids:
        raise ValueError("geometry audit does not cover all frozen study sources")

    claims = _unique([c["claim_id"] for c in documents["claim_registry"]["claims"]], "claim IDs")
    manuscript = documents["manuscript"]
    if _unique(manuscript["claims"], "manuscript claims") != claims:
        raise ValueError("manuscript claims differ from the selected claim registry")
    tables = manuscript["tables"]
    if not isinstance(tables, list) or not tables:
        raise ValueError("manuscript must list result tables")
    _unique([row["table_id"] for row in tables], "table IDs")
    for row in tables:
        fields(row, {"table_id", "results"})
        table_path = _reference(paths["manuscript"].parent, row["results"])
        if fingerprint_file(table_path)["sha256"] != fingerprint_file(paths["results"])["sha256"]:
            raise ValueError("manuscript table points to unrelated results")

    membership = {}
    for sample in split["samples"]:
        membership.setdefault((sample["sample_id"], sample["group_id"]), []).append(
            sample["manifest_sha256"]
        )
    audits = documents["label_audit"]["records"]
    if not isinstance(audits, list) or not audits:
        raise ValueError("audit records are missing")
    seen_audits = set()
    for row in audits:
        candidates = membership.get((row["sample_id"], row["group_id"]), [])
        anchor = row.get("dataset_manifest_sha256")
        if anchor is not None:
            candidates = [p for p in candidates if p == digest(anchor)]
        if len(candidates) != 1:
            raise ValueError("audit sample is ambiguous or outside the frozen study")
        key = (candidates[0], row["sample_id"])
        if key in seen_audits:
            raise ValueError("duplicate audit sample identity")
        seen_audits.add(key)

    global_events = _cost_events(paths["costs"])
    used_events = set()
    producers = results["producers"]
    if not isinstance(producers, Mapping) or not producers:
        raise ValueError("result producers are missing")
    for binding in producers.values():
        if binding["metric_field"] != protocol["primary_metric"]:
            raise ValueError("result metric field differs from the primary metric")
        producer = _reference(paths["results"].parent, binding["manifest"])
        producer_document = _object(producer)
        registered = producer_document["references"]["datasets"]
        if (
            producer_document.get("invocation", {})
            .get("config", {})
            .get("sample_consumption_required")
        ):
            from .consumption_verify import verify_producer_consumption

            trace = verify_producer_consumption(producer.parent)
            registered = [{"manifest": r} for r in trace["source_manifests"]]
        if not isinstance(registered, list) or not registered:
            raise ValueError("producer has no bound dataset sources")
        for source in registered:
            ref = source["manifest"]
            _reference(producer.parent, ref)
            if ref["sha256"] not in source_ids:
                raise ValueError("producer dataset is outside the frozen study")
        ledger = _reference(paths["results"].parent, binding["cost_ledger"])
        for event_id, event in _cost_events(ledger).items():
            if event_id in used_events:
                raise ValueError("cost event counted in multiple producers")
            if event != global_events.get(event_id):
                raise ValueError("per-run cost event omitted or changed in study ledger")
            used_events.add(event_id)
    if used_events != global_events.keys():
        raise ValueError("unallocated study cost events would disappear from method budgets")
    return {
        "status": "passed",
        "sample_count": len(split["samples"]),
        "holdout_count": split["holdout_count"],
        "holdout_sha256": split["holdout_sha256"],
        "audited_sample_count": len(seen_audits),
        "bound_cost_events": len(used_events),
        "additional_cost_events": len(global_events.keys() - used_events),
        "scope": "cross_artifact_consistency_not_empirical_validation",
    }
