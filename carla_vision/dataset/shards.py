"""Bounded-sample shards that reuse the canonical writer and semantic verifier.

Every sealed shard is immutable. A crash/failed release may be inventoried, but
is never silently promoted. Continuation starts a new explicitly derived release;
this API intentionally does not reconstruct a partial sensor frame from disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from ..artifacts import fingerprint_file
from ..reproduction.review_contracts import digest, fields, integer, read_json, relative_path
from .lineage import scene_group_key
from .verified import load_verified_dataset
from .writer import DatasetWriter, _write_json


class ShardedDatasetWriter:
    def __init__(self, datasets_root, *, dataset_id, shard_samples, **writer_options):
        integer(shard_samples, "shard_samples", minimum=1)
        if shard_samples > 100_000:
            raise ValueError("shard_samples exceeds the canonical writer sample bound")
        # Use the existing ID validation instead of allowing a path as dataset ID.
        from ..artifacts import RunArtifactTracker

        RunArtifactTracker._validate_run_id(dataset_id)
        self.dataset_dir = Path(datasets_root).expanduser().resolve() / dataset_id
        self.dataset_dir.mkdir(parents=True, exist_ok=False)
        (self.dataset_dir / "shards").mkdir()
        self.dataset_id, self.shard_samples = dataset_id, shard_samples
        self.options = dict(writer_options)
        self.shards, self.auxiliary, self.metadata = [], [], {}
        self.current = None
        self._current_count = 0
        self.entered = False
        self.closed = False
        self.groups, self.last_frames = {}, {}
        self._publish("created")

    @property
    def buffered_sample_count(self):
        return self._current_count

    def _publish(self, status, *, failure=None):
        _write_json(
            self.dataset_dir / "shards.json",
            {
                "schema_version": "1.0",
                "object_type": "sharded_dataset",
                "dataset_id": self.dataset_id,
                "status": status,
                "shard_sample_limit": self.shard_samples,
                "sample_count": sum(s["sample_count"] for s in self.shards),
                "shards": self.shards,
                "auxiliary": self.auxiliary,
                "release_metadata": self.metadata,
                "failure": failure,
                "live_capture_verified": False,
            },
        )

    def __enter__(self):
        if self.entered:
            raise RuntimeError("sharded writer already entered")
        self.entered = True
        self._publish("running")
        return self

    def _start_shard(self):
        name = f"shard-{len(self.shards):06d}"
        options = dict(self.options)
        options["maximum_samples"] = self.shard_samples
        self.current = DatasetWriter(self.dataset_dir / "shards", dataset_id=name, **options)
        self.current.__enter__()
        self._current_count = 0

    def _seal(self):
        if self.current is None:
            return
        writer = self.current
        local = {
            "shard_parent_id": self.dataset_id,
            "episode_ids": sorted({s.episode_id for s in writer.samples}),
            "control_mode": self.options.get("config", {}).get("control_mode"),
        }
        writer.set_release_metadata(local)
        count = len(writer.samples)
        writer.__exit__(None, None, None)
        reference = fingerprint_file(writer.manifest_path)
        reference["path"] = writer.manifest_path.relative_to(self.dataset_dir).as_posix()
        self.shards.append({"manifest": reference, "sample_count": count})
        self.current = None
        self._current_count = 0
        self._publish("running")

    def add_pair(self, pair, *, split, scenario_id, episode_id, context=None, **kwargs):
        if not self.entered or self.closed:
            raise RuntimeError("writer is not active")
        identity = (scenario_id, episode_id)
        root_id = scene_group_key(context or {}, episode_id=f"{scenario_id}/{episode_id}")
        previous = self.groups.get(root_id)
        if previous is not None and previous != split:
            raise ValueError("scene group crosses shard partitions")
        last = self.last_frames.get(identity)
        if last is not None and pair.rgb.frame <= last:
            raise ValueError("frame is repeated/out of order across shard boundary")
        if self.current is None:
            self._start_shard()
        result = self.current.add_pair(
            pair,
            split=split,
            scenario_id=scenario_id,
            episode_id=episode_id,
            context=context,
            **kwargs,
        )
        self.groups[root_id] = split
        self.last_frames[identity] = pair.rgb.frame
        self._current_count += 1
        if self._current_count == self.shard_samples:
            self._seal()
        return result

    def add_auxiliary_json(self, path, payload, *, role, metadata=None):
        if not self.entered or self.closed:
            raise RuntimeError("writer is not active")
        target = relative_path(self.dataset_dir, str(path), must_exist=False)
        if target.name == "shards.json" or target.is_relative_to(self.dataset_dir / "shards"):
            raise ValueError("reserved sharded artifact path")
        if target.exists():
            raise FileExistsError(target)
        _write_json(target, dict(payload))
        reference = fingerprint_file(target)
        reference["path"] = target.relative_to(self.dataset_dir).as_posix()
        self.auxiliary.append(dict(reference, role=role, metadata=dict(metadata or {})))
        self._publish("running")
        return target

    def set_release_metadata(self, metadata):
        if self.closed:
            raise RuntimeError("writer closed")
        self.metadata = dict(metadata)

    def __exit__(self, kind, error, traceback):
        if self.closed:
            raise RuntimeError("writer already closed")
        try:
            if error is None:
                self._seal()
                if not self.shards:
                    raise ValueError("empty sharded release")
                self._publish("complete")
            else:
                if self.current is not None:
                    self.current.__exit__(kind, error, traceback)
                    self.current = None
                self._publish("failed", failure=type(error).__name__)
        except BaseException as secondary:
            try:
                self._publish("failed", failure=type(secondary).__name__)
            except BaseException as final_error:
                secondary.add_note(
                    "Shard failure status also failed: " + type(final_error).__name__
                )
            if error is None:
                raise
            error.add_note("Sharded finalization also failed: " + type(secondary).__name__)
        finally:
            self.closed = True


def iter_verified_shards(root):
    """Yield one semantically verified shard at a time; never materialize all RGB."""
    root = Path(root).resolve(strict=True)
    data = read_json(root / "shards.json")
    fields(
        data,
        {
            "schema_version",
            "object_type",
            "dataset_id",
            "status",
            "shard_sample_limit",
            "sample_count",
            "shards",
            "auxiliary",
            "release_metadata",
            "failure",
            "live_capture_verified",
        },
    )
    if (
        data["schema_version"] != "1.0"
        or data["object_type"] != "sharded_dataset"
        or data["status"] != "complete"
    ):
        raise ValueError("sharded release is not complete")
    limit = integer(data["shard_sample_limit"], "shard limit", minimum=1)
    integer(data["sample_count"], "sample count", minimum=1)
    if not isinstance(data["shards"], list) or not data["shards"]:
        raise ValueError("empty shard inventory")
    expected_dirs = {f"shard-{i:06d}" for i in range(len(data["shards"]))}
    if {p.name for p in (root / "shards").iterdir()} != expected_dirs:
        raise ValueError("unregistered or missing shard directory")
    for artifact in data["auxiliary"]:
        target = relative_path(root, artifact["path"])
        actual = fingerprint_file(target)
        if any(actual[k] != artifact[k] for k in ("sha256", "size_bytes")):
            raise ValueError("auxiliary digest changed")
    expected_aux = {a["path"] for a in data["auxiliary"]} | {"shards.json"}
    if len(expected_aux) != len(data["auxiliary"]) + 1:
        raise ValueError("duplicate auxiliary artifact")

    def outside_files(directory):
        for p in directory.iterdir():
            if p.is_symlink():
                raise ValueError("symlink in sharded release")
            if p == root / "shards":
                continue
            if p.is_dir():
                yield from outside_files(p)
            elif p.is_file():
                yield p.relative_to(root).as_posix()

    if set(outside_files(root)) != expected_aux:
        raise ValueError("unregistered root artifact in sharded release")
    groups, frames, count = {}, {}, 0
    for i, shard in enumerate(data["shards"]):
        fields(shard, {"manifest", "sample_count"})
        integer(shard["sample_count"], "shard sample count", minimum=1)
        ref = fields(shard["manifest"], {"path", "sha256", "size_bytes"})
        digest(ref["sha256"])
        integer(ref["size_bytes"], "manifest size", minimum=1)
        if ref["path"] != f"shards/shard-{i:06d}/manifest.json":
            raise ValueError("shard manifest path/order differs")
        path = relative_path(root, ref["path"])
        actual = fingerprint_file(path)
        if any(actual[k] != ref[k] for k in ("sha256", "size_bytes")):
            raise ValueError("shard manifest digest/size mismatch")
        verified = load_verified_dataset(path.parent)
        samples = verified.dataset["samples"]
        if len(samples) != shard["sample_count"] or not 0 < len(samples) <= limit:
            raise ValueError("shard sample count mismatch/overflow")
        count += len(samples)
        for sample in samples:
            metadata = read_json(relative_path(verified.root, sample["metadata"]["path"]))
            key = scene_group_key(
                metadata.get("context", {}),
                episode_id=f"{sample['scenario_id']}/{sample['episode_id']}",
            )
            previous = groups.setdefault(key, sample["split"])
            if previous != sample["split"]:
                raise ValueError("group partition changes across shards")
            identity = (sample["scenario_id"], sample["episode_id"])
            last = frames.get(identity)
            if last is not None and sample["carla_frame"] <= last:
                raise ValueError("frame repeated/out of order across shards")
            frames[identity] = sample["carla_frame"]
        yield verified
    if count != data["sample_count"]:
        raise ValueError("total shard sample count differs")


def recover_shards(root):
    """Read-only crash inventory; incomplete/changed shards are retained, not repaired."""
    root = Path(root).resolve(strict=True)
    valid, incomplete = [], []
    for path in sorted((root / "shards").iterdir()):
        try:
            if path.is_symlink() or not path.is_dir():
                raise ValueError("invalid shard path")
            verified = load_verified_dataset(path)
            valid.append(
                {
                    "path": path.name,
                    "sample_count": len(verified.dataset["samples"]),
                    "manifest": fingerprint_file(path / "manifest.json"),
                }
            )
        except (OSError, ValueError, RuntimeError) as error:
            incomplete.append({"path": path.name, "error": type(error).__name__})
    return {
        "status": "incomplete",
        "auto_promoted": False,
        "sealed": valid,
        "incomplete": incomplete,
        "action": "retain_original_and_start_explicit_new_release",
    }


def expand_dataset_roots(roots):
    """Canonical consumer bridge. Fully validate an index before returning any root."""
    result = []
    for raw in roots:
        path = Path(raw).resolve(strict=True)
        if (path / "shards.json").is_file():
            result.extend(s.root for s in iter_verified_shards(path))
        else:
            result.append(path)
    if len(result) != len(set(result)):
        raise ValueError("duplicate source dataset/shard")
    return tuple(result)


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--recover-inventory", action="store_true")
    args = parser.parse_args(argv)
    if args.recover_inventory:
        report = recover_shards(args.root)
    else:
        counts = [len(s.dataset["samples"]) for s in iter_verified_shards(args.root)]
        report = {"status": "passed", "shards": len(counts), "samples": sum(counts)}
    print(json.dumps(report, sort_keys=True, indent=2))
    return 1 if args.recover_inventory else 0


if __name__ == "__main__":
    sys.exit(main())
