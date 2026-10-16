"""Read verified shards into a backend-neutral training manifest, without RGB copies."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import fingerprint_file
from ..dataset.lineage import scene_group_key
from ..dataset.shards import iter_verified_shards
from ..dataset.verified import DatasetIntegrityError, VerifiedDataset, load_verified_dataset
from ..reproduction.review_contracts import read_json


@dataclass(frozen=True)
class VerifiedShardedTrainingSet:
    root: Path
    dataset_id: str
    sample_count: int
    partitions: dict[str, int]
    categories: dict[int, str]
    image_directories: dict[str, tuple[Path, ...]]
    reference: dict[str, Any]

    def resolve_training_partitions(self, **kwargs):
        # One canonical train/val role policy for single releases and shards.
        return VerifiedDataset.resolve_training_partitions(self, **kwargs)


def load_training_dataset(path):
    root = Path(path).expanduser().resolve(strict=True)
    if not (root / "shards.json").is_file():
        return load_verified_dataset(root)
    counts, categories, directories = Counter(), None, {}
    groups, rgb_hashes, references = {}, {}, []
    for dataset in iter_verified_shards(root):
        if categories is None:
            categories = dict(dataset.categories)
        elif categories != dict(dataset.categories):
            raise DatasetIntegrityError("shards use inconsistent class ontology")
        counts.update(dataset.partitions)
        references.append(dataset.reference)
        for partition in dataset.partitions:
            image_dir = dataset.root / "images" / partition
            label_dir = dataset.root / "labels" / partition
            if not image_dir.is_dir() or not label_dir.is_dir():
                raise DatasetIntegrityError("shard lacks canonical images/labels directories")
            directories.setdefault(partition, []).append(image_dir)
        for sample in dataset.dataset["samples"]:
            metadata = read_json(dataset.root / sample["metadata"]["path"])
            group = scene_group_key(
                metadata["context"], episode_id=f"{sample['scenario_id']}/{sample['episode_id']}"
            )
            partition = sample["split"]
            if groups.setdefault(group, partition) != partition:
                raise DatasetIntegrityError("scene group crosses shard partitions")
            digest = sample["rgb"]["sha256"]
            if rgb_hashes.setdefault(digest, partition) != partition:
                raise DatasetIntegrityError("exact RGB hash crosses shard partitions")
    index = read_json(root / "shards.json")
    return VerifiedShardedTrainingSet(
        root,
        index["dataset_id"],
        sum(counts.values()),
        dict(counts),
        categories or {},
        {k: tuple(v) for k, v in directories.items()},
        {
            "kind": "verified_sharded_training_release",
            "dataset_id": index["dataset_id"],
            "index": fingerprint_file(root / "shards.json"),
            "shards": references,
            "sample_count": sum(counts.values()),
            "partitions": dict(counts),
        },
    )
