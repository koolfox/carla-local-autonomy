"""Storage/lineage tests use deterministic pixels, not simulated driving evidence."""

from __future__ import annotations

import json

import pytest


def pair(frame):
    from test_dataset_writer import synchronized_pair

    return synchronized_pair(carla_frame=frame, timestamp=frame * 0.1)[0]


def test_shards_seal_rollover_and_stream_read(tmp_path):
    from carla_vision.dataset.shards import ShardedDatasetWriter, iter_verified_shards

    with ShardedDatasetWriter(
        tmp_path,
        dataset_id="demo",
        shard_samples=2,
        carla_endpoint={"host": "fixture"},
        carla_version="0.9.16",
        carla_map="fixture",
        config={"synthetic": True},
    ) as writer:
        for i in range(1, 6):
            writer.add_pair(pair(i), split="train", scenario_id="s", episode_id="e")
        assert writer.buffered_sample_count <= 2
    shards = list(iter_verified_shards(tmp_path / "demo"))
    assert [len(s.dataset["samples"]) for s in shards] == [2, 2, 1]
    assert sum(len(s.dataset["samples"]) for s in shards) == 5
    raw = json.loads((tmp_path / "demo/shards.json").read_text())
    assert raw["status"] == "complete"
    with pytest.raises(FileExistsError):
        ShardedDatasetWriter(
            tmp_path,
            dataset_id="demo",
            shard_samples=2,
            carla_endpoint={},
            carla_version="x",
            carla_map="x",
        )


def test_failed_sharded_capture_never_looks_complete(tmp_path):
    from carla_vision.dataset.shards import (
        ShardedDatasetWriter,
        iter_verified_shards,
        recover_shards,
    )

    with pytest.raises(RuntimeError, match="injected"):
        with ShardedDatasetWriter(
            tmp_path,
            dataset_id="bad",
            shard_samples=1,
            carla_endpoint={},
            carla_version="x",
            carla_map="x",
        ) as writer:
            writer.add_pair(pair(1), split="train", scenario_id="s", episode_id="e")
            raise RuntimeError("injected")
    with pytest.raises(ValueError, match="complete"):
        list(iter_verified_shards(tmp_path / "bad"))
    report = recover_shards(tmp_path / "bad")
    assert report["status"] == "incomplete"
    assert report["auto_promoted"] is False


def test_shards_cross_boundary_frame_and_partition_checks(tmp_path):
    from carla_vision.dataset.shards import ShardedDatasetWriter

    with ShardedDatasetWriter(
        tmp_path,
        dataset_id="x",
        shard_samples=1,
        carla_endpoint={},
        carla_version="x",
        carla_map="x",
    ) as writer:
        writer.add_pair(pair(2), split="train", scenario_id="s", episode_id="e")
        with pytest.raises(ValueError, match="frame"):
            writer.add_pair(pair(1), split="train", scenario_id="s", episode_id="e")
        with pytest.raises(ValueError, match="partition"):
            writer.add_pair(pair(3), split="test", scenario_id="s", episode_id="e")


def test_shards_reject_unregistered_and_changed_shard(tmp_path):
    from carla_vision.dataset.shards import ShardedDatasetWriter, iter_verified_shards

    with ShardedDatasetWriter(
        tmp_path,
        dataset_id="x",
        shard_samples=1,
        carla_endpoint={},
        carla_version="x",
        carla_map="x",
    ) as writer:
        writer.add_pair(pair(2), split="train", scenario_id="s", episode_id="e")
    extra = tmp_path / "x/shards/not-registered"
    extra.mkdir()
    with pytest.raises(ValueError, match="unregistered"):
        list(iter_verified_shards(tmp_path / "x"))
