"""Dataset loading for temporal RGB-only voxel occupancy training."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ...dataset.lineage import partition_role
from ...dataset.semantic_version import COMPACT_VOXEL_TAGS
from ...reproduction.review_contracts import integer, number, relative_path
from ..contracts import OCCUPIED, SEMANTIC_UNKNOWN, VoxelGridSpec

Partition = Literal["train", "val", "test"]
COMPACT_SEMANTIC_NAMES = (
    "building",
    "road",
    "sidewalk",
    "vehicle",
    "pedestrian",
    "vegetation",
    "other",
)
COMPACT_SEMANTIC_IGNORE = 255

# Only occupied voxels are supervised; this is the supported 0.9.16 table.

_SEMANTIC_TAG_TO_COMPACT = COMPACT_VOXEL_TAGS


@dataclass(frozen=True, slots=True)
class VoxelEpisode:
    root: Path
    episode_id: str
    group_key: str
    spec: VoxelGridSpec
    records: tuple[dict[str, Any], ...]
    future_targets: dict[int, dict[float, int | None]]
    source_partition: str | None = None


@dataclass(frozen=True, slots=True)
class SampleReference:
    episode_index: int
    anchor_index: int
    target_frames: tuple[int, ...]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"required voxel artifact is missing: {path}") from error


def _spec_from_payload(payload: dict[str, Any]) -> VoxelGridSpec:
    required = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "resolution")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError("voxel grid is missing fields: " + ", ".join(missing))
    return VoxelGridSpec(**{name: float(payload[name]) for name in required})


def _episode_group_key(root: Path, manifest: dict[str, Any]) -> tuple[str, str]:
    episode_id = str(manifest.get("episode_id") or manifest.get("run_id") or root.name)
    route = str(
        manifest.get("route_id")
        or manifest.get("scenario_id")
        or manifest.get("map")
        or manifest.get("map_name")
        or episode_id
    )
    lineage = manifest.get("lineage")
    if isinstance(lineage, dict) and isinstance(lineage.get("scene_root_id"), str):
        if not lineage["scene_root_id"].strip():
            raise ValueError("empty voxel scene root")
        return episode_id, "scene-root:" + lineage["scene_root_id"]
    # A route with a declared partition must stay grouped across traffic seeds.
    if manifest.get("split") is not None:
        return episode_id, "source-route:" + route
    seed = manifest.get("seed")
    if seed is None:
        seed = manifest.get("scenario_seed")
    if seed is None:
        seed = manifest.get("traffic_seed")
    if seed is None:
        seed = episode_id
    return episode_id, f"{route}|{seed}"


def load_voxel_episode(root: Path | str) -> VoxelEpisode:
    resolved = Path(root).expanduser().resolve(strict=True)
    sequence = _read_json(resolved / "sequence.json")
    manifest = _read_json(resolved / "manifest.json")
    if not isinstance(sequence, dict) or not isinstance(manifest, dict):
        raise ValueError("sequence.json and manifest.json must contain JSON objects")
    spec = _spec_from_payload(dict(sequence.get("grid") or {}))
    raw_records = sequence.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(f"voxel sequence contains no records: {resolved}")
    records = tuple(dict(record) for record in raw_records)
    for record in records:
        for name in ("rgb", "teacher_voxel", "teacher_flow"):
            if record.get(name) is not None:
                relative_path(resolved, record[name])
    source_partition = manifest.get("split")
    record_splits = {r["split"] for r in records if "split" in r}
    if record_splits:
        if len(record_splits) != 1 or any("split" not in r for r in records):
            raise ValueError("voxel temporal windows cannot cross source partitions")
        record_partition = next(iter(record_splits))
        if source_partition is not None and source_partition != record_partition:
            raise ValueError("voxel manifest and record source partitions disagree")
        source_partition = record_partition
    if source_partition is not None:
        partition_role(source_partition)
    frames = [integer(record["frame"], "voxel frame") for record in records]
    timestamps = [number(record["timestamp"], "voxel timestamp", minimum=0) for record in records]
    if any(b <= a for a, b in zip(timestamps, timestamps[1:], strict=False)):
        raise ValueError("voxel timestamps must be strictly increasing")
    time_by_frame = dict(zip(frames, timestamps, strict=True))
    if len(frames) != len(set(frames)):
        raise ValueError(f"duplicate frame IDs in voxel sequence: {resolved}")
    if frames != sorted(frames):
        raise ValueError(f"voxel records must be sorted by frame: {resolved}")

    future_targets: dict[int, dict[float, int | None]] = {}
    raw_future = sequence.get("future_targets")
    if not isinstance(raw_future, list):
        raise ValueError(f"voxel sequence future_targets must be a list: {resolved}")
    for item in raw_future:
        if not isinstance(item, dict) or "frame" not in item:
            raise ValueError("invalid future target record")
        mapping = item.get("future_targets")
        if not isinstance(mapping, dict):
            raise ValueError("future_targets mapping must be an object")
        converted: dict[float, int | None] = {}
        for horizon, target in mapping.items():
            value = float(horizon)
            if not math.isfinite(value) or value < 0:
                raise ValueError("future target horizons must be finite and non-negative")
            if value in converted:
                raise ValueError("duplicate numeric future horizon")
            converted[value] = None if target is None else integer(target, "future target frame")
        anchor = integer(item["frame"], "future anchor frame")
        if anchor not in time_by_frame or anchor in future_targets:
            raise ValueError("unknown/duplicate future target anchor")
        for horizon, target in converted.items():
            expected = next(
                (f for f in frames if time_by_frame[f] + 1e-9 >= time_by_frame[anchor] + horizon),
                None,
            )
            if target != expected:
                raise ValueError(
                    "future target must be the first recorded frame at/after its horizon"
                )
        future_targets[anchor] = converted

    episode_id, group_key = _episode_group_key(resolved, dict(manifest, split=source_partition))
    return VoxelEpisode(
        root=resolved,
        episode_id=episode_id,
        group_key=group_key,
        spec=spec,
        records=records,
        future_targets=future_targets,
        source_partition=source_partition,
    )


def stable_partition(
    group_key: str,
    *,
    split_seed: int = 17,
    val_fraction: float = 0.2,
    test_fraction: float = 0.1,
) -> Partition:
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("split fractions must be in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be less than 1")
    digest = hashlib.sha256(f"{split_seed}:{group_key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < test_fraction:
        return "test"
    if value < test_fraction + val_fraction:
        return "val"
    return "train"


def compact_semantic_labels(tags: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
    tag_array = np.asarray(tags, dtype=np.uint8)
    occupancy_array = np.asarray(occupancy, dtype=np.int8)
    if tag_array.shape != occupancy_array.shape:
        raise ValueError("semantic tags and occupancy must have the same shape")
    result = np.full(tag_array.shape, COMPACT_SEMANTIC_IGNORE, dtype=np.uint8)
    occupied = occupancy_array == OCCUPIED
    if not np.any(occupied):
        return result
    result[occupied] = len(COMPACT_SEMANTIC_NAMES) - 1
    for source, target in _SEMANTIC_TAG_TO_COMPACT.items():
        result[occupied & (tag_array == source)] = target
    result[tag_array == SEMANTIC_UNKNOWN] = COMPACT_SEMANTIC_IGNORE
    return result


def _target_for_horizon(mapping: dict[float, int | None], horizon: float) -> int | None:
    for available, target in mapping.items():
        if abs(available - horizon) <= 1e-4:
            return target
    return None


class TemporalVoxelDataset(Dataset[dict[str, Any]]):
    """Build temporal RGB samples without exposing privileged teacher inputs."""

    def __init__(
        self,
        roots: Sequence[Path | str],
        *,
        partition: Partition,
        history_frames: int = 4,
        horizons_s: Sequence[float] = (0.0, 0.5, 1.0, 2.0),
        image_size: tuple[int, int] = (192, 320),
        split_seed: int = 17,
        val_fraction: float = 0.2,
        test_fraction: float = 0.1,
        max_samples: int | None = None,
        allow_legacy_splits: bool = False,
    ) -> None:
        if history_frames <= 0:
            raise ValueError("history_frames must be positive")
        height, width = image_size
        if height <= 0 or width <= 0:
            raise ValueError("image_size must contain positive height and width")
        horizons = tuple(float(value) for value in horizons_s)
        if not horizons or any(not math.isfinite(value) or value < 0 for value in horizons):
            raise ValueError("horizons must be finite and non-negative")
        if tuple(sorted(horizons)) != horizons or len(set(horizons)) != len(horizons):
            raise ValueError("horizons must be unique and increasing")
        episodes = tuple(load_voxel_episode(root) for root in roots)
        if not episodes:
            raise ValueError("at least one voxel episode is required")
        first_spec = episodes[0].spec
        if any(episode.spec != first_spec for episode in episodes[1:]):
            raise ValueError("all voxel episodes must use the same grid specification")
        if partition not in {"train", "val", "test"}:
            raise ValueError("partition must be train, val or test")
        if any(e.source_partition is None for e in episodes):
            if not allow_legacy_splits:
                raise ValueError(
                    "voxel source partition missing; explicitly opt into development legacy splits"
                )
            warnings.warn(
                "legacy voxel repartitioning is development-only", UserWarning, stacklevel=2
            )
        self.legacy_split_used = any(e.source_partition is None for e in episodes)
        selected = tuple(
            episode
            for episode in episodes
            if (
                partition_role(episode.source_partition)
                if episode.source_partition is not None
                else stable_partition(
                    episode.group_key,
                    split_seed=split_seed,
                    val_fraction=val_fraction,
                    test_fraction=test_fraction,
                )
            )
            == partition
        )
        assignment = {}
        for episode in episodes:
            if episode.source_partition is not None:
                prior = assignment.setdefault(episode.group_key, episode.source_partition)
                if prior != episode.source_partition:
                    raise ValueError("voxel scene group crosses source partitions")
        self.episodes = selected
        self.spec = first_spec
        self.partition = partition
        self.history_frames = history_frames
        self.horizons_s = horizons
        self.image_size = (height, width)
        self.samples = self._build_samples()
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            self.samples = self.samples[:max_samples]

    def _build_samples(self) -> list[SampleReference]:
        samples: list[SampleReference] = []
        for episode_index, episode in enumerate(self.episodes):
            by_frame = {int(record["frame"]): record for record in episode.records}
            for anchor_index in range(self.history_frames - 1, len(episode.records)):
                anchor = episode.records[anchor_index]
                frame = int(anchor["frame"])
                mapping = episode.future_targets.get(frame)
                if mapping is None:
                    continue
                history = episode.records[anchor_index - self.history_frames + 1 : anchor_index + 1]
                if any(not record.get("rgb") for record in history):
                    continue
                target_frames: list[int] = []
                valid = True
                for horizon in self.horizons_s:
                    target_frame = _target_for_horizon(mapping, horizon)
                    target_record = None if target_frame is None else by_frame.get(target_frame)
                    if target_record is None or not target_record.get("teacher_voxel"):
                        valid = False
                        break
                    target_frames.append(int(target_frame))
                if valid:
                    samples.append(
                        SampleReference(
                            episode_index=episode_index,
                            anchor_index=anchor_index,
                            target_frames=tuple(target_frames),
                        )
                    )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, path: Path) -> torch.Tensor:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"failed to read RGB frame: {path}")
        height, width = self.image_size
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        array = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return torch.from_numpy(array)

    def __getitem__(self, index: int) -> dict[str, Any]:
        reference = self.samples[index]
        episode = self.episodes[reference.episode_index]
        history_records = episode.records[
            reference.anchor_index - self.history_frames + 1 : reference.anchor_index + 1
        ]
        rgb_history = torch.stack(
            [
                self._load_rgb(relative_path(episode.root, record["rgb"]))
                for record in history_records
            ]
        )
        by_frame = {int(record["frame"]): record for record in episode.records}
        occupancies: list[torch.Tensor] = []
        semantics: list[torch.Tensor] = []
        for target_frame in reference.target_frames:
            record = by_frame[target_frame]
            target_path = relative_path(episode.root, record["teacher_voxel"])
            with np.load(target_path, allow_pickle=False) as payload:
                occupancy = np.asarray(payload["occupancy"], dtype=np.int8)
                semantic_tags = np.asarray(payload["semantics"], dtype=np.uint8)
            if occupancy.shape != self.spec.shape or semantic_tags.shape != self.spec.shape:
                raise ValueError(f"teacher voxel shape does not match grid spec: {target_path}")
            occupancies.append(torch.from_numpy(occupancy.copy()))
            semantics.append(torch.from_numpy(compact_semantic_labels(semantic_tags, occupancy)))
        anchor = history_records[-1]
        return {
            "rgb_history": rgb_history,
            "occupancy": torch.stack(occupancies),
            "semantics": torch.stack(semantics),
            "horizons_s": torch.tensor(self.horizons_s, dtype=torch.float32),
            "frame": int(anchor["frame"]),
            "episode_id": episode.episode_id,
            "group_key": episode.group_key,
        }


def split_summary(
    roots: Iterable[Path | str],
    *,
    split_seed: int = 17,
    val_fraction: float = 0.2,
    test_fraction: float = 0.1,
    allow_legacy_splits: bool = False,
) -> dict[str, list[str]]:
    """Describe the *same* source partitions consumed by temporal datasets."""
    result: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    groups: dict[str, str] = {}
    for root in roots:
        episode = load_voxel_episode(root)
        if episode.source_partition is None:
            if not allow_legacy_splits:
                raise ValueError("voxel source partition missing; development opt-in required")
            partition = stable_partition(
                episode.group_key,
                split_seed=split_seed,
                val_fraction=val_fraction,
                test_fraction=test_fraction,
            )
        else:
            partition = partition_role(episode.source_partition)
        previous = groups.setdefault(episode.group_key, partition)
        if previous != partition:
            raise ValueError("voxel source group crosses partitions")
        result.setdefault(partition, []).append(episode.episode_id)
    return result
