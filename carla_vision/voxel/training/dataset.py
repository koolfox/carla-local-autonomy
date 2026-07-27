"""Dataset loading for temporal RGB-only voxel occupancy training."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

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

# CARLA 0.9.x semantic tag IDs. Only occupied voxels are supervised.
_SEMANTIC_TAG_TO_COMPACT = {
    1: 0,  # Building
    6: 1,  # RoadLine
    7: 1,  # Road
    8: 2,  # SideWalk
    10: 3,  # Vehicles
    4: 4,  # Pedestrian
    9: 5,  # Vegetation
}


@dataclass(frozen=True, slots=True)
class VoxelEpisode:
    root: Path
    episode_id: str
    group_key: str
    spec: VoxelGridSpec
    records: tuple[dict[str, Any], ...]
    future_targets: dict[int, dict[float, int | None]]


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
    frames = [int(record["frame"]) for record in records]
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
            converted[value] = None if target is None else int(target)
        future_targets[int(item["frame"])] = converted

    episode_id, group_key = _episode_group_key(resolved, manifest)
    return VoxelEpisode(
        root=resolved,
        episode_id=episode_id,
        group_key=group_key,
        spec=spec,
        records=records,
        future_targets=future_targets,
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
        selected = tuple(
            episode
            for episode in episodes
            if stable_partition(
                episode.group_key,
                split_seed=split_seed,
                val_fraction=val_fraction,
                test_fraction=test_fraction,
            )
            == partition
        )
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
            [self._load_rgb(episode.root / str(record["rgb"])) for record in history_records]
        )
        by_frame = {int(record["frame"]): record for record in episode.records}
        occupancies: list[torch.Tensor] = []
        semantics: list[torch.Tensor] = []
        for target_frame in reference.target_frames:
            record = by_frame[target_frame]
            target_path = episode.root / str(record["teacher_voxel"])
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
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for root in roots:
        episode = load_voxel_episode(root)
        partition = stable_partition(
            episode.group_key,
            split_seed=split_seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
        result[partition].append(episode.episode_id)
    return result
