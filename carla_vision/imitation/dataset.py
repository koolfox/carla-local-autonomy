"""Route/seed-grouped BehaviorAgent teacher dataset for imitation learning."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..native.teacher_verify import verify_teacher_dataset

Partition = Literal["train", "val", "test"]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _stable_unit_interval(value: str) -> float:
    prefix = hashlib.sha256(value.encode("utf-8")).digest()[:8]
    return int.from_bytes(prefix, byteorder="big", signed=False) / float(2**64)


def _group_partition(
    group_key: str,
    *,
    split_seed: int,
    val_fraction: float,
    test_fraction: float,
) -> Partition:
    unit = _stable_unit_interval(f"imitation-split-v1\0{split_seed}\0{group_key}")
    if unit < test_fraction:
        return "test"
    if unit < test_fraction + val_fraction:
        return "val"
    return "train"


def _artifact_path(root: Path, artifact: Mapping[str, Any], name: str) -> Path:
    relative = artifact.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{name} artifact path is missing")
    path = (root / relative).resolve(strict=True)
    path.relative_to(root.resolve())
    return path


def _route_group_key(
    *,
    context: Mapping[str, Any],
    episode_id: str,
) -> str:
    route = context.get("route")
    group = context.get("group")
    route_data = dict(route) if isinstance(route, Mapping) else {}
    group_data = dict(group) if isinstance(group, Mapping) else {}
    payload = {
        "map_family": context.get("map_family"),
        "scenario_recipe_id": context.get("scenario_recipe_id"),
        "route_seed": route_data.get("route_seed"),
        "start_spawn_index": route_data.get("start_spawn_index"),
        "destination_spawn_index": route_data.get("destination_spawn_index"),
        "static_layout_seed": group_data.get("static_layout_seed"),
        "fallback_episode_id": episode_id if route_data.get("route_seed") is None else None,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class ImitationSampleRef:
    dataset_root: Path
    dataset_id: str
    sample_id: str
    episode_id: str
    carla_frame: int
    image_path: Path
    metadata_path: Path
    speed_mps: float
    steer: float
    longitudinal: float
    throttle: float
    brake: float
    route_group_key: str
    partition: Partition


class BehaviorImitationDataset(Dataset[dict[str, Any]]):
    """Load RGB, ego speed, and BehaviorAgent controls from teacher datasets.

    Split assignment is derived from a stable route/seed group key so all
    samples from the same route and static layout remain in one partition even
    when multiple dataset roots are combined.
    """

    def __init__(
        self,
        roots: Sequence[str | Path],
        *,
        partition: Partition,
        image_size: tuple[int, int] = (192, 320),
        speed_scale_mps: float = 20.0,
        split_seed: int = 17,
        val_fraction: float = 0.2,
        test_fraction: float = 0.1,
        augment: bool = False,
        augmentation_seed: int = 23,
        brightness: float = 0.12,
        contrast: float = 0.12,
        horizontal_flip_probability: float = 0.0,
        verify: bool = True,
        max_samples: int | None = None,
    ) -> None:
        if partition not in {"train", "val", "test"}:
            raise ValueError("partition must be train, val, or test")
        if not roots:
            raise ValueError("at least one dataset root is required")
        height, width = image_size
        if height <= 0 or width <= 0:
            raise ValueError("image_size dimensions must be positive")
        if not math.isfinite(speed_scale_mps) or speed_scale_mps <= 0.0:
            raise ValueError("speed_scale_mps must be finite and positive")
        if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
            raise ValueError("split fractions must be in [0, 1)")
        if val_fraction + test_fraction >= 1.0:
            raise ValueError("val_fraction + test_fraction must be less than 1")
        for name, value in (
            ("brightness", brightness),
            ("contrast", contrast),
            ("horizontal_flip_probability", horizontal_flip_probability),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if horizontal_flip_probability > 1.0:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive")

        self.partition = partition
        self.image_size = (int(height), int(width))
        self.speed_scale_mps = float(speed_scale_mps)
        self.augment = bool(augment)
        self.augmentation_seed = int(augmentation_seed)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.samples: list[ImitationSampleRef] = []
        self.group_partitions: dict[str, Partition] = {}

        for raw_root in roots:
            root = Path(raw_root).expanduser().resolve(strict=True)
            if verify:
                report = verify_teacher_dataset(root)
                if report["status"] != "passed":
                    raise ValueError(
                        f"teacher dataset verification failed for {root}: {report['errors']}"
                    )
            dataset = _read_json(root / "dataset.json")
            dataset_id = str(dataset.get("dataset_id") or root.name)
            raw_samples = dataset.get("samples")
            if not isinstance(raw_samples, list):
                raise ValueError(f"dataset.samples must be a list: {root}")
            for raw_sample in raw_samples:
                if not isinstance(raw_sample, Mapping):
                    raise ValueError(f"dataset sample must be an object: {root}")
                metadata_artifact = raw_sample.get("metadata")
                rgb_artifact = raw_sample.get("rgb")
                if not isinstance(metadata_artifact, Mapping) or not isinstance(
                    rgb_artifact, Mapping
                ):
                    raise ValueError("teacher sample must contain rgb and metadata artifacts")
                metadata_path = _artifact_path(root, metadata_artifact, "metadata")
                image_path = _artifact_path(root, rgb_artifact, "rgb")
                metadata = _read_json(metadata_path)
                context = metadata.get("context")
                if not isinstance(context, Mapping):
                    raise ValueError(f"sample context is missing: {metadata_path}")
                if context.get("control_mode") != "behavior_agent_teacher":
                    raise ValueError(f"sample is not BehaviorAgent teacher data: {metadata_path}")
                control = context.get("privileged_teacher_control")
                ego_state = context.get("privileged_evaluation")
                if not isinstance(control, Mapping) or not isinstance(ego_state, Mapping):
                    raise ValueError(f"sample control or ego state is missing: {metadata_path}")
                velocity = ego_state.get("velocity_mps")
                if not isinstance(velocity, Mapping):
                    raise ValueError(f"sample ego velocity is missing: {metadata_path}")
                speed = float(velocity["speed"])
                throttle = float(control["throttle"])
                steer = float(control["steer"])
                brake = float(control["brake"])
                if not math.isfinite(speed) or speed < 0.0:
                    raise ValueError(f"invalid ego speed: {metadata_path}")
                if not 0.0 <= throttle <= 1.0 or not -1.0 <= steer <= 1.0:
                    raise ValueError(f"invalid teacher throttle/steer: {metadata_path}")
                if not 0.0 <= brake <= 1.0:
                    raise ValueError(f"invalid teacher brake: {metadata_path}")
                episode_id = str(raw_sample["episode_id"])
                group_key = _route_group_key(context=context, episode_id=episode_id)
                assigned = _group_partition(
                    group_key,
                    split_seed=split_seed,
                    val_fraction=val_fraction,
                    test_fraction=test_fraction,
                )
                prior = self.group_partitions.setdefault(group_key, assigned)
                if prior != assigned:
                    raise RuntimeError("one route group was assigned to multiple partitions")
                if assigned != partition:
                    continue
                self.samples.append(
                    ImitationSampleRef(
                        dataset_root=root,
                        dataset_id=dataset_id,
                        sample_id=str(raw_sample["sample_id"]),
                        episode_id=episode_id,
                        carla_frame=int(raw_sample["carla_frame"]),
                        image_path=image_path,
                        metadata_path=metadata_path,
                        speed_mps=speed,
                        steer=steer,
                        longitudinal=throttle - brake,
                        throttle=throttle,
                        brake=brake,
                        route_group_key=group_key,
                        partition=assigned,
                    )
                )
        self.samples.sort(
            key=lambda item: (
                item.route_group_key,
                item.episode_id,
                item.carla_frame,
                item.dataset_id,
                item.sample_id,
            )
        )
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _augmentation_rng(self, sample: ImitationSampleRef) -> random.Random:
        identity = (
            f"{sample.dataset_id}/{sample.episode_id}/{sample.sample_id}/"
            f"{sample.carla_frame}"
        )
        digest = hashlib.sha256(
            f"imitation-augmentation-v1\0{self.augmentation_seed}\0{identity}".encode()
        ).digest()[:8]
        return random.Random(int.from_bytes(digest, byteorder="big", signed=False))

    def _load_image(self, sample: ImitationSampleRef) -> tuple[np.ndarray, float]:
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise RuntimeError(f"could not decode RGB image {sample.image_path}")
        height, width = self.image_size
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        steer = sample.steer
        if self.augment:
            rng = self._augmentation_rng(sample)
            if self.brightness:
                image = image.astype(np.float32) * rng.uniform(
                    1.0 - self.brightness, 1.0 + self.brightness
                )
            if self.contrast:
                contrast = rng.uniform(1.0 - self.contrast, 1.0 + self.contrast)
                mean = image.mean(axis=(0, 1), keepdims=True)
                image = (image - mean) * contrast + mean
            image = np.clip(image, 0.0, 255.0).astype(np.uint8)
            if rng.random() < self.horizontal_flip_probability:
                image = cv2.flip(image, 1)
                steer = -steer
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return chw, float(steer)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image, steer = self._load_image(sample)
        speed_normalized = min(sample.speed_mps / self.speed_scale_mps, 2.0)
        target = np.asarray([steer, sample.longitudinal], dtype=np.float32)
        return {
            "image": torch.from_numpy(image),
            "speed": torch.tensor([speed_normalized], dtype=torch.float32),
            "target": torch.from_numpy(target),
            "raw_control": torch.tensor(
                [sample.throttle, steer, sample.brake], dtype=torch.float32
            ),
            "sample_id": sample.sample_id,
            "episode_id": sample.episode_id,
            "route_group_key": sample.route_group_key,
            "carla_frame": sample.carla_frame,
        }


def split_summary(datasets: Mapping[Partition, BehaviorImitationDataset]) -> dict[str, Any]:
    groups_by_partition = {
        name: {sample.route_group_key for sample in dataset.samples}
        for name, dataset in datasets.items()
    }
    names = tuple(groups_by_partition)
    overlaps: dict[str, list[str]] = {}
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            shared = groups_by_partition[left_name] & groups_by_partition[right_name]
            if shared:
                overlaps[f"{left_name}:{right_name}"] = sorted(shared)
    return {
        "sample_counts": {name: len(dataset) for name, dataset in datasets.items()},
        "route_group_counts": {
            name: len(groups) for name, groups in groups_by_partition.items()
        },
        "route_group_overlap": overlaps,
        "control_counts": {
            name: dict(
                Counter(
                    "brake" if sample.longitudinal < -0.05 else "drive"
                    for sample in dataset.samples
                )
            )
            for name, dataset in datasets.items()
        },
    }


__all__ = [
    "BehaviorImitationDataset",
    "ImitationSampleRef",
    "Partition",
    "split_summary",
]
