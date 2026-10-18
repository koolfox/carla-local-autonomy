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

from ..artifacts import fingerprint_file
from ..dataset.lineage import canonical_digest, partition_role, scene_group_key
from ..dataset.shards import expand_dataset_roots
from ..native.teacher_timing import validate_decision_target
from ..native.teacher_verify import verify_teacher_dataset
from ..navigation_intent import NAVIGATION_COMMAND_ORDER, NavigationIntent
from ..reproduction.consumption import checked_image, sample_key

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
    return scene_group_key(context, episode_id=episode_id)


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
    source_partition: str
    rgb_sha256: str
    navigation_intent: NavigationIntent | None
    manifest_sha256: str = ""


class BehaviorImitationDataset(Dataset[dict[str, Any]]):
    """Load RGB and privileged teacher labels from BehaviorAgent datasets.

    Published source partitions are authoritative and never rehashed. Route
    identities detect conflicts across roots; split_seed/val_fraction/test_fraction
    are legacy API parameters and cannot change the published membership. Ego speed and navigation intent
    are retained as privileged labels/optional experiment features; their
    presence does not make them inputs to the strict front-RGB model track.
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
        require_navigation_intent: bool = False,
        attest_consumption: bool = False,
        allow_legacy_action_targets: bool = False,
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

        self.attest_consumption = attest_consumption
        self.partition = partition
        self.image_size = (int(height), int(width))
        self.speed_scale_mps = float(speed_scale_mps)
        self.augment = bool(augment)
        self.augmentation_seed = int(augmentation_seed)
        self._augmentation_epoch = 0
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.require_navigation_intent = bool(require_navigation_intent)
        self.samples: list[ImitationSampleRef] = []
        self.group_partitions: dict[str, Partition] = {}
        self.source_references: list[dict[str, Any]] = []
        self.allow_legacy_action_targets = bool(allow_legacy_action_targets)

        for raw_root in expand_dataset_roots(roots):
            root = Path(raw_root).expanduser().resolve(strict=True)
            if verify:
                report = verify_teacher_dataset(root)
                if report["status"] != "passed":
                    raise ValueError(
                        f"teacher dataset verification failed for {root}: {report['errors']}"
                    )
            dataset = _read_json(root / "dataset.json")
            manifest_sha = fingerprint_file(
                root / ("manifest.json" if (root / "manifest.json").is_file() else "dataset.json")
            )["sha256"]
            source_ref = fingerprint_file(root / "dataset.json")
            if any(ref["sha256"] == source_ref["sha256"] for ref in self.source_references):
                raise ValueError("duplicate source dataset release")
            self.source_references.append(source_ref)
            dataset_id = str(dataset.get("dataset_id") or root.name)
            raw_samples = dataset.get("samples")
            if not isinstance(raw_samples, list):
                raise ValueError(f"dataset.samples must be a list: {root}")
            for raw_sample in raw_samples:
                if not isinstance(raw_sample, Mapping):
                    raise ValueError(f"dataset sample must be an object: {root}")
                # Select from the published split before opening target metadata.
                source_partition = str(raw_sample.get("split", ""))
                assigned = partition_role(source_partition)
                if assigned in {"unassigned", "acquisition_pool"}:
                    raise ValueError(
                        f"{source_partition} is not a frozen training/evaluation partition"
                    )
                if assigned != partition:
                    continue
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
                if not self.allow_legacy_action_targets:
                    validate_decision_target(context, int(raw_sample["carla_frame"]))
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
                raw_navigation = context.get("navigation_intent")
                if raw_navigation is None:
                    navigation_intent = None
                    if self.require_navigation_intent:
                        raise ValueError(
                            f"route-conditioned sample lacks navigation intent: {metadata_path}"
                        )
                elif not isinstance(raw_navigation, Mapping):
                    raise ValueError(f"sample navigation intent is invalid: {metadata_path}")
                else:
                    navigation_intent = NavigationIntent.from_mapping(raw_navigation)
                    if navigation_intent.source_frame_id != int(raw_sample["carla_frame"]):
                        raise ValueError(
                            f"navigation intent frame does not match sample: {metadata_path}"
                        )
                group_key = _route_group_key(context=context, episode_id=episode_id)
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
                        source_partition=source_partition,
                        rgb_sha256=fingerprint_file(image_path)["sha256"],
                        navigation_intent=navigation_intent,
                        manifest_sha256=manifest_sha,
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

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("augmentation epoch must be a non-negative integer")
        self._augmentation_epoch = epoch

    def _augmentation_rng(self, sample: ImitationSampleRef) -> random.Random:
        identity = (
            f"{sample.dataset_id}/{sample.episode_id}/{sample.sample_id}/{sample.carla_frame}"
        )
        digest = hashlib.sha256(
            f"imitation-augmentation-v2\0{self.augmentation_seed}\0{self._augmentation_epoch}\0{identity}".encode()
        ).digest()[:8]
        return random.Random(int.from_bytes(digest, byteorder="big", signed=False))

    def _load_image(self, sample: ImitationSampleRef) -> tuple[np.ndarray, float, bool]:
        if self.attest_consumption:
            image, _ = checked_image(sample.image_path, sample.rgb_sha256)
        else:
            image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise RuntimeError(f"could not decode RGB image {sample.image_path}")
        height, width = self.image_size
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        steer = sample.steer
        flipped = False
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
                flipped = True
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return chw, float(steer), flipped

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image, steer, flipped = self._load_image(sample)
        speed_normalized = min(sample.speed_mps / self.speed_scale_mps, 2.0)
        target = np.asarray([steer, sample.longitudinal], dtype=np.float32)
        navigation_intent = sample.navigation_intent
        if navigation_intent is not None and flipped:
            navigation_intent = navigation_intent.mirrored()
        if navigation_intent is None:
            navigation_command = -1
            navigation_features = np.zeros(
                len(NAVIGATION_COMMAND_ORDER) + 3,
                dtype=np.float32,
            )
            navigation_direction = np.zeros(2, dtype=np.float32)
            navigation_distance = np.zeros(1, dtype=np.float32)
        else:
            navigation_command = navigation_intent.command_index
            navigation_features = np.asarray(navigation_intent.model_features(), dtype=np.float32)
            navigation_direction = np.asarray(navigation_intent.direction, dtype=np.float32)
            navigation_distance = np.asarray(
                [navigation_intent.distance_to_maneuver_m], dtype=np.float32
            )
        return {
            "consumption_key": sample_key(sample.manifest_sha256, sample.sample_id),
            "consumption_rgb_sha256": sample.rgb_sha256 if self.attest_consumption else "",
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
            "navigation_available": torch.tensor(navigation_intent is not None),
            "navigation_command": torch.tensor(navigation_command, dtype=torch.long),
            "navigation_features": torch.from_numpy(navigation_features),
            "navigation_direction": torch.from_numpy(navigation_direction),
            "navigation_distance_m": torch.from_numpy(navigation_distance),
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
    rgb_partitions: dict[str, set[str]] = {}
    for name, dataset in datasets.items():
        for sample in dataset.samples:
            rgb_partitions.setdefault(sample.rgb_sha256, set()).add(name)
    rgb_overlap = {
        digest: sorted(names) for digest, names in rgb_partitions.items() if len(names) > 1
    }
    return {
        "rgb_hash_overlap": rgb_overlap,
        "group_ids_by_partition": {
            name: sorted(groups) for name, groups in groups_by_partition.items()
        },
        "rgb_hashes_by_partition": {
            name: sorted({sample.rgb_sha256 for sample in dataset.samples})
            for name, dataset in datasets.items()
        },
        "sample_counts": {name: len(dataset) for name, dataset in datasets.items()},
        "route_group_counts": {name: len(groups) for name, groups in groups_by_partition.items()},
        "route_group_overlap": overlaps,
        "source_partition_counts": {
            name: dict(Counter(sample.source_partition for sample in dataset.samples))
            for name, dataset in datasets.items()
        },
        "split_manifest_sha256": canonical_digest(
            sorted(
                (
                    sample.dataset_id,
                    sample.sample_id,
                    sample.source_partition,
                    sample.route_group_key,
                )
                for dataset in datasets.values()
                for sample in dataset.samples
            )
        ),
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
