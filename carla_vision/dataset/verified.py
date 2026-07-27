"""Integrity-checked dataset releases for training and evaluation consumers."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import fingerprint_file
from .writer import DATASET_SCHEMA_VERSION

_REPEATABLE_TRACKER_ROLES = frozenset({"native_episode_provenance"})


class DatasetIntegrityError(RuntimeError):
    """Raised when a dataset release cannot be trusted by downstream jobs."""


def _requires_privileged_teacher_control(
    dataset: Mapping[str, Any],
) -> bool:
    release_metadata = dataset.get("release_metadata")
    if not isinstance(release_metadata, Mapping):
        return False
    if release_metadata.get("collector") != "native_official_pythonapi":
        return False
    contract = release_metadata.get("privileged_teacher_control")
    if not isinstance(contract, Mapping):
        raise DatasetIntegrityError(
            "native dataset release is missing its privileged teacher-control contract"
        )
    if set(contract) != {
        "schema_version",
        "available_for_every_sample",
        "source",
        "sample_alignment",
        "runtime_model_input",
    }:
        raise DatasetIntegrityError(
            "native privileged teacher-control contract fields differ from schema"
        )
    if (
        contract.get("schema_version") != "1.0"
        or contract.get("available_for_every_sample") is not True
        or contract.get("source") != "carla.Vehicle.get_control"
        or contract.get("sample_alignment")
        != "queried_after_exact_sensor_frame_before_sample_write"
        or contract.get("runtime_model_input") is not False
    ):
        raise DatasetIntegrityError("native privileged teacher-control contract is invalid")
    return True


def _verify_privileged_teacher_control(
    metadata: Mapping[str, Any],
    *,
    sample_id: str,
    carla_frame: int,
) -> None:
    context = metadata.get("context")
    target = context.get("privileged_teacher_control") if isinstance(context, Mapping) else None
    required = {
        "schema_version",
        "privileged",
        "purpose",
        "source",
        "carla_frame",
        "throttle",
        "steer",
        "brake",
        "hand_brake",
        "reverse",
        "manual_gear_shift",
        "gear",
    }
    if not isinstance(target, Mapping) or set(target) != required:
        raise DatasetIntegrityError(
            f"sample {sample_id} privileged teacher control fields differ from schema"
        )
    if (
        target.get("schema_version") != "1.0"
        or target.get("privileged") is not True
        or target.get("purpose") != "offline_teacher_action_target_only"
        or target.get("source") != "carla.Vehicle.get_control"
        or target.get("carla_frame") != carla_frame
    ):
        raise DatasetIntegrityError(
            f"sample {sample_id} privileged teacher control identity is invalid"
        )
    for name, minimum, maximum in (
        ("throttle", 0.0, 1.0),
        ("steer", -1.0, 1.0),
        ("brake", 0.0, 1.0),
    ):
        raw = target.get(name)
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or not minimum <= float(raw) <= maximum
        ):
            raise DatasetIntegrityError(
                f"sample {sample_id} privileged teacher control {name} is invalid"
            )
    for name in ("hand_brake", "reverse", "manual_gear_shift"):
        if not isinstance(target.get(name), bool):
            raise DatasetIntegrityError(
                f"sample {sample_id} privileged teacher control {name} is invalid"
            )
    gear = target.get("gear")
    if isinstance(gear, bool) or not isinstance(gear, int):
        raise DatasetIntegrityError(
            f"sample {sample_id} privileged teacher control gear is invalid"
        )


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DatasetIntegrityError(f"could not read {name}: {error}") from error
    if not isinstance(payload, Mapping):
        raise DatasetIntegrityError(f"{name} must contain a JSON object")
    return payload


def _path_inside(root: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute():
        raise DatasetIntegrityError(f"absolute dataset path is forbidden: {relative}")
    path = (root / raw).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise DatasetIntegrityError(f"dataset path escapes release root: {relative}") from error
    if not path.is_file():
        raise DatasetIntegrityError(f"dataset path is not a file: {relative}")
    return path


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise DatasetIntegrityError(f"could not hash {path}: {error}") from error
    return digest.hexdigest()


def _verify_tracker_artifacts(
    root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if manifest.get("status") != "success":
        raise DatasetIntegrityError("dataset tracker manifest status is not success")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise DatasetIntegrityError("dataset tracker artifacts must be an array")
    roles: dict[str, Mapping[str, Any]] = {}
    for index, entry in enumerate(artifacts):
        if not isinstance(entry, Mapping):
            raise DatasetIntegrityError(f"dataset tracker artifact {index} is invalid")
        path_value = entry.get("path")
        digest_value = entry.get("sha256")
        role = str(entry.get("role", ""))
        if not isinstance(path_value, str) or not isinstance(digest_value, str) or not role:
            raise DatasetIntegrityError(f"dataset tracker artifact {index} is incomplete")
        path = _path_inside(root, path_value)
        if _digest(path) != digest_value or path.stat().st_size != entry.get("size_bytes"):
            raise DatasetIntegrityError(
                f"dataset tracker artifact fingerprint mismatch: {path_value}"
            )
        if role in roles and role not in _REPEATABLE_TRACKER_ROLES:
            raise DatasetIntegrityError(f"duplicate dataset tracker artifact role: {role}")
        roles.setdefault(role, entry)
    required = {
        "dataset_manifest",
        "canonical_coco_annotations",
        "dataset_ontology",
        "ultralytics_dataset_config",
        "dataset_checksum_index",
    }
    missing = sorted(required - roles.keys())
    if missing:
        raise DatasetIntegrityError(
            "dataset tracker is missing required artifact roles: " + ", ".join(missing)
        )
    return roles


def _verify_checksum_index(root: Path) -> dict[str, str]:
    index_path = root / "checksums.sha256"
    try:
        lines = index_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise DatasetIntegrityError(f"could not read checksum index: {error}") from error
    if not lines:
        raise DatasetIntegrityError("dataset checksum index is empty")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        expected, separator, relative = line.partition("  ")
        if (
            not separator
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            or not relative
        ):
            raise DatasetIntegrityError(f"invalid dataset checksum line {line_number}")
        if relative in entries:
            raise DatasetIntegrityError(f"duplicate dataset checksum path: {relative}")
        path = _path_inside(root, relative)
        if _digest(path) != expected:
            raise DatasetIntegrityError(f"dataset checksum mismatch: {relative}")
        entries[relative] = expected
    return entries


def _sample_reference(
    root: Path,
    checksums: Mapping[str, str],
    raw: Any,
    *,
    sample_id: str,
    field: str,
) -> tuple[str, str]:
    if not isinstance(raw, Mapping):
        raise DatasetIntegrityError(f"sample {sample_id} {field} reference must be an object")
    relative = str(raw.get("path", ""))
    expected = str(raw.get("sha256", ""))
    if relative not in checksums:
        raise DatasetIntegrityError(f"sample {sample_id} {field} is absent from checksum index")
    path = _path_inside(root, relative)
    if checksums[relative] != expected or _digest(path) != expected:
        raise DatasetIntegrityError(f"sample {sample_id} {field} fingerprint is inconsistent")
    if path.stat().st_size != raw.get("size_bytes"):
        raise DatasetIntegrityError(f"sample {sample_id} {field} size is inconsistent")
    return relative, expected


def _verify_native_episode_provenance(
    root: Path,
    manifest: Mapping[str, Any],
    dataset: Mapping[str, Any],
    *,
    episode_partitions: Mapping[str, str],
    episode_sample_counts: Mapping[str, int],
    episode_annotation_counts: Mapping[str, int],
) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Sequence):
        raise DatasetIntegrityError("dataset tracker artifacts must be an array")
    entries = [
        entry
        for entry in artifacts
        if isinstance(entry, Mapping) and entry.get("role") == "native_episode_provenance"
    ]
    if not entries:
        return
    if len(entries) != len(episode_partitions):
        raise DatasetIntegrityError(
            "native episode provenance count disagrees with dataset episodes"
        )
    observed_ids: set[str] = set()
    for entry in entries:
        payload = _load_json(
            _path_inside(root, str(entry.get("path", ""))),
            "native episode provenance",
        )
        episode = payload.get("episode")
        samples = payload.get("samples")
        actors = payload.get("actors")
        if not all(isinstance(value, Mapping) for value in (episode, samples, actors)):
            raise DatasetIntegrityError("native episode provenance is incomplete")
        episode_id = str(episode.get("episode_id", ""))
        partition = str(episode.get("partition", ""))
        if not episode_id or episode_id in observed_ids:
            raise DatasetIntegrityError("native episode provenance ID is missing or duplicated")
        if episode_partitions.get(episode_id) != partition:
            raise DatasetIntegrityError(
                f"native episode {episode_id} partition disagrees with dataset samples"
            )
        if samples.get("count") != episode_sample_counts.get(episode_id):
            raise DatasetIntegrityError(
                f"native episode {episode_id} sample count disagrees with dataset"
            )
        if samples.get("annotation_count") != episode_annotation_counts.get(episode_id):
            raise DatasetIntegrityError(
                f"native episode {episode_id} annotation count disagrees with dataset"
            )
        cleanup = actors.get("cleanup")
        if not isinstance(cleanup, Mapping) or cleanup.get("success") is not True:
            raise DatasetIntegrityError(f"native episode {episode_id} cleanup did not succeed")
        if payload.get("status") != "complete":
            raise DatasetIntegrityError(f"native episode {episode_id} status is not complete")
        observed_ids.add(episode_id)
    if observed_ids != set(episode_partitions):
        raise DatasetIntegrityError("native episode provenance IDs are incomplete")
    release_metadata = dataset.get("release_metadata")
    if not isinstance(release_metadata, Mapping):
        raise DatasetIntegrityError("native dataset release metadata is missing")
    if release_metadata.get("episode_count") != len(episode_partitions):
        raise DatasetIntegrityError("native release episode count is inconsistent")
    if release_metadata.get("world_restored_to_asynchronous_mode") is not True:
        raise DatasetIntegrityError("native dataset did not record asynchronous world restoration")


@dataclass(frozen=True)
class VerifiedDataset:
    root: Path
    dataset_id: str
    manifest: Mapping[str, Any]
    dataset: Mapping[str, Any]
    coco: Mapping[str, Any]
    ontology: Mapping[str, Any]
    partitions: Mapping[str, int]
    categories: Mapping[int, str]
    checksum_entries: Mapping[str, str]
    reference: Mapping[str, Any]

    @property
    def sample_count(self) -> int:
        return int(self.dataset["sample_count"])

    def resolve_training_partitions(
        self,
        *,
        training_partitions: Sequence[str] = ("train",),
        validation_partitions: Sequence[str] = ("val_seen", "val"),
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        train = tuple(
            dict.fromkeys(
                partition
                for partition in training_partitions
                if self.partitions.get(partition, 0) > 0
            )
        )
        validation = tuple(
            dict.fromkeys(
                partition
                for partition in validation_partitions
                if self.partitions.get(partition, 0) > 0
            )
        )
        if not train:
            raise DatasetIntegrityError(
                "dataset has no samples in the requested training partitions"
            )
        if not validation:
            raise DatasetIntegrityError(
                "dataset has no samples in the requested validation partitions"
            )
        forbidden = set(train) & set(validation)
        if forbidden:
            raise DatasetIntegrityError(
                "training and validation partitions overlap: " + ", ".join(sorted(forbidden))
            )
        return train, validation


def load_verified_dataset(path: str | Path) -> VerifiedDataset:
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise DatasetIntegrityError(f"dataset release is not a directory: {root}")
    manifest_path = root / "manifest.json"
    dataset_path = root / "dataset.json"
    manifest = _load_json(manifest_path, "dataset tracker manifest")
    roles = _verify_tracker_artifacts(root, manifest)
    dataset = _load_json(dataset_path, "dataset manifest")
    if dataset.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise DatasetIntegrityError(f"unsupported dataset schema {dataset.get('schema_version')!r}")
    if dataset.get("status") != "complete":
        raise DatasetIntegrityError("dataset manifest status is not complete")
    dataset_id = str(dataset.get("dataset_id", ""))
    if not dataset_id or dataset_id != manifest.get("run_id"):
        raise DatasetIntegrityError("dataset identity disagrees with tracker manifest")
    checksum_entries = _verify_checksum_index(root)
    checksum_reference = dataset.get("checksum_index")
    if not isinstance(checksum_reference, Mapping):
        raise DatasetIntegrityError("dataset checksum_index reference is missing")
    checksum_path = root / "checksums.sha256"
    if (
        checksum_reference.get("path") != "checksums.sha256"
        or checksum_reference.get("sha256") != _digest(checksum_path)
        or checksum_reference.get("size_bytes") != checksum_path.stat().st_size
    ):
        raise DatasetIntegrityError("dataset checksum_index reference is inconsistent")

    coco_path = _path_inside(
        root,
        str(roles["canonical_coco_annotations"]["path"]),
    )
    ontology_path = _path_inside(root, str(roles["dataset_ontology"]["path"]))
    coco = _load_json(coco_path, "canonical COCO annotations")
    ontology = _load_json(ontology_path, "dataset ontology")
    samples = dataset.get("samples")
    if not isinstance(samples, list):
        raise DatasetIntegrityError("dataset samples must be an array")
    if dataset.get("sample_count") != len(samples):
        raise DatasetIntegrityError("dataset sample_count is inconsistent")
    requires_teacher_control = _requires_privileged_teacher_control(dataset)

    partitions: Counter[str] = Counter()
    episode_splits: dict[tuple[str, str], str] = {}
    episode_partitions: dict[str, str] = {}
    episode_sample_counts: Counter[str] = Counter()
    episode_annotation_counts: Counter[str] = Counter()
    rgb_hash_splits: dict[str, set[str]] = defaultdict(set)
    sample_ids: set[str] = set()
    image_ids: set[int] = set()
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise DatasetIntegrityError("dataset sample must be an object")
        sample_id = str(sample.get("sample_id", ""))
        image_id = int(sample.get("image_id", -1))
        if not sample_id or sample_id in sample_ids:
            raise DatasetIntegrityError(f"duplicate or missing sample ID: {sample_id!r}")
        if image_id <= 0 or image_id in image_ids:
            raise DatasetIntegrityError(f"duplicate or invalid image ID: {image_id}")
        split = str(sample.get("split", ""))
        scenario_id = str(sample.get("scenario_id", ""))
        episode_id = str(sample.get("episode_id", ""))
        if not split or not scenario_id or not episode_id:
            raise DatasetIntegrityError(f"sample {sample_id} grouping fields are incomplete")
        episode_key = (scenario_id, episode_id)
        prior_split = episode_splits.setdefault(episode_key, split)
        if prior_split != split:
            raise DatasetIntegrityError(f"episode split leakage detected for {episode_key}")
        prior_episode_partition = episode_partitions.setdefault(episode_id, split)
        if prior_episode_partition != split:
            raise DatasetIntegrityError(f"episode ID {episode_id} crosses dataset partitions")
        rgb_path, rgb_digest = _sample_reference(
            root,
            checksum_entries,
            sample.get("rgb"),
            sample_id=sample_id,
            field="rgb",
        )
        _sample_reference(
            root,
            checksum_entries,
            sample.get("instance_mask"),
            sample_id=sample_id,
            field="instance_mask",
        )
        _sample_reference(
            root,
            checksum_entries,
            sample.get("yolo_label"),
            sample_id=sample_id,
            field="yolo_label",
        )
        metadata_relative, _metadata_digest = _sample_reference(
            root,
            checksum_entries,
            sample.get("metadata"),
            sample_id=sample_id,
            field="metadata",
        )
        if requires_teacher_control:
            metadata = _load_json(
                _path_inside(root, metadata_relative),
                f"sample {sample_id} metadata",
            )
            _verify_privileged_teacher_control(
                metadata,
                sample_id=sample_id,
                carla_frame=int(sample.get("carla_frame", -1)),
            )
        if not rgb_path.startswith(f"images/{split}/"):
            raise DatasetIntegrityError(f"sample {sample_id} RGB path does not match its partition")
        rgb_hash_splits[rgb_digest].add(split)
        partitions[split] += 1
        episode_sample_counts[episode_id] += 1
        episode_annotation_counts[episode_id] += int(sample.get("annotation_count", 0))
        sample_ids.add(sample_id)
        image_ids.add(image_id)
    leaking_hashes = sorted(digest for digest, splits in rgb_hash_splits.items() if len(splits) > 1)
    if leaking_hashes:
        raise DatasetIntegrityError(
            f"{len(leaking_hashes)} exact RGB hash group(s) cross dataset partitions"
        )
    expected_split_counts = {
        str(key): int(value) for key, value in dict(dataset.get("split_counts", {})).items()
    }
    if dict(sorted(partitions.items())) != dict(sorted(expected_split_counts.items())):
        raise DatasetIntegrityError("dataset split_counts are inconsistent")

    categories_raw = coco.get("categories")
    images_raw = coco.get("images")
    annotations_raw = coco.get("annotations")
    if not all(isinstance(value, list) for value in (categories_raw, images_raw, annotations_raw)):
        raise DatasetIntegrityError("canonical COCO arrays are incomplete")
    if len(images_raw) != len(samples):
        raise DatasetIntegrityError("COCO image count disagrees with dataset samples")
    if dataset.get("annotation_count") != len(annotations_raw):
        raise DatasetIntegrityError("dataset annotation_count disagrees with COCO")
    categories: dict[int, str] = {}
    for category in categories_raw:
        if not isinstance(category, Mapping):
            raise DatasetIntegrityError("COCO category must be an object")
        zero_based_id = int(category["id"]) - 1
        if zero_based_id < 0 or zero_based_id in categories:
            raise DatasetIntegrityError("COCO category IDs are invalid or duplicated")
        categories[zero_based_id] = str(category["name"])
    ontology_categories = ontology.get("detector_categories")
    if not isinstance(ontology_categories, list):
        raise DatasetIntegrityError("dataset ontology detector_categories is missing")
    expected_categories = {
        int(category["id"]): str(category["name"])
        for category in ontology_categories
        if isinstance(category, Mapping)
    }
    if categories != expected_categories:
        raise DatasetIntegrityError("COCO categories disagree with dataset ontology")
    _verify_native_episode_provenance(
        root,
        manifest,
        dataset,
        episode_partitions=episode_partitions,
        episode_sample_counts=episode_sample_counts,
        episode_annotation_counts=episode_annotation_counts,
    )

    manifest_reference = fingerprint_file(manifest_path)
    dataset_reference = fingerprint_file(dataset_path)
    return VerifiedDataset(
        root=root,
        dataset_id=dataset_id,
        manifest=manifest,
        dataset=dataset,
        coco=coco,
        ontology=ontology,
        partitions=dict(sorted(partitions.items())),
        categories=dict(sorted(categories.items())),
        checksum_entries=checksum_entries,
        reference={
            "kind": "verified_dataset_release",
            "dataset_id": dataset_id,
            "manifest": manifest_reference,
            "dataset_manifest": dataset_reference,
            "sample_count": len(samples),
            "annotation_count": len(annotations_raw),
            "partitions": dict(sorted(partitions.items())),
            "privileged_teacher_control_targets": requires_teacher_control,
        },
    )


__all__ = [
    "DatasetIntegrityError",
    "VerifiedDataset",
    "load_verified_dataset",
]
