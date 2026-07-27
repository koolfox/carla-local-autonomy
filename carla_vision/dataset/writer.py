"""Immutable pilot-dataset writer with COCO and YOLO detection exports."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import cv2

from ..artifacts import RunArtifactTracker, fingerprint_file
from .instance_labels import InstanceLabel, extract_instance_labels
from .ontology import CARLA_SEMANTIC_TAGS, DETECTOR_CATEGORIES
from .sync import SynchronizedFramePair

DATASET_SCHEMA_VERSION = "1.0"
ANNOTATION_SCHEMA_VERSION = "1.0"
DATASET_PARTITIONS = frozenset(
    {
        "train",
        "val",
        "test",
        "val_seen",
        "test_seen",
        "val_map_ood",
        "test_map_ood",
        "val_weather_ood",
        "test_weather_ood",
        "unassigned",
    }
)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=f".tmp{path.suffix}",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write_text(path: Path, payload: str) -> None:
    _atomic_write_bytes(path, payload.encode("utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    _atomic_write_text(path, serialized + "\n")


def _write_png(path: Path, image: Any) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"OpenCV could not encode {path}")
    _atomic_write_bytes(path, encoded.tobytes())


@dataclass(frozen=True)
class SampleArtifact:
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class DatasetSample:
    sample_id: str
    image_id: int
    carla_frame: int
    source_timestamp: float
    split: str
    scenario_id: str
    episode_id: str
    rgb: SampleArtifact
    instance_mask: SampleArtifact
    yolo_label: SampleArtifact
    metadata: SampleArtifact
    annotation_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuxiliaryArtifact:
    path: str
    role: str
    metadata: dict[str, Any]


class DatasetWriter:
    """Write one checksum-indexed dataset release without overwriting files."""

    def __init__(
        self,
        datasets_root: str | Path,
        *,
        dataset_id: str,
        carla_endpoint: Mapping[str, Any],
        carla_version: str,
        carla_map: str,
        cli_args: Sequence[str] | Mapping[str, Any] = (),
        config: Mapping[str, Any] | None = None,
        repository_root: str | Path | None = None,
        minimum_pixels: int = 16,
        minimum_box_width: int = 2,
        minimum_box_height: int = 2,
    ) -> None:
        self._tracker = RunArtifactTracker(
            datasets_root,
            run_id=dataset_id,
            cli_args=cli_args,
            config={
                "object_type": "dataset",
                "dataset_schema_version": DATASET_SCHEMA_VERSION,
                "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
                "teacher_sensor": "sensor.camera.instance_segmentation",
                "runtime_sensor_contract": "front_monocular_rgb_only",
                "minimum_pixels": minimum_pixels,
                "minimum_box_width": minimum_box_width,
                "minimum_box_height": minimum_box_height,
                **dict(config or {}),
            },
            repository_root=repository_root,
            carla_endpoint=dict(carla_endpoint),
            carla_version=carla_version,
            carla_map=carla_map,
        )
        self.dataset_id = dataset_id
        self.dataset_dir = self._tracker.run_dir
        self.manifest_path = self._tracker.manifest_path
        self.minimum_pixels = minimum_pixels
        self.minimum_box_width = minimum_box_width
        self.minimum_box_height = minimum_box_height
        self._samples: list[DatasetSample] = []
        self._coco_images: list[dict[str, Any]] = []
        self._coco_annotations: list[dict[str, Any]] = []
        self._episode_splits: dict[tuple[str, str], str] = {}
        self._last_frame_by_episode: dict[tuple[str, str], int] = {}
        self._seen_frame_keys: set[tuple[str, str, int]] = set()
        self._auxiliary_artifacts: list[AuxiliaryArtifact] = []
        self._auxiliary_paths: set[str] = set()
        self._release_metadata: dict[str, Any] = {}
        self._entered = False
        self._closed = False

    def __enter__(self) -> Self:
        if self._entered:
            raise RuntimeError("dataset writer has already been entered")
        self._tracker.__enter__()
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_value is not None:
            self._tracker.__exit__(exc_type, exc_value, traceback)
            return
        try:
            self.close()
        except BaseException as error:
            self._tracker.__exit__(
                type(error),
                error,
                error.__traceback__,
            )
            raise
        self._tracker.__exit__(None, None, None)

    @property
    def samples(self) -> tuple[DatasetSample, ...]:
        return tuple(self._samples)

    def set_release_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Set final collection metadata before the writer leaves its context."""

        if self._closed:
            raise RuntimeError("dataset writer is closed")
        self._release_metadata = dict(metadata)

    def add_auxiliary_json(
        self,
        relative_path: str | Path,
        payload: Mapping[str, Any],
        *,
        role: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Add a checksum-tracked provenance/episode JSON artifact."""

        if not self._entered:
            raise RuntimeError("dataset writer must be used as a context manager")
        if self._closed:
            raise RuntimeError("dataset writer is closed")
        raw = Path(relative_path)
        if raw.is_absolute():
            raise ValueError("auxiliary artifact path must be relative")
        path = (self.dataset_dir / raw).resolve(strict=False)
        try:
            relative = path.relative_to(self.dataset_dir.resolve()).as_posix()
        except ValueError as error:
            raise ValueError("auxiliary artifact must stay inside the dataset") from error
        if relative == "manifest.json" or relative in self._auxiliary_paths:
            raise ValueError(f"duplicate or reserved auxiliary artifact path: {relative}")
        normalized_role = role.strip()
        if not normalized_role:
            raise ValueError("auxiliary artifact role must not be empty")
        if path.exists():
            raise FileExistsError(f"auxiliary artifact already exists: {path}")
        _write_json(path, payload)
        self._auxiliary_paths.add(relative)
        self._auxiliary_artifacts.append(
            AuxiliaryArtifact(
                path=relative,
                role=normalized_role,
                metadata=dict(metadata or {}),
            )
        )
        return path

    def add_pair(
        self,
        pair: SynchronizedFramePair,
        *,
        split: str,
        scenario_id: str,
        episode_id: str,
        context: Mapping[str, Any] | None = None,
    ) -> DatasetSample:
        if not self._entered:
            raise RuntimeError("dataset writer must be used as a context manager")
        if self._closed:
            raise RuntimeError("dataset writer is closed")
        if split not in DATASET_PARTITIONS:
            raise ValueError("split must be one of: " + ", ".join(sorted(DATASET_PARTITIONS)))
        if not scenario_id or not episode_id:
            raise ValueError("scenario_id and episode_id must not be empty")
        self._validate_pair(pair)
        episode_key = (scenario_id, episode_id)
        prior_split = self._episode_splits.get(episode_key)
        if prior_split is not None and prior_split != split:
            raise ValueError(
                "one episode cannot cross dataset splits: "
                f"{episode_id!r} was {prior_split!r}, got {split!r}"
            )
        prior_frame = self._last_frame_by_episode.get(episode_key)
        if prior_frame is not None and pair.carla_frame <= prior_frame:
            raise ValueError("CARLA frame IDs must increase strictly within an episode")
        frame_key = (*episode_key, pair.carla_frame)
        if frame_key in self._seen_frame_keys:
            raise ValueError(
                f"duplicate CARLA frame {pair.carla_frame} within episode {episode_id!r}"
            )

        rgb = pair.rgb.bgr()
        teacher_bgra = pair.teacher.bgra_array()
        labels = extract_instance_labels(
            teacher_bgra,
            minimum_pixels=self.minimum_pixels,
            minimum_box_width=self.minimum_box_width,
            minimum_box_height=self.minimum_box_height,
        )
        image_id = len(self._samples) + 1
        sample_id = f"{image_id:08d}"
        rgb_path = self.dataset_dir / "images" / split / f"{sample_id}.png"
        mask_path = self.dataset_dir / "teacher" / "instance" / split / f"{sample_id}.png"
        yolo_path = self.dataset_dir / "labels" / split / f"{sample_id}.txt"
        metadata_path = self.dataset_dir / "metadata" / f"{sample_id}.json"
        for path in (rgb_path, mask_path, yolo_path, metadata_path):
            if path.exists():
                raise FileExistsError(f"dataset sample artifact already exists: {path}")

        _write_png(rgb_path, rgb)
        _write_png(mask_path, teacher_bgra)
        _atomic_write_text(
            yolo_path,
            self._yolo_text(labels, width=pair.rgb.width, height=pair.rgb.height),
        )

        metadata_payload = {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "dataset_id": self.dataset_id,
            "sample_id": sample_id,
            "image_id": image_id,
            "split": split,
            "scenario_id": scenario_id,
            "episode_id": episode_id,
            "carla_frame": pair.carla_frame,
            "simulation_timestamp_seconds": pair.rgb.timestamp,
            "camera": {
                "width": pair.rgb.width,
                "height": pair.rgb.height,
                "fov_degrees": pair.rgb.fov,
                "world_transform": list(pair.rgb.transform),
            },
            "synchronization": {
                "rgb_sequence": pair.rgb.sequence,
                "teacher_sequence": pair.teacher.sequence,
                "rgb_skipped_before_pair": pair.rgb_skipped,
                "teacher_skipped_before_pair": pair.teacher_skipped,
                "exact_carla_frame_match": True,
            },
            "teacher": {
                "privileged": True,
                "sensor_type": "sensor.camera.instance_segmentation",
                "encoding": {
                    "pixel_order": "BGRA",
                    "semantic_tag": "R",
                    "actor_id_low_byte": "G",
                    "actor_id_high_byte": "B",
                },
            },
            "annotations": [label.as_dict() for label in labels],
            "context": dict(context or {}),
        }
        _write_json(metadata_path, metadata_payload)

        rgb_artifact = self._sample_artifact(rgb_path)
        mask_artifact = self._sample_artifact(mask_path)
        yolo_artifact = self._sample_artifact(yolo_path)
        metadata_artifact = self._sample_artifact(metadata_path)
        sample = DatasetSample(
            sample_id=sample_id,
            image_id=image_id,
            carla_frame=pair.carla_frame,
            source_timestamp=pair.rgb.timestamp,
            split=split,
            scenario_id=scenario_id,
            episode_id=episode_id,
            rgb=rgb_artifact,
            instance_mask=mask_artifact,
            yolo_label=yolo_artifact,
            metadata=metadata_artifact,
            annotation_count=len(labels),
        )
        self._samples.append(sample)
        self._append_coco(sample, labels, pair.rgb.width, pair.rgb.height)
        self._episode_splits[episode_key] = split
        self._last_frame_by_episode[episode_key] = pair.carla_frame
        self._seen_frame_keys.add(frame_key)
        return sample

    def close(self) -> None:
        if not self._entered:
            raise RuntimeError("dataset writer was not entered")
        if self._closed:
            return
        self._closed = True

        coco_path = self.dataset_dir / "annotations" / "instances.coco.json"
        ontology_path = self.dataset_dir / "ontology.json"
        yaml_path = self.dataset_dir / "data.yaml"
        checksums_path = self.dataset_dir / "checksums.sha256"
        dataset_path = self.dataset_dir / "dataset.json"

        _write_json(
            coco_path,
            {
                "info": {
                    "description": "CARLA synchronized RGB detection dataset",
                    "version": DATASET_SCHEMA_VERSION,
                    "dataset_id": self.dataset_id,
                },
                "licenses": [],
                "images": self._coco_images,
                "annotations": self._coco_annotations,
                "categories": [
                    {
                        "id": category.id + 1,
                        "name": category.name,
                        "supercategory": "road_user_or_control",
                        "carla_semantic_tag": int(category.semantic_tag),
                    }
                    for category in DETECTOR_CATEGORIES
                ],
            },
        )
        _write_json(
            ontology_path,
            {
                "schema_version": "1.0",
                "carla_version": "0.9.16",
                "semantic_tags": [
                    {"id": int(tag), "name": tag.name.lower()} for tag in CARLA_SEMANTIC_TAGS
                ],
                "detector_categories": [category.as_dict() for category in DETECTOR_CATEGORIES],
                "policy": (
                    "Official CARLA thing tags form the v0 detector ontology; "
                    "dense scene classes remain in the privileged mask."
                ),
            },
        )
        _atomic_write_text(yaml_path, self._data_yaml())

        indexed_paths = sorted(
            (
                *(
                    self.dataset_dir / sample_path.path
                    for sample in self._samples
                    for sample_path in (
                        sample.rgb,
                        sample.instance_mask,
                        sample.yolo_label,
                        sample.metadata,
                    )
                ),
                coco_path,
                ontology_path,
                yaml_path,
                *(self.dataset_dir / artifact.path for artifact in self._auxiliary_artifacts),
            ),
            key=lambda path: path.relative_to(self.dataset_dir).as_posix(),
        )
        checksum_lines = [
            f"{fingerprint_file(path)['sha256']}  {path.relative_to(self.dataset_dir).as_posix()}"
            for path in indexed_paths
        ]
        _atomic_write_text(checksums_path, "\n".join(checksum_lines) + "\n")

        class_counts = Counter(annotation["category_id"] for annotation in self._coco_annotations)
        split_counts = Counter(sample.split for sample in self._samples)
        checksum_reference = fingerprint_file(checksums_path)
        _write_json(
            dataset_path,
            {
                "schema_version": DATASET_SCHEMA_VERSION,
                "object_type": "dataset",
                "dataset_id": self.dataset_id,
                "status": "complete",
                "runtime_sensor_contract": "front_monocular_rgb_only",
                "teacher_sensor": {
                    "type": "sensor.camera.instance_segmentation",
                    "privileged": True,
                    "retained": True,
                },
                "sample_count": len(self._samples),
                "annotation_count": len(self._coco_annotations),
                "split_counts": dict(sorted(split_counts.items())),
                "category_counts": {
                    str(category.id): class_counts.get(category.id + 1, 0)
                    for category in DETECTOR_CATEGORIES
                },
                "samples": [sample.as_dict() for sample in self._samples],
                "release_metadata": self._release_metadata,
                "checksum_index": {
                    "path": "checksums.sha256",
                    "sha256": checksum_reference["sha256"],
                    "size_bytes": checksum_reference["size_bytes"],
                    "algorithm": "sha256",
                },
            },
        )

        for path, role in (
            (dataset_path, "dataset_manifest"),
            (coco_path, "canonical_coco_annotations"),
            (ontology_path, "dataset_ontology"),
            (yaml_path, "ultralytics_dataset_config"),
            (checksums_path, "dataset_checksum_index"),
        ):
            self._tracker.register_artifact(
                path,
                role=role,
                metadata={
                    "dataset_id": self.dataset_id,
                    "sample_count": len(self._samples),
                    "annotation_count": len(self._coco_annotations),
                },
            )
        for artifact in self._auxiliary_artifacts:
            self._tracker.register_artifact(
                self.dataset_dir / artifact.path,
                role=artifact.role,
                metadata=artifact.metadata,
            )

    def _sample_artifact(self, path: Path) -> SampleArtifact:
        fingerprint = fingerprint_file(path)
        return SampleArtifact(
            path=path.relative_to(self.dataset_dir).as_posix(),
            sha256=str(fingerprint["sha256"]),
            size_bytes=int(fingerprint["size_bytes"]),
        )

    @staticmethod
    def _validate_pair(pair: SynchronizedFramePair) -> None:
        rgb = pair.rgb
        teacher = pair.teacher
        if rgb.frame != teacher.frame:
            raise ValueError("RGB and teacher must have the same CARLA frame")
        if not math.isclose(
            rgb.timestamp,
            teacher.timestamp,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("RGB and teacher timestamps differ")
        if (rgb.width, rgb.height) != (teacher.width, teacher.height):
            raise ValueError("RGB and teacher dimensions differ")
        if not math.isclose(rgb.fov, teacher.fov, rel_tol=0.0, abs_tol=1e-4):
            raise ValueError("RGB and teacher FOV differ")
        if any(
            abs(left - right) > 1e-4
            for left, right in zip(
                rgb.transform,
                teacher.transform,
                strict=True,
            )
        ):
            raise ValueError("RGB and teacher camera transforms differ")

    def _append_coco(
        self,
        sample: DatasetSample,
        labels: Sequence[InstanceLabel],
        width: int,
        height: int,
    ) -> None:
        self._coco_images.append(
            {
                "id": sample.image_id,
                "file_name": sample.rgb.path,
                "width": width,
                "height": height,
                "carla_frame": sample.carla_frame,
                "simulation_timestamp_seconds": sample.source_timestamp,
                "split": sample.split,
                "scenario_id": sample.scenario_id,
                "episode_id": sample.episode_id,
            }
        )
        for label in labels:
            x1, y1, x2, y2 = label.bbox_xyxy
            self._coco_annotations.append(
                {
                    "id": len(self._coco_annotations) + 1,
                    "image_id": sample.image_id,
                    "category_id": label.category_id + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": label.visible_area_pixels,
                    "iscrowd": 0,
                    "attributes": {
                        "carla_actor_id": label.actor_id,
                        "carla_semantic_tag": int(label.semantic_tag),
                        "visible_area_pixels": label.visible_area_pixels,
                        "bbox_area_pixels": label.bbox_area_pixels,
                        "mask_fill_ratio": label.fill_ratio,
                        "truncated": label.truncated,
                        "teacher_method": "instance_segmentation_visible_mask",
                    },
                }
            )

    @staticmethod
    def _yolo_text(
        labels: Sequence[InstanceLabel],
        *,
        width: int,
        height: int,
    ) -> str:
        lines: list[str] = []
        for label in labels:
            x1, y1, x2, y2 = label.bbox_xyxy
            box_width = x2 - x1
            box_height = y2 - y1
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            lines.append(
                f"{label.category_id} "
                f"{center_x / width:.8f} {center_y / height:.8f} "
                f"{box_width / width:.8f} {box_height / height:.8f}"
            )
        return "\n".join(lines) + ("\n" if lines else "")

    def _data_yaml(self) -> str:
        names = "\n".join(f"  {category.id}: {category.name}" for category in DETECTOR_CATEGORIES)
        present = {sample.split for sample in self._samples}

        def paths(partitions: Sequence[str]) -> str:
            selected = [partition for partition in partitions if partition in present]
            if not selected:
                return " []"
            return "\n" + "\n".join(f"  - images/{partition}" for partition in selected)

        train_paths = paths(("train",))
        validation_paths = paths(("val", "val_seen", "val_map_ood", "val_weather_ood"))
        test_paths = paths(("test", "test_seen", "test_map_ood", "test_weather_ood"))
        return (
            f"path: .\ntrain:{train_paths}\nval:{validation_paths}\n"
            f"test:{test_paths}\nnames:\n{names}\n"
        )


__all__ = [
    "ANNOTATION_SCHEMA_VERSION",
    "AuxiliaryArtifact",
    "DATASET_PARTITIONS",
    "DATASET_SCHEMA_VERSION",
    "DatasetSample",
    "DatasetWriter",
    "SampleArtifact",
]
