"""Read-only integrity and label-quality audit for CARLA dataset releases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402

from ..artifacts import (
    DEFAULT_PACKAGE_NAMES,
    RunArtifactTracker,
    fingerprint_file,
)
from .instance_labels import extract_instance_labels

DATASET_AUDIT_SCHEMA_VERSION = "1.0"


class DatasetAuditError(ValueError):
    """Raised when a dataset fails integrity or schema validation."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DatasetAuditError(f"required file is missing: {path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DatasetAuditError(f"could not read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DatasetAuditError(f"{path} must contain a JSON object")
    return payload


def _source_path(root: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute():
        raise DatasetAuditError(f"absolute path is forbidden in dataset: {relative}")
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise DatasetAuditError(f"dataset path escapes release root: {relative}") from error
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise DatasetAuditError(f"could not hash {path}: {error}") from error
    return digest.hexdigest()


def _verify_checksum_index(root: Path) -> dict[str, str]:
    index_path = root / "checksums.sha256"
    try:
        lines = index_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise DatasetAuditError(f"could not read {index_path}: {error}") from error
    if not lines:
        raise DatasetAuditError("checksum index is empty")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
        ):
            raise DatasetAuditError(f"invalid checksums.sha256 line {line_number}")
        if relative in entries:
            raise DatasetAuditError(f"duplicate checksum path at line {line_number}: {relative}")
        path = _source_path(root, relative)
        if not path.is_file():
            raise DatasetAuditError(f"checksum target is missing: {relative}")
        actual = _sha256(path)
        if actual != digest:
            raise DatasetAuditError(f"checksum mismatch: {relative}")
        entries[relative] = digest
    return entries


def _verify_registered_artifacts(
    root: Path,
    manifest: Mapping[str, Any],
) -> None:
    if manifest.get("status") != "success":
        raise DatasetAuditError("source dataset run manifest is not successful")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise DatasetAuditError("source manifest artifacts must be an array")
    for entry in artifacts:
        if not isinstance(entry, Mapping):
            raise DatasetAuditError("source manifest artifact must be an object")
        relative = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise DatasetAuditError("source manifest artifact identity is invalid")
        path = _source_path(root, relative)
        if not path.is_file() or _sha256(path) != expected:
            raise DatasetAuditError(f"registered artifact checksum mismatch: {relative}")


def _require_list(payload: Mapping[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise DatasetAuditError(f"COCO field {key!r} must be an array")
    return value


def _expected_yolo(
    annotations: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
) -> str:
    lines: list[str] = []
    for annotation in annotations:
        category_id = int(annotation["category_id"]) - 1
        x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
        lines.append(
            f"{category_id} "
            f"{(x + box_width / 2.0) / width:.8f} "
            f"{(y + box_height / 2.0) / height:.8f} "
            f"{box_width / width:.8f} {box_height / height:.8f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _descriptive(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p95": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
    }


def _audit_records(
    root: Path,
    dataset: Mapping[str, Any],
    coco: Mapping[str, Any],
    tracker_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    samples = dataset.get("samples")
    if not isinstance(samples, list):
        raise DatasetAuditError("dataset.json samples must be an array")
    images = _require_list(coco, "images")
    annotations = _require_list(coco, "annotations")
    categories = _require_list(coco, "categories")
    if len(images) != len(samples):
        raise DatasetAuditError("COCO image count disagrees with dataset samples")

    category_names: dict[int, str] = {}
    for category in categories:
        if not isinstance(category, Mapping):
            raise DatasetAuditError("COCO category must be an object")
        category_id = int(category["id"])
        if category_id in category_names:
            raise DatasetAuditError(f"duplicate COCO category {category_id}")
        category_names[category_id] = str(category["name"])

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    normalized_areas: list[float] = []
    truncated_count = 0
    for raw_annotation in annotations:
        if not isinstance(raw_annotation, dict):
            raise DatasetAuditError("COCO annotation must be an object")
        image_id = int(raw_annotation["image_id"])
        annotations_by_image[image_id].append(raw_annotation)

    config = tracker_manifest.get("invocation", {}).get("config", {})
    minimum_pixels = int(config.get("minimum_pixels", 1))
    minimum_box_width = int(config.get("minimum_box_width", 1))
    minimum_box_height = int(config.get("minimum_box_height", 1))
    episode_splits: dict[tuple[str, str], str] = {}
    rgb_hash_groups: dict[str, list[str]] = defaultdict(list)

    for sample, image_entry in zip(samples, images, strict=True):
        if not isinstance(sample, Mapping) or not isinstance(
            image_entry,
            Mapping,
        ):
            raise DatasetAuditError("dataset sample and COCO image must be objects")
        image_id = int(sample["image_id"])
        if image_id != int(image_entry["id"]):
            raise DatasetAuditError("dataset and COCO image order/IDs disagree")
        width = int(image_entry["width"])
        height = int(image_entry["height"])
        rgb_ref = sample["rgb"]
        mask_ref = sample["instance_mask"]
        label_ref = sample["yolo_label"]
        metadata_ref = sample["metadata"]
        if not all(
            isinstance(value, Mapping) for value in (rgb_ref, mask_ref, label_ref, metadata_ref)
        ):
            raise DatasetAuditError("sample artifact references must be objects")

        rgb_path = _source_path(root, str(rgb_ref["path"]))
        mask_path = _source_path(root, str(mask_ref["path"]))
        yolo_path = _source_path(root, str(label_ref["path"]))
        metadata_path = _source_path(root, str(metadata_ref["path"]))
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or rgb.shape != (height, width, 3):
            raise DatasetAuditError(f"invalid RGB image: {rgb_path}")
        if mask is None or mask.shape != (height, width, 4):
            raise DatasetAuditError(f"invalid instance mask: {mask_path}")

        metadata = _load_json(metadata_path)
        if int(metadata.get("carla_frame", -1)) != int(sample["carla_frame"]):
            raise DatasetAuditError("sample metadata CARLA frame mismatch")
        extracted = extract_instance_labels(
            mask,
            minimum_pixels=minimum_pixels,
            minimum_box_width=minimum_box_width,
            minimum_box_height=minimum_box_height,
        )
        if [label.as_dict() for label in extracted] != metadata.get("annotations"):
            raise DatasetAuditError(
                f"teacher mask does not reproduce metadata: {metadata_path.name}"
            )

        image_annotations = annotations_by_image.get(image_id, [])
        if len(image_annotations) != len(extracted):
            raise DatasetAuditError(f"COCO annotation count mismatch for image {image_id}")
        expected_yolo = _expected_yolo(
            image_annotations,
            width=width,
            height=height,
        )
        if yolo_path.read_text(encoding="utf-8") != expected_yolo:
            raise DatasetAuditError(f"YOLO export disagrees with COCO: {yolo_path.name}")

        episode_key = (str(sample["scenario_id"]), str(sample["episode_id"]))
        split = str(sample["split"])
        prior_split = episode_splits.setdefault(episode_key, split)
        if prior_split != split:
            raise DatasetAuditError(f"episode split leakage detected for {episode_key}")
        rgb_hash_groups[str(rgb_ref["sha256"])].append(str(sample["sample_id"]))

        for annotation in image_annotations:
            category_id = int(annotation["category_id"])
            if category_id not in category_names:
                raise DatasetAuditError(f"unknown COCO category ID {category_id}")
            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise DatasetAuditError("COCO bbox must contain four values")
            x, y, box_width, box_height = (float(value) for value in bbox)
            if (
                min(x, y, box_width, box_height) < 0.0
                or box_width <= 0.0
                or box_height <= 0.0
                or x + box_width > width
                or y + box_height > height
            ):
                raise DatasetAuditError(f"out-of-bounds COCO bbox in image {image_id}")
            attributes = annotation.get("attributes", {})
            truncated = bool(attributes.get("truncated", False))
            truncated_count += truncated
            class_name = category_names[category_id]
            class_counts[class_name] += 1
            normalized_area = box_width * box_height / (width * height)
            normalized_areas.append(normalized_area)
            rows.append(
                {
                    "image_id": image_id,
                    "sample_id": sample["sample_id"],
                    "split": split,
                    "scenario_id": sample["scenario_id"],
                    "episode_id": sample["episode_id"],
                    "carla_frame": sample["carla_frame"],
                    "annotation_id": annotation["id"],
                    "category_id": category_id,
                    "category_name": class_name,
                    "actor_id": attributes.get("carla_actor_id"),
                    "x": x,
                    "y": y,
                    "width": box_width,
                    "height": box_height,
                    "normalized_bbox_area": normalized_area,
                    "visible_area_pixels": attributes.get("visible_area_pixels"),
                    "mask_fill_ratio": attributes.get("mask_fill_ratio"),
                    "truncated": truncated,
                }
            )

    duplicate_groups = [
        sample_ids for sample_ids in rgb_hash_groups.values() if len(sample_ids) > 1
    ]
    summary = {
        "schema_version": DATASET_AUDIT_SCHEMA_VERSION,
        "audit_type": "carla_detection_dataset",
        "valid": True,
        "dataset_id": dataset.get("dataset_id"),
        "samples": len(samples),
        "annotations": len(annotations),
        "checks": {
            "teacher_masks_reproduced": len(samples),
            "yolo_exports_reproduced": len(samples),
            "episode_split_leakage": False,
            "exact_rgb_duplicate_groups": duplicate_groups,
        },
        "class_frequency": dict(sorted(class_counts.items())),
        "bbox_normalized_area": _descriptive(normalized_areas),
        "truncated_annotations": truncated_count,
        "split_frequency": dict(
            sorted(Counter(str(sample["split"]) for sample in samples).items())
        ),
        "scenario_frequency": dict(
            sorted(Counter(str(sample["scenario_id"]) for sample in samples).items())
        ),
    }
    return summary, rows, annotations_by_image


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = (
        tuple(rows[0].keys())
        if rows
        else (
            "image_id",
            "sample_id",
            "split",
            "scenario_id",
            "episode_id",
            "carla_frame",
            "annotation_id",
            "category_id",
            "category_name",
            "actor_id",
            "x",
            "y",
            "width",
            "height",
            "normalized_bbox_area",
            "visible_area_pixels",
            "mask_fill_ratio",
            "truncated",
        )
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_class_frequency(path: Path, frequency: Mapping[str, int]) -> None:
    labels = list(frequency)
    values = [frequency[label] for label in labels]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(labels, values, color="#22b5c3")
    axis.set_title("Dataset class frequency")
    axis.set_xlabel("Visible instances")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_bbox_area(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = [float(row["normalized_bbox_area"]) for row in rows]
    figure, axis = plt.subplots(figsize=(10, 6))
    if values:
        axis.hist(values, bins=min(30, max(5, round(math.sqrt(len(values))))), color="#8f63bd")
        axis.set_xscale("log")
    axis.set_title("Bounding-box area distribution")
    axis.set_xlabel("Box area / image area (log scale)")
    axis.set_ylabel("Annotations")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_montage(
    path: Path,
    root: Path,
    dataset: Mapping[str, Any],
    annotations_by_image: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    count: int,
) -> int:
    samples = dataset["samples"][:count]
    if not samples:
        raise DatasetAuditError("cannot create a montage for an empty dataset")
    columns = min(3, len(samples))
    rows = math.ceil(len(samples) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.5 * columns, 3.8 * rows),
        squeeze=False,
    )
    palette = plt.get_cmap("tab10")
    for index, sample in enumerate(samples):
        axis = axes[index // columns][index % columns]
        image = cv2.imread(
            str(_source_path(root, str(sample["rgb"]["path"]))),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise DatasetAuditError("could not read montage image")
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for annotation in annotations_by_image[int(sample["image_id"])]:
            x, y, width, height = (float(value) for value in annotation["bbox"])
            class_index = int(annotation["category_id"]) - 1
            color = palette(class_index % 10)
            rectangle = plt.Rectangle(
                (x, y),
                width,
                height,
                fill=False,
                linewidth=1.5,
                color=color,
            )
            axis.add_patch(rectangle)
        axis.set_title(
            f"{sample['sample_id']} | frame {sample['carla_frame']} | "
            f"{sample['annotation_count']} boxes"
        )
        axis.axis("off")
    for index in range(len(samples), rows * columns):
        axes[index // columns][index % columns].axis("off")
    figure.suptitle("Dataset QA sample: visible teacher boxes", fontsize=14)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return len(samples)


def audit_dataset(
    dataset_dir: str | Path,
    *,
    runs_root: str | Path = "runs",
    run_id: str | None = None,
    montage_count: int = 9,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    if montage_count <= 0:
        raise ValueError("montage_count must be positive")
    try:
        source_root = Path(dataset_dir).expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise DatasetAuditError(f"dataset directory does not exist: {dataset_dir}") from error
    if not source_root.is_dir():
        raise DatasetAuditError(f"dataset path is not a directory: {source_root}")

    manifest_path = source_root / "manifest.json"
    dataset_path = source_root / "dataset.json"
    coco_path = source_root / "annotations" / "instances.coco.json"
    source_before = {
        "manifest": fingerprint_file(manifest_path),
        "dataset": fingerprint_file(dataset_path),
        "coco": fingerprint_file(coco_path),
        "checksums": fingerprint_file(source_root / "checksums.sha256"),
    }
    tracker_manifest = _load_json(manifest_path)
    dataset = _load_json(dataset_path)
    coco = _load_json(coco_path)
    _verify_registered_artifacts(source_root, tracker_manifest)
    checksum_entries = _verify_checksum_index(source_root)
    summary, rows, annotations_by_image = _audit_records(
        source_root,
        dataset,
        coco,
        tracker_manifest,
    )
    summary["checks"]["checksum_entries_verified"] = len(checksum_entries)

    source_reference = {
        "type": "source_dataset",
        "dataset_id": dataset.get("dataset_id"),
        "path": source_root.as_posix(),
        "files": source_before,
    }
    carla = tracker_manifest.get("carla", {})
    environment_packages = (*DEFAULT_PACKAGE_NAMES, "matplotlib")
    tracker = RunArtifactTracker(
        runs_root,
        run_id=run_id,
        cli_args=(),
        config={
            "analysis_type": "dataset_quality_audit",
            "schema_version": DATASET_AUDIT_SCHEMA_VERSION,
            "source_dataset": source_reference,
            "montage_count": montage_count,
        },
        repository_root=repository_root,
        carla_endpoint=carla.get("endpoint"),
        carla_version=carla.get("version"),
        carla_map=carla.get("map"),
        dataset_refs=(source_reference,),
        package_names=environment_packages,
    )
    with tracker:
        summary["audit_run_id"] = tracker.run_id
        summary["source_dataset"] = source_reference
        summary_path = tracker.artifact_path("summary_dataset_qa.json")
        rows_path = tracker.artifact_path("annotations.csv")
        class_path = tracker.artifact_path("plots/class_frequency.png")
        area_path = tracker.artifact_path("plots/bbox_area_distribution.png")
        montage_path = tracker.artifact_path("plots/qa_montage.png")
        _write_rows(rows_path, rows)
        _plot_class_frequency(class_path, summary["class_frequency"])
        _plot_bbox_area(area_path, rows)
        montage_samples = _write_montage(
            montage_path,
            source_root,
            dataset,
            annotations_by_image,
            count=montage_count,
        )
        summary["checks"]["montage_samples"] = montage_samples
        _write_json(summary_path, summary)
        for path, role in (
            (summary_path, "dataset_qa_summary"),
            (rows_path, "dataset_annotation_table"),
            (class_path, "dataset_qa_class_plot"),
            (area_path, "dataset_qa_bbox_plot"),
            (montage_path, "dataset_qa_montage"),
        ):
            tracker.register_artifact(
                path,
                role=role,
                metadata={
                    "source_dataset_id": dataset.get("dataset_id"),
                    "audit_schema_version": DATASET_AUDIT_SCHEMA_VERSION,
                },
            )

        source_after = {
            "manifest": fingerprint_file(manifest_path),
            "dataset": fingerprint_file(dataset_path),
            "coco": fingerprint_file(coco_path),
            "checksums": fingerprint_file(source_root / "checksums.sha256"),
        }
        if source_before != source_after:
            raise DatasetAuditError("source dataset changed during audit")

    return {
        "run_id": tracker.run_id,
        "run_dir": tracker.run_dir.as_posix(),
        "summary": summary,
        "manifest": tracker.manifest_path.as_posix(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a CARLA dataset and create a read-only QA analysis run"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run-id")
    parser.add_argument("--montage-count", type=int, default=9)
    args = parser.parse_args(argv)
    if args.montage_count <= 0:
        parser.error("--montage-count must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit_dataset(
        args.dataset,
        runs_root=args.runs_root,
        run_id=args.run_id,
        montage_count=args.montage_count,
        repository_root=Path.cwd(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


__all__ = [
    "DATASET_AUDIT_SCHEMA_VERSION",
    "DatasetAuditError",
    "audit_dataset",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    raise SystemExit(main())
