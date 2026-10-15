"""Create immutable training-side FREEZE/RELABEL releases using the canonical writer.

RECAPTURE/NEW_SCENE are deliberately not emulated by copying old RGB: their
outputs must come from the existing native collector, with new capture evidence.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2

from ..artifacts import fingerprint_file
from ..bridge import CarlaImageFrame
from .lineage import canonical_digest, require_train_parent, scene_group_key, validate_derivation
from .sync import SynchronizedFramePair
from .verified import load_verified_dataset
from .writer import DatasetWriter


def _frame(pixels, metadata, *, sequence: int, sensor_type: int, calibration=None, transform=None):
    camera = metadata["camera"]
    return CarlaImageFrame(
        sequence=sequence,
        sensor_type=sensor_type,
        frame=metadata["carla_frame"],
        timestamp=metadata["simulation_timestamp_seconds"],
        transform=tuple(transform if transform is not None else camera["world_transform"]),
        width=pixels.shape[1],
        height=pixels.shape[0],
        fov=calibration["fov_degrees"] if calibration is not None else camera["fov_degrees"],
        bgra=pixels.tobytes(),
        received_monotonic=0.0,
    )


def derive_dataset(
    parent: str | Path,
    *,
    datasets_root: str | Path,
    dataset_id: str,
    action: str,
    sample_ids: Sequence[str] | None = None,
    review_path: str | Path | None = None,
) -> dict[str, Any]:
    source = load_verified_dataset(parent)
    if action not in {"FREEZE", "RELABEL"}:
        raise ValueError("recapture/new-scene requires genuine collector output, not copied RGB")
    selected = (
        set(sample_ids)
        if sample_ids is not None
        else {str(s["sample_id"]) for s in source.dataset["samples"]}
    )
    available = {str(s["sample_id"]) for s in source.dataset["samples"]}
    if not selected or not selected <= available:
        raise ValueError("selection must contain existing source sample IDs")
    if sample_ids is not None and len(sample_ids) != len(selected):
        raise ValueError("duplicate selected sample IDs")
    samples = [s for s in source.dataset["samples"] if str(s["sample_id"]) in selected]
    for sample in samples:
        require_train_parent(sample["split"])
        child_split = "train" if action == "FREEZE" else sample["split"]
        validate_derivation(sample["split"], child_split, action)
    corrections = {}
    review_ref = None
    if action == "RELABEL":
        if review_path is None:
            raise ValueError("RELABEL requires an independent review manifest")
        review_ref = fingerprint_file(review_path)
        review = json.loads(Path(review_path).read_text(encoding="utf-8"))
        if (
            set(review)
            != {"schema_version", "parent_manifest", "blind_to_model", "reviewer_id", "corrections"}
            or review["schema_version"] != "1.0"
            or review["parent_manifest"] != source.reference["manifest"]
            or review["blind_to_model"] is not True
            or not review["reviewer_id"]
        ):
            raise ValueError("review must identify the exact parent and independent reviewer")
        corrections = review["corrections"]
        if not isinstance(corrections, dict) or set(corrections) != selected:
            raise ValueError("each selected relabel must reference its corrected mask")
        for reference in corrections.values():
            if not isinstance(reference, dict) or fingerprint_file(reference["path"]) != reference:
                raise ValueError("corrected mask fingerprint changed")
    elif review_path is not None:
        raise ValueError("FREEZE cannot silently apply relabels")
    config = source.manifest["invocation"]["config"]
    provenance = {
        "schema_version": "1.0",
        "action": action,
        "parent": dict(source.reference),
        "review": review_ref,
        "selected_source_sample_ids": sorted(selected),
        "original_bytes_modified": False,
        "new_capture_performed": False,
    }
    carla = source.manifest.get("carla", {})
    with DatasetWriter(
        datasets_root,
        dataset_id=dataset_id,
        carla_endpoint=carla.get("endpoint"),
        carla_version=carla.get("version"),
        carla_map=carla.get("map"),
        config={"derivation": provenance, "collector": "immutable_training_derivation"},
        minimum_pixels=config.get("minimum_pixels", 16),
        minimum_box_width=config.get("minimum_box_width", 2),
        minimum_box_height=config.get("minimum_box_height", 2),
    ) as writer:
        writer.add_auxiliary_json(
            "provenance/derivation.json", provenance, role="dataset_derivation"
        )
        for sequence, sample in enumerate(samples, start=1):
            metadata = json.loads(
                (source.root / sample["metadata"]["path"]).read_text(encoding="utf-8")
            )
            rgb = cv2.imread(str(source.root / sample["rgb"]["path"]), cv2.IMREAD_COLOR)
            mask_path = (
                Path(corrections[sample["sample_id"]]["path"])
                if corrections
                else source.root / sample["instance_mask"]["path"]
            )
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if rgb is None or mask is None or mask.shape != (*rgb.shape[:2], 4):
                raise ValueError("corrected mask must be aligned BGRA at original dimensions")
            rgb_frame = _frame(
                cv2.cvtColor(rgb, cv2.COLOR_BGR2BGRA), metadata, sequence=sequence, sensor_type=0
            )
            teacher = _frame(mask, metadata, sequence=sequence, sensor_type=1)
            views, calibrations = None, None
            if metadata.get("rgb_views"):
                views, calibrations = {"front": rgb_frame}, {}
                for name, view in metadata["rgb_views"].items():
                    calibrations[name] = view["calibration"]
                    if name != "front":
                        pixels = cv2.imread(
                            str(source.root / view["image"]["path"]), cv2.IMREAD_COLOR
                        )
                        if pixels is None:
                            raise ValueError("source view no longer decodes")
                        views[name] = _frame(
                            cv2.cvtColor(pixels, cv2.COLOR_BGR2BGRA),
                            metadata,
                            sequence=sequence,
                            sensor_type=0,
                            calibration=view["calibration"],
                            transform=view["privileged_camera_world_transform"],
                        )
            context = dict(metadata.get("context", {}))
            existing = context.get("lineage", {})
            root_id = existing.get("scene_root_id") or scene_group_key(
                context, episode_id=sample["episode_id"]
            )
            context["lineage"] = {
                "schema_version": "1.0",
                "scene_root_id": root_id,
                "parent_dataset_id": source.dataset_id,
                "parent_sample_id": sample["sample_id"],
                "parent_partition": sample["split"],
                "parent_manifest_sha256": source.reference["manifest"]["sha256"],
                "action": action,
            }
            writer.add_pair(
                SynchronizedFramePair(rgb_frame, teacher, 0, 0),
                rgb_views=views,
                camera_calibrations=calibrations,
                split="train" if action == "FREEZE" else sample["split"],
                scenario_id=sample["scenario_id"],
                episode_id=sample["episode_id"],
                context=context,
            )
        upstream = source.dataset.get("release_metadata", {})
        writer.set_release_metadata(
            {
                "derivation": provenance,
                "control_mode": upstream.get("control_mode"),
                "episode_ids": sorted({s["episode_id"] for s in samples}),
                "sample_count": len(samples),
                "capture_evidence_inherited_not_rerun": True,
            }
        )
        # Recheck every correction/review as well as the parent before success.
        for reference in [*corrections.values(), *([review_ref] if review_ref else [])]:
            if fingerprint_file(reference["path"]) != reference:
                raise RuntimeError("review or corrected mask changed while deriving release")
        # Fail the tracker before finalization if the source changed mid-copy.
        if load_verified_dataset(parent).reference != source.reference:
            raise RuntimeError("parent changed while deriving release")
    result = load_verified_dataset(Path(datasets_root) / dataset_id)
    return {
        "dataset": dict(result.reference),
        "sample_count": len(samples),
        "derivation_sha256": canonical_digest(provenance),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--datasets-root", default="datasets")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--action", choices=("FREEZE", "RELABEL"), required=True)
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--review")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            derive_dataset(
                args.parent,
                datasets_root=args.datasets_root,
                dataset_id=args.dataset_id,
                action=args.action,
                sample_ids=args.sample_id,
                review_path=args.review,
            ),
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
