"""Consumer-side semantic verification for paired replay releases."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import fingerprint_file
from ..verification import ArtifactIntegrityError, verify_research_object
from .contracts import ReplayConfig
from .runner import REPLAY_RUN_SCHEMA_VERSION


class ReplayIntegrityError(RuntimeError):
    """Raised when a paired replay release cannot be trusted."""


@dataclass(frozen=True)
class VerifiedReplay:
    root: Path
    replay_id: str
    descriptor: Mapping[str, Any]
    config: ReplayConfig
    reference: Mapping[str, Any]


def _load_json(path: Path, name: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReplayIntegrityError(f"could not read {name}: {error}") from error


def _artifact_path(
    root: Path,
    manifest: Mapping[str, Any],
    role: str,
) -> Path:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ReplayIntegrityError("replay tracker artifacts must be an array")
    matches = [
        entry for entry in artifacts if isinstance(entry, Mapping) and entry.get("role") == role
    ]
    if len(matches) != 1:
        raise ReplayIntegrityError(f"replay release must contain exactly one {role!r} artifact")
    return (root / str(matches[0]["path"])).resolve(strict=True)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ReplayIntegrityError(f"replay CSV has no header: {path.name}")
            return list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ReplayIntegrityError(f"could not read replay CSV {path.name}: {error}") from error


def _order_digest(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_order_rows(path: Path) -> list[dict[str, Any]]:
    rows = _csv_rows(path)
    expected_fields = {
        "position",
        "sample_id",
        "image_id",
        "partition",
        "scenario_id",
        "episode_id",
        "carla_frame",
        "rgb_sha256",
    }
    result = []
    prior_position = 0
    seen_images: set[int] = set()
    for row in rows:
        if set(row) != expected_fields:
            raise ReplayIntegrityError("replay sample-order CSV schema is invalid")
        try:
            position = int(row["position"])
            image_id = int(row["image_id"])
            carla_frame = int(row["carla_frame"])
        except ValueError as error:
            raise ReplayIntegrityError("replay sample-order numeric field is invalid") from error
        if position != prior_position + 1 or image_id <= 0 or image_id in seen_images:
            raise ReplayIntegrityError("replay sample order is not contiguous and unique")
        rgb_digest = row["rgb_sha256"]
        if len(rgb_digest) != 64 or any(
            character not in "0123456789abcdef" for character in rgb_digest
        ):
            raise ReplayIntegrityError("replay RGB SHA-256 is invalid")
        if any(
            not row[field]
            for field in (
                "sample_id",
                "partition",
                "scenario_id",
                "episode_id",
            )
        ):
            raise ReplayIntegrityError("replay sample-order grouping field is empty")
        result.append(
            {
                "position": position,
                "sample_id": row["sample_id"],
                "image_id": image_id,
                "partition": row["partition"],
                "scenario_id": row["scenario_id"],
                "episode_id": row["episode_id"],
                "carla_frame": carla_frame,
                "rgb_sha256": rgb_digest,
            }
        )
        prior_position = position
        seen_images.add(image_id)
    if not result:
        raise ReplayIntegrityError("replay sample order is empty")
    return result


def _child_frame_order(
    path: Path,
    rgb_by_image: Mapping[int, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ReplayIntegrityError(
                        f"child frame log line {line_number} is not an object"
                    )
                image_id = int(value["image_id"])
                if image_id not in rgb_by_image:
                    raise ReplayIntegrityError(
                        f"child frame log contains unexpected image {image_id}"
                    )
                rows.append(
                    {
                        "position": int(value["position"]),
                        "sample_id": str(value["sample_id"]),
                        "image_id": image_id,
                        "partition": str(value["partition"]),
                        "scenario_id": str(value["scenario_id"]),
                        "episode_id": str(value["episode_id"]),
                        "carla_frame": int(value["carla_frame"]),
                        "rgb_sha256": rgb_by_image[image_id],
                    }
                )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        if isinstance(error, ReplayIntegrityError):
            raise
        raise ReplayIntegrityError(f"could not validate child frame log: {error}") from error
    return rows


def load_verified_replay(path: str | Path) -> VerifiedReplay:
    try:
        verification = verify_research_object(
            path,
            verify_references=True,
            deep=False,
            reject_unregistered=True,
        )
    except ArtifactIntegrityError as error:
        raise ReplayIntegrityError(str(error)) from error
    root = Path(verification.root)
    manifest = _load_json(root / "manifest.json", "replay tracker manifest")
    if not isinstance(manifest, Mapping):
        raise ReplayIntegrityError("replay tracker manifest must be an object")
    descriptor_path = _artifact_path(root, manifest, "replay_release_manifest")
    resolved_path = _artifact_path(
        root,
        manifest,
        "resolved_replay_configuration",
    )
    sample_order_path = _artifact_path(root, manifest, "replay_sample_order")
    child_runs_path = _artifact_path(root, manifest, "replay_child_evaluations")
    aggregate_path = _artifact_path(root, manifest, "replay_aggregate_metrics")
    per_image_path = _artifact_path(root, manifest, "replay_per_image_metrics")
    paired_path = _artifact_path(root, manifest, "replay_paired_differences")
    disagreements_path = _artifact_path(root, manifest, "replay_disagreements")
    bootstrap_path = _artifact_path(root, manifest, "replay_paired_bootstrap")
    descriptor = _load_json(descriptor_path, "replay release manifest")
    resolved = _load_json(resolved_path, "resolved replay configuration")
    child_runs = _load_json(child_runs_path, "replay child evaluations")
    bootstrap = _load_json(bootstrap_path, "replay paired bootstrap")
    if not all(
        isinstance(value, Mapping) for value in (descriptor, resolved, bootstrap)
    ) or not isinstance(child_runs, list):
        raise ReplayIntegrityError("replay JSON artifact envelope is invalid")
    replay_raw = resolved.get("replay")
    if not isinstance(replay_raw, Mapping):
        raise ReplayIntegrityError("resolved replay contract is missing")
    try:
        config = ReplayConfig.from_mapping(replay_raw, base=root)
    except (TypeError, ValueError, OSError) as error:
        raise ReplayIntegrityError(f"resolved replay contract is invalid: {error}") from error
    if (
        descriptor.get("schema_version") != REPLAY_RUN_SCHEMA_VERSION
        or descriptor.get("object_type") != "paired_detector_replay_release"
        or descriptor.get("status") != "complete"
    ):
        raise ReplayIntegrityError("replay release descriptor envelope is invalid")
    replay_id = str(descriptor.get("replay_id", ""))
    if replay_id != verification.run_id or replay_id != config.replay_id:
        raise ReplayIntegrityError("replay identity disagrees across package files")
    if descriptor.get("purpose") != config.purpose:
        raise ReplayIntegrityError("replay purpose disagrees with preregistration")
    if descriptor.get("runtime_sensor_contract") != "front_monocular_rgb_only":
        raise ReplayIntegrityError("replay runtime sensor contract is not monocular RGB")
    if config.purpose == "confirmatory":
        try:
            verify_research_object(
                root,
                verify_references=True,
                deep=False,
                reject_unregistered=True,
                require_clean_git=True,
            )
        except ArtifactIntegrityError as error:
            raise ReplayIntegrityError(
                f"confirmatory replay clean-git gate failed: {error}"
            ) from error

    order_rows = _canonical_order_rows(sample_order_path)
    order_digest = _order_digest(order_rows)
    if (
        descriptor.get("sample_count") != len(order_rows)
        or descriptor.get("sample_order_sha256") != order_digest
        or resolved.get("sample_order_sha256") != order_digest
        or resolved.get("selected_sample_count") != len(order_rows)
    ):
        raise ReplayIntegrityError("replay sample count or order digest is inconsistent")
    rgb_by_image = {int(row["image_id"]): str(row["rgb_sha256"]) for row in order_rows}

    model_ids = [model.model_id for model in config.models]
    descriptor_models = descriptor.get("models")
    resolved_models = resolved.get("models")
    if (
        not isinstance(descriptor_models, list)
        or not isinstance(resolved_models, list)
        or [str(model.get("model_id", "")) for model in descriptor_models] != model_ids
        or [
            str(model.get("replay_model_id", ""))
            for model in resolved_models
            if isinstance(model, Mapping)
        ]
        != model_ids
    ):
        raise ReplayIntegrityError("replay model order or identity is inconsistent")
    if (
        descriptor.get("model_count") != len(model_ids)
        or len(child_runs) != len(model_ids)
        or descriptor.get("child_runs") != child_runs
    ):
        raise ReplayIntegrityError("replay model or child-run count is inconsistent")
    child_ids: set[str] = set()
    for child in child_runs:
        if not isinstance(child, Mapping):
            raise ReplayIntegrityError("replay child descriptor must be an object")
        model_id = str(child.get("model_id", ""))
        run_id = str(child.get("run_id", ""))
        root_value = child.get("root")
        manifest_reference = child.get("manifest")
        if (
            model_id not in model_ids
            or model_id in child_ids
            or not run_id
            or not isinstance(root_value, str)
            or not isinstance(manifest_reference, Mapping)
        ):
            raise ReplayIntegrityError("replay child identity is invalid or duplicated")
        try:
            child_verification = verify_research_object(
                root_value,
                verify_references=True,
                deep=True,
                reject_unregistered=True,
                require_clean_git=config.purpose == "confirmatory",
            )
        except ArtifactIntegrityError as error:
            raise ReplayIntegrityError(
                f"replay child {model_id!r} failed verification: {error}"
            ) from error
        child_root = Path(child_verification.root)
        actual_manifest = fingerprint_file(child_root / "manifest.json")
        if (
            child_verification.run_id != run_id
            or manifest_reference.get("sha256") != actual_manifest["sha256"]
            or manifest_reference.get("size_bytes") != actual_manifest["size_bytes"]
            or Path(str(manifest_reference.get("path", ""))).resolve()
            != (child_root / "manifest.json").resolve()
        ):
            raise ReplayIntegrityError(f"replay child {model_id!r} identity or manifest changed")
        child_manifest = _load_json(
            child_root / "manifest.json",
            f"{model_id} child manifest",
        )
        if not isinstance(child_manifest, Mapping):
            raise ReplayIntegrityError("child tracker manifest must be an object")
        frame_path = _artifact_path(
            child_root,
            child_manifest,
            "per_image_prediction_log",
        )
        child_order = _child_frame_order(frame_path, rgb_by_image)
        if child_order != order_rows or child.get("sample_order_sha256") != order_digest:
            raise ReplayIntegrityError(f"replay child {model_id!r} sample order differs")
        child_ids.add(model_id)
    if child_ids != set(model_ids):
        raise ReplayIntegrityError("replay child model set is incomplete")

    aggregate_rows = _csv_rows(aggregate_path)
    per_image_rows = _csv_rows(per_image_path)
    paired_rows = _csv_rows(paired_path)
    disagreement_rows = _csv_rows(disagreements_path)
    expected_pair_metrics = 7
    if len(aggregate_rows) != len(model_ids):
        raise ReplayIntegrityError("replay aggregate table row count is inconsistent")
    if len(per_image_rows) != len(model_ids) * len(order_rows):
        raise ReplayIntegrityError("replay per-image table row count is inconsistent")
    if len(paired_rows) != (len(model_ids) - 1) * expected_pair_metrics:
        raise ReplayIntegrityError("replay paired-difference row count is inconsistent")
    if len(disagreement_rows) != (len(model_ids) - 1) * len(order_rows):
        raise ReplayIntegrityError("replay disagreement row count is inconsistent")
    if descriptor.get("paired_comparison_count") != len(paired_rows) or descriptor.get(
        "disagreement_row_count"
    ) != len(disagreement_rows):
        raise ReplayIntegrityError("replay descriptor table counts are inconsistent")
    aggregate_model_ids = [row.get("model_id") for row in aggregate_rows]
    if aggregate_model_ids != model_ids:
        raise ReplayIntegrityError("replay aggregate model order changed")
    child_run_by_model = {str(child["model_id"]): str(child["run_id"]) for child in child_runs}
    if any(
        row.get("child_run_id") != child_run_by_model[row["model_id"]] for row in aggregate_rows
    ):
        # Schema 1.0 originally emitted ``evaluation_id/run_id`` in one
        # development pilot. Accept that explicit legacy spelling while still
        # requiring the immutable child run ID as the suffix.
        if any(
            not str(row.get("child_run_id", "")).endswith("/" + child_run_by_model[row["model_id"]])
            for row in aggregate_rows
        ):
            raise ReplayIntegrityError("replay aggregate child-run identity changed")

    expected_order_projection = [
        {
            "position": str(row["position"]),
            "sample_id": str(row["sample_id"]),
            "image_id": str(row["image_id"]),
            "partition": str(row["partition"]),
            "scenario_id": str(row["scenario_id"]),
            "episode_id": str(row["episode_id"]),
            "carla_frame": str(row["carla_frame"]),
            "rgb_sha256": str(row["rgb_sha256"]),
        }
        for row in order_rows
    ]
    for model_index, model_id in enumerate(model_ids):
        start = model_index * len(order_rows)
        model_rows = per_image_rows[start : start + len(order_rows)]
        if [row.get("model_id") for row in model_rows] != [model_id] * len(order_rows):
            raise ReplayIntegrityError("replay per-image model grouping changed")
        projection = [
            {key: row.get(key) for key in expected_order_projection[0]} for row in model_rows
        ]
        if projection != expected_order_projection:
            raise ReplayIntegrityError("replay per-image sample order changed")

    expected_metrics = {
        "precision",
        "recall",
        "f1",
        "fp",
        "fn",
        "error_count",
        "latency_mean_ms",
    }
    candidate_ids = [model_id for model_id in model_ids if model_id != config.reference_model_id]
    for candidate_id in candidate_ids:
        candidate_rows = [
            row for row in paired_rows if row.get("candidate_model_id") == candidate_id
        ]
        if (
            len(candidate_rows) != len(expected_metrics)
            or {row.get("metric") for row in candidate_rows} != expected_metrics
            or any(
                row.get("reference_model_id") != config.reference_model_id for row in candidate_rows
            )
        ):
            raise ReplayIntegrityError("replay paired metric set is inconsistent")
        candidate_disagreements = [
            row for row in disagreement_rows if row.get("candidate_model_id") == candidate_id
        ]
        if (
            len(candidate_disagreements) != len(order_rows)
            or [row.get("image_id") for row in candidate_disagreements]
            != [str(row["image_id"]) for row in order_rows]
            or any(
                row.get("reference_model_id") != config.reference_model_id
                for row in candidate_disagreements
            )
        ):
            raise ReplayIntegrityError("replay disagreement sample pairing is inconsistent")
    outcome_count = sum(
        str(row.get("outcome_disagreement", "")).casefold() == "true" for row in disagreement_rows
    )
    if descriptor.get("outcome_disagreement_count") != outcome_count:
        raise ReplayIntegrityError("replay disagreement summary count changed")
    if bootstrap.get("reference_model_id") != config.reference_model_id or set(
        dict(bootstrap.get("pairs", {}))
    ) != set(candidate_ids):
        raise ReplayIntegrityError("replay bootstrap pair set is inconsistent")
    for candidate_id, pair in dict(bootstrap["pairs"]).items():
        if (
            not isinstance(pair, Mapping)
            or pair.get("replicates") != config.bootstrap_replicates
            or pair.get("confidence") != config.bootstrap_confidence
            or set(dict(pair.get("metrics", {}))) != expected_metrics
        ):
            raise ReplayIntegrityError(f"replay bootstrap contract changed for {candidate_id!r}")

    return VerifiedReplay(
        root=root,
        replay_id=replay_id,
        descriptor=descriptor,
        config=config,
        reference={
            "kind": "verified_paired_replay",
            "replay_id": replay_id,
            "manifest": fingerprint_file(root / "manifest.json"),
            "replay_manifest": fingerprint_file(descriptor_path),
            "sample_order": fingerprint_file(sample_order_path),
        },
    )


__all__ = [
    "ReplayIntegrityError",
    "VerifiedReplay",
    "load_verified_replay",
]
