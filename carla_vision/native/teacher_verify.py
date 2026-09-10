"""Verify BehaviorAgent teacher episodes and their immutable DatasetWriter artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2

from ..dataset.camera_views import verify_camera_views, view_path
from .teacher_routes import NAVIGATION_INTENT_SCHEMA_VERSION, validate_behavior_sample_context

TEACHER_EPISODE_VERIFICATION_SCHEMA_VERSION = "1.0"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checksum_index(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or not relative:
            raise ValueError(f"invalid checksum line {line_number}: {raw!r}")
        if relative in checksums:
            raise ValueError(f"duplicate checksum path: {relative}")
        checksums[relative] = digest
    return checksums


def _artifact_path(dataset_dir: Path, artifact: Mapping[str, Any], name: str) -> Path:
    relative = artifact.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"sample {name} artifact path is missing")
    path = (dataset_dir / relative).resolve(strict=False)
    try:
        path.relative_to(dataset_dir.resolve())
    except ValueError as error:
        raise ValueError(f"sample {name} artifact escapes dataset directory") from error
    return path


def verify_teacher_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    """Return a detailed report and raise no exception for content failures."""

    root = Path(dataset_dir).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    dataset_path = root / "dataset.json"
    checksum_path = root / "checksums.sha256"
    try:
        dataset = _read_json(dataset_path)
    except (FileNotFoundError, ValueError) as error:
        return {
            "schema_version": TEACHER_EPISODE_VERIFICATION_SCHEMA_VERSION,
            "dataset_dir": str(root),
            "status": "failed",
            "sample_count": 0,
            "episode_count": 0,
            "errors": [str(error)],
            "warnings": [],
        }

    try:
        checksums = _load_checksum_index(checksum_path)
    except (FileNotFoundError, OSError, ValueError) as error:
        checksums = {}
        errors.append(str(error))

    for relative, expected in sorted(checksums.items()):
        try:
            path = view_path(root, relative)
        except (OSError, ValueError) as error:
            errors.append(f"checksum artifact {relative}: {error}")
            continue
        actual = _sha256(path)
        if actual != expected:
            errors.append(f"checksum mismatch: {relative}")

    samples = dataset.get("samples")
    if not isinstance(samples, list):
        samples = []
        errors.append("dataset.samples must be a list")
    declared_count = dataset.get("sample_count")
    if declared_count != len(samples):
        errors.append(
            f"dataset.sample_count={declared_count!r} does not match {len(samples)} samples"
        )
    if dataset.get("status") != "complete":
        errors.append("dataset.status must be complete")

    frames_by_episode: dict[str, list[int]] = defaultdict(list)
    timestamps_by_episode: dict[str, list[float]] = defaultdict(list)
    route_ids: set[str] = set()
    navigation_intent_count = 0

    for index, sample in enumerate(samples):
        prefix = f"sample[{index}]"
        if not isinstance(sample, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        episode_id = sample.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            errors.append(f"{prefix}.episode_id must be a non-empty string")
            continue
        frame = sample.get("carla_frame")
        timestamp = sample.get("source_timestamp")
        if not isinstance(frame, int) or frame < 0:
            errors.append(f"{prefix}.carla_frame must be a non-negative integer")
            continue
        if not isinstance(timestamp, (int, float)) or not math.isfinite(float(timestamp)):
            errors.append(f"{prefix}.source_timestamp must be finite")
            continue
        frames_by_episode[episode_id].append(frame)
        timestamps_by_episode[episode_id].append(float(timestamp))

        for artifact_name in ("rgb", "metadata"):
            artifact = sample.get(artifact_name)
            if not isinstance(artifact, Mapping):
                errors.append(f"{prefix}.{artifact_name} must be an artifact object")
                continue
            try:
                path = _artifact_path(root, artifact, artifact_name)
            except ValueError as error:
                errors.append(f"{prefix}: {error}")
                continue
            if not path.is_file():
                errors.append(f"{prefix} missing {artifact_name}: {path.relative_to(root)}")
                continue
            expected_sha = artifact.get("sha256")
            if isinstance(expected_sha, str) and _sha256(path) != expected_sha:
                errors.append(f"{prefix} {artifact_name} sha256 mismatch")

        rgb_artifact = sample.get("rgb")
        if isinstance(rgb_artifact, Mapping):
            try:
                rgb_path = _artifact_path(root, rgb_artifact, "rgb")
                image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                if image is None or image.size == 0:
                    errors.append(f"{prefix} RGB image cannot be decoded")
            except ValueError:
                image = None
        else:
            image = None

        metadata_artifact = sample.get("metadata")
        if not isinstance(metadata_artifact, Mapping):
            continue
        try:
            metadata_path = _artifact_path(root, metadata_artifact, "metadata")
            metadata = _read_json(metadata_path)
        except (FileNotFoundError, ValueError) as error:
            errors.append(f"{prefix} invalid metadata: {error}")
            continue
        if metadata.get("carla_frame") != frame:
            errors.append(f"{prefix} metadata frame does not match dataset sample")
        errors.extend(
            f"{prefix}: {message}" for message in verify_camera_views(
                root, sample, metadata, dataset.get("rgb_camera_ids"), checksums,
            )
        )
        metadata_timestamp = metadata.get("simulation_timestamp_seconds")
        if not isinstance(metadata_timestamp, (int, float)) or not math.isclose(
            float(metadata_timestamp), float(timestamp), rel_tol=0.0, abs_tol=1e-6
        ):
            errors.append(f"{prefix} metadata timestamp does not match dataset sample")
        camera = metadata.get("camera")
        if isinstance(camera, Mapping) and image is not None:
            expected_size = (camera.get("height"), camera.get("width"))
            if expected_size != image.shape[:2]:
                errors.append(
                    f"{prefix} RGB dimensions {image.shape[:2]} do not match metadata {expected_size}"
                )
        synchronization = metadata.get("synchronization")
        if not isinstance(synchronization, Mapping) or not synchronization.get(
            "exact_carla_frame_match"
        ):
            errors.append(f"{prefix} lacks exact RGB/teacher frame synchronization")
        context = metadata.get("context")
        if not isinstance(context, Mapping):
            errors.append(f"{prefix}.context must be an object")
            continue
        errors.extend(
            f"{prefix}: {message}"
            for message in validate_behavior_sample_context(context, carla_frame=frame)
        )
        route = context.get("route")
        if isinstance(route, Mapping) and isinstance(route.get("route_id"), str):
            route_ids.add(str(route["route_id"]))
        if isinstance(context.get("navigation_intent"), Mapping):
            navigation_intent_count += 1
        ego_state = context.get("privileged_evaluation")
        speed = None
        if isinstance(ego_state, Mapping):
            velocity = ego_state.get("velocity_mps")
            if isinstance(velocity, Mapping):
                speed = velocity.get("speed")
        if not isinstance(speed, (int, float)) or not math.isfinite(float(speed)) or speed < 0:
            errors.append(f"{prefix} privileged ego speed is missing or invalid")

    for episode_id, frames in frames_by_episode.items():
        if any(right <= left for left, right in zip(frames, frames[1:], strict=False)):
            errors.append(f"episode {episode_id} frame IDs are not strictly increasing")
    for episode_id, timestamps in timestamps_by_episode.items():
        if any(right < left for left, right in zip(timestamps, timestamps[1:], strict=False)):
            errors.append(f"episode {episode_id} timestamps are not monotonic")

    release = dataset.get("release_metadata")
    if not isinstance(release, Mapping):
        errors.append("dataset.release_metadata must be an object")
    else:
        if release.get("control_mode") != "behavior_agent_teacher":
            errors.append("release_metadata.control_mode must be behavior_agent_teacher")
        declared_episodes = release.get("episode_ids")
        actual_episodes = sorted(frames_by_episode)
        if isinstance(declared_episodes, list) and sorted(declared_episodes) != actual_episodes:
            errors.append("release_metadata.episode_ids does not match sample episodes")
        navigation_release = release.get("navigation_intent")
        if isinstance(navigation_release, Mapping):
            if navigation_release.get("schema_version") != NAVIGATION_INTENT_SCHEMA_VERSION:
                errors.append("release navigation_intent schema_version is unsupported")
            if navigation_release.get("privileged") is not True:
                errors.append("release navigation_intent must declare privileged=true")
            if navigation_release.get("runtime_model_input") is not False:
                errors.append(
                    "release navigation_intent must declare runtime_model_input=false"
                )
            if navigation_release.get("strict_rgb_only_input") is not False:
                errors.append(
                    "release navigation_intent must declare strict_rgb_only_input=false"
                )
            if navigation_release.get("route_conditioned_vision_input") is not True:
                errors.append(
                    "release navigation_intent must declare "
                    "route_conditioned_vision_input=true"
                )
            if navigation_release.get("available_for_every_sample") is True and (
                navigation_intent_count != len(samples)
            ):
                errors.append(
                    "release declares navigation intent for every sample but "
                    f"found {navigation_intent_count}/{len(samples)}"
                )

    if not route_ids and samples:
        warnings.append("no route IDs were observed")

    return {
        "schema_version": TEACHER_EPISODE_VERIFICATION_SCHEMA_VERSION,
        "dataset_dir": str(root),
        "dataset_id": dataset.get("dataset_id"),
        "status": "passed" if not errors else "failed",
        "sample_count": len(samples),
        "episode_count": len(frames_by_episode),
        "route_leg_count": len(route_ids),
        "navigation_intent_count": navigation_intent_count,
        "rgb_camera_ids": dataset.get("rgb_camera_ids", ["front"]),
        "checksum_entry_count": len(checksums),
        "errors": errors,
        "warnings": warnings,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify BehaviorAgent teacher dataset images, metadata, routes, and checksums."
    )
    parser.add_argument("dataset_dir")
    parser.add_argument("--report")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = verify_teacher_dataset(args.dataset_dir)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["status"] == "passed" else 1


__all__ = [
    "TEACHER_EPISODE_VERIFICATION_SCHEMA_VERSION",
    "main",
    "parse_args",
    "verify_teacher_dataset",
]


if __name__ == "__main__":
    raise SystemExit(main())
