"""Compare repeated BehaviorAgent teacher datasets for deterministic evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .teacher_verify import verify_teacher_dataset

TEACHER_EPISODE_COMPARISON_SCHEMA_VERSION = "1.0"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _metadata_for_sample(root: Path, sample: Mapping[str, Any]) -> dict[str, Any]:
    artifact = sample.get("metadata")
    if not isinstance(artifact, Mapping):
        raise ValueError("sample metadata artifact is missing")
    relative = artifact.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("sample metadata path is missing")
    path = (root / relative).resolve(strict=True)
    path.relative_to(root.resolve())
    return _read_json(path)


def _sample_records(root: Path) -> dict[str, list[dict[str, Any]]]:
    dataset = _read_json(root / "dataset.json")
    samples = dataset.get("samples")
    if not isinstance(samples, list):
        raise ValueError("dataset.samples must be a list")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ordinal, raw_sample in enumerate(samples):
        if not isinstance(raw_sample, Mapping):
            raise ValueError(f"sample[{ordinal}] must be an object")
        episode_id = raw_sample.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError(f"sample[{ordinal}] episode_id is missing")
        metadata = _metadata_for_sample(root, raw_sample)
        context = metadata.get("context")
        if not isinstance(context, Mapping):
            raise ValueError(f"sample[{ordinal}] context is missing")
        route = context.get("route")
        control = context.get("privileged_teacher_control")
        ego_state = context.get("privileged_evaluation")
        if not isinstance(route, Mapping) or not isinstance(control, Mapping):
            raise ValueError(f"sample[{ordinal}] route/control metadata is missing")
        speed = None
        if isinstance(ego_state, Mapping):
            velocity = ego_state.get("velocity_mps")
            if isinstance(velocity, Mapping):
                speed = velocity.get("speed")
        rgb = raw_sample.get("rgb")
        rgb_sha = rgb.get("sha256") if isinstance(rgb, Mapping) else None
        grouped[episode_id].append(
            {
                "ordinal": ordinal,
                "carla_frame": int(raw_sample["carla_frame"]),
                "timestamp": float(raw_sample["source_timestamp"]),
                "route_id": route.get("route_id"),
                "leg_index": route.get("leg_index"),
                "destination_spawn_index": route.get("destination_spawn_index"),
                "throttle": float(control["throttle"]),
                "steer": float(control["steer"]),
                "brake": float(control["brake"]),
                "speed_mps": None if speed is None else float(speed),
                "rgb_sha256": rgb_sha,
            }
        )
    return dict(grouped)


def _close(left: float, right: float, tolerance: float) -> bool:
    return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tolerance


def compare_teacher_datasets(
    left_dataset_dir: str | Path,
    right_dataset_dir: str | Path,
    *,
    control_tolerance: float = 1e-6,
    timestamp_tolerance: float = 1e-6,
    state_tolerance: float = 1e-4,
    require_identical_rgb: bool = False,
) -> dict[str, Any]:
    """Compare two collections aligned by episode and sample ordinal.

    Absolute CARLA frame IDs and timestamps may restart at different values after
    map loading, so comparison uses offsets relative to each episode's first
    sample. Route choices, frame cadence, controls, and ego speed are compared.
    """

    for name, value in (
        ("control_tolerance", control_tolerance),
        ("timestamp_tolerance", timestamp_tolerance),
        ("state_tolerance", state_tolerance),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    left_root = Path(left_dataset_dir).expanduser().resolve()
    right_root = Path(right_dataset_dir).expanduser().resolve()
    left_verification = verify_teacher_dataset(left_root)
    right_verification = verify_teacher_dataset(right_root)
    mismatches: list[dict[str, Any]] = []
    if left_verification["status"] != "passed":
        mismatches.append(
            {"kind": "left_verification_failed", "errors": left_verification["errors"]}
        )
    if right_verification["status"] != "passed":
        mismatches.append(
            {"kind": "right_verification_failed", "errors": right_verification["errors"]}
        )
    if mismatches:
        return {
            "schema_version": TEACHER_EPISODE_COMPARISON_SCHEMA_VERSION,
            "status": "failed",
            "left_dataset_dir": str(left_root),
            "right_dataset_dir": str(right_root),
            "compared_episode_count": 0,
            "compared_sample_count": 0,
            "mismatches": mismatches,
        }

    left = _sample_records(left_root)
    right = _sample_records(right_root)
    left_episodes = set(left)
    right_episodes = set(right)
    if left_episodes != right_episodes:
        mismatches.append(
            {
                "kind": "episode_set",
                "left_only": sorted(left_episodes - right_episodes),
                "right_only": sorted(right_episodes - left_episodes),
            }
        )

    compared_samples = 0
    for episode_id in sorted(left_episodes & right_episodes):
        left_samples = left[episode_id]
        right_samples = right[episode_id]
        if len(left_samples) != len(right_samples):
            mismatches.append(
                {
                    "kind": "sample_count",
                    "episode_id": episode_id,
                    "left": len(left_samples),
                    "right": len(right_samples),
                }
            )
        pair_count = min(len(left_samples), len(right_samples))
        compared_samples += pair_count
        if pair_count == 0:
            continue
        left_frame_origin = left_samples[0]["carla_frame"]
        right_frame_origin = right_samples[0]["carla_frame"]
        left_time_origin = left_samples[0]["timestamp"]
        right_time_origin = right_samples[0]["timestamp"]
        for index in range(pair_count):
            left_sample = left_samples[index]
            right_sample = right_samples[index]
            fields: list[tuple[str, Any, Any, float | None]] = [
                (
                    "relative_frame",
                    left_sample["carla_frame"] - left_frame_origin,
                    right_sample["carla_frame"] - right_frame_origin,
                    None,
                ),
                (
                    "relative_timestamp",
                    left_sample["timestamp"] - left_time_origin,
                    right_sample["timestamp"] - right_time_origin,
                    timestamp_tolerance,
                ),
                ("route_id", left_sample["route_id"], right_sample["route_id"], None),
                ("leg_index", left_sample["leg_index"], right_sample["leg_index"], None),
                (
                    "destination_spawn_index",
                    left_sample["destination_spawn_index"],
                    right_sample["destination_spawn_index"],
                    None,
                ),
                (
                    "throttle",
                    left_sample["throttle"],
                    right_sample["throttle"],
                    control_tolerance,
                ),
                ("steer", left_sample["steer"], right_sample["steer"], control_tolerance),
                ("brake", left_sample["brake"], right_sample["brake"], control_tolerance),
            ]
            if left_sample["speed_mps"] is not None and right_sample["speed_mps"] is not None:
                fields.append(
                    (
                        "speed_mps",
                        left_sample["speed_mps"],
                        right_sample["speed_mps"],
                        state_tolerance,
                    )
                )
            if require_identical_rgb:
                fields.append(
                    (
                        "rgb_sha256",
                        left_sample["rgb_sha256"],
                        right_sample["rgb_sha256"],
                        None,
                    )
                )
            for field, left_value, right_value, tolerance in fields:
                matched = (
                    left_value == right_value
                    if tolerance is None
                    else _close(float(left_value), float(right_value), tolerance)
                )
                if not matched:
                    mismatches.append(
                        {
                            "kind": "sample_field",
                            "episode_id": episode_id,
                            "sample_index": index,
                            "field": field,
                            "left": left_value,
                            "right": right_value,
                            "tolerance": tolerance,
                        }
                    )

    return {
        "schema_version": TEACHER_EPISODE_COMPARISON_SCHEMA_VERSION,
        "status": "matched" if not mismatches else "failed",
        "left_dataset_dir": str(left_root),
        "right_dataset_dir": str(right_root),
        "compared_episode_count": len(left_episodes & right_episodes),
        "compared_sample_count": compared_samples,
        "require_identical_rgb": bool(require_identical_rgb),
        "tolerances": {
            "control": control_tolerance,
            "timestamp": timestamp_tolerance,
            "state": state_tolerance,
        },
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two repeated BehaviorAgent teacher datasets for determinism."
    )
    parser.add_argument("left_dataset_dir")
    parser.add_argument("right_dataset_dir")
    parser.add_argument("--control-tolerance", type=float, default=1e-6)
    parser.add_argument("--timestamp-tolerance", type=float, default=1e-6)
    parser.add_argument("--state-tolerance", type=float, default=1e-4)
    parser.add_argument("--require-identical-rgb", action="store_true")
    parser.add_argument("--report")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = compare_teacher_datasets(
        args.left_dataset_dir,
        args.right_dataset_dir,
        control_tolerance=args.control_tolerance,
        timestamp_tolerance=args.timestamp_tolerance,
        state_tolerance=args.state_tolerance,
        require_identical_rgb=args.require_identical_rgb,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        path = Path(args.report).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["status"] == "matched" else 1


__all__ = [
    "TEACHER_EPISODE_COMPARISON_SCHEMA_VERSION",
    "compare_teacher_datasets",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    raise SystemExit(main())
