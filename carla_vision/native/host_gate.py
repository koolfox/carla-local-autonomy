"""Bind a destructive native collection command to one verified ready preflight."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .verified_preflight import (
    NativePreflightIntegrityError,
    load_verified_native_preflight,
)


class NativeHostGateError(RuntimeError):
    """Raised when a preflight does not authorize the configured native run."""


def _load_object(path: Path, name: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativeHostGateError(f"could not read {name}: {error}") from error
    if not isinstance(payload, Mapping):
        raise NativeHostGateError(f"{name} must contain a JSON object")
    return payload


def _strict_episode_ids(raw: Any, *, context: str) -> tuple[str, ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise NativeHostGateError(f"{context} must be an array")
    result: list[str] = []
    for index, value in enumerate(raw):
        if not isinstance(value, str) or not value:
            raise NativeHostGateError(f"{context}[{index}] must be a non-empty string")
        result.append(value)
    if len(result) != len(set(result)):
        raise NativeHostGateError(f"{context} contains duplicate episode IDs")
    return tuple(result)


def validate_ready_preflight(
    *,
    kit_plan_path: str | Path,
    preflight_path: str | Path,
) -> dict[str, Any]:
    """Verify that a ready preflight exactly matches a portable kit plan."""

    kit_plan_file = Path(kit_plan_path).expanduser().resolve(strict=True)
    kit_plan = _load_object(kit_plan_file, "native host kit plan")
    if kit_plan.get("object_type") != "native_host_kit_payload":
        raise NativeHostGateError("kit plan object_type is invalid")
    if kit_plan.get("schema_version") != "1.0":
        raise NativeHostGateError("kit plan schema version is unsupported")

    try:
        preflight = load_verified_native_preflight(preflight_path)
    except NativePreflightIntegrityError as error:
        raise NativeHostGateError(f"preflight integrity verification failed: {error}") from error
    summary = preflight.summary
    selected = preflight.selected_episodes
    if summary.get("ready_for_native_execution") is not True:
        raise NativeHostGateError("preflight is verified but not ready for native execution")
    if summary.get("read_only") is not True or summary.get("simulator_mutated") is not False:
        raise NativeHostGateError("preflight does not retain the required read-only safety claim")

    expected_dataset = kit_plan.get("dataset")
    if not isinstance(expected_dataset, Mapping):
        raise NativeHostGateError("kit plan dataset section is invalid")
    observed_dataset = summary.get("dataset")
    if not isinstance(observed_dataset, Mapping):
        raise NativeHostGateError("preflight dataset section is invalid")
    dataset_id = expected_dataset.get("dataset_id")
    if observed_dataset.get("dataset_id") != dataset_id:
        raise NativeHostGateError("preflight dataset ID differs from the kit plan")
    if observed_dataset.get("output_available") is not True:
        raise NativeHostGateError("preflight dataset output is no longer available")

    expected_endpoint = kit_plan.get("endpoint")
    observed_endpoint = summary.get("endpoint")
    if not isinstance(expected_endpoint, Mapping) or not isinstance(observed_endpoint, Mapping):
        raise NativeHostGateError("kit/preflight endpoint section is invalid")
    if observed_endpoint.get("host") != expected_endpoint.get("host") or observed_endpoint.get(
        "port"
    ) != expected_endpoint.get("port"):
        raise NativeHostGateError("preflight endpoint differs from the kit plan")

    expected_plan = kit_plan.get("scenario_plan")
    observed_plan = selected.get("scenario_plan")
    if not isinstance(expected_plan, Mapping) or not isinstance(observed_plan, Mapping):
        raise NativeHostGateError("kit/preflight scenario-plan reference is invalid")
    for key in ("run_id", "sha256", "size_bytes"):
        if observed_plan.get(key) != expected_plan.get(key):
            raise NativeHostGateError(f"preflight scenario-plan {key} differs from the kit plan")

    expected_selection = kit_plan.get("selection")
    if not isinstance(expected_selection, Mapping):
        raise NativeHostGateError("kit plan selection section is invalid")
    expected_ids = _strict_episode_ids(
        expected_selection.get("episode_ids"),
        context="kit plan episode_ids",
    )
    raw_episodes = selected.get("episodes")
    if isinstance(raw_episodes, (str, bytes)) or not isinstance(raw_episodes, Sequence):
        raise NativeHostGateError("preflight selected episodes are invalid")
    observed_ids = _strict_episode_ids(
        [episode.get("episode_id") for episode in raw_episodes if isinstance(episode, Mapping)],
        context="preflight episode_ids",
    )
    if len(observed_ids) != len(raw_episodes) or observed_ids != expected_ids:
        raise NativeHostGateError("preflight episode selection differs from the kit plan")
    if selected.get("planned_capture_count") != expected_selection.get("planned_capture_count"):
        raise NativeHostGateError("preflight capture count differs from the kit plan")

    return {
        "status": "authorized",
        "kit_id": str(kit_plan.get("kit_id", "")),
        "preflight_run_id": preflight.run_id,
        "dataset_id": str(dataset_id),
        "host": str(expected_endpoint.get("host")),
        "port": int(expected_endpoint.get("port")),
        "episode_ids": list(expected_ids),
        "planned_capture_count": int(expected_selection["planned_capture_count"]),
        "read_only_preflight_verified": True,
        "manual_readiness_verified": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that a checksum-tracked native preflight is ready and exactly "
            "matches one portable native-host kit plan"
        )
    )
    parser.add_argument("--kit-plan", required=True)
    parser.add_argument("--preflight", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_ready_preflight(
        kit_plan_path=args.kit_plan,
        preflight_path=args.preflight,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


__all__ = [
    "NativeHostGateError",
    "main",
    "parse_args",
    "validate_ready_preflight",
]


if __name__ == "__main__":
    raise SystemExit(main())
