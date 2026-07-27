"""Semantic verification for retained native-collection preflight artifacts."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..scenarios.verified_plan import load_verified_scenario_plan
from .preflight import NATIVE_PREFLIGHT_SCHEMA_VERSION
from .worker import select_episodes


class NativePreflightIntegrityError(RuntimeError):
    """Raised when a native preflight is internally inconsistent."""


@dataclass(frozen=True)
class VerifiedNativePreflight:
    root: Path
    run_id: str
    summary: Mapping[str, Any]
    checks: tuple[Mapping[str, Any], ...]
    selected_episodes: Mapping[str, Any]


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativePreflightIntegrityError(f"could not read {name}: {error}") from error
    if not isinstance(payload, Mapping):
        raise NativePreflightIntegrityError(f"{name} must contain an object")
    return payload


def _role_path(
    root: Path,
    manifest: Mapping[str, Any],
    role: str,
) -> Path:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Sequence):
        raise NativePreflightIntegrityError("manifest artifacts must be an array")
    matches = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping) and artifact.get("role") == role
    ]
    if len(matches) != 1:
        raise NativePreflightIntegrityError(
            f"native preflight must contain exactly one {role!r} artifact"
        )
    candidate = (root / str(matches[0].get("path", ""))).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise NativePreflightIntegrityError(f"{role} artifact escapes object root") from error
    return candidate


def _capture_count(episode: Any) -> int:
    capture = episode.recipe.capture
    return 1 + (capture.duration_ticks - 1) // capture.capture_every_ticks


def _strict_boolean(raw: Any, name: str) -> bool:
    if not isinstance(raw, bool):
        raise NativePreflightIntegrityError(f"{name} must be boolean")
    return raw


def load_verified_native_preflight(root: str | Path) -> VerifiedNativePreflight:
    resolved = Path(root).expanduser().resolve(strict=True)
    manifest = _load_json(resolved / "manifest.json", "preflight manifest")
    summary = _load_json(
        _role_path(resolved, manifest, "native_preflight_summary"),
        "preflight summary",
    )
    checks_payload = _load_json(
        _role_path(resolved, manifest, "native_preflight_checks"),
        "preflight checks",
    )
    selected = _load_json(
        _role_path(resolved, manifest, "native_preflight_selected_episodes"),
        "preflight selected episodes",
    )
    for name, payload in (
        ("summary", summary),
        ("checks", checks_payload),
        ("selection", selected),
    ):
        if payload.get("schema_version") != NATIVE_PREFLIGHT_SCHEMA_VERSION:
            raise NativePreflightIntegrityError(f"{name} schema version is unsupported")
    if summary.get("object_type") != "native_collection_preflight":
        raise NativePreflightIntegrityError("summary object_type is invalid")
    if checks_payload.get("object_type") != "native_preflight_checks":
        raise NativePreflightIntegrityError("checks object_type is invalid")
    if selected.get("object_type") != "native_preflight_selection":
        raise NativePreflightIntegrityError("selection object_type is invalid")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or summary.get("run_id") != run_id:
        raise NativePreflightIntegrityError("summary run_id does not match the manifest")
    if summary.get("status") not in {"ready", "not_ready"}:
        raise NativePreflightIntegrityError("summary readiness status is invalid")
    if _strict_boolean(summary.get("read_only"), "summary.read_only") is not True:
        raise NativePreflightIntegrityError("native preflight must be read-only")
    if _strict_boolean(summary.get("simulator_mutated"), "summary.simulator_mutated"):
        raise NativePreflightIntegrityError("native preflight cannot report simulator mutation")

    raw_checks = checks_payload.get("checks")
    if isinstance(raw_checks, (str, bytes)) or not isinstance(raw_checks, Sequence):
        raise NativePreflightIntegrityError("checks must be an array")
    checks: list[Mapping[str, Any]] = []
    check_ids: set[str] = set()
    for index, check in enumerate(raw_checks):
        if not isinstance(check, Mapping):
            raise NativePreflightIntegrityError(f"check {index} must be an object")
        check_id = check.get("check_id")
        if not isinstance(check_id, str) or not check_id or check_id in check_ids:
            raise NativePreflightIntegrityError(f"check {index} has an invalid or duplicate ID")
        if check.get("category") not in {"automated", "manual", "advisory"}:
            raise NativePreflightIntegrityError(f"check {check_id} category is invalid")
        if check.get("status") not in {"pass", "fail", "skip", "pending", "warn"}:
            raise NativePreflightIntegrityError(f"check {check_id} status is invalid")
        _strict_boolean(check.get("required"), f"check {check_id}.required")
        passed = _strict_boolean(check.get("passed"), f"check {check_id}.passed")
        if (check.get("status") == "pass") != passed:
            raise NativePreflightIntegrityError(f"check {check_id} passed flag and status disagree")
        check_ids.add(check_id)
        checks.append(check)

    automated_required = [
        check for check in checks if check["category"] == "automated" and check["required"]
    ]
    manual_required = [
        check for check in checks if check["category"] == "manual" and check["required"]
    ]
    automated_ready = all(bool(check["passed"]) for check in automated_required)
    manual_ready = all(bool(check["passed"]) for check in manual_required)
    ready = automated_ready and manual_ready
    if (
        _strict_boolean(summary.get("automated_ready"), "summary.automated_ready")
        != automated_ready
    ):
        raise NativePreflightIntegrityError("automated readiness does not reproduce")
    if _strict_boolean(summary.get("manual_ready"), "summary.manual_ready") != manual_ready:
        raise NativePreflightIntegrityError("manual readiness does not reproduce")
    if (
        _strict_boolean(
            summary.get("ready_for_native_execution"),
            "summary.ready_for_native_execution",
        )
        != ready
    ):
        raise NativePreflightIntegrityError("overall native readiness does not reproduce")
    if summary["status"] != ("ready" if ready else "not_ready"):
        raise NativePreflightIntegrityError("summary readiness label does not reproduce")

    counts = summary.get("checks")
    if not isinstance(counts, Mapping):
        raise NativePreflightIntegrityError("summary check counts must be an object")
    status_counts = Counter(str(check["status"]) for check in checks)
    expected_counts = {
        "count": len(checks),
        "passed": status_counts["pass"],
        "failed": status_counts["fail"],
        "pending": status_counts["pending"],
        "skipped": status_counts["skip"],
        "warnings": status_counts["warn"],
    }
    if dict(counts) != expected_counts:
        raise NativePreflightIntegrityError("summary check counts do not reproduce")

    plan_reference = selected.get("scenario_plan")
    if not isinstance(plan_reference, Mapping):
        raise NativePreflightIntegrityError("selection scenario plan reference is missing")
    manifest_path = plan_reference.get("path")
    if not isinstance(manifest_path, str):
        raise NativePreflightIntegrityError("scenario plan manifest path is missing")
    plan = load_verified_scenario_plan(Path(manifest_path).expanduser().resolve(strict=True).parent)
    if plan.run_id != plan_reference.get("run_id"):
        raise NativePreflightIntegrityError("scenario plan run_id does not match")
    request = selected.get("selection_request")
    if not isinstance(request, Mapping):
        raise NativePreflightIntegrityError("selection request is missing")
    raw_episode_ids = request.get("episode_ids")
    raw_partitions = request.get("partitions")
    if not isinstance(raw_episode_ids, Sequence) or isinstance(raw_episode_ids, (str, bytes)):
        raise NativePreflightIntegrityError("selection episode IDs must be an array")
    if not isinstance(raw_partitions, Sequence) or isinstance(raw_partitions, (str, bytes)):
        raise NativePreflightIntegrityError("selection partitions must be an array")
    max_episodes = request.get("max_episodes")
    if max_episodes is not None and (
        isinstance(max_episodes, bool) or not isinstance(max_episodes, int)
    ):
        raise NativePreflightIntegrityError("selection max_episodes is invalid")
    recomputed = select_episodes(
        plan,
        episode_ids=tuple(str(value) for value in raw_episode_ids),
        partitions=tuple(str(value) for value in raw_partitions),
        max_episodes=max_episodes,
    )
    raw_selected = selected.get("episodes")
    if isinstance(raw_selected, (str, bytes)) or not isinstance(raw_selected, Sequence):
        raise NativePreflightIntegrityError("selected episodes must be an array")
    selected_ids = [
        row.get("episode_id") if isinstance(row, Mapping) else None for row in raw_selected
    ]
    expected_ids = [episode.episode_id for episode in recomputed]
    if selected_ids != expected_ids:
        raise NativePreflightIntegrityError("selected episode IDs do not reproduce")
    capture_count = sum(_capture_count(episode) for episode in recomputed)
    if selected.get("selected_episode_count") != len(recomputed):
        raise NativePreflightIntegrityError("selected episode count does not reproduce")
    if selected.get("planned_capture_count") != capture_count:
        raise NativePreflightIntegrityError("planned capture count does not reproduce")
    selection_summary = summary.get("selection")
    if not isinstance(selection_summary, Mapping):
        raise NativePreflightIntegrityError("summary selection is missing")
    if selection_summary.get("selected_episode_count") != len(recomputed):
        raise NativePreflightIntegrityError("summary episode count does not reproduce")
    if selection_summary.get("planned_capture_count") != capture_count:
        raise NativePreflightIntegrityError("summary capture count does not reproduce")

    return VerifiedNativePreflight(
        root=resolved,
        run_id=run_id,
        summary=summary,
        checks=tuple(checks),
        selected_episodes=selected,
    )


__all__ = [
    "NativePreflightIntegrityError",
    "VerifiedNativePreflight",
    "load_verified_native_preflight",
]
