"""Integrity validation for scenario-plan runs consumed by native workers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import fingerprint_file
from .contracts import ScenarioSuite
from .planner import EpisodePlan, expand_scenario_suite
from .splits import SplitPlan

_REQUIRED_ROLES = {
    "resolved_scenario_suite",
    "resolved_split_plan",
    "planned_episodes_jsonl",
    "scenario_plan_summary",
}


class ScenarioPlanIntegrityError(RuntimeError):
    """Raised when a scenario-plan run is incomplete or has changed."""


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ScenarioPlanIntegrityError(f"could not read {name}: {error}") from error
    if not isinstance(value, Mapping):
        raise ScenarioPlanIntegrityError(f"{name} must contain a JSON object")
    return value


def _safe_artifact_path(plan_dir: Path, relative_path: str) -> Path:
    candidate = (plan_dir / relative_path).resolve(strict=True)
    try:
        candidate.relative_to(plan_dir)
    except ValueError as error:
        raise ScenarioPlanIntegrityError(
            f"scenario artifact escapes its run directory: {relative_path}"
        ) from error
    if not candidate.is_file():
        raise ScenarioPlanIntegrityError(f"scenario artifact is not a file: {relative_path}")
    return candidate


def _verify_artifacts(
    plan_dir: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Path]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ScenarioPlanIntegrityError("scenario manifest artifacts must be an array")
    roles: dict[str, Path] = {}
    for index, raw in enumerate(artifacts):
        if not isinstance(raw, Mapping):
            raise ScenarioPlanIntegrityError(f"scenario artifact {index} must be an object")
        role = str(raw.get("role", ""))
        relative_path = str(raw.get("path", ""))
        expected_digest = str(raw.get("sha256", ""))
        expected_size = raw.get("size_bytes")
        if not role or not relative_path:
            raise ScenarioPlanIntegrityError(f"scenario artifact {index} is incomplete")
        if role in roles:
            raise ScenarioPlanIntegrityError(f"duplicate scenario artifact role {role!r}")
        path = _safe_artifact_path(plan_dir, relative_path)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_digest:
            raise ScenarioPlanIntegrityError(
                f"scenario artifact checksum mismatch: {relative_path}"
            )
        if path.stat().st_size != expected_size:
            raise ScenarioPlanIntegrityError(f"scenario artifact size mismatch: {relative_path}")
        roles[role] = path
    missing = sorted(_REQUIRED_ROLES - roles.keys())
    if missing:
        raise ScenarioPlanIntegrityError(
            "scenario plan is missing required artifact roles: " + ", ".join(missing)
        )
    return roles


@dataclass(frozen=True)
class VerifiedScenarioPlan:
    plan_dir: Path
    run_id: str
    suite: ScenarioSuite
    split_plan: SplitPlan
    episodes: tuple[EpisodePlan, ...]
    summary: Mapping[str, Any]
    reference: Mapping[str, Any]


def load_verified_scenario_plan(path: str | Path) -> VerifiedScenarioPlan:
    plan_dir = Path(path).expanduser().resolve(strict=True)
    if not plan_dir.is_dir():
        raise ScenarioPlanIntegrityError(f"scenario plan is not a directory: {plan_dir}")
    manifest_path = plan_dir / "manifest.json"
    manifest = _load_json(manifest_path, "scenario manifest")
    if manifest.get("status") != "success":
        raise ScenarioPlanIntegrityError(
            f"scenario plan status must be success, got {manifest.get('status')!r}"
        )
    roles = _verify_artifacts(plan_dir, manifest)
    suite = ScenarioSuite.from_mapping(
        _load_json(roles["resolved_scenario_suite"], "resolved scenario suite")
    )
    split_plan = SplitPlan.from_mapping(
        _load_json(roles["resolved_split_plan"], "resolved split plan")
    )
    expected = expand_scenario_suite(suite, split_plan)
    actual_payloads: list[Mapping[str, Any]] = []
    try:
        with roles["planned_episodes_jsonl"].open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ScenarioPlanIntegrityError(
                        f"blank line in episodes JSONL at line {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ScenarioPlanIntegrityError(
                        f"episode line {line_number} must be a JSON object"
                    )
                actual_payloads.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ScenarioPlanIntegrityError(f"could not parse episodes JSONL: {error}") from error
    expected_payloads = [episode.as_dict() for episode in expected]
    if actual_payloads != expected_payloads:
        raise ScenarioPlanIntegrityError(
            "episodes JSONL does not reproduce from the resolved suite and split plan"
        )
    summary = _load_json(roles["scenario_plan_summary"], "scenario plan summary")
    if summary.get("episode_count") != len(expected):
        raise ScenarioPlanIntegrityError("scenario summary episode_count is inconsistent")
    run_id = str(manifest.get("run_id", ""))
    if not run_id:
        raise ScenarioPlanIntegrityError("scenario manifest run_id is missing")
    manifest_reference = fingerprint_file(manifest_path)
    return VerifiedScenarioPlan(
        plan_dir=plan_dir,
        run_id=run_id,
        suite=suite,
        split_plan=split_plan,
        episodes=expected,
        summary=summary,
        reference={
            "kind": "verified_scenario_plan",
            "run_id": run_id,
            **manifest_reference,
        },
    )


__all__ = [
    "ScenarioPlanIntegrityError",
    "VerifiedScenarioPlan",
    "load_verified_scenario_plan",
]
