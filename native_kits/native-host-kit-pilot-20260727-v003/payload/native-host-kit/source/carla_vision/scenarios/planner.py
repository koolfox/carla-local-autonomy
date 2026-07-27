"""Resolve scenario recipes into immutable, split-assigned episode plans."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import RunArtifactTracker, fingerprint_file
from .contracts import ScenarioRecipe, ScenarioSuite, load_scenario_suite
from .seeds import SeedBundle, derive_seed_bundle
from .splits import GroupIdentity, SplitAssignment, SplitPlan, load_split_plan

EPISODE_PLAN_SCHEMA_VERSION = "1.0"


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


@dataclass(frozen=True)
class EpisodePlan:
    suite_id: str
    episode_id: str
    scenario_id: str
    repetition_index: int
    recipe: ScenarioRecipe
    seeds: SeedBundle
    group: GroupIdentity
    split: SplitAssignment
    schema_version: str = EPISODE_PLAN_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "scenario_id": self.scenario_id,
            "episode_id": self.episode_id,
            "repetition_index": self.repetition_index,
            "partition": self.split.partition,
            "split_assignment": self.split.as_dict(),
            "group": self.group.as_dict(),
            "seeds": self.seeds.as_dict(),
            "recipe": self.recipe.as_dict(),
        }


def expand_scenario_suite(
    suite: ScenarioSuite,
    split_plan: SplitPlan,
) -> tuple[EpisodePlan, ...]:
    episodes: list[EpisodePlan] = []
    seen_episode_ids: set[str] = set()
    for recipe in suite.recipes:
        for repetition_index in range(recipe.repetitions):
            episode_id = f"ep-{recipe.recipe_id}-r{repetition_index:03d}"
            if episode_id in seen_episode_ids:
                raise RuntimeError(f"duplicate generated episode ID {episode_id!r}")
            namespace = f"episode/{recipe.recipe_id}/r{repetition_index:03d}"
            seeds = derive_seed_bundle(
                suite.master_seed,
                namespace_prefix=namespace,
            )
            scenario_id = f"scn-{recipe.recipe_id}-s{seeds.values['world']}"
            group = GroupIdentity(
                simulator_build=suite.carla_version,
                map_family=recipe.map_name,
                route_or_road_region_id=recipe.route_region_id,
                scenario_recipe_id=recipe.recipe_id,
                static_layout_seed=seeds.values["props"],
                episode_id=episode_id,
            )
            split = split_plan.assign(group, recipe.weather.weather_id)
            episodes.append(
                EpisodePlan(
                    suite_id=suite.suite_id,
                    episode_id=episode_id,
                    scenario_id=scenario_id,
                    repetition_index=repetition_index,
                    recipe=recipe,
                    seeds=seeds,
                    group=group,
                    split=split,
                )
            )
            seen_episode_ids.add(episode_id)
    return tuple(episodes)


def _summary(
    suite: ScenarioSuite,
    split_plan: SplitPlan,
    episodes: Sequence[EpisodePlan],
) -> dict[str, Any]:
    partition_counts = Counter(episode.split.partition for episode in episodes)
    map_counts = Counter(episode.group.as_dict()["map_family"] for episode in episodes)
    weather_counts = Counter(episode.recipe.weather.weather_id for episode in episodes)
    planned_samples = sum(
        1
        + (episode.recipe.capture.duration_ticks - 1) // episode.recipe.capture.capture_every_ticks
        for episode in episodes
    )
    return {
        "schema_version": EPISODE_PLAN_SCHEMA_VERSION,
        "object_type": "scenario_plan",
        "suite_id": suite.suite_id,
        "split_plan_id": split_plan.plan_id,
        "carla_version": suite.carla_version,
        "master_seed": suite.master_seed,
        "recipe_count": len(suite.recipes),
        "episode_count": len(episodes),
        "planned_capture_count": planned_samples,
        "partition_counts": dict(sorted(partition_counts.items())),
        "map_family_counts": dict(sorted(map_counts.items())),
        "weather_counts": dict(sorted(weather_counts.items())),
        "runtime_sensor_contract": "front_monocular_rgb_only",
        "teacher_data_policy": (
            "Privileged sensors and simulator state may create labels and evaluation "
            "metadata but are forbidden as deployable model inputs."
        ),
        "determinism": {
            "single_tick_owner_required": True,
            "synchronous_world_required": True,
            "synchronous_traffic_manager_required": True,
            "reload_world_per_repetition_required": True,
            "fixed_delta_required": True,
            "exact_sensor_frame_match_required": True,
        },
    }


def plan_scenarios(
    *,
    suite_path: str | Path,
    split_plan_path: str | Path,
    runs_root: str | Path = "runs",
    run_id: str | None = None,
    cli_args: Sequence[str] | Mapping[str, Any] = (),
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    resolved_suite_path = Path(suite_path).expanduser().resolve(strict=True)
    resolved_split_path = Path(split_plan_path).expanduser().resolve(strict=True)
    suite = load_scenario_suite(resolved_suite_path)
    split_plan = load_split_plan(resolved_split_path)
    episodes = expand_scenario_suite(suite, split_plan)
    suite_ref = fingerprint_file(resolved_suite_path)
    split_ref = fingerprint_file(resolved_split_path)

    tracker = RunArtifactTracker(
        runs_root,
        run_id=run_id,
        cli_args=cli_args,
        config={
            "object_type": "scenario_plan",
            "scenario_suite": suite_ref,
            "split_plan": split_ref,
            "runtime_sensor_contract": "front_monocular_rgb_only",
        },
        repository_root=repository_root,
        carla_version=suite.carla_version,
    )
    with tracker:
        resolved_suite_output = tracker.artifact_path("resolved_suite.json")
        split_plan_output = tracker.artifact_path("resolved_split_plan.json")
        episodes_output = tracker.artifact_path("episodes.jsonl")
        summary_output = tracker.artifact_path("summary.json")
        _write_json(resolved_suite_output, suite.as_dict())
        _write_json(split_plan_output, split_plan.as_dict())
        _atomic_write_text(
            episodes_output,
            "".join(
                json.dumps(
                    episode.as_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
                for episode in episodes
            ),
        )
        summary = _summary(suite, split_plan, episodes)
        _write_json(summary_output, summary)
        for path, role in (
            (resolved_suite_output, "resolved_scenario_suite"),
            (split_plan_output, "resolved_split_plan"),
            (episodes_output, "planned_episodes_jsonl"),
            (summary_output, "scenario_plan_summary"),
        ):
            tracker.register_artifact(
                path,
                role=role,
                metadata={
                    "suite_id": suite.suite_id,
                    "split_plan_id": split_plan.plan_id,
                    "episode_count": len(episodes),
                },
            )
    return {
        "run_id": tracker.run_id,
        "run_dir": str(tracker.run_dir),
        "manifest": str(tracker.manifest_path),
        "summary": summary,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve a deterministic CARLA scenario suite into immutable, "
            "episode-level split assignments"
        )
    )
    parser.add_argument("--suite", required=True)
    parser.add_argument("--split-plan", required=True)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run-id")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = plan_scenarios(
        suite_path=args.suite,
        split_plan_path=args.split_plan,
        runs_root=args.runs_root,
        run_id=args.run_id,
        cli_args=vars(args),
        repository_root=Path.cwd(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


__all__ = [
    "EPISODE_PLAN_SCHEMA_VERSION",
    "EpisodePlan",
    "expand_scenario_suite",
    "main",
    "parse_args",
    "plan_scenarios",
]
