from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from carla_vision.scenarios.contracts import ScenarioSuite
from carla_vision.scenarios.planner import expand_scenario_suite, plan_scenarios
from carla_vision.scenarios.seeds import (
    REQUIRED_SEED_NAMESPACES,
    derive_seed,
    derive_seed_bundle,
)
from carla_vision.scenarios.splits import (
    GroupIdentity,
    SplitPlan,
    canonical_map_family,
)


def weather(*, weather_id: str = "clear-day") -> dict[str, object]:
    return {
        "weather_id": weather_id,
        "light": "day",
        "cloudiness": 10.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 5.0,
        "sun_azimuth_angle": 45.0,
        "sun_altitude_angle": 55.0,
        "fog_density": 0.0,
        "fog_distance": 0.0,
        "wetness": 0.0,
        "fog_falloff": 0.2,
        "scattering_intensity": 1.0,
        "mie_scattering_scale": 0.03,
        "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 0.0,
    }


def recipe(
    *,
    recipe_id: str = "town10-clear",
    map_name: str = "Town10HD_Opt",
    weather_id: str = "clear-day",
    repetitions: int = 2,
) -> dict[str, object]:
    return {
        "recipe_id": recipe_id,
        "map_name": map_name,
        "route_region_id": "spawn-0-loop",
        "ego_blueprint": "vehicle.tesla.model3",
        "ego_spawn_index": 0,
        "fixed_delta_seconds": 0.05,
        "repetitions": repetitions,
        "weather": weather(weather_id=weather_id),
        "traffic": {
            "vehicle_count": 12,
            "walker_count": 8,
            "pedestrian_crossing_factor": 0.1,
            "vehicle_filter": "vehicle.*",
            "walker_filter": "walker.pedestrian.*",
            "vehicle_generation": "All",
            "walker_generation": "2",
            "global_speed_difference_percent": 20.0,
            "global_distance_to_leading_vehicle": 2.5,
            "automatic_vehicle_lights": True,
        },
        "camera": {
            "width": 1280,
            "height": 720,
            "fov_degrees": 90.0,
            "sensor_tick_seconds": 0.1,
            "gamma": 2.2,
            "enable_postprocess_effects": True,
            "mount": {
                "x": 1.5,
                "y": 0.0,
                "z": 1.7,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
            },
        },
        "capture": {
            "warmup_ticks": 20,
            "duration_ticks": 100,
            "capture_every_ticks": 4,
            "minimum_visible_pixels": 16,
            "minimum_box_width": 2,
            "minimum_box_height": 2,
        },
        "props": [
            {
                "blueprint_id": "static.prop.trafficcone01",
                "relative_to": "ego_start",
                "transform": {
                    "x": 20.0,
                    "y": 2.0,
                    "z": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "roll": 0.0,
                },
            }
        ],
    }


def suite(*recipes: dict[str, object]) -> dict[str, object]:
    return {
        "suite_id": "suite-test-v1",
        "schema_version": "1.0",
        "carla_version": "0.9.16",
        "master_seed": 20260726,
        "traffic_manager_port": 8000,
        "recipes": list(recipes or (recipe(),)),
    }


def split_plan() -> dict[str, object]:
    return {
        "plan_id": "split-test-v1",
        "schema_version": "1.0",
        "validation_map_families": ["Town02"],
        "test_map_families": ["Town04", "Town07"],
        "validation_weather_ids": ["soft-rain-sunset"],
        "test_weather_ids": ["fog-night"],
    }


class SeedTests(unittest.TestCase):
    def test_protocol_seed_values_are_stable_and_namespaced(self) -> None:
        self.assertEqual(derive_seed(20260726, "python"), 1365744867)
        self.assertEqual(derive_seed(20260726, "world"), 906446181)
        self.assertEqual(derive_seed(20260726, "traffic_manager"), 898087000)
        self.assertNotEqual(
            derive_seed(20260726, "world"),
            derive_seed(20260726, "episode/a/world"),
        )

    def test_seed_bundle_contains_every_required_unique_seed(self) -> None:
        bundle = derive_seed_bundle(7, namespace_prefix="episode/example/r000")
        self.assertEqual(set(bundle.values), set(REQUIRED_SEED_NAMESPACES))
        self.assertEqual(len(set(bundle.values.values())), len(REQUIRED_SEED_NAMESPACES))
        self.assertEqual(bundle.namespace_prefix, "episode/example/r000")

    def test_seed_inputs_reject_ambiguous_or_invalid_values(self) -> None:
        for invalid in (-1, 2**63, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    derive_seed(invalid, "world")
        for invalid in ("", "  ", "bad\0namespace"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    derive_seed(1, invalid)


class SplitTests(unittest.TestCase):
    def test_map_family_collapses_paths_and_opt_variants(self) -> None:
        self.assertEqual(canonical_map_family("Town10HD_Opt"), "Town10HD")
        self.assertEqual(
            canonical_map_family("/Game/Carla/Maps/Town10HD_Opt.umap"),
            "Town10HD",
        )
        self.assertEqual(canonical_map_family("Town10HD"), "Town10HD")

    def test_group_key_and_bucket_are_versioned_and_stable(self) -> None:
        group = GroupIdentity(
            simulator_build="0.9.16",
            map_family="Carla/Maps/Town10HD_Opt",
            route_or_road_region_id="route-a",
            scenario_recipe_id="recipe-a",
            static_layout_seed=123,
            episode_id="ep-a",
        )
        self.assertEqual(group.bucket(), 85)
        self.assertEqual(
            group.as_dict()["canonical_key_sha256"],
            "4c16b6970c309a2d4aa98b5aed6d6cb42948ca17e8cf1521208ceb2f285b83d9",
        )

    def test_split_assignment_prioritizes_held_out_map_then_weather(self) -> None:
        plan = SplitPlan.from_mapping(split_plan())
        base = dict(
            simulator_build="0.9.16",
            route_or_road_region_id="route",
            scenario_recipe_id="recipe",
            static_layout_seed=1,
            episode_id="episode",
        )
        test_map = GroupIdentity(map_family="Town04_Opt", **base)
        val_map = GroupIdentity(map_family="Town02", **base)
        seen_map = GroupIdentity(map_family="Town10HD", **base)
        self.assertEqual(plan.assign(test_map, "soft-rain-sunset").partition, "test_map_ood")
        self.assertEqual(plan.assign(val_map, "fog-night").partition, "val_map_ood")
        self.assertEqual(plan.assign(seen_map, "fog-night").partition, "test_weather_ood")
        self.assertEqual(
            plan.assign(seen_map, "soft-rain-sunset").partition,
            "val_weather_ood",
        )

    def test_split_plan_rejects_overlapping_holdouts(self) -> None:
        invalid = split_plan()
        invalid["test_map_families"] = ["Town02_Opt"]
        with self.assertRaisesRegex(ValueError, "map families overlap"):
            SplitPlan.from_mapping(invalid)


class ScenarioContractTests(unittest.TestCase):
    def test_suite_parses_and_round_trips_without_losing_contract_fields(self) -> None:
        parsed = ScenarioSuite.from_mapping(suite())
        self.assertEqual(parsed.suite_id, "suite-test-v1")
        self.assertEqual(parsed.recipes[0].camera.width, 1280)
        self.assertEqual(parsed.recipes[0].props[0].blueprint_id, "static.prop.trafficcone01")
        self.assertEqual(ScenarioSuite.from_mapping(parsed.as_dict()), parsed)

    def test_sensor_tick_must_be_integer_multiple_of_fixed_delta(self) -> None:
        invalid_recipe = recipe()
        invalid_recipe["camera"]["sensor_tick_seconds"] = 0.075
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            ScenarioSuite.from_mapping(suite(invalid_recipe))

    def test_capture_cadence_must_preserve_sensor_phase(self) -> None:
        invalid_recipe = recipe()
        invalid_recipe["capture"]["capture_every_ticks"] = 3
        with self.assertRaisesRegex(ValueError, "camera sensor period"):
            ScenarioSuite.from_mapping(suite(invalid_recipe))

    def test_recipe_rejects_unknown_fields_and_non_prop_blueprints(self) -> None:
        with_unknown = recipe()
        with_unknown["surprise"] = True
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            ScenarioSuite.from_mapping(suite(with_unknown))

        invalid_prop = recipe()
        invalid_prop["props"][0]["blueprint_id"] = "vehicle.tesla.model3"
        with self.assertRaisesRegex(ValueError, "static.prop"):
            ScenarioSuite.from_mapping(suite(invalid_prop))


class ScenarioPlannerTests(unittest.TestCase):
    def test_expansion_is_deterministic_unique_and_split_at_episode_level(self) -> None:
        parsed_suite = ScenarioSuite.from_mapping(
            suite(
                recipe(repetitions=2),
                recipe(
                    recipe_id="town04-clear",
                    map_name="Town04_Opt",
                    repetitions=1,
                ),
            )
        )
        parsed_split = SplitPlan.from_mapping(split_plan())
        first = expand_scenario_suite(parsed_suite, parsed_split)
        second = expand_scenario_suite(parsed_suite, parsed_split)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len({episode.episode_id for episode in first}), 3)
        self.assertEqual(first[-1].split.partition, "test_map_ood")
        self.assertNotEqual(first[0].seeds.values["world"], first[1].seeds.values["world"])

    def test_plan_run_writes_hash_tracked_inputs_episodes_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite_path = root / "suite.json"
            split_path = root / "split.json"
            suite_path.write_text(json.dumps(suite()), encoding="utf-8")
            split_path.write_text(json.dumps(split_plan()), encoding="utf-8")
            result = plan_scenarios(
                suite_path=suite_path,
                split_plan_path=split_path,
                runs_root=root / "runs",
                run_id="scenario-plan-test",
                repository_root=root,
            )
            run_dir = Path(result["run_dir"])
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            episodes = (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()

            self.assertEqual(manifest["status"], "success")
            self.assertEqual(len(manifest["artifacts"]), 4)
            self.assertEqual(summary["episode_count"], 2)
            self.assertEqual(summary["planned_capture_count"], 50)
            self.assertEqual(len(episodes), 2)
            for artifact in manifest["artifacts"]:
                path = run_dir / artifact["path"]
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    artifact["sha256"],
                )


if __name__ == "__main__":
    unittest.main()
