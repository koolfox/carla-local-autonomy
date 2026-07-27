from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from carla_vision.native.synchronization import (
    NativeSensorQueue,
    SensorFrameError,
    compose_relative_transform,
    image_to_bridge_frame,
)
from carla_vision.native.worker import (
    _privileged_teacher_control,
    collect_native,
    parse_args,
    select_episodes,
)
from carla_vision.scenarios.contracts import TransformRecipe
from carla_vision.scenarios.planner import plan_scenarios
from carla_vision.scenarios.verified_plan import (
    ScenarioPlanIntegrityError,
    load_verified_scenario_plan,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = REPOSITORY_ROOT / "configs" / "scenarios" / "thesis_pilot_v1.json"
SPLIT_PATH = REPOSITORY_ROOT / "configs" / "scenarios" / "split_plan_thesis_pilot_v1.json"


class FakeImage:
    def __init__(self, frame: int, *, width: int = 2, height: int = 1) -> None:
        self.frame = frame
        self.timestamp = frame / 20.0
        self.width = width
        self.height = height
        self.raw_data = bytes(range(width * height * 4))
        self.transform = SimpleNamespace(
            location=SimpleNamespace(x=1.0, y=2.0, z=3.0),
            rotation=SimpleNamespace(pitch=4.0, yaw=5.0, roll=6.0),
        )


def make_verified_plan(root: Path) -> Path:
    result = plan_scenarios(
        suite_path=SUITE_PATH,
        split_plan_path=SPLIT_PATH,
        runs_root=root,
        run_id="verified-plan",
        repository_root=REPOSITORY_ROOT,
    )
    return Path(result["run_dir"])


class NativeSynchronizationTests(unittest.TestCase):
    def test_teacher_control_is_serialized_as_an_exact_frame_privileged_target(self) -> None:
        ego = SimpleNamespace(
            get_control=lambda: SimpleNamespace(
                throttle=0.25,
                steer=-0.4,
                brake=0.0,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
                gear=1,
            )
        )

        target = _privileged_teacher_control(ego, carla_frame=1234)

        self.assertEqual(
            target,
            {
                "schema_version": "1.0",
                "privileged": True,
                "purpose": "offline_teacher_action_target_only",
                "source": "carla.Vehicle.get_control",
                "carla_frame": 1234,
                "throttle": 0.25,
                "steer": -0.4,
                "brake": 0.0,
                "hand_brake": False,
                "reverse": False,
                "manual_gear_shift": False,
                "gear": 1,
            },
        )

    def test_invalid_teacher_control_is_rejected_before_sample_write(self) -> None:
        ego = SimpleNamespace(
            get_control=lambda: SimpleNamespace(
                throttle=float("nan"),
                steer=0.0,
                brake=0.0,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
                gear=0,
            )
        )

        with self.assertRaisesRegex(RuntimeError, "throttle"):
            _privileged_teacher_control(ego, carla_frame=1)

    def test_sensor_queue_discards_old_frames_and_returns_exact_target(self) -> None:
        sensor_queue = NativeSensorQueue("rgb")
        sensor_queue.callback(FakeImage(10))
        sensor_queue.callback(FakeImage(11))
        selected = sensor_queue.get_exact(11, timeout=0.1)
        self.assertEqual(selected.image.frame, 11)
        self.assertEqual(selected.sequence, 2)
        self.assertEqual(sensor_queue.received, 2)
        self.assertEqual(sensor_queue.discarded, 1)

    def test_sensor_queue_exposes_the_next_frame_for_phase_discovery(self) -> None:
        sensor_queue = NativeSensorQueue("rgb")
        sensor_queue.callback(FakeImage(7))
        selected = sensor_queue.get_next(timeout=0.1)
        self.assertEqual(selected.image.frame, 7)
        self.assertEqual(selected.sequence, 1)

    def test_sensor_queue_fails_if_the_stream_skips_the_target(self) -> None:
        sensor_queue = NativeSensorQueue("teacher")
        sensor_queue.callback(FakeImage(12))
        with self.assertRaisesRegex(SensorFrameError, "skipped required CARLA frame 11"):
            sensor_queue.get_exact(11, timeout=0.1)

    def test_image_conversion_preserves_bgra_pose_and_source_frame(self) -> None:
        sensor_queue = NativeSensorQueue("rgb")
        image = FakeImage(20)
        sensor_queue.callback(image)
        queued = sensor_queue.get_exact(20, timeout=0.1)
        frame = image_to_bridge_frame(
            queued,
            fov_degrees=90.0,
            sensor_type=0,
        )
        self.assertEqual(frame.frame, 20)
        self.assertEqual(frame.bgra, image.raw_data)
        self.assertEqual(frame.transform, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        self.assertEqual((frame.width, frame.height, frame.fov), (2, 1, 90.0))

    def test_relative_transform_rotates_local_offset_with_origin(self) -> None:
        origin = TransformRecipe(10.0, 20.0, 1.0, 0.0, 90.0, 0.0)
        relative = TransformRecipe(5.0, 2.0, 3.0, 0.0, 10.0, 0.0)
        result = compose_relative_transform(origin, relative)
        self.assertAlmostEqual(result.x, 8.0)
        self.assertAlmostEqual(result.y, 25.0)
        self.assertAlmostEqual(result.z, 4.0)
        self.assertEqual(result.yaw, 100.0)


class VerifiedScenarioPlanTests(unittest.TestCase):
    def test_plan_recomputes_exactly_and_supports_stable_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan_dir = make_verified_plan(Path(temporary))
            plan = load_verified_scenario_plan(plan_dir)
            self.assertEqual(plan.run_id, "verified-plan")
            self.assertEqual(len(plan.episodes), 23)
            selected = select_episodes(
                plan,
                partitions=("test_map_ood",),
                max_episodes=2,
            )
            self.assertEqual(len(selected), 2)
            self.assertTrue(all(episode.split.partition == "test_map_ood" for episode in selected))

    def test_plan_rejects_an_artifact_changed_after_manifest_registration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan_dir = make_verified_plan(Path(temporary))
            episodes_path = plan_dir / "episodes.jsonl"
            episodes_path.write_text(
                episodes_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ScenarioPlanIntegrityError,
                "checksum mismatch",
            ):
                load_verified_scenario_plan(plan_dir)

    def test_dry_run_never_imports_carla_or_mutates_the_simulator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan_dir = make_verified_plan(Path(temporary))
            args = parse_args(
                [
                    "--scenario-plan",
                    str(plan_dir),
                    "--dataset-id",
                    "dry-run-dataset",
                    "--partition",
                    "test_seen",
                    "--dry-run",
                ]
            )
            with (
                patch(
                    "carla_vision.native.worker._load_carla_module",
                    side_effect=AssertionError("CARLA must not be imported"),
                ),
                patch("sys.stdout", new=io.StringIO()),
            ):
                result = collect_native(args)
            self.assertEqual(result["mode"], "dry_run")
            self.assertEqual(result["selected_episode_count"], 1)
            self.assertTrue(result["would_mutate_simulator"])

    def test_real_run_requires_explicit_tick_owner_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan_dir = make_verified_plan(Path(temporary))
            args = parse_args(
                [
                    "--scenario-plan",
                    str(plan_dir),
                    "--dataset-id",
                    "blocked-run",
                    "--max-episodes",
                    "1",
                ]
            )
            with self.assertRaisesRegex(RuntimeError, "exclusive-tick-owner"):
                collect_native(args)

    def test_verified_plan_reference_points_to_the_manifest_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan_dir = make_verified_plan(Path(temporary))
            plan = load_verified_scenario_plan(plan_dir)
            manifest = json.loads((plan_dir / "manifest.json").read_text())
            self.assertEqual(plan.reference["run_id"], manifest["run_id"])
            self.assertEqual(plan.reference["path"], str(plan_dir / "manifest.json"))
            self.assertEqual(len(plan.reference["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
