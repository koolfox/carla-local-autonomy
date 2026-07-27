from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from carla_vision.bridge import CarlaImageFrame
from carla_vision.dataset.qa import audit_dataset
from carla_vision.dataset.sync import SynchronizedFramePair
from carla_vision.native.worker import collect_native, parse_args
from carla_vision.scenarios.planner import plan_scenarios
from carla_vision.scenarios.verified_plan import load_verified_scenario_plan
from carla_vision.training.runner import run_training
from carla_vision.verification import verify_research_object

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = REPOSITORY_ROOT / "configs" / "scenarios" / "thesis_pilot_v1.json"
SPLIT_PATH = REPOSITORY_ROOT / "configs" / "scenarios" / "split_plan_thesis_pilot_v1.json"


def frame(sequence: int, carla_frame: int, pixels: np.ndarray, sensor_type: int) -> CarlaImageFrame:
    height, width, _ = pixels.shape
    return CarlaImageFrame(
        sequence=sequence,
        sensor_type=sensor_type,
        frame=carla_frame,
        timestamp=carla_frame / 20.0,
        transform=(0.0, 0.0, 1.7, 0.0, 0.0, 0.0),
        width=width,
        height=height,
        fov=90.0,
        bgra=pixels.tobytes(),
        received_monotonic=carla_frame / 20.0,
    )


def pair(sequence: int, carla_frame: int, rgb_value: int) -> SynchronizedFramePair:
    rgb = np.full((12, 16, 4), rgb_value, dtype=np.uint8)
    rgb[:, :, 3] = 255
    teacher = np.zeros((12, 16, 4), dtype=np.uint8)
    teacher[2:10, 3:13, 2] = 0x12
    teacher[2:10, 3:13, 1] = 0x34
    teacher[2:10, 3:13, 0] = 14
    teacher[:, :, 3] = 255
    return SynchronizedFramePair(
        rgb=frame(sequence, carla_frame, rgb, 0),
        teacher=frame(sequence, carla_frame, teacher, 1),
        rgb_skipped=0,
        teacher_skipped=0,
    )


class FakeNativeSession:
    def __init__(self, *_: object, **__: object) -> None:
        self.server_version = "0.9.16"
        self.client_version = "0.9.16"
        self.restored = False
        self.counter = 0

    def run_episode(self, episode: object, writer: object) -> dict[str, object]:
        self.counter += 1
        carla_frame = 100 + self.counter
        sample = writer.add_pair(
            pair(self.counter, carla_frame, 20 + self.counter),
            split=episode.split.partition,
            scenario_id=episode.scenario_id,
            episode_id=episode.episode_id,
            context={
                "map": episode.recipe.map_name,
                "weather_recipe_id": episode.recipe.weather.weather_id,
                "runtime_sensor_contract": "front_monocular_rgb_only",
                "teacher_uses_privileged_simulator_state": True,
                "privileged_teacher_control": {
                    "schema_version": "1.0",
                    "privileged": True,
                    "purpose": "offline_teacher_action_target_only",
                    "source": "carla.Vehicle.get_control",
                    "carla_frame": carla_frame,
                    "throttle": 0.2,
                    "steer": 0.0,
                    "brake": 0.0,
                    "hand_brake": False,
                    "reverse": False,
                    "manual_gear_shift": False,
                    "gear": 1,
                },
            },
        )
        return {
            "schema_version": "1.0",
            "status": "complete",
            "episode": episode.as_dict(),
            "carla": {
                "client_version": self.client_version,
                "server_version": self.server_version,
                "map": episode.recipe.map_name,
            },
            "samples": {
                "count": 1,
                "sample_ids": [sample.sample_id],
                "carla_frames": [carla_frame],
                "annotation_count": sample.annotation_count,
            },
            "actors": {
                "spawn_failures": [],
                "cleanup": {"success": True},
            },
        }

    def restore_asynchronous_mode(self) -> None:
        self.restored = True


def training_config() -> dict[str, object]:
    return {
        "experiment_id": "exp-native-handoff-test",
        "schema_version": "1.0",
        "backend": "ultralytics-rtdetr",
        "custom_factory": None,
        "master_seed": 20260727,
        "epochs": 1,
        "image_size": 320,
        "batch_size": 1,
        "device": "cpu",
        "workers": 0,
        "patience": 0,
        "optimizer": "AdamW",
        "initial_learning_rate": 0.0001,
        "final_learning_rate_fraction": 0.01,
        "weight_decay": 0.0005,
        "warmup_epochs": 0.0,
        "amp": False,
        "cache": False,
        "deterministic": True,
        "save_period": 1,
        "training_partitions": ["train"],
        "validation_partitions": ["val_seen"],
        "extra_options": {},
    }


class NativeTrainingHandoffTests(unittest.TestCase):
    def test_plan_to_native_dataset_to_qa_to_training_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_result = plan_scenarios(
                suite_path=SUITE_PATH,
                split_plan_path=SPLIT_PATH,
                runs_root=root / "runs",
                run_id="handoff-plan",
                repository_root=REPOSITORY_ROOT,
            )
            plan_dir = Path(plan_result["run_dir"])
            plan = load_verified_scenario_plan(plan_dir)
            train_episode = next(
                episode for episode in plan.episodes if episode.split.partition == "train"
            )
            val_episode = next(
                episode for episode in plan.episodes if episode.split.partition == "val_seen"
            )
            args = parse_args(
                [
                    "--scenario-plan",
                    str(plan_dir),
                    "--dataset-id",
                    "ds-native-handoff",
                    "--datasets-root",
                    str(root / "datasets"),
                    "--episode-id",
                    train_episode.episode_id,
                    "--episode-id",
                    val_episode.episode_id,
                    "--acknowledge-exclusive-tick-owner",
                ]
            )
            with (
                patch("carla_vision.native.worker._load_carla_module", return_value=object()),
                patch(
                    "carla_vision.native.worker.NativeCarlaSession",
                    FakeNativeSession,
                ),
                patch("sys.stdout", new=io.StringIO()),
            ):
                collection = collect_native(args)
            dataset = Path(collection["dataset_dir"])
            before = hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest()
            verified_dataset = verify_research_object(
                dataset,
                reject_unregistered=True,
            )
            self.assertEqual(verified_dataset.deep_verification["sample_count"], 2)
            self.assertEqual(
                verified_dataset.deep_verification["partitions"],
                {"train": 1, "val_seen": 1},
            )
            self.assertTrue(
                verified_dataset.deep_verification["privileged_teacher_control_targets"]
            )

            qa_result = audit_dataset(
                dataset,
                runs_root=root / "runs",
                run_id="handoff-qa",
                montage_count=2,
                repository_root=REPOSITORY_ROOT,
            )
            verify_research_object(
                qa_result["run_dir"],
                reject_unregistered=True,
            )
            self.assertEqual(
                hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest(),
                before,
            )

            config_path = root / "training.json"
            config_path.write_text(json.dumps(training_config()), encoding="utf-8")
            weights = root / "rtdetr-l.pt"
            weights.write_bytes(b"development-pretrained-placeholder")
            training = run_training(
                config_path=config_path,
                dataset_path=dataset,
                pretrained_weights=weights,
                runs_root=root / "runs",
                run_id="handoff-training-dry-run",
                dry_run=True,
                repository_root=REPOSITORY_ROOT,
            )
            training_run = Path(training["run_dir"])
            resolved = json.loads(
                (training_run / "resolved_training_config.json").read_text(encoding="utf-8")
            )
            self.assertFalse(resolved["resolved_partitions"]["locked_test_exposed_to_trainer"])
            self.assertEqual(resolved["resolved_partitions"]["train"], ["train"])
            self.assertEqual(resolved["resolved_partitions"]["validation"], ["val_seen"])
            self.assertFalse(training["summary"]["training_executed"])
            verify_research_object(
                training_run,
                reject_unregistered=True,
            )


if __name__ == "__main__":
    unittest.main()
