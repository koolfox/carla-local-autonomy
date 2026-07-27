from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from carla_vision.artifacts import RunArtifactTracker
from carla_vision.bridge import CarlaImageFrame
from carla_vision.dataset.sync import SynchronizedFramePair
from carla_vision.dataset.verified import load_verified_dataset
from carla_vision.dataset.writer import DatasetWriter
from carla_vision.model_release.contracts import ModelReleaseConfig
from carla_vision.model_release.package import package_model
from carla_vision.model_release.verified import ModelIntegrityError, load_verified_model
from carla_vision.verification import verify_research_object


def synchronized_pair(sequence: int, value: int) -> SynchronizedFramePair:
    rgb = np.full((6, 8, 4), value, dtype=np.uint8)
    rgb[:, :, 3] = 255
    instance = np.zeros((6, 8, 4), dtype=np.uint8)
    instance[1:5, 2:7, 2] = 0x12
    instance[1:5, 2:7, 1] = 0x34
    instance[1:5, 2:7, 0] = 14
    instance[:, :, 3] = 255

    def frame(sensor_type: int, pixels: np.ndarray) -> CarlaImageFrame:
        return CarlaImageFrame(
            sequence=sequence,
            sensor_type=sensor_type,
            frame=sequence * 10,
            timestamp=sequence / 2,
            transform=(1.5, 0.0, 1.7, 0.0, 0.0, 0.0),
            width=8,
            height=6,
            fov=90.0,
            bgra=pixels.tobytes(),
            received_monotonic=float(sequence),
        )

    return SynchronizedFramePair(
        rgb=frame(0, rgb),
        teacher=frame(1, instance),
        rgb_skipped=0,
        teacher_skipped=0,
    )


def build_dataset(root: Path) -> Path:
    with DatasetWriter(
        root / "datasets",
        dataset_id="ds-model-release",
        carla_endpoint={"host": "localhost", "port": 2000},
        carla_version="0.9.16",
        carla_map="Town10HD_Opt",
        repository_root=root,
        minimum_pixels=4,
    ) as writer:
        writer.add_pair(
            synchronized_pair(1, 20),
            split="train",
            scenario_id="scn-train",
            episode_id="ep-train",
        )
        writer.add_pair(
            synchronized_pair(2, 30),
            split="val_seen",
            scenario_id="scn-val",
            episode_id="ep-val",
        )
    return root / "datasets" / "ds-model-release"


def build_training_run(root: Path) -> Path:
    dataset = load_verified_dataset(build_dataset(root))
    tracker = RunArtifactTracker(
        root / "runs",
        run_id="train-source",
        cli_args=(),
        config={"object_type": "detector_training"},
        repository_root=root,
        dataset_refs=[dataset.reference],
        package_names=(),
    )
    with tracker:
        resolved_path = tracker.artifact_path("resolved_training_config.json")
        summary_path = tracker.artifact_path("training_summary.json")
        best_path = tracker.artifact_path("backend/weights/selected.ckpt")
        last_path = tracker.artifact_path("backend/weights/final.ckpt")
        resolved_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "object_type": "detector_training",
                    "mode": "training",
                    "experiment_id": "exp-model-release-test",
                    "config": {
                        "backend": "custom",
                        "image_size": 64,
                    },
                    "resolved_partitions": {
                        "train": ["train"],
                        "validation": ["val_seen"],
                        "locked_test_exposed_to_trainer": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": "success",
                    "training_executed": True,
                    "metrics": {"map50": 0.42, "loss": 0.3},
                    "seed_state": {"master_seed": 42},
                }
            ),
            encoding="utf-8",
        )
        best_path.write_bytes(b"selected-model-weights")
        last_path.write_bytes(b"last-model-weights")
        for path, role in (
            (best_path, "best_checkpoint"),
            (last_path, "last_checkpoint"),
            (resolved_path, "resolved_training_configuration"),
            (summary_path, "training_summary"),
        ):
            tracker.register_artifact(path, role=role)
    return tracker.run_dir


def release_config(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "1.0",
        "model_id": "mdl-custom-pilot-s42-best",
        "architecture": "test-detector",
        "backend": "custom",
        "checkpoint_role": "best",
        "input": {
            "color_order": "BGR",
            "image_size": 64,
            "resize_method": "letterbox",
            "normalization": "backend-managed uint8",
            "temporal_context_frames": 1,
        },
        "inference": {
            "detector_backend": "custom",
            "detector_factory": "example.detector:create",
            "options": {"postprocess": "test"},
        },
        "output_contract": "carla-vision-detection-v1",
        "selection": {
            "metric": "map50",
            "mode": "max",
            "partitions": ["val_seen"],
        },
        "intended_use": "Test-only monocular CARLA object detection.",
        "limitations": [
            "Synthetic test package with no accuracy claim.",
            "Not for real-world driving.",
        ],
        "license": {
            "weights": "Test-Only",
            "code": "BSD-3-Clause",
            "upstream": "Test-Only",
        },
        "upstream_source": "local deterministic test backend",
    }
    value.update(overrides)
    return value


class ModelReleaseContractTests(unittest.TestCase):
    def test_contract_rejects_locked_selection_and_license_placeholders(self) -> None:
        config = release_config()
        config["selection"] = {
            "metric": "map50",
            "mode": "max",
            "partitions": ["test_seen"],
        }
        with self.assertRaisesRegex(ValueError, "locked test"):
            ModelReleaseConfig.from_mapping(config)
        config = release_config()
        config["license"] = {
            "weights": "TBD",
            "code": "BSD-3-Clause",
            "upstream": "Test-Only",
        }
        with self.assertRaisesRegex(ValueError, "placeholders"):
            ModelReleaseConfig.from_mapping(config)


class ModelPackagingTests(unittest.TestCase):
    def test_package_is_checksum_indexed_and_consumer_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_run = build_training_run(root)
            config_path = root / "release.json"
            config_path.write_text(json.dumps(release_config()), encoding="utf-8")
            result = package_model(
                config_path=config_path,
                training_run=training_run,
                models_root=root / "models",
                repository_root=root,
            )
            model_dir = Path(result["model_dir"])
            verified = load_verified_model(model_dir)
            generic = verify_research_object(model_dir, reject_unregistered=True)
            self.assertEqual(verified.model_id, "mdl-custom-pilot-s42-best")
            self.assertEqual(verified.backend, "custom")
            self.assertEqual(verified.image_size, 64)
            self.assertEqual(verified.weights_path.read_bytes(), b"selected-model-weights")
            detector_config = verified.detector_config(device="cpu", confidence=0.2)
            self.assertEqual(detector_config.factory, "example.detector:create")
            self.assertEqual(detector_config.options, {"postprocess": "test"})
            self.assertEqual(generic.deep_verification["kind"], "model")
            self.assertEqual(generic.checksum_index_entries, 4)
            self.assertEqual(generic.artifact_count, 5)
            self.assertEqual(generic.unregistered_file_count, 0)
            self.assertIn("Checkpoint formats", (model_dir / "model-card.md").read_text())

    def test_training_contract_mismatch_fails_before_model_directory_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_run = build_training_run(root)
            config = release_config()
            config["backend"] = "rtdetr"
            config_path = root / "release.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagrees with training backend"):
                package_model(
                    config_path=config_path,
                    training_run=training_run,
                    models_root=root / "models",
                    repository_root=root,
                )
            self.assertFalse((root / "models").exists())

    def test_packaged_weight_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_run = build_training_run(root)
            config_path = root / "release.json"
            config_path.write_text(json.dumps(release_config()), encoding="utf-8")
            result = package_model(
                config_path=config_path,
                training_run=training_run,
                models_root=root / "models",
                repository_root=root,
            )
            model_dir = Path(result["model_dir"])
            weights = model_dir / "weights" / "model.ckpt"
            weights.write_bytes(b"tampered")
            with self.assertRaisesRegex(ModelIntegrityError, "fingerprint mismatch"):
                load_verified_model(model_dir)


if __name__ == "__main__":
    unittest.main()
