from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from carla_vision.bridge import CarlaImageFrame
from carla_vision.dataset.sync import SynchronizedFramePair
from carla_vision.dataset.writer import DatasetWriter
from carla_vision.training.contracts import (
    TrainingConfig,
    TrainingRequest,
    TrainingResult,
)
from carla_vision.training.runner import _Tee, run_training


def image_frame(
    sequence: int,
    carla_frame: int,
    pixels: np.ndarray,
) -> CarlaImageFrame:
    height, width, _ = pixels.shape
    return CarlaImageFrame(
        sequence=sequence,
        sensor_type=0,
        frame=carla_frame,
        timestamp=carla_frame / 20.0,
        transform=(0.0, 0.0, 1.7, 0.0, 0.0, 0.0),
        width=width,
        height=height,
        fov=90.0,
        bgra=pixels.tobytes(),
        received_monotonic=carla_frame / 20.0,
    )


def sample_pair(
    sequence: int,
    carla_frame: int,
    rgb_value: int,
) -> SynchronizedFramePair:
    rgb = np.full((8, 12, 4), rgb_value, dtype=np.uint8)
    rgb[:, :, 3] = 255
    teacher = np.zeros((8, 12, 4), dtype=np.uint8)
    teacher[2:7, 3:10, 2] = 0x12
    teacher[2:7, 3:10, 1] = 0x34
    teacher[2:7, 3:10, 0] = 14
    teacher[:, :, 3] = 255
    return SynchronizedFramePair(
        rgb=image_frame(sequence, carla_frame, rgb),
        teacher=image_frame(sequence, carla_frame, teacher),
        rgb_skipped=0,
        teacher_skipped=0,
    )


def build_trainable_dataset(root: Path) -> Path:
    with DatasetWriter(
        root / "datasets",
        dataset_id="ds-trainable",
        carla_endpoint={"host": "localhost", "port": 2000},
        carla_version="0.9.16",
        carla_map="multi-map",
        repository_root=root,
        minimum_pixels=4,
    ) as writer:
        writer.add_pair(
            sample_pair(1, 10, 20),
            split="train",
            scenario_id="scn-train",
            episode_id="ep-train",
        )
        writer.add_pair(
            sample_pair(2, 10, 30),
            split="val_seen",
            scenario_id="scn-val",
            episode_id="ep-val",
        )
    return root / "datasets" / "ds-trainable"


def training_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "experiment_id": "exp-test-v1",
        "schema_version": "1.0",
        "backend": "custom",
        "custom_factory": "example.module:create",
        "master_seed": 123,
        "epochs": 2,
        "image_size": 320,
        "batch_size": 1,
        "device": "cpu",
        "workers": 0,
        "patience": 1,
        "optimizer": "AdamW",
        "initial_learning_rate": 0.001,
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
    config.update(overrides)
    return config


class FakeBackend:
    @property
    def name(self) -> str:
        return "fake-pytorch-detector"

    def train(self, request: TrainingRequest) -> TrainingResult:
        output = request.output_root / "fit"
        weights = output / "weights"
        weights.mkdir(parents=True)
        best = weights / "selected.ckpt"
        last = weights / "final.ckpt"
        best.write_bytes(b"best-checkpoint")
        last.write_bytes(b"last-checkpoint")
        (output / "history.csv").write_text(
            "epoch,train_loss,val_map50\n1,1.0,0.1\n2,0.5,0.2\n",
            encoding="utf-8",
        )
        (output / "curve.svg").write_text("<svg></svg>\n", encoding="utf-8")
        return TrainingResult(
            backend_name=self.name,
            output_dir=output,
            best_checkpoint=best,
            last_checkpoint=last,
            metrics={"map50": 0.2, "loss": 0.5},
            metadata={"framework": "test"},
        )


class TrainingConfigTests(unittest.TestCase):
    def test_tee_flush_tolerates_a_closed_log_during_interpreter_cleanup(self) -> None:
        console = io.StringIO()
        log = io.StringIO()
        tee = _Tee(console, log)
        tee.write("training\n")
        log.close()
        tee.flush()
        tee.close()
        self.assertEqual(console.getvalue(), "training\n")

    def test_config_rejects_locked_test_use_and_controlled_option_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "locked test"):
            TrainingConfig.from_mapping(training_config(validation_partitions=["test_seen"]))
        with self.assertRaisesRegex(ValueError, "controlled fields"):
            TrainingConfig.from_mapping(training_config(extra_options={"seed": 999}))

    def test_config_requires_custom_factory_only_for_custom_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires custom_factory"):
            TrainingConfig.from_mapping(training_config(custom_factory=None))
        with self.assertRaisesRegex(ValueError, "only valid"):
            TrainingConfig.from_mapping(training_config(backend="rtdetr", custom_factory="x:y"))

    def test_config_rejects_invalid_batch_and_empty_backend_values(self) -> None:
        for batch_size in (0, -2):
            with (
                self.subTest(batch_size=batch_size),
                self.assertRaisesRegex(ValueError, "batch_size"),
            ):
                TrainingConfig.from_mapping(training_config(batch_size=batch_size))
        for field in ("device", "optimizer"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                TrainingConfig.from_mapping(training_config(**{field: "  "}))
        with self.assertRaisesRegex(ValueError, "initial_learning_rate"):
            TrainingConfig.from_mapping(training_config(initial_learning_rate=0.0))
        with self.assertRaisesRegex(ValueError, "cache"):
            TrainingConfig.from_mapping(training_config(cache=""))


class TrainingRunnerTests(unittest.TestCase):
    def test_dry_run_validates_every_input_and_writes_reproducible_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = build_trainable_dataset(root)
            config_path = root / "training.json"
            config_path.write_text(json.dumps(training_config()), encoding="utf-8")
            weights = root / "pretrained.pt"
            weights.write_bytes(b"pretrained")
            result = run_training(
                config_path=config_path,
                dataset_path=dataset,
                pretrained_weights=weights,
                runs_root=root / "runs",
                run_id="training-dry-run",
                dry_run=True,
                repository_root=root,
            )
            run_dir = Path(result["run_dir"])
            manifest = json.loads((run_dir / "manifest.json").read_text())
            resolved = json.loads((run_dir / "resolved_training_config.json").read_text())
            self.assertEqual(manifest["status"], "success")
            self.assertEqual(result["summary"]["status"], "dry_run")
            self.assertFalse(result["summary"]["training_executed"])
            self.assertFalse(resolved["resolved_partitions"]["locked_test_exposed_to_trainer"])
            self.assertEqual(
                resolved["pretrained_weights"]["sha256"],
                hashlib.sha256(b"pretrained").hexdigest(),
            )
            data_yaml = (run_dir / "data" / "training_data.yaml").read_text()
            self.assertIn("  - images/train", data_yaml)
            self.assertIn("  - images/val_seen", data_yaml)
            self.assertIn("test: []", data_yaml)

    def test_fake_backend_run_registers_logs_history_plots_and_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = build_trainable_dataset(root)
            config_path = root / "training.json"
            config_path.write_text(json.dumps(training_config()), encoding="utf-8")
            weights = root / "pretrained.pt"
            weights.write_bytes(b"pretrained")
            result = run_training(
                config_path=config_path,
                dataset_path=dataset,
                pretrained_weights=weights,
                runs_root=root / "runs",
                run_id="training-fake",
                repository_root=root,
                backend_override=FakeBackend(),
            )
            run_dir = Path(result["run_dir"])
            manifest = json.loads((run_dir / "manifest.json").read_text())
            summary = result["summary"]
            self.assertEqual(manifest["status"], "success")
            self.assertEqual(summary["backend"], "fake-pytorch-detector")
            self.assertEqual(summary["metrics"]["map50"], 0.2)
            roles = {artifact["role"] for artifact in manifest["artifacts"]}
            self.assertTrue(
                {
                    "best_checkpoint",
                    "last_checkpoint",
                    "training_history",
                    "training_plot_or_panel",
                    "training_log",
                    "training_summary",
                }.issubset(roles)
            )
            for artifact in manifest["artifacts"]:
                path = run_dir / artifact["path"]
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    artifact["sha256"],
                )
            trained_refs = [
                reference
                for reference in manifest["references"]["models"]
                if isinstance(reference, dict) and reference.get("kind") == "trained_checkpoint"
            ]
            self.assertEqual(len(trained_refs), 2)


if __name__ == "__main__":
    unittest.main()
