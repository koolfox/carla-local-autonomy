from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from carla_vision.bridge import CarlaImageFrame
from carla_vision.contracts import Detection, DetectorMetadata
from carla_vision.dataset.sync import SynchronizedFramePair
from carla_vision.dataset.writer import DatasetWriter
from carla_vision.replay.contracts import ReplayConfig
from carla_vision.replay.runner import run_replay
from carla_vision.replay.verified import (
    ReplayIntegrityError,
    load_verified_replay,
)
from carla_vision.verification import verify_research_object


def _camera_frame(
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
        transform=(1.0, 2.0, 3.0, 0.0, 0.0, 0.0),
        width=width,
        height=height,
        fov=90.0,
        bgra=pixels.tobytes(),
        received_monotonic=carla_frame / 20.0,
    )


def _pair(sequence: int, carla_frame: int, value: int) -> SynchronizedFramePair:
    rgb = np.full((12, 16, 4), value, dtype=np.uint8)
    rgb[:, :, 3] = 255
    teacher = np.zeros((12, 16, 4), dtype=np.uint8)
    teacher[2:8, 3:11, 2] = 14
    teacher[2:8, 3:11, 1] = sequence
    teacher[:, :, 3] = 255
    return SynchronizedFramePair(
        rgb=_camera_frame(sequence, carla_frame, rgb),
        teacher=_camera_frame(sequence, carla_frame, teacher),
        rgb_skipped=0,
        teacher_skipped=0,
    )


def _dataset(root: Path) -> Path:
    with DatasetWriter(
        root / "datasets",
        dataset_id="ds-replay",
        carla_endpoint={"host": "localhost", "port": 2000},
        carla_version="0.9.16",
        carla_map="Town10HD_Opt",
        repository_root=root,
        minimum_pixels=4,
    ) as writer:
        for sequence, episode_id in ((1, "episode-a"), (2, "episode-b")):
            writer.add_pair(
                _pair(sequence, sequence * 10, sequence * 20),
                split="train",
                scenario_id="scenario-replay",
                episode_id=episode_id,
                context={
                    "map_family": "Town10HD",
                    "weather_recipe_id": "clear-day",
                    "light": "day",
                },
            )
    return root / "datasets" / "ds-replay"


def _evaluation_config() -> dict[str, object]:
    return {
        "evaluation_id": "eval-replay-test",
        "schema_version": "1.0",
        "purpose": "development",
        "master_seed": 99,
        "partitions": ["train"],
        "minimum_prediction_confidence": 0.05,
        "operating_confidence": 0.25,
        "matching_iou": 0.5,
        "calibration_bins": 5,
        "bootstrap_replicates": 20,
        "bootstrap_confidence": 0.95,
        "montage_count": 2,
        "class_aliases": {"automobile": "car"},
    }


def _model(model_id: str, label: str) -> dict[str, object]:
    return {
        "model_id": model_id,
        "label": label,
        "device": "cpu",
        "model_package": None,
        "detector": {
            "backend": "custom",
            "weights": None,
            "image_size": 320,
            "factory": "unused:create",
            "options": {},
        },
    }


def _replay_config() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "replay_id": "replay-test",
        "title": "Controlled paired replay",
        "purpose": "development",
        "master_seed": 1234,
        "reference_model_id": "reference",
        "bootstrap_replicates": 30,
        "bootstrap_confidence": 0.95,
        "latency_exclude_first_images": 1,
        "montage_count": 2,
        "models": [
            _model("reference", "Reference"),
            _model("candidate", "Candidate"),
        ],
    }


class _FakeDetector:
    def __init__(self, *, detects: bool) -> None:
        self.detects = detects
        self.closed = False
        self.seen_values: list[int] = []

    @property
    def name(self) -> str:
        return "controlled-fake"

    @property
    def metadata(self) -> DetectorMetadata:
        return DetectorMetadata(
            name=self.name,
            backend="fake",
            weights=None,
            device="cpu",
            image_size=320,
            confidence=0.05,
        )

    def infer(self, image_bgr: np.ndarray) -> tuple[Detection, ...]:
        self.seen_values.append(int(image_bgr[0, 0, 0]))
        if not self.detects:
            return ()
        return (
            Detection(
                class_id=5,
                source_class_id=5,
                label="automobile",
                confidence=0.9,
                xyxy=(3.0, 2.0, 11.0, 8.0),
            ),
        )

    def close(self) -> None:
        self.closed = True


class ReplayContractTests(unittest.TestCase):
    def test_strict_contract_rejects_unknown_fields_and_confirmatory_loose_weights(
        self,
    ) -> None:
        raw = _replay_config()
        raw["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            ReplayConfig.from_mapping(raw, base=Path.cwd())

        raw = _replay_config()
        raw["purpose"] = "confirmatory"
        with self.assertRaisesRegex(ValueError, "verified model packages"):
            ReplayConfig.from_mapping(raw, base=Path.cwd())


class ReplayRunnerTests(unittest.TestCase):
    def test_paired_replay_retains_children_tables_plots_and_detects_tampering(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = _dataset(root)
            evaluation_path = root / "evaluation.json"
            replay_path = root / "replay.json"
            evaluation_path.write_text(
                json.dumps(_evaluation_config()),
                encoding="utf-8",
            )
            replay_path.write_text(
                json.dumps(_replay_config()),
                encoding="utf-8",
            )
            reference = _FakeDetector(detects=True)
            candidate = _FakeDetector(detects=False)
            result = run_replay(
                config_path=replay_path,
                dataset_path=dataset,
                evaluation_config_path=evaluation_path,
                runs_root=root / "runs",
                repository_root=root,
                detector_overrides={
                    "reference": reference,
                    "candidate": candidate,
                },
            )
            run_dir = Path(result["run_dir"])
            self.assertTrue(reference.closed)
            self.assertTrue(candidate.closed)
            self.assertEqual(reference.seen_values, candidate.seen_values)
            self.assertEqual(reference.seen_values, [20, 40])

            verified = load_verified_replay(run_dir)
            generic = verify_research_object(
                run_dir,
                deep=True,
                reject_unregistered=True,
            )
            self.assertEqual(verified.replay_id, "replay-test")
            self.assertEqual(generic.deep_verification["kind"], "paired_replay")
            self.assertEqual(generic.deep_verification["model_count"], 2)
            self.assertGreaterEqual(generic.checksum_index_entries, 18)

            summary = result["summary"]
            aggregate = {row["model_id"]: row for row in summary["aggregate_metrics"]}
            self.assertEqual(aggregate["reference"]["operating_tp"], 2)
            self.assertEqual(aggregate["reference"]["operating_fn"], 0)
            self.assertEqual(aggregate["candidate"]["operating_tp"], 0)
            self.assertEqual(aggregate["candidate"]["operating_fn"], 2)
            self.assertEqual(len(summary["child_runs"]), 2)
            for child in summary["child_runs"]:
                child_root = Path(child["root"])
                self.assertTrue((child_root / "summary_metrics.json").is_file())

            roles = {
                artifact["role"]
                for artifact in json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))[
                    "artifacts"
                ]
            }
            self.assertTrue(
                {
                    "replay_release_manifest",
                    "replay_paired_differences",
                    "replay_disagreements",
                    "replay_paired_bootstrap",
                    "replay_qualitative_panel",
                    "replay_plot",
                    "replay_checksum_index",
                }.issubset(roles)
            )
            self.assertEqual(len(list((run_dir / "plots").glob("*.png"))), 4)
            self.assertEqual(len(list((run_dir / "plots").glob("*.svg"))), 4)

            per_image = run_dir / "tables" / "per_image_metrics.csv"
            per_image.write_text(
                per_image.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ReplayIntegrityError,
                "fingerprint mismatch|checksum mismatch",
            ):
                load_verified_replay(run_dir)


if __name__ == "__main__":
    unittest.main()
