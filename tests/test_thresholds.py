from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from carla_vision.bridge import CarlaImageFrame
from carla_vision.contracts import (
    Detection,
    DetectorConfig,
    DetectorMetadata,
)
from carla_vision.dataset.sync import SynchronizedFramePair
from carla_vision.dataset.writer import DatasetWriter
from carla_vision.evaluation.runner import run_evaluation
from carla_vision.thresholds.contracts import ThresholdSelectionConfig
from carla_vision.thresholds.selector import run_threshold_selection
from carla_vision.thresholds.verified import (
    ThresholdSelectionIntegrityError,
    load_verified_threshold_selection,
)
from carla_vision.verification import verify_research_object


def _camera(
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
        transform=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        width=width,
        height=height,
        fov=90.0,
        bgra=pixels.tobytes(),
        received_monotonic=carla_frame / 20.0,
    )


def _pair(sequence: int, value: int) -> SynchronizedFramePair:
    rgb = np.full((12, 16, 4), value, dtype=np.uint8)
    rgb[:, :, 3] = 255
    teacher = np.zeros((12, 16, 4), dtype=np.uint8)
    teacher[2:8, 3:11, 2] = 14
    teacher[2:8, 3:11, 1] = sequence
    teacher[:, :, 3] = 255
    return SynchronizedFramePair(
        rgb=_camera(sequence, sequence * 10, rgb),
        teacher=_camera(sequence, sequence * 10, teacher),
        rgb_skipped=0,
        teacher_skipped=0,
    )


def _dataset(root: Path) -> Path:
    with DatasetWriter(
        root / "datasets",
        dataset_id="ds-threshold",
        carla_endpoint={"host": "localhost", "port": 2000},
        carla_version="0.9.16",
        carla_map="Town10HD_Opt",
        repository_root=root,
        minimum_pixels=4,
    ) as writer:
        for sequence, value in enumerate((20, 40, 60, 80), start=1):
            writer.add_pair(
                _pair(sequence, value),
                split="val_seen",
                scenario_id="scenario-threshold",
                episode_id="episode-a" if sequence <= 2 else "episode-b",
                context={
                    "map_family": "Town10HD",
                    "weather_recipe_id": "clear-day",
                    "light": "day",
                },
            )
    return root / "datasets" / "ds-threshold"


class _ScoredDetector:
    correct = {20: 0.9, 40: 0.8, 60: 0.4, 80: 0.3}
    false = {20: 0.7, 40: 0.6, 60: 0.2, 80: 0.1}

    @property
    def name(self) -> str:
        return "scored-fake"

    @property
    def metadata(self) -> DetectorMetadata:
        return DetectorMetadata(
            name=self.name,
            backend="fake",
            weights=None,
            device="cpu",
            image_size=320,
            confidence=0.1,
        )

    def infer(self, image_bgr: np.ndarray) -> tuple[Detection, ...]:
        value = int(image_bgr[0, 0, 0])
        return (
            Detection(
                class_id=5,
                source_class_id=5,
                label="car",
                confidence=self.correct[value],
                xyxy=(3.0, 2.0, 11.0, 8.0),
            ),
            Detection(
                class_id=5,
                source_class_id=5,
                label="car",
                confidence=self.false[value],
                xyxy=(0.0, 0.0, 2.0, 2.0),
            ),
        )

    def close(self) -> None:
        pass


def _evaluation_config() -> dict[str, object]:
    return {
        "evaluation_id": "eval-threshold-source",
        "schema_version": "1.0",
        "purpose": "development",
        "master_seed": 100,
        "partitions": ["val_seen"],
        "minimum_prediction_confidence": 0.1,
        "operating_confidence": 0.25,
        "matching_iou": 0.5,
        "calibration_bins": 5,
        "bootstrap_replicates": 20,
        "bootstrap_confidence": 0.95,
        "montage_count": 2,
        "class_aliases": {},
    }


def _selection_config() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "selection_id": "threshold-test",
        "title": "Validation threshold test",
        "purpose": "development",
        "master_seed": 999,
        "objective": "maximize_f1",
        "target_recall": None,
        "target_precision": None,
        "false_positive_cost": 1.0,
        "false_negative_cost": 2.0,
        "tie_breaker": "highest_threshold",
        "grid": {
            "minimum": 0.1,
            "maximum": 0.9,
            "steps": 5,
            "scale": "linear",
        },
        "bootstrap_replicates": 40,
        "bootstrap_confidence": 0.95,
    }


class ThresholdContractTests(unittest.TestCase):
    def test_objective_specific_targets_are_strict(self) -> None:
        raw = _selection_config()
        raw["target_recall"] = 0.8
        with self.assertRaisesRegex(ValueError, "only valid"):
            ThresholdSelectionConfig.from_mapping(raw)


class ThresholdSelectionTests(unittest.TestCase):
    def test_validation_sweep_selects_and_verifies_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = _dataset(root)
            evaluation_config = root / "evaluation.json"
            evaluation_config.write_text(
                json.dumps(_evaluation_config()),
                encoding="utf-8",
            )
            evaluation = run_evaluation(
                config_path=evaluation_config,
                dataset_path=dataset,
                detector_config=DetectorConfig(
                    backend="custom",
                    factory="unused:create",
                    confidence=0.1,
                ),
                runs_root=root / "runs",
                run_id="evaluation-threshold-source",
                repository_root=root,
                detector_override=_ScoredDetector(),
            )
            selection_config = root / "selection.json"
            selection_config.write_text(
                json.dumps(_selection_config()),
                encoding="utf-8",
            )
            result = run_threshold_selection(
                config_path=selection_config,
                evaluation_run=evaluation["run_dir"],
                runs_root=root / "runs",
                repository_root=root,
            )
            run_dir = Path(result["run_dir"])
            verified = load_verified_threshold_selection(run_dir)
            generic = verify_research_object(
                run_dir,
                deep=True,
                reject_unregistered=True,
            )
            self.assertAlmostEqual(verified.operating_confidence, 0.3)
            self.assertEqual(
                generic.deep_verification["kind"],
                "threshold_selection",
            )
            self.assertEqual(generic.checksum_index_entries, 12)
            selected = json.loads((run_dir / "selected_threshold.json").read_text(encoding="utf-8"))
            self.assertEqual(selected["operating_point"]["tp"], 4)
            self.assertEqual(selected["operating_point"]["fp"], 2)
            self.assertEqual(selected["operating_point"]["fn"], 0)
            self.assertEqual(len(list((run_dir / "plots").glob("*.png"))), 3)
            self.assertEqual(len(list((run_dir / "plots").glob("*.svg"))), 3)

            sweep = run_dir / "tables" / "threshold_sweep.csv"
            sweep.write_text(
                sweep.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ThresholdSelectionIntegrityError,
                "fingerprint mismatch|checksum mismatch",
            ):
                load_verified_threshold_selection(run_dir)


if __name__ == "__main__":
    unittest.main()
