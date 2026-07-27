from __future__ import annotations

import csv
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
from carla_vision.failure_mining.contracts import FailureMiningConfig
from carla_vision.failure_mining.miner import run_failure_mining
from carla_vision.failure_mining.review import run_failure_review
from carla_vision.failure_mining.review_contracts import FailureReviewConfig
from carla_vision.failure_mining.review_verified import (
    FailureReviewIntegrityError,
    load_verified_failure_review,
)
from carla_vision.failure_mining.verified import (
    FailureMiningIntegrityError,
    load_verified_failure_mining,
)
from carla_vision.verification import verify_research_object


def _frame(
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
        rgb=_frame(sequence, sequence * 10, rgb),
        teacher=_frame(sequence, sequence * 10, teacher),
        rgb_skipped=0,
        teacher_skipped=0,
    )


def _dataset(root: Path) -> Path:
    with DatasetWriter(
        root / "datasets",
        dataset_id="ds-failure",
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
                scenario_id="scenario-failure",
                episode_id="episode-a" if sequence <= 2 else "episode-b",
                context={
                    "map_family": "Town10HD",
                    "weather_recipe_id": "clear-day",
                    "light": "day",
                },
            )
    return root / "datasets" / "ds-failure"


class _FailureDetector:
    @property
    def name(self) -> str:
        return "failure-fake"

    @property
    def metadata(self) -> DetectorMetadata:
        return DetectorMetadata(
            name=self.name,
            backend="fake",
            weights=None,
            device="cpu",
            image_size=320,
            confidence=0.5,
        )

    def infer(self, image_bgr: np.ndarray) -> tuple[Detection, ...]:
        value = int(image_bgr[0, 0, 0])
        if value == 20:
            return (
                Detection(
                    class_id=5,
                    source_class_id=5,
                    label="car",
                    confidence=0.9,
                    xyxy=(3.0, 2.0, 11.0, 8.0),
                ),
                Detection(
                    class_id=5,
                    source_class_id=5,
                    label="car",
                    confidence=0.8,
                    xyxy=(0.0, 0.0, 2.0, 2.0),
                ),
            )
        if value == 40:
            return (
                Detection(
                    class_id=3,
                    source_class_id=0,
                    label="person",
                    confidence=0.9,
                    xyxy=(3.0, 2.0, 11.0, 8.0),
                ),
            )
        if value == 60:
            return (
                Detection(
                    class_id=5,
                    source_class_id=5,
                    label="car",
                    confidence=0.9,
                    xyxy=(7.0, 2.0, 15.0, 8.0),
                ),
            )
        return ()

    def close(self) -> None:
        pass


def _evaluation_config() -> dict[str, object]:
    return {
        "evaluation_id": "eval-failure-source",
        "schema_version": "1.0",
        "purpose": "development",
        "master_seed": 100,
        "partitions": ["val_seen"],
        "minimum_prediction_confidence": 0.5,
        "operating_confidence": 0.5,
        "matching_iou": 0.5,
        "calibration_bins": 5,
        "bootstrap_replicates": 20,
        "bootstrap_confidence": 0.95,
        "montage_count": 2,
        "class_aliases": {"person": "pedestrian"},
    }


def _mining_config() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "mining_id": "failure-mining-test",
        "title": "Controlled validation failure mining",
        "purpose": "development",
        "master_seed": 9876,
        "threshold_source": "evaluation_config",
        "failure_types": [
            "false_negative",
            "false_positive",
            "misclassification",
            "localization",
        ],
        "localization_iou_minimum": 0.2,
        "max_candidates": 10,
        "per_image_limit": 2,
        "per_episode_limit": 4,
        "montage_count": 4,
        "severity_weights": {
            "false_negative": 4.0,
            "false_positive": 1.0,
            "misclassification": 3.0,
            "localization": 2.0,
        },
        "critical_categories": ["car", "pedestrian"],
        "critical_multiplier": 2.0,
        "rarity_weight": 0.5,
    }


def _review_config() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "review_id": "failure-review-test",
        "title": "Controlled completed failure review",
        "purpose": "development",
        "reviewers": ["reviewer-a"],
        "require_complete": True,
    }


class FailureMiningContractTests(unittest.TestCase):
    def test_mining_is_development_only(self) -> None:
        raw = _mining_config()
        raw["purpose"] = "confirmatory"
        with self.assertRaisesRegex(ValueError, "development workflow"):
            FailureMiningConfig.from_mapping(raw)

        review = _review_config()
        review["require_complete"] = False
        with self.assertRaisesRegex(ValueError, "must be true"):
            FailureReviewConfig.from_mapping(review)


class FailureMiningTests(unittest.TestCase):
    def test_four_failure_types_queue_review_template_and_verifier(self) -> None:
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
                    confidence=0.5,
                ),
                runs_root=root / "runs",
                run_id="evaluation-failure-source",
                repository_root=root,
                detector_override=_FailureDetector(),
            )
            mining_config = root / "mining.json"
            mining_config.write_text(
                json.dumps(_mining_config()),
                encoding="utf-8",
            )
            result = run_failure_mining(
                config_path=mining_config,
                evaluation_run=evaluation["run_dir"],
                runs_root=root / "runs",
                repository_root=root,
            )
            run_dir = Path(result["run_dir"])
            verified = load_verified_failure_mining(run_dir)
            generic = verify_research_object(
                run_dir,
                deep=True,
                reject_unregistered=True,
            )
            self.assertEqual(
                generic.deep_verification["kind"],
                "failure_mining",
            )
            self.assertEqual(generic.checksum_index_entries, 15)
            self.assertEqual(len(verified.all_failures), 4)
            self.assertEqual(len(verified.review_queue), 4)
            self.assertEqual(
                {row["failure_type"] for row in verified.review_queue},
                {
                    "false_negative",
                    "false_positive",
                    "misclassification",
                    "localization",
                },
            )
            self.assertEqual(
                [row["selected_rank"] for row in verified.review_queue],
                [1, 2, 3, 4],
            )
            template = run_dir / "review" / "review_template.csv"
            with template.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(not row["decision"] for row in rows))
            self.assertEqual(len(list((run_dir / "plots").glob("*.png"))), 3)
            self.assertEqual(len(list((run_dir / "plots").glob("*.svg"))), 3)

            completed_review = root / "completed_review.csv"
            decisions = (
                "confirmed",
                "label_issue",
                "rejected",
                "duplicate",
            )
            for row, decision in zip(rows, decisions, strict=True):
                row["decision"] = decision
                row["reviewer"] = "reviewer-a"
                row["review_notes"] = f"reviewed as {decision}"
                row["proposed_remedy"] = (
                    "generate a new scenario" if decision in {"confirmed", "label_issue"} else ""
                )
            with completed_review.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            review_config = root / "review_config.json"
            review_config.write_text(
                json.dumps(_review_config()),
                encoding="utf-8",
            )
            review_result = run_failure_review(
                config_path=review_config,
                mining_run=run_dir,
                review_csv=completed_review,
                runs_root=root / "runs",
                repository_root=root,
            )
            review_dir = Path(review_result["run_dir"])
            reviewed = load_verified_failure_review(review_dir)
            generic_review = verify_research_object(
                review_dir,
                deep=True,
                reject_unregistered=True,
            )
            self.assertEqual(
                generic_review.deep_verification["kind"],
                "failure_review",
            )
            self.assertEqual(generic_review.checksum_index_entries, 15)
            self.assertEqual(len(reviewed.reviews), 4)
            self.assertEqual(len(reviewed.confirmed), 1)
            self.assertEqual(len(reviewed.label_issues), 1)
            self.assertEqual(
                len(list((review_dir / "plots").glob("*.png"))),
                3,
            )
            normalized_reviews = review_dir / "reviews" / "reviews.jsonl"
            normalized_reviews.write_text(
                normalized_reviews.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                FailureReviewIntegrityError,
                "fingerprint mismatch|checksum mismatch",
            ):
                load_verified_failure_review(review_dir)

            template.write_text(
                template.read_text(encoding="utf-8").replace(
                    ",,,,,\n",
                    ",confirmed,,reviewer,looks valid,generate variant\n",
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                FailureMiningIntegrityError,
                "fingerprint mismatch|checksum mismatch",
            ):
                load_verified_failure_mining(run_dir)


if __name__ == "__main__":
    unittest.main()
