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
from carla_vision.contracts import (
    Detection,
    DetectorConfig,
    DetectorMetadata,
)
from carla_vision.dataset.sync import SynchronizedFramePair
from carla_vision.dataset.writer import DatasetWriter
from carla_vision.evaluation.metrics import (
    bbox_iou_xywh,
    calibration_metrics,
    operating_metrics,
)
from carla_vision.evaluation.runner import parse_args, run_evaluation
from carla_vision.evaluation.verified import (
    EvaluationIntegrityError,
    load_verified_evaluation,
)


def camera_frame(
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


def make_pair(sequence: int, carla_frame: int, value: int) -> SynchronizedFramePair:
    rgb = np.full((12, 16, 4), value, dtype=np.uint8)
    rgb[:, :, 3] = 255
    teacher = np.zeros((12, 16, 4), dtype=np.uint8)
    teacher[2:8, 3:11, 2] = 14
    teacher[2:8, 3:11, 1] = 0x34
    teacher[2:8, 3:11, 0] = 0x12
    teacher[:, :, 3] = 255
    return SynchronizedFramePair(
        rgb=camera_frame(sequence, carla_frame, rgb),
        teacher=camera_frame(sequence, carla_frame, teacher),
        rgb_skipped=0,
        teacher_skipped=0,
    )


def build_dataset(root: Path, *, split: str = "train") -> Path:
    with DatasetWriter(
        root / "datasets",
        dataset_id="ds-eval",
        carla_endpoint={"host": "localhost", "port": 2000},
        carla_version="0.9.16",
        carla_map="Town10HD_Opt",
        repository_root=root,
        minimum_pixels=4,
    ) as writer:
        writer.add_pair(
            make_pair(1, 10, 20),
            split=split,
            scenario_id="scn-eval",
            episode_id="ep-eval",
            context={
                "map_family": "Town10HD",
                "weather_recipe_id": "clear-day",
                "light": "day",
            },
        )
    return root / "datasets" / "ds-eval"


def evaluation_config(partition: str = "train") -> dict[str, object]:
    return {
        "evaluation_id": "eval-test-v1",
        "schema_version": "1.0",
        "purpose": "development",
        "master_seed": 42,
        "partitions": [partition],
        "minimum_prediction_confidence": 0.05,
        "operating_confidence": 0.25,
        "matching_iou": 0.5,
        "calibration_bins": 5,
        "bootstrap_replicates": 20,
        "bootstrap_confidence": 0.95,
        "montage_count": 2,
        "class_aliases": {"automobile": "car"},
    }


class FakeDetector:
    def __init__(self) -> None:
        self.closed = False

    @property
    def name(self) -> str:
        return "fake-detector"

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
        self.assert_image(image_bgr)
        return (
            Detection(
                class_id=99,
                source_class_id=99,
                label="automobile",
                confidence=0.9,
                xyxy=(3.0, 2.0, 11.0, 8.0),
            ),
        )

    @staticmethod
    def assert_image(image_bgr: np.ndarray) -> None:
        if image_bgr.shape != (12, 16, 3):
            raise AssertionError("unexpected image")

    def close(self) -> None:
        self.closed = True


class MetricTests(unittest.TestCase):
    def test_iou_operating_counts_confusion_and_calibration(self) -> None:
        self.assertEqual(bbox_iou_xywh((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        ground_truth = [{"image_id": 1, "category_id": 5, "bbox": [0, 0, 10, 10]}]
        predictions = [
            {"image_id": 1, "category_id": 5, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"image_id": 1, "category_id": 5, "bbox": [20, 20, 2, 2], "score": 0.8},
        ]
        metrics = operating_metrics(
            ground_truth,
            predictions,
            image_ids=[1],
            categories={5: "car"},
            confidence_threshold=0.25,
            iou_threshold=0.5,
        )
        self.assertEqual(metrics.overall["tp"], 1)
        self.assertEqual(metrics.overall["fp"], 1)
        self.assertEqual(metrics.overall["fn"], 0)
        self.assertEqual(metrics.confusion_matrix.tolist(), [[1, 0], [1, 0]])
        calibration, rows = calibration_metrics(metrics.prediction_outcomes, bins=5)
        self.assertEqual(calibration["count"], 2)
        self.assertEqual(sum(row["count"] for row in rows), 2)


class EvaluationRunnerTests(unittest.TestCase):
    def test_model_package_is_mutually_exclusive_with_loose_model_options(self) -> None:
        base = [
            "--config",
            "evaluation.json",
            "--dataset",
            "dataset",
            "--model-package",
            "models/example",
        ]
        for option in (
            ["--detector", "rtdetr"],
            ["--weights", "model.pt"],
            ["--detector-factory", "example:create"],
            ["--image-size", "320"],
        ):
            with self.subTest(option=option), patch("sys.stderr", new=io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args([*base, *option])

    def test_fake_detector_produces_canonical_metrics_tables_plots_and_panel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = build_dataset(root)
            config_path = root / "evaluation.json"
            config_path.write_text(json.dumps(evaluation_config()), encoding="utf-8")
            detector = FakeDetector()
            result = run_evaluation(
                config_path=config_path,
                dataset_path=dataset,
                detector_config=DetectorConfig(
                    backend="custom",
                    factory="unused:create",
                    confidence=0.05,
                ),
                runs_root=root / "runs",
                run_id="evaluation-fake",
                repository_root=root,
                detector_override=detector,
            )
            run_dir = Path(result["run_dir"])
            manifest = json.loads((run_dir / "manifest.json").read_text())
            summary = result["summary"]
            self.assertTrue(detector.closed)
            self.assertEqual(manifest["status"], "success")
            self.assertEqual(summary["operating_point"]["tp"], 1)
            self.assertEqual(summary["operating_point"]["fp"], 0)
            self.assertEqual(summary["operating_point"]["fn"], 0)
            self.assertAlmostEqual(summary["coco"]["ap_50"], 1.0)
            self.assertEqual(summary["prediction_mapping"]["accepted_predictions"], 1)
            predictions = json.loads(
                (run_dir / "predictions" / "predictions.coco.json").read_text()
            )
            self.assertEqual(
                predictions,
                [
                    {
                        "image_id": 1,
                        "category_id": 5,
                        "bbox": [3.0, 2.0, 8.0, 6.0],
                        "score": 0.9,
                    }
                ],
            )
            roles = {artifact["role"] for artifact in manifest["artifacts"]}
            self.assertTrue(
                {
                    "canonical_coco_predictions",
                    "per_class_metrics",
                    "stratified_metrics",
                    "calibration_bins",
                    "episode_bootstrap_metrics",
                    "qualitative_failure_panel",
                    "evaluation_plot",
                    "evaluation_summary",
                }.issubset(roles)
            )
            for artifact in manifest["artifacts"]:
                path = run_dir / artifact["path"]
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    artifact["sha256"],
                )
            verified = load_verified_evaluation(run_dir)
            self.assertEqual(verified.run_id, "evaluation-fake")
            self.assertEqual(len(verified.predictions), 1)
            predictions_path = run_dir / "predictions" / "predictions.coco.json"
            predictions_path.write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(
                EvaluationIntegrityError,
                "fingerprint mismatch",
            ):
                load_verified_evaluation(run_dir)

    def test_locked_test_partition_requires_explicit_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = build_dataset(root, split="test_seen")
            config_path = root / "evaluation.json"
            config_path.write_text(
                json.dumps(evaluation_config("test_seen")),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "acknowledge-locked-test"):
                run_evaluation(
                    config_path=config_path,
                    dataset_path=dataset,
                    detector_config=DetectorConfig(
                        backend="custom",
                        factory="unused:create",
                    ),
                    runs_root=root / "runs",
                    run_id="must-not-exist",
                    repository_root=root,
                    detector_override=FakeDetector(),
                )
            self.assertFalse((root / "runs" / "must-not-exist").exists())


if __name__ == "__main__":
    unittest.main()
