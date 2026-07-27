from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from carla_vision.analysis import (
    ANALYSIS_SCHEMA_VERSION,
    AnalysisInputError,
    analyze_run,
    parse_args,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detection(
    *,
    class_id: int,
    label: str,
    confidence: float,
    hazard: bool = False,
) -> dict[str, object]:
    return {
        "class_id": class_id,
        "source_class_id": class_id,
        "label": label,
        "confidence": confidence,
        "xyxy": [10.0, 20.0, 50.0, 80.0],
        "attributes": {},
        "risk": {
            "in_driving_corridor": hazard,
            "visually_close": hazard,
            "hazard": hazard,
            "confidence_threshold": 0.45,
        },
    }


def frame(
    *,
    sequence: int,
    carla_frame: int,
    source_timestamp: float,
    received: float,
    inference_seconds: float,
    detections: list[dict[str, object]],
    speed: float,
) -> dict[str, object]:
    return {
        "sequence": sequence,
        "carla_frame": carla_frame,
        "source_timestamp": source_timestamp,
        "source_received_monotonic": received,
        "completed_monotonic": received + inference_seconds,
        "pipeline_latency_seconds": inference_seconds,
        "model_inference_seconds": inference_seconds * 0.8,
        "detector": "rtdetr:test",
        "detections": detections,
        "hazard": any(
            bool(item["risk"]["hazard"])  # type: ignore[index]
            for item in detections
        ),
        "mode": "PERCEPTION ONLY",
        "simulator_speed_mps": speed,
        "route_progress": "-",
    }


def create_source_run(
    root: Path,
    *,
    frames: list[dict[str, object]],
    run_id: str = "source-run",
) -> Path:
    run_dir = root / run_id
    log_path = run_dir / "logs" / "detections.jsonl"
    log_path.parent.mkdir(parents=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as stream:
        for payload in frames:
            stream.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            stream.write("\n")
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "success",
        "carla": {
            "endpoint": {"host": "172.20.10.7", "port": 2000},
            "version": "0.9.16",
            "map": "Town10HD_Opt",
        },
        "artifacts": [
            {
                "path": "logs/detections.jsonl",
                "role": "detections_jsonl",
                "sha256": sha256(log_path),
                "size_bytes": log_path.stat().st_size,
                "mime_type": "application/x-ndjson",
                "metadata": {"schema_version": "1.0"},
            }
        ],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir


def tree_snapshot(root: Path) -> dict[str, tuple[str, int]]:
    return {
        path.relative_to(root).as_posix(): (sha256(path), path.stat().st_size)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class PostHocAnalysisTests(unittest.TestCase):
    def sample_frames(self) -> list[dict[str, object]]:
        return [
            frame(
                sequence=1,
                carla_frame=101,
                source_timestamp=10.0,
                received=100.0,
                inference_seconds=0.04,
                detections=[
                    detection(
                        class_id=2,
                        label="car",
                        confidence=0.80,
                    )
                ],
                speed=0.0,
            ),
            frame(
                sequence=3,
                carla_frame=103,
                source_timestamp=10.1,
                received=100.1,
                inference_seconds=0.06,
                detections=[
                    detection(
                        class_id=0,
                        label="person",
                        confidence=0.91,
                        hazard=True,
                    ),
                    detection(
                        class_id=2,
                        label="car",
                        confidence=0.74,
                    ),
                ],
                speed=2.0,
            ),
            frame(
                sequence=5,
                carla_frame=105,
                source_timestamp=10.25,
                received=100.25,
                inference_seconds=0.03,
                detections=[],
                speed=1.0,
            ),
        ]

    def test_analysis_creates_derived_run_with_metrics_plots_and_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_run = create_source_run(
                root / "source-runs",
                frames=self.sample_frames(),
            )
            source_before = tree_snapshot(source_run)
            source_manifest_hash = sha256(source_run / "manifest.json")
            source_log_hash = sha256(source_run / "logs" / "detections.jsonl")

            result = analyze_run(
                source_run,
                runs_root=root / "derived-runs",
                run_id="analysis-001",
                cli_args=(
                    "--source-run",
                    str(source_run),
                    "--run-id",
                    "analysis-001",
                ),
                repository_root=root,
            )

            self.assertEqual(tree_snapshot(source_run), source_before)
            self.assertEqual(result.run_id, "analysis-001")
            self.assertEqual(
                result.run_dir,
                (root / "derived-runs" / "analysis-001").resolve(),
            )

            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["schema_version"], ANALYSIS_SCHEMA_VERSION)
            self.assertEqual(summary["analysis_run_id"], "analysis-001")
            self.assertEqual(summary["source_run"]["run_id"], "source-run")
            self.assertEqual(
                summary["source_run"]["manifest"]["sha256"],
                source_manifest_hash,
            )
            self.assertEqual(
                summary["source_run"]["detections"]["sha256"],
                source_log_hash,
            )
            metrics = summary["metrics"]
            self.assertEqual(metrics["frames"]["count"], 3)
            self.assertAlmostEqual(metrics["frames"]["effective_source_fps"], 8.0)
            self.assertEqual(metrics["detections"]["total"], 3)
            self.assertEqual(metrics["detections"]["hazard_frame_count"], 1)
            self.assertAlmostEqual(
                metrics["detections"]["hazard_frame_rate"],
                1.0 / 3.0,
            )
            self.assertEqual(
                metrics["detections"]["class_frequency"],
                {"car": 2, "person": 1},
            )
            self.assertAlmostEqual(
                metrics["timing"]["inference_ms"]["mean"],
                130.0 / 3.0,
            )
            self.assertAlmostEqual(
                metrics["timing"]["pipeline_latency_ms"]["mean"],
                130.0 / 3.0,
            )
            self.assertAlmostEqual(
                metrics["timing"]["model_inference_ms"]["mean"],
                104.0 / 3.0,
            )

            with (result.run_dir / "frames.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["sequence"], "1")
            self.assertEqual(rows[0]["source_cadence_ms"], "")
            self.assertAlmostEqual(float(rows[1]["source_cadence_ms"]), 100.0)
            self.assertAlmostEqual(float(rows[1]["pipeline_latency_ms"]), 60.0)
            self.assertAlmostEqual(float(rows[1]["model_inference_ms"]), 48.0)
            self.assertEqual(rows[1]["hazard"], "true")
            self.assertEqual(
                json.loads(rows[1]["class_counts_json"]),
                {"car": 1, "person": 1},
            )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "success")
            self.assertEqual(
                manifest["references"]["datasets"],
                [summary["source_run"]],
            )
            expected_paths = {
                "frames.csv",
                "plots/class_frequency.png",
                "plots/detections_hazards.png",
                "plots/latency_cadence.png",
                "plots/simulator_speed.png",
                "summary_metrics.json",
            }
            self.assertEqual(
                {entry["path"] for entry in manifest["artifacts"]},
                expected_paths,
            )
            for entry in manifest["artifacts"]:
                artifact_path = result.run_dir / entry["path"]
                self.assertTrue(artifact_path.is_file())
                self.assertEqual(entry["sha256"], sha256(artifact_path))
                self.assertEqual(entry["size_bytes"], artifact_path.stat().st_size)
                if artifact_path.suffix == ".png":
                    self.assertEqual(
                        artifact_path.read_bytes()[:8],
                        b"\x89PNG\r\n\x1a\n",
                    )

    def test_bad_frame_schema_is_reported_before_output_run_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bad_frames = self.sample_frames()
            bad_frames[1]["detections"][0]["confidence"] = 1.5  # type: ignore[index]
            source_run = create_source_run(
                root / "source-runs",
                frames=bad_frames,
            )
            derived_root = root / "derived-runs"

            with self.assertRaisesRegex(
                AnalysisInputError,
                r"line 2 detections\[0\].*confidence.*at most 1.0",
            ):
                analyze_run(
                    source_run,
                    runs_root=derived_root,
                    run_id="must-not-exist",
                    repository_root=root,
                )

            self.assertFalse(derived_root.exists())

    def test_manifest_checksum_mismatch_is_rejected_without_mutating_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_run = create_source_run(
                root / "source-runs",
                frames=self.sample_frames(),
            )
            log_path = source_run / "logs" / "detections.jsonl"
            serialized = log_path.read_text(encoding="utf-8")
            log_path.write_text(
                serialized.replace(
                    '"simulator_speed_mps": 2.0',
                    '"simulator_speed_mps": 2.5',
                    1,
                ),
                encoding="utf-8",
            )
            source_before = tree_snapshot(source_run)

            with self.assertRaisesRegex(
                AnalysisInputError,
                "checksum does not match",
            ):
                analyze_run(
                    source_run,
                    runs_root=root / "derived-runs",
                    run_id="must-not-exist",
                    repository_root=root,
                )

            self.assertEqual(tree_snapshot(source_run), source_before)
            self.assertFalse((root / "derived-runs").exists())

    def test_cli_contract_requires_source_and_accepts_output_identity(self) -> None:
        args = parse_args(
            [
                "--source-run",
                "/tmp/source-run",
                "--runs-root",
                "/tmp/derived-runs",
                "--run-id",
                "analysis-pilot",
            ]
        )
        self.assertEqual(args.source_run, "/tmp/source-run")
        self.assertEqual(args.runs_root, "/tmp/derived-runs")
        self.assertEqual(args.run_id, "analysis-pilot")

    def test_legacy_inference_seconds_field_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            frames = self.sample_frames()
            for payload in frames:
                payload["inference_seconds"] = payload.pop("pipeline_latency_seconds")
                payload.pop("model_inference_seconds")
            source_run = create_source_run(root / "source-runs", frames=frames)

            result = analyze_run(
                source_run,
                runs_root=root / "derived-runs",
                run_id="legacy-analysis",
                repository_root=root,
            )

            timing = result.summary["metrics"]["timing"]
            self.assertEqual(timing["pipeline_latency_ms"]["count"], 3)
            self.assertEqual(timing["model_inference_ms"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
