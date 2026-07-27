from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from carla_vision.artifacts import RunArtifactTracker
from carla_vision.reporting.builder import build_report
from carla_vision.reporting.contracts import ReportConfig
from carla_vision.reporting.verified import ReportIntegrityError, load_verified_report
from carla_vision.verification import verify_research_object


def build_source(
    root: Path,
    *,
    run_id: str,
    role: str,
    payload: dict[str, object],
) -> Path:
    tracker = RunArtifactTracker(
        root / "runs",
        run_id=run_id,
        cli_args=(),
        repository_root=root,
        package_names=(),
    )
    with tracker:
        summary = tracker.artifact_path("summary.json")
        summary.write_text(json.dumps(payload), encoding="utf-8")
        tracker.register_artifact(summary, role=role)
    return tracker.run_dir


def build_sources(root: Path) -> tuple[Path, Path, Path, Path]:
    evaluation = build_source(
        root,
        run_id="eval-source",
        role="evaluation_summary",
        payload={
            "selection": {"images": 10, "ground_truth_annotations": 25},
            "bootstrap": {"episode_count": 3},
            "coco": {"ap_50_95": 0.4, "ap_50": 0.6, "ap_75": 0.3},
            "operating_point": {
                "precision": 0.8,
                "recall": 0.5,
                "false_negative_rate": 0.5,
                "f1": 0.615,
            },
            "latency_ms": {"excluding_first_image": {"median": 12.5, "p95": 18.0}},
        },
    )
    runtime = build_source(
        root,
        run_id="runtime-source",
        role="analysis_summary_metrics",
        payload={
            "metrics": {
                "frames": {"count": 20, "effective_source_fps": 9.5},
                "detections": {"total": 40, "hazard_frame_rate": 0.1},
                "timing": {
                    "pipeline_latency_ms": {"median": 20.0, "p95": 30.0},
                    "model_inference_ms": {"median": 15.0},
                },
                "simulator_speed_mps": {"estimated_distance_metres": 12.0},
            }
        },
    )
    vision_shadow = build_source(
        root,
        run_id="vision-shadow-source",
        role="run_summary",
        payload={
            "elapsed": 12.0,
            "distance_travelled": 15.0,
            "max_simulator_speed": 2.5,
            "perception_stats": {
                "submitted": 80,
                "processed": 75,
                "dropped_before_inference": 4,
            },
            "recording_stats": {"written": 74},
            "vision_shadow_actuation_authorized": False,
            "vision_shadow_stats": {
                "proposal_count": 74,
                "throttle_proposal_count": 70,
                "braking_proposal_count": 4,
                "latency_median_ms": 0.02,
                "latency_p95_ms": 0.04,
            },
        },
    )
    shadow_matrix = build_source(
        root,
        run_id="shadow-matrix-source",
        role="shadow_matrix_release_manifest",
        payload={
            "planned_run_count": 2,
            "successful_run_count": 2,
            "failed_run_count": 0,
            "vision_policy_actuation_authorized": False,
        },
    )
    return evaluation, runtime, vision_shadow, shadow_matrix


def build_operator_source(root: Path) -> Path:
    return build_source(
        root,
        run_id="operator-source",
        role="operator_job_status",
        payload={
            "job_id": "op-test-plan",
            "kind": "scenario_plan",
            "status": "success",
            "returncode": 0,
            "motion_authorized": False,
            "destructive": False,
            "stop_requested": False,
            "expected_output": str(root / "runs" / "plan"),
            "expected_output_exists": True,
        },
    )


def build_native_preflight_source(root: Path) -> Path:
    return build_source(
        root,
        run_id="native-preflight-source",
        role="native_preflight_summary",
        payload={
            "read_only": True,
            "simulator_contacted": True,
            "simulator_mutated": False,
            "automated_ready": False,
            "manual_ready": False,
            "ready_for_native_execution": False,
            "selection": {
                "selected_episode_count": 1,
                "planned_capture_count": 150,
            },
            "checks": {
                "count": 14,
                "passed": 7,
                "failed": 1,
                "pending": 2,
                "skipped": 3,
                "warnings": 1,
            },
            "endpoint": {
                "socket": {"reachable": True, "latency_ms": 1.0},
                "bridge_read_only_rpc": {"latency_ms": 2.0},
                "native_pythonapi": {"importable": False},
            },
        },
    )


def report_config(
    evaluation: Path,
    runtime: Path,
    vision_shadow: Path | None = None,
    shadow_matrix: Path | None = None,
    operator_session: Path | None = None,
    native_preflight: Path | None = None,
    *,
    purpose: str = "development",
) -> dict[str, object]:
    sources = [
        {
            "path": str(evaluation),
            "kind": "evaluation",
            "label": "Detector evaluation",
        },
        {
            "path": str(runtime),
            "kind": "runtime_analysis",
            "label": "Runtime analysis",
        },
    ]
    if vision_shadow is not None:
        sources.append(
            {
                "path": str(vision_shadow),
                "kind": "vision_shadow",
                "label": "Vision shadow run",
            }
        )
    if shadow_matrix is not None:
        sources.append(
            {
                "path": str(shadow_matrix),
                "kind": "shadow_matrix",
                "label": "Shadow matrix",
            }
        )
    if operator_session is not None:
        sources.append(
            {
                "path": str(operator_session),
                "kind": "operator_session",
                "label": "Operator scenario planning session",
            }
        )
    if native_preflight is not None:
        sources.append(
            {
                "path": str(native_preflight),
                "kind": "native_preflight",
                "label": "Native collection preflight",
            }
        )
    return {
        "schema_version": "1.0",
        "report_id": "rpt-test-v001",
        "title": "Deterministic Test Report",
        "authors": ["CARLA Vision Research Project"],
        "purpose": purpose,
        "thesis_context": "Tests manifest-driven evidence aggregation.",
        "sources": sources,
        "claims": ["All reported values are extracted from verified machine-readable inputs."],
        "limitations": ["Synthetic unit-test sources are not scientific evidence."],
        "include_plots": True,
    }


class ReportContractTests(unittest.TestCase):
    def test_contract_rejects_duplicate_labels_and_unknown_source_kind(self) -> None:
        source = {
            "path": "run",
            "kind": "evaluation",
            "label": "same",
        }
        config = report_config(Path("a"), Path("b"))
        config["sources"] = [source, source]
        with self.assertRaisesRegex(ValueError, "labels must be unique"):
            ReportConfig.from_mapping(config)
        config = report_config(Path("a"), Path("b"))
        config["sources"][0]["kind"] = "spreadsheet"
        with self.assertRaisesRegex(ValueError, "unsupported report source"):
            ReportConfig.from_mapping(config)


class ReportBuilderTests(unittest.TestCase):
    def test_evidence_registry_source_metrics_are_extracted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = build_source(
                root,
                run_id="registry-source",
                role="evidence_registry_release_manifest",
                payload={
                    "source_count": 7,
                    "source_manifest_count": 6,
                    "verified_count": 5,
                    "verification_failed_count": 2,
                    "registered_artifact_count": 42,
                    "failed_source_tree_entry_count": 9,
                    "failed_source_tree_file_count": 7,
                    "failed_source_tree_bytes": 2048,
                    "excluded_registry_count": 3,
                    "generation": {
                        "carla_contacted": False,
                        "simulator_mutated": False,
                        "source_objects_modified": False,
                    },
                },
            )
            config = report_config(Path("unused-evaluation"), Path("unused-runtime"))
            config["sources"] = [
                {
                    "path": str(registry),
                    "kind": "evidence_registry",
                    "label": "Evidence registry",
                }
            ]
            config["include_plots"] = False
            config_path = root / "report.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            result = build_report(
                config_path=config_path,
                reports_root=root / "reports",
                repository_root=root,
            )
            metrics_path = Path(result["report_dir"]) / "tables" / "metrics.csv"
            with metrics_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))

            metrics = {row["metric"]: (row["value"], row["unit"]) for row in rows}
            self.assertEqual(
                metrics,
                {
                    "objects": ("7", "objects"),
                    "source_manifests": ("6", "objects"),
                    "verified_objects": ("5", "objects"),
                    "verification_failures": ("2", "objects"),
                    "registered_artifacts": ("42", "artifacts"),
                    "failed_source_tree_entries": ("9", "entries"),
                    "failed_source_tree_files": ("7", "files"),
                    "failed_source_tree_bytes": ("2048", "bytes"),
                    "excluded_registries": ("3", "objects"),
                    "carla_contacted": ("0", "boolean"),
                    "simulator_mutated": ("0", "boolean"),
                    "source_objects_modified": ("0", "boolean"),
                },
            )

    def test_report_tables_plots_checksums_and_source_graph_verify(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation, runtime, vision_shadow, shadow_matrix = build_sources(root)
            operator_session = build_operator_source(root)
            native_preflight = build_native_preflight_source(root)
            config_path = root / "report.json"
            config_path.write_text(
                json.dumps(
                    report_config(
                        evaluation,
                        runtime,
                        vision_shadow,
                        shadow_matrix,
                        operator_session,
                        native_preflight,
                    )
                ),
                encoding="utf-8",
            )
            result = build_report(
                config_path=config_path,
                reports_root=root / "reports",
                repository_root=root,
            )
            report_dir = Path(result["report_dir"])
            verified = load_verified_report(report_dir)
            generic = verify_research_object(report_dir, reject_unregistered=True)
            self.assertEqual(verified.report_id, "rpt-test-v001")
            self.assertEqual(generic.deep_verification["kind"], "report")
            self.assertEqual(generic.deep_verification["source_count"], 6)
            self.assertEqual(generic.unregistered_file_count, 0)
            self.assertGreaterEqual(generic.checksum_index_entries, 10)
            markdown = (report_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("Detector evaluation", markdown)
            self.assertIn("coco_ap_50_95", markdown)
            self.assertIn("vision_policy_actuation_authorized", markdown)
            self.assertIn("job_succeeded", markdown)
            self.assertIn("expected_output_exists", markdown)
            self.assertIn("ready_for_native_execution", markdown)
            self.assertTrue((report_dir / "plots" / "evaluation_summary.svg").is_file())
            self.assertTrue((report_dir / "plots" / "runtime_performance.svg").is_file())

    def test_source_tampering_invalidates_the_derived_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation, runtime, _, _ = build_sources(root)
            config_path = root / "report.json"
            config_path.write_text(
                json.dumps(report_config(evaluation, runtime)),
                encoding="utf-8",
            )
            result = build_report(
                config_path=config_path,
                reports_root=root / "reports",
                repository_root=root,
            )
            (evaluation / "summary.json").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ReportIntegrityError, "source.*failed verification"):
                load_verified_report(result["report_dir"])

    def test_confirmatory_report_rejects_dirty_or_uncommitted_sources_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation, runtime, _, _ = build_sources(root)
            config_path = root / "report.json"
            config_path.write_text(
                json.dumps(report_config(evaluation, runtime, purpose="confirmatory")),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Exception, "clean-git gate"):
                build_report(
                    config_path=config_path,
                    reports_root=root / "reports",
                    repository_root=root,
                )
            self.assertFalse((root / "reports").exists())


if __name__ == "__main__":
    unittest.main()
