from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from carla_vision.artifacts import RunArtifactTracker, fingerprint_file
from carla_vision.verification import ArtifactIntegrityError, verify_research_object


def build_run(
    root: Path,
    *,
    external_reference: Path | None = None,
) -> Path:
    references = [fingerprint_file(external_reference)] if external_reference is not None else None
    tracker = RunArtifactTracker(
        root / "runs",
        run_id="verified-run",
        cli_args=(),
        repository_root=root,
        model_refs=references,
        package_names=(),
    )
    with tracker:
        artifact = tracker.artifact_path("tables/metrics.csv")
        artifact.write_text("metric,value\nap,0.5\n", encoding="utf-8")
        tracker.register_artifact(
            artifact,
            role="metrics_table",
            metadata={"schema_version": "1.0"},
        )
    return tracker.run_dir


class ResearchObjectVerificationTests(unittest.TestCase):
    def test_successful_run_verifies_by_directory_or_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = build_run(Path(temporary))
            by_directory = verify_research_object(run_dir)
            by_manifest = verify_research_object(run_dir / "manifest.json")
            self.assertEqual(by_directory.run_id, "verified-run")
            self.assertEqual(by_directory.status, "success")
            self.assertEqual(by_directory.artifact_count, 1)
            self.assertEqual(by_directory.role_counts, {"metrics_table": 1})
            self.assertEqual(by_directory.deep_verification["kind"], "tracked_run")
            self.assertEqual(by_directory.as_dict(), by_manifest.as_dict())

    def test_artifact_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = build_run(Path(temporary))
            (run_dir / "tables" / "metrics.csv").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactIntegrityError, "fingerprint mismatch"):
                verify_research_object(run_dir)

    def test_external_reference_is_verified_and_can_be_skipped_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "weights.pt"
            weights.write_bytes(b"trusted weights")
            run_dir = build_run(root, external_reference=weights)
            self.assertEqual(
                verify_research_object(run_dir).external_reference_count,
                1,
            )
            weights.write_bytes(b"changed weights")
            with self.assertRaisesRegex(ArtifactIntegrityError, "external reference"):
                verify_research_object(run_dir)
            self.assertEqual(
                verify_research_object(
                    run_dir,
                    verify_references=False,
                ).external_reference_count,
                0,
            )

    def test_relative_nested_reference_uses_its_declared_source_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            source_file = source / "payload.json"
            source_file.write_text("{}\n", encoding="utf-8")
            tracker = RunArtifactTracker(
                root / "runs",
                run_id="nested-reference",
                cli_args=(),
                repository_root=root,
                input_refs=[
                    {
                        "path": str(source),
                        "payload": {
                            **fingerprint_file(source_file),
                            "path": "payload.json",
                        },
                    }
                ],
                package_names=(),
            )
            with tracker:
                pass
            self.assertEqual(
                verify_research_object(tracker.run_dir).external_reference_count,
                1,
            )

    def test_unregistered_file_gate_and_clean_git_gate_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = build_run(Path(temporary))
            (run_dir / "unregistered.txt").write_text("not tracked", encoding="utf-8")
            result = verify_research_object(run_dir)
            self.assertEqual(result.unregistered_file_count, 1)
            with self.assertRaisesRegex(ArtifactIntegrityError, "unregistered files"):
                verify_research_object(run_dir, reject_unregistered=True)
            with self.assertRaisesRegex(ArtifactIntegrityError, "clean-git gate"):
                verify_research_object(run_dir, require_clean_git=True)

    def test_manifest_path_escape_and_unsorted_artifacts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = build_run(root)
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            outside = root / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            manifest["artifacts"][0]["path"] = "../outside.txt"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ArtifactIntegrityError, "unsafe relative path"):
                verify_research_object(run_dir)

            run_dir = build_run(root / "second")
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            first = dict(manifest["artifacts"][0])
            second_path = run_dir / "a.txt"
            second_path.write_text("a", encoding="utf-8")
            second = {
                **first,
                "path": "a.txt",
                **fingerprint_file(second_path),
            }
            second.pop("path")
            second["path"] = "a.txt"
            manifest["artifacts"].append(second)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ArtifactIntegrityError, "sorted by path"):
                verify_research_object(run_dir)


if __name__ == "__main__":
    unittest.main()
