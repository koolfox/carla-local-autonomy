from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from carla_vision.artifacts import RunArtifactTracker
from carla_vision.evidence.builder import build_evidence_registry
from carla_vision.evidence.verified import (
    EvidenceRegistryIntegrityError,
    verify_evidence_registry,
)
from carla_vision.verification import verify_research_object


def build_source(
    workspace: Path,
    *,
    root_kind: str,
    run_id: str,
    fail_lifecycle: bool = False,
) -> Path:
    tracker = RunArtifactTracker(
        workspace / root_kind,
        run_id=run_id,
        cli_args=(),
        repository_root=workspace,
        package_names=(),
    )
    try:
        with tracker:
            summary = tracker.artifact_path("summary.json")
            summary.write_text(
                json.dumps({"run_id": run_id, "value": 1}) + "\n",
                encoding="utf-8",
            )
            tracker.register_artifact(summary, role="test_summary")
            if fail_lifecycle:
                raise RuntimeError("retained expected test failure")
    except RuntimeError:
        if not fail_lifecycle:
            raise
    return tracker.run_dir


def object_rows(registry: Path) -> list[dict[str, str]]:
    with (registry / "tables" / "objects.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        return list(csv.DictReader(stream))


def failed_tree_rows(registry: Path) -> list[dict[str, str]]:
    with (registry / "tables" / "failed_source_tree.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        return list(csv.DictReader(stream))


class EvidenceRegistryTests(unittest.TestCase):
    def test_valid_invalid_and_non_success_sources_are_all_retained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            valid = build_source(
                workspace,
                root_kind="datasets",
                run_id="z-valid-source",
            )
            invalid = build_source(
                workspace,
                root_kind="runs",
                run_id="a-invalid-source",
            )
            (invalid / "unregistered.txt").write_text("not registered\n", encoding="utf-8")
            build_source(
                workspace,
                root_kind="reports",
                run_id="m-retained-failure",
                fail_lifecycle=True,
            )

            result = build_evidence_registry(
                workspace=workspace,
                registry_id="evidence-registry-test-v001",
            )
            registry = Path(result["registry_dir"])
            summary = result["summary"]
            self.assertEqual(summary["source_count"], 3)
            self.assertEqual(summary["verified_count"], 2)
            self.assertEqual(summary["verification_failed_count"], 1)
            self.assertEqual(summary["declared_non_success_count"], 1)
            self.assertEqual(summary["registered_artifact_count"], 3)
            rows = object_rows(registry)
            self.assertEqual(
                [row["relative_path"] for row in rows],
                [
                    "datasets/z-valid-source",
                    "reports/m-retained-failure",
                    "runs/a-invalid-source",
                ],
            )
            by_id = {row["object_id"]: row for row in rows}
            self.assertEqual(by_id["z-valid-source"]["verification_ok"], "True")
            self.assertEqual(by_id["m-retained-failure"]["verification_ok"], "True")
            self.assertEqual(by_id["m-retained-failure"]["declared_status"], "failed")
            self.assertEqual(by_id["a-invalid-source"]["verification_ok"], "False")
            self.assertIn(
                "unregistered files",
                by_id["a-invalid-source"]["verification_error"],
            )
            with (registry / "tables" / "artifacts.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            ) as stream:
                artifacts = list(csv.DictReader(stream))
            self.assertEqual(len(artifacts), 3)
            self.assertEqual(
                {row["source_id"] for row in artifacts},
                {"a-invalid-source", "m-retained-failure", "z-valid-source"},
            )
            self.assertTrue((registry / "plots" / "root_inventory.png").is_file())
            self.assertTrue((registry / "plots" / "root_inventory.svg").is_file())
            self.assertFalse(result["descriptor"]["generation"]["carla_contacted"])
            self.assertFalse(result["descriptor"]["generation"]["simulator_mutated"])

            generic = verify_research_object(
                registry,
                reject_unregistered=True,
            )
            self.assertEqual(generic.external_reference_count, 3)
            self.assertEqual(
                generic.deep_verification,
                {
                    "kind": "evidence_registry",
                    "registry_id": "evidence-registry-test-v001",
                    "source_count": 3,
                    "verified_count": 2,
                    "verification_failed_count": 1,
                    "registered_artifact_count": 3,
                },
            )
            verified = verify_evidence_registry(registry)
            self.assertEqual(verified.registry_id, "evidence-registry-test-v001")
            self.assertEqual(verified.source_count, 3)
            self.assertEqual(verified.verification_failed_count, 1)
            self.assertEqual(verified.registered_artifact_count, 3)
            self.assertEqual(valid.name, "z-valid-source")

    def test_ordering_is_deterministic_and_existing_registries_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            build_source(workspace, root_kind="runs", run_id="z-source")
            build_source(workspace, root_kind="runs", run_id="a-source")
            first = Path(
                build_evidence_registry(
                    workspace=workspace,
                    registry_id="evidence-registry-first",
                )["registry_dir"]
            )
            first_rows = object_rows(first)
            self.assertEqual(
                [row["relative_path"] for row in first_rows],
                ["runs/a-source", "runs/z-source"],
            )

            second_result = build_evidence_registry(
                workspace=workspace,
                registry_id="evidence-registry-second",
            )
            second = Path(second_result["registry_dir"])
            self.assertEqual(second_result["summary"]["source_count"], 2)
            self.assertEqual(
                second_result["summary"]["excluded_registry_paths"],
                ["runs/evidence-registry-first"],
            )
            self.assertNotIn(
                "runs/evidence-registry-first",
                [row["relative_path"] for row in object_rows(second)],
            )
            verified = verify_evidence_registry(second)
            self.assertEqual(
                verified.config.excluded_registry_paths,
                ("runs/evidence-registry-first",),
            )
            self.assertEqual(verify_evidence_registry(first).source_count, 2)

    def test_recorded_source_drift_fails_but_later_additions_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source = build_source(workspace, root_kind="datasets", run_id="source-a")
            registry = Path(
                build_evidence_registry(
                    workspace=workspace,
                    registry_id="evidence-registry-drift",
                )["registry_dir"]
            )
            (source / "summary.json").write_text('{"changed":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                EvidenceRegistryIntegrityError,
                "does not reproduce",
            ):
                verify_evidence_registry(registry)

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            build_source(workspace, root_kind="datasets", run_id="source-a")
            registry = Path(
                build_evidence_registry(
                    workspace=workspace,
                    registry_id="evidence-registry-new-source",
                )["registry_dir"]
            )
            build_source(workspace, root_kind="models", run_id="source-b")
            build_evidence_registry(
                workspace=workspace,
                registry_id="evidence-registry-later-snapshot",
            )
            verified = verify_evidence_registry(registry)
            self.assertEqual(verified.source_count, 1)
            self.assertEqual(verified.config.source_paths, ("datasets/source-a",))

    def test_registry_payload_tampering_is_rejected_before_semantic_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            build_source(workspace, root_kind="runs", run_id="source-a")
            registry = Path(
                build_evidence_registry(
                    workspace=workspace,
                    registry_id="evidence-registry-tamper",
                )["registry_dir"]
            )
            summary = registry / "summary.json"
            summary.write_text(summary.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                EvidenceRegistryIntegrityError,
                "fingerprint mismatch",
            ):
                verify_evidence_registry(registry)

    def test_symlinked_canonical_root_is_rejected_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            workspace = base / "workspace"
            outside = base / "outside"
            workspace.mkdir()
            outside.mkdir()
            (workspace / "datasets").symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(
                RuntimeError,
                "canonical evidence root must not be a symbolic link",
            ):
                build_evidence_registry(
                    workspace=workspace,
                    registry_id="evidence-registry-canonical-symlink",
                )
            self.assertFalse((workspace / "runs").exists())
            self.assertEqual(list(outside.iterdir()), [])

    def test_source_and_manifest_symlinks_are_retained_without_dereferencing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            workspace = base / "workspace"
            outside = base / "outside"
            workspace.mkdir()
            outside.mkdir()
            outside_source = build_source(
                outside,
                root_kind="datasets",
                run_id="outside-source",
            )
            (workspace / "datasets").mkdir()
            (workspace / "datasets" / "linked-source").symlink_to(
                outside_source,
                target_is_directory=True,
            )
            (workspace / "datasets" / "dangling-source").symlink_to(
                outside / "missing-target",
                target_is_directory=True,
            )
            manifest_link_source = workspace / "models" / "manifest-link"
            manifest_link_source.mkdir(parents=True)
            (manifest_link_source / "local.txt").write_text("local\n", encoding="utf-8")
            (manifest_link_source / "manifest.json").symlink_to(outside_source / "manifest.json")

            registry = Path(
                build_evidence_registry(
                    workspace=workspace,
                    registry_id="evidence-registry-source-symlinks",
                )["registry_dir"]
            )
            rows = object_rows(registry)
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["verification_ok"] == "False" for row in rows))
            self.assertTrue(
                all(row["verification_error_type"] == "UnsafeSourcePathError" for row in rows)
            )

            tree = failed_tree_rows(registry)
            tree_index = {(row["source_relative_path"], row["path"]): row for row in tree}
            self.assertEqual(
                tree_index[("datasets/linked-source", ".")]["entry_type"],
                "symlink",
            )
            self.assertEqual(
                tree_index[("datasets/dangling-source", ".")]["entry_type"],
                "symlink",
            )
            self.assertEqual(
                tree_index[("models/manifest-link", "manifest.json")]["entry_type"],
                "symlink",
            )

            tracker_manifest = json.loads((registry / "manifest.json").read_text(encoding="utf-8"))
            input_references = tracker_manifest["references"]["inputs"]
            self.assertEqual(input_references, [])
            self.assertNotIn(str(outside), json.dumps(tracker_manifest["references"]))
            verify_evidence_registry(registry)

            (outside_source / "summary.json").write_text(
                '{"outside":"changed"}\n',
                encoding="utf-8",
            )
            verify_evidence_registry(registry)

    def test_failed_source_tree_detects_mutations_in_each_invalid_source_kind(self) -> None:
        for invalid_kind in ("unregistered", "missing_manifest"):
            with self.subTest(invalid_kind=invalid_kind):
                with tempfile.TemporaryDirectory() as temporary:
                    workspace = Path(temporary)
                    if invalid_kind == "unregistered":
                        source = build_source(
                            workspace,
                            root_kind="datasets",
                            run_id="invalid-source",
                        )
                        mutable = source / "unregistered.txt"
                    else:
                        source = workspace / "datasets" / "invalid-source"
                        source.mkdir(parents=True)
                        mutable = source / "payload.txt"
                    mutable.write_text("before\n", encoding="utf-8")
                    registry = Path(
                        build_evidence_registry(
                            workspace=workspace,
                            registry_id=(f"evidence-registry-{invalid_kind.replace('_', '-')}"),
                        )["registry_dir"]
                    )
                    verify_evidence_registry(registry)
                    row = object_rows(registry)[0]
                    self.assertEqual(row["verification_ok"], "False")
                    self.assertTrue(row["failed_tree_sha256"])

                    mutable.write_text("after with different bytes\n", encoding="utf-8")
                    with self.assertRaisesRegex(
                        EvidenceRegistryIntegrityError,
                        "does not reproduce",
                    ):
                        verify_evidence_registry(registry)

    def test_nonfinite_json_manifests_are_failed_and_raw_bytes_are_retained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            raw_by_id = {
                "nan-source": b'{"run_id":"nan-source","value":NaN}\n',
                "infinity-source": b'{"run_id":"infinity-source","value":Infinity}\n',
                "negative-infinity-source": (
                    b'{"run_id":"negative-infinity-source","value":-Infinity}\n'
                ),
            }
            for source_id, raw in raw_by_id.items():
                source = workspace / "datasets" / source_id
                source.mkdir(parents=True)
                (source / "manifest.json").write_bytes(raw)

            registry = Path(
                build_evidence_registry(
                    workspace=workspace,
                    registry_id="evidence-registry-nonfinite-json",
                )["registry_dir"]
            )
            rows = object_rows(registry)
            self.assertEqual(len(rows), 3)
            for row in rows:
                source_id = row["object_id"]
                raw = raw_by_id[source_id]
                self.assertEqual(row["verification_ok"], "False")
                self.assertEqual(row["verification_error_type"], "InvalidManifest")
                self.assertIn("non-finite JSON number", row["verification_error"])
                self.assertEqual(row["manifest_present"], "True")
                self.assertEqual(row["manifest_sha256"], hashlib.sha256(raw).hexdigest())
                self.assertEqual(row["manifest_size_bytes"], str(len(raw)))

            tracker_manifest = json.loads((registry / "manifest.json").read_text(encoding="utf-8"))
            references = tracker_manifest["references"]["inputs"]
            self.assertEqual(len(references), 3)
            self.assertEqual(
                {reference["source_id"] for reference in references},
                set(raw_by_id),
            )
            verified = verify_evidence_registry(registry)
            self.assertEqual(verified.source_count, 3)
            self.assertEqual(verified.verification_failed_count, 3)

    def test_verifier_rejects_manifest_replaced_by_symlink_before_reference_check(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source = build_source(
                workspace,
                root_kind="datasets",
                run_id="source-a",
            )
            registry = Path(
                build_evidence_registry(
                    workspace=workspace,
                    registry_id="evidence-registry-manifest-replaced",
                )["registry_dir"]
            )
            outside_manifest = workspace / "outside-manifest.json"
            (source / "manifest.json").replace(outside_manifest)
            (source / "manifest.json").symlink_to(outside_manifest)

            with self.assertRaisesRegex(
                EvidenceRegistryIntegrityError,
                "reference set changed",
            ):
                verify_evidence_registry(registry)


if __name__ == "__main__":
    unittest.main()
