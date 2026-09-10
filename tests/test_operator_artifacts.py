"""Saved-results behavior without an HTTP server, simulator, or model runtime."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from carla_vision.operator.artifacts import ArtifactStore


class ArtifactStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.directory = self.root / "runs" / "sample"
        self.directory.mkdir(parents=True)
        self.manifest: dict[str, object] = {
            "run_id": "sample",
            "status": "success",
            "invocation": {"config": {"object_type": "drive_session"}},
            "artifacts": [
                {"path": "raw.mp4", "role": "raw_video", "size_bytes": 5},
                {"path": "model.pt", "role": "weights"},
                {"path": "missing.csv", "role": "metrics"},
            ],
        }
        (self.directory / "raw.mp4").write_bytes(b"video")
        (self.directory / "model.pt").write_bytes(b"model")
        self.save_manifest()
        self.store = ArtifactStore(self.root)

    def save_manifest(self) -> None:
        (self.directory / "manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )

    def test_inspection_is_declarations_not_verification(self) -> None:
        result = self.store.inspect_research_object("runs/sample")
        self.assertEqual(result["source"], "manifest_declarations")
        self.assertEqual(result["verification"]["status"], "not_checked")
        self.assertFalse(result["verification"]["performed"])
        self.assertEqual(result["object"]["object_type"], "drive_session")
        self.assertEqual(result["object"]["artifact_count"], 3)
        self.assertEqual(result["object"]["available_artifact_count"], 2)
        self.assertEqual(result["object"]["declared_artifact_bytes"], 5)
        by_path = {item["path"]: item for item in result["artifacts"]}
        self.assertEqual(by_path["raw.mp4"]["preview_kind"], "video")
        self.assertEqual(by_path["model.pt"]["mime_type"], "application/octet-stream")
        self.assertEqual(by_path["missing.csv"]["availability"], "missing")

    def test_resolution_preserves_inline_and_download_policy(self) -> None:
        self.assertEqual(
            self.store.artifact_path("runs/sample", "raw.mp4"),
            (self.directory / "raw.mp4", False),
        )
        self.assertEqual(
            self.store.artifact_path("runs/sample", "model.pt"),
            (self.directory / "model.pt", True),
        )
        with self.assertRaises(FileNotFoundError):
            self.store.artifact_path("runs/sample", "missing.csv")
        (self.directory / "unregistered.txt").write_text("not public", encoding="utf-8")
        with self.assertRaises(KeyError):
            self.store.artifact_path("runs/sample", "unregistered.txt")

    def test_unsafe_paths_and_roots_are_rejected(self) -> None:
        for value in ("../sample", "/runs/sample", "runs/sample/nested", "secrets/sample"):
            with self.subTest(object_path=value), self.assertRaises(ValueError):
                self.store.inspect_research_object(value)
        for value in ("../raw.mp4", "%2e%2e/raw.mp4", "C:/raw.mp4", "sub\\raw.mp4"):
            with self.subTest(artifact_path=value), self.assertRaises(ValueError):
                self.store.artifact_path("runs/sample", value)

    def test_symlink_replacement_is_rechecked_after_inspection(self) -> None:
        self.store.inspect_research_object("runs/sample")
        outside = self.root / "outside.mp4"
        outside.write_bytes(b"private")
        (self.directory / "raw.mp4").unlink()
        (self.directory / "raw.mp4").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            self.store.artifact_path("runs/sample", "raw.mp4")

    def test_replaced_manifest_is_not_cached(self) -> None:
        self.store.inspect_research_object("runs/sample")
        self.manifest["artifacts"] = []
        self.save_manifest()
        with self.assertRaises(KeyError):
            self.store.artifact_path("runs/sample", "raw.mp4")

    def test_malformed_manifests_keep_explicit_errors(self) -> None:
        for value, message in (("[]", "contain an object"), ("{", "valid UTF-8 JSON")):
            (self.directory / "manifest.json").write_text(value, encoding="utf-8")
            with self.subTest(manifest=value), self.assertRaisesRegex(ValueError, message):
                self.store.inspect_research_object("runs/sample")
        self.manifest["artifacts"] = [
            {"path": "raw.mp4", "role": "video"},
            {"path": "raw.mp4", "role": "duplicate"},
        ]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "duplicate artifact"):
            self.store.inspect_research_object("runs/sample")

    def test_import_does_not_load_transport_or_ml(self) -> None:
        # A fresh interpreter proves the dependency boundary even when other
        # tests in this process have already imported the application.
        result = subprocess.run(
            [
                sys.executable, "-c",
                "import sys; from carla_vision.operator.artifacts import ArtifactStore; "
                "print(sorted(set(sys.modules) & "
                "{'carla', 'torch', 'ultralytics', 'carla_vision.operator.server', "
                "'carla_vision.operator.drive'}))",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        self.assertEqual(result.stdout.strip(), "[]")
