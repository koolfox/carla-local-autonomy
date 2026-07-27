from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

_MODULE_PATH = Path(__file__).resolve().parents[1] / "carla_vision" / "artifacts.py"
_SPEC = importlib.util.spec_from_file_location("carla_vision_artifacts_under_test", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"could not load artifact module from {_MODULE_PATH}")
_ARTIFACTS_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ARTIFACTS_MODULE)

RunArtifactTracker = _ARTIFACTS_MODULE.RunArtifactTracker
SCHEMA_VERSION = _ARTIFACTS_MODULE.SCHEMA_VERSION


class SequenceClock:
    def __init__(self, *values: datetime) -> None:
        self._values = iter(values)

    def __call__(self) -> datetime:
        try:
            return next(self._values)
        except StopIteration as error:
            raise AssertionError("test clock was called more often than expected") from error


def at(second: int) -> datetime:
    return datetime(2026, 7, 26, 10, 0, second, 123456, tzinfo=timezone.utc)


def fixed_git_probe(_repository_root: Path) -> dict[str, object]:
    return {
        "available": True,
        "commit": "0123456789abcdef0123456789abcdef01234567",
        "dirty": True,
        "unexpected": "must not be retained",
    }


def fixed_environment_probe(package_names: object) -> dict[str, object]:
    names = tuple(package_names)
    versions = {"torch": "2.7.1", "ultralytics": "8.4.0"}
    return {
        "python": {
            "version": "3.12.4",
            "implementation": "CPython",
            "executable": "/private/python",
        },
        "platform": {
            "system": "Darwin",
            "release": "25.0",
            "machine": "arm64",
            "hostname": "private-host",
        },
        "packages": {name: versions.get(name) for name in names},
        "environment": {"SECRET_TOKEN": "must-not-be-retained"},
    }


class RunArtifactTrackerTests(unittest.TestCase):
    def test_success_manifest_tracks_provenance_and_artifact_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            clock = SequenceClock(at(0), at(1), at(2), at(3))
            secret_from_environment = "environment-secret-value"

            with patch.dict(
                os.environ,
                {"SUPER_SECRET_TOKEN": secret_from_environment},
                clear=False,
            ):
                tracker = RunArtifactTracker(
                    root / "runs",
                    run_id="pilot-run",
                    cli_args=(
                        "train",
                        "--epochs",
                        "10",
                        "--token",
                        "cli-secret",
                        "--api-key=config-secret",
                    ),
                    config={
                        "seed": 42,
                        "output": Path("outputs"),
                        "password": "configuration-secret",
                        "nested": {"access_token": "nested-secret"},
                        "wandb_api_key": "wandb-secret",
                    },
                    repository_root=root,
                    carla_endpoint={"host": "172.20.10.7", "port": 2000},
                    carla_version="0.9.16",
                    carla_map="Town10HD_Opt",
                    model_refs=({"name": "rtdetr-l", "revision": "base"},),
                    dataset_refs=({"name": "pilot", "version": "v0"},),
                    input_refs=({"name": "scenario-plan", "version": "v1"},),
                    package_names=("ultralytics", "torch"),
                    clock=clock,
                    git_probe=fixed_git_probe,
                    environment_probe=fixed_environment_probe,
                )

                self.assertEqual(tracker.run_id, "pilot-run")
                with tracker as active_run:
                    artifact = active_run.artifact_path("plots/loss.png")
                    artifact.write_bytes(b"deterministic-plot-bytes")
                    entry = active_run.register_artifact(
                        artifact,
                        role="training_curve",
                        metadata={"epoch": 10},
                    )

            manifest = json.loads(tracker.manifest_path.read_text(encoding="utf-8"))
            serialized_manifest = tracker.manifest_path.read_text(encoding="utf-8")

            self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
            self.assertEqual(manifest["run_id"], "pilot-run")
            self.assertEqual(manifest["status"], "success")
            self.assertIsNone(manifest["failure"])
            self.assertEqual(
                manifest["timestamps"],
                {
                    "created_at": "2026-07-26T10:00:00.123456Z",
                    "started_at": "2026-07-26T10:00:01.123456Z",
                    "finished_at": "2026-07-26T10:00:03.123456Z",
                    "updated_at": "2026-07-26T10:00:03.123456Z",
                },
            )
            self.assertEqual(
                manifest["git"],
                {
                    "available": True,
                    "commit": "0123456789abcdef0123456789abcdef01234567",
                    "dirty": True,
                },
            )
            self.assertEqual(
                manifest["environment"],
                {
                    "python": {
                        "version": "3.12.4",
                        "implementation": "CPython",
                    },
                    "platform": {
                        "system": "Darwin",
                        "release": "25.0",
                        "machine": "arm64",
                    },
                    "packages": {
                        "torch": "2.7.1",
                        "ultralytics": "8.4.0",
                    },
                },
            )
            self.assertEqual(
                manifest["carla"],
                {
                    "endpoint": {"host": "172.20.10.7", "port": 2000},
                    "version": "0.9.16",
                    "map": "Town10HD_Opt",
                },
            )
            self.assertEqual(manifest["references"]["models"][0]["name"], "rtdetr-l")
            self.assertEqual(manifest["references"]["datasets"][0]["name"], "pilot")
            self.assertEqual(
                manifest["references"]["inputs"][0]["name"],
                "scenario-plan",
            )
            self.assertEqual(
                manifest["invocation"]["cli_args"],
                [
                    "train",
                    "--epochs",
                    "10",
                    "--token",
                    "[REDACTED]",
                    "--api-key=[REDACTED]",
                ],
            )
            self.assertEqual(manifest["invocation"]["config"]["password"], "[REDACTED]")
            self.assertEqual(
                manifest["invocation"]["config"]["nested"]["access_token"],
                "[REDACTED]",
            )
            self.assertEqual(
                manifest["invocation"]["config"]["wandb_api_key"],
                "[REDACTED]",
            )

            expected_hash = hashlib.sha256(b"deterministic-plot-bytes").hexdigest()
            self.assertEqual(
                entry,
                {
                    "path": "plots/loss.png",
                    "role": "training_curve",
                    "sha256": expected_hash,
                    "size_bytes": 24,
                    "mime_type": "image/png",
                    "registered_at": "2026-07-26T10:00:02.123456Z",
                    "metadata": {"epoch": 10},
                },
            )
            self.assertEqual(manifest["artifacts"], [entry])
            self.assertNotIn(secret_from_environment, serialized_manifest)
            self.assertNotIn("configuration-secret", serialized_manifest)
            self.assertNotIn("nested-secret", serialized_manifest)
            self.assertNotIn("wandb-secret", serialized_manifest)
            self.assertNotIn("cli-secret", serialized_manifest)
            self.assertNotIn("config-secret", serialized_manifest)
            self.assertNotIn("private-host", serialized_manifest)
            self.assertFalse(
                list(tracker.run_dir.glob(".manifest.json.*.tmp")),
                "atomic manifest writer left a temporary file behind",
            )

    def test_failure_is_recorded_and_original_exception_is_propagated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tracker = RunArtifactTracker(
                Path(temporary_directory) / "runs",
                run_id="failed-run",
                cli_args=(),
                package_names=(),
                clock=SequenceClock(at(0), at(1), at(2)),
                git_probe=fixed_git_probe,
                environment_probe=fixed_environment_probe,
            )

            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with tracker:
                    raise RuntimeError(
                        "training failed token=do-not-store "
                        "at https://user:password@example.test/api"
                    )

            manifest = tracker.read_manifest()
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(
                manifest["timestamps"]["finished_at"], at(2).isoformat().replace("+00:00", "Z")
            )
            self.assertEqual(
                manifest["failure"],
                {
                    "type": "RuntimeError",
                    "module": "builtins",
                    "message": (
                        "training failed token=[REDACTED] at https://[REDACTED]@example.test/api"
                    ),
                },
            )
            serialized = tracker.manifest_path.read_text(encoding="utf-8")
            self.assertNotIn("do-not-store", serialized)
            self.assertNotIn("user:password", serialized)

    def test_generated_run_ids_are_unique_and_remain_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runs_root = Path(temporary_directory) / "runs"
            first = RunArtifactTracker(
                runs_root,
                cli_args=(),
                package_names=(),
                clock=SequenceClock(at(0), at(1), at(2)),
                id_factory=lambda: "aaaaaaaaaaaaaaaa",
                git_probe=fixed_git_probe,
                environment_probe=fixed_environment_probe,
            )
            second = RunArtifactTracker(
                runs_root,
                cli_args=(),
                package_names=(),
                clock=SequenceClock(at(0), at(1), at(2)),
                id_factory=lambda: "bbbbbbbbbbbbbbbb",
                git_probe=fixed_git_probe,
                environment_probe=fixed_environment_probe,
            )

            first_id = first.run_id
            second_id = second.run_id
            with first:
                pass
            with second:
                pass

            self.assertNotEqual(first_id, second_id)
            self.assertEqual(first.run_id, first_id)
            self.assertEqual(second.run_id, second_id)
            self.assertTrue((runs_root / first_id / "manifest.json").is_file())
            self.assertTrue((runs_root / second_id / "manifest.json").is_file())

            with self.assertRaises(FileExistsError):
                RunArtifactTracker(
                    runs_root,
                    run_id=first_id,
                    cli_args=(),
                    package_names=(),
                    clock=SequenceClock(at(3)),
                    git_probe=fixed_git_probe,
                    environment_probe=fixed_environment_probe,
                )

    def test_artifacts_cannot_escape_run_directory_or_register_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tracker = RunArtifactTracker(
                root / "runs",
                run_id="contained-run",
                cli_args=(),
                package_names=(),
                clock=SequenceClock(at(0), at(1), at(2)),
                git_probe=fixed_git_probe,
                environment_probe=fixed_environment_probe,
            )

            with tracker:
                with self.assertRaisesRegex(ValueError, "inside run directory"):
                    tracker.artifact_path("../outside.txt")
                outside = root / "outside.txt"
                outside.write_text("external", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "inside run directory"):
                    tracker.register_artifact(outside, role="invalid")
                with self.assertRaisesRegex(ValueError, "cannot register itself"):
                    tracker.register_artifact(
                        tracker.manifest_path,
                        role="invalid",
                    )


if __name__ == "__main__":
    unittest.main()
