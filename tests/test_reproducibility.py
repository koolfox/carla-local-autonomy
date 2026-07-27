from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from carla_vision.artifacts import RunArtifactTracker
from carla_vision.reproducibility import (
    CANONICAL_JSON_ALGORITHM,
    canonical_experiment_id,
    canonical_json_bytes,
    configuration_identity,
    dependency_lock_snapshot,
    safe_hardware_snapshot,
)
from carla_vision.verification import ArtifactIntegrityError, verify_research_object


def _hardware_probe() -> dict[str, object]:
    return {
        "cpu": {
            "architecture": "arm64",
            "model": "test-cpu",
            "logical_count": 8,
            "hostname": "must-not-be-retained",
        },
        "memory": {"total_bytes": 16 * 1024**3, "secret": "discard"},
        "accelerators": {
            "available": True,
            "version": "2.7.1",
            "cuda": {
                "available": False,
                "runtime_version": None,
                "device_count": 0,
                "devices": [],
            },
            "cudnn": {"available": False, "version": None},
            "mps": {"available": True, "built": True},
            "rocm_version": None,
            "determinism": {
                "deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
            },
        },
        "tools": {
            "python_compiler": "Clang test",
            "git": "git version test",
            "uv": "uv test",
            "private_tool": "discard",
        },
        "container": {"detected": False, "image_digest": None},
        "environment": {"TOKEN": "must-not-be-retained"},
    }


class ReproducibilityTests(unittest.TestCase):
    def test_canonical_configuration_is_order_independent_and_has_two_identities(self) -> None:
        first = {
            "z": [1, {"β": "دید"}],
            "run_id": "first",
            "output_dir": "runs/first",
            "a": 0.0,
        }
        second = {
            "a": -0.0,
            "output_dir": "different",
            "run_id": "second",
            "z": [1, {"β": "دید"}],
        }
        first_identity = configuration_identity(first)
        second_identity = configuration_identity(second)

        self.assertEqual(
            canonical_json_bytes({"b": 2, "a": 1}),
            b'{"a":1,"b":2}',
        )
        self.assertEqual(
            first_identity["canonicalization"],
            CANONICAL_JSON_ALGORITHM,
        )
        self.assertNotEqual(
            first_identity["resolved_sha256"],
            second_identity["resolved_sha256"],
        )
        self.assertEqual(
            first_identity["identity_sha256"],
            second_identity["identity_sha256"],
        )

    def test_canonical_experiment_id_uses_controlled_stage_model_config_and_seed(self) -> None:
        started = datetime(2026, 7, 26, 12, 34, 56, tzinfo=timezone.utc)
        config = {"epochs": 10, "master_seed": 42}
        digest = configuration_identity(config)["identity_sha256"]
        experiment_id = canonical_experiment_id(
            started_at=started,
            stage="train",
            model="RT-DETR L",
            config=config,
            seed=42,
        )
        self.assertEqual(
            experiment_id,
            f"exp-20260726T123456Z-train-rt-detr-l-{digest[:8]}-s42",
        )
        with self.assertRaisesRegex(ValueError, "unsupported experiment stage"):
            canonical_experiment_id(
                started_at=started,
                stage="unknown",
                model="model",
                config=config,
                seed=42,
            )

    def test_lock_and_hardware_snapshots_are_hashed_and_allow_listed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            payload = b"version = 1\n"
            (root / "uv.lock").write_bytes(payload)
            locks = dependency_lock_snapshot(root)
            self.assertEqual(
                locks,
                [
                    {
                        "relative_path": "uv.lock",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size_bytes": len(payload),
                    }
                ],
            )
            hardware = safe_hardware_snapshot(_hardware_probe)
            serialized = json.dumps(hardware)
            self.assertNotIn("hostname", serialized)
            self.assertNotIn("TOKEN", serialized)
            self.assertNotIn("private_tool", serialized)
            self.assertEqual(hardware["cpu"]["logical_count"], 8)
            self.assertTrue(hardware["accelerators"]["mps"]["available"])

    def test_tracker_can_allocate_and_verify_a_canonical_experiment_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "uv.lock").write_text("lock\n", encoding="utf-8")
            created = datetime(2026, 7, 26, 12, 0, 0, tzinfo=timezone.utc)
            values = iter(
                (
                    created,
                    datetime(2026, 7, 26, 12, 0, 1, tzinfo=timezone.utc),
                    datetime(2026, 7, 26, 12, 0, 2, tzinfo=timezone.utc),
                )
            )
            config = {"epochs": 2, "master_seed": 7, "output_dir": "ignored"}
            tracker = RunArtifactTracker(
                root / "runs",
                config=config,
                repository_root=root,
                package_names=(),
                clock=lambda: next(values),
                hardware_probe=_hardware_probe,
                experiment_stage="train",
                experiment_model="rtdetr-l",
                master_seed=7,
            )
            expected_id = canonical_experiment_id(
                started_at=created,
                stage="train",
                model="rtdetr-l",
                config=config,
                seed=7,
            )
            self.assertEqual(tracker.run_id, expected_id)
            with tracker:
                pass
            result = verify_research_object(
                tracker.run_dir,
                reject_unregistered=True,
            )
            self.assertEqual(result.run_id, expected_id)
            manifest = tracker.read_manifest()
            self.assertEqual(
                manifest["reproducibility"]["configuration"],
                configuration_identity(manifest["invocation"]["config"]),
            )
            self.assertEqual(
                manifest["reproducibility"]["dependency_locks"][0]["relative_path"],
                "uv.lock",
            )

            manifest["invocation"]["config"]["epochs"] = 3
            tracker.manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ArtifactIntegrityError,
                "configuration identity",
            ):
                verify_research_object(tracker.run_dir)


if __name__ == "__main__":
    unittest.main()
