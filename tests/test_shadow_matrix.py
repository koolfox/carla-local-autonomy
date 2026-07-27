from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from carla_vision.shadow import (
    ShadowMatrixConfig,
    load_verified_shadow_matrix,
    plan_or_run_shadow_matrix,
)
from carla_vision.verification import ArtifactIntegrityError, verify_research_object


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _config(weights: str, *, control: str = "none") -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "matrix_id": "shadow-matrix-test-v1",
        "title": "Vision shadow test matrix",
        "purpose": "development",
        "host": "172.20.10.7",
        "port": 2000,
        "vehicle_id": 24,
        "camera_id": 25,
        "expected_map": "Town10HD_Opt",
        "resolution": [640, 384],
        "camera_fps": 10.0,
        "camera_fov": 90.0,
        "duration_seconds": 2.0,
        "max_stale_seconds": 0.5,
        "control": control,
        "view": "none",
        "record_video": False,
        "continue_on_failure": False,
        "cells": [
            {
                "cell_id": "rtdetr-hazard",
                "label": "RT-DETR with hazard-stop shadow",
                "repetitions": 2,
                "detector": {
                    "backend": "rtdetr",
                    "weights": weights,
                    "factory": None,
                    "model_package": None,
                    "device": "cpu",
                    "image_size": 640,
                    "confidence": 0.2,
                },
                "policy": {
                    "backend": "hazard-stop",
                    "factory": None,
                    "checkpoint": None,
                    "device": "cpu",
                    "options": {"confidence": 0.4},
                },
            }
        ],
    }


def _rehash_with_checksum(run_dir: Path, relative: str) -> None:
    target = run_dir / relative
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    checksum_path = run_dir / "checksums.sha256"
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    checksum_path.write_text(
        "\n".join(
            f"{digest}  {relative}" if line.endswith(f"  {relative}") else line for line in lines
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    replacements = {
        relative: (
            digest,
            target.stat().st_size,
        ),
        "checksums.sha256": (
            hashlib.sha256(checksum_path.read_bytes()).hexdigest(),
            checksum_path.stat().st_size,
        ),
    }
    for artifact in manifest["artifacts"]:
        replacement = replacements.get(artifact["path"])
        if replacement is not None:
            artifact["sha256"], artifact["size_bytes"] = replacement
    _write_json(manifest_path, manifest)


class ShadowMatrixTests(unittest.TestCase):
    def test_contract_rejects_vision_control_and_confirmatory_loose_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "control must be none or teacher"):
            ShadowMatrixConfig.from_mapping(
                _config("weights.pt", control="vision"),
                base=Path.cwd(),
            )
        confirmatory = _config("weights.pt")
        confirmatory["purpose"] = "confirmatory"
        with tempfile.TemporaryDirectory() as temporary_directory:
            weights = Path(temporary_directory) / "weights.pt"
            weights.write_bytes(b"weights")
            confirmatory["cells"][0]["detector"]["weights"] = str(weights)
            with self.assertRaisesRegex(ValueError, "verified model packages"):
                ShadowMatrixConfig.from_mapping(
                    confirmatory,
                    base=Path(temporary_directory),
                )

    def test_plan_reproduces_commands_and_passes_deep_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            weights = root / "weights.pt"
            weights.write_bytes(b"deterministic-weights")
            config_path = root / "configs" / "shadow.json"
            _write_json(config_path, _config(str(weights)))
            result = plan_or_run_shadow_matrix(
                config_path=config_path,
                runs_root=root / "runs",
                executable="/usr/bin/python-shadow-test",
                repository_root=root,
            )
            run_dir = Path(result["run_dir"])
            verified = load_verified_shadow_matrix(run_dir)
            self.assertEqual(verified.descriptor["status"], "planned")
            self.assertEqual(len(verified.plan_rows), 2)
            for row in verified.plan_rows:
                command = row["command"]
                self.assertIn("--shadow-policy", command)
                control = command[command.index("--control") + 1]
                self.assertEqual(control, "none")
                self.assertNotEqual(control, "vision")

            generic = verify_research_object(
                run_dir,
                reject_unregistered=True,
            )
            self.assertEqual(
                generic.deep_verification["kind"],
                "vision_shadow_matrix",
            )
            self.assertEqual(generic.checksum_index_entries, 6)
            self.assertEqual(generic.external_reference_count, 2)

    def test_rehashed_command_tampering_fails_semantic_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            weights = root / "weights.pt"
            weights.write_bytes(b"weights")
            config_path = root / "shadow.json"
            _write_json(config_path, _config(str(weights)))
            result = plan_or_run_shadow_matrix(
                config_path=config_path,
                runs_root=root / "runs",
                executable="/usr/bin/python-shadow-test",
                repository_root=root,
            )
            run_dir = Path(result["run_dir"])
            plan_path = run_dir / "planned_runs.jsonl"
            rows = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["command"].extend(("--control", "vision"))
            plan_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            _rehash_with_checksum(run_dir, "planned_runs.jsonl")

            shallow = verify_research_object(
                run_dir,
                deep=False,
                reject_unregistered=True,
            )
            self.assertEqual(shallow.deep_verification["kind"], "skipped")
            with self.assertRaisesRegex(
                ArtifactIntegrityError,
                "do not reproduce",
            ):
                load_verified_shadow_matrix(run_dir)

    def test_teacher_execution_requires_explicit_motion_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            weights = root / "weights.pt"
            weights.write_bytes(b"weights")
            config_path = root / "shadow.json"
            _write_json(config_path, _config(str(weights), control="teacher"))
            with self.assertRaisesRegex(PermissionError, "acknowledge"):
                plan_or_run_shadow_matrix(
                    config_path=config_path,
                    runs_root=root / "runs",
                    execute=True,
                    acknowledge_teacher_motion=False,
                    repository_root=root,
                )


if __name__ == "__main__":
    unittest.main()
