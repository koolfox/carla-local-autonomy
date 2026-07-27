from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from carla_vision.artifacts import RunArtifactTracker
from carla_vision.contracts import PerceptionResult
from carla_vision.policy import (
    ShadowPolicyRunner,
    VisionPolicyConfig,
    create_vision_policy,
    load_verified_vision_shadow_run,
)
from carla_vision.verification import ArtifactIntegrityError, verify_research_object


def _result(sequence: int) -> PerceptionResult:
    return PerceptionResult(
        sequence=sequence,
        carla_frame=100 + sequence,
        source_timestamp=float(sequence) / 10.0,
        source_received_monotonic=10.0 + sequence,
        inference_started_monotonic=10.01 + sequence,
        completed_monotonic=10.02 + sequence,
        detections=(),
        source_bgr=np.full((40, 60, 3), sequence, dtype=np.uint8),
        detector_name="fake",
        source_transform=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        source_fov=90.0,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rehash_manifest_artifact(run_dir: Path, relative: str) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = run_dir / relative
    for artifact in manifest["artifacts"]:
        if artifact["path"] == relative:
            artifact["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
            artifact["size_bytes"] = target.stat().st_size
            break
    else:
        raise AssertionError(f"artifact not found: {relative}")
    _write_json(manifest_path, manifest)


class VisionShadowVerificationTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        tracker = RunArtifactTracker(
            root / "runs",
            run_id="shadow-test-v1",
            config={
                "runtime_sensor_contract": "front_monocular_rgb_only",
                "control_mode": "none",
                "shadow_policy": {"backend": "hazard-stop"},
            },
            package_names=(),
        )
        policy = create_vision_policy(VisionPolicyConfig(backend="hazard-stop"))
        runner = ShadowPolicyRunner(policy)
        records: list[dict[str, object]] = []
        detections: list[dict[str, object]] = []
        for sequence in (1, 2):
            proposal, record = runner.propose(_result(sequence))
            records.append(record)
            detections.append(
                {
                    "sequence": sequence,
                    "decision_context": {
                        "vision_shadow": {
                            **proposal.as_dict(),
                            "actuation_applied": False,
                        }
                    },
                }
            )
        stats = runner.stats().as_dict()
        audit = runner.audit.as_dict()
        runner.close()

        with tracker:
            audit_path = tracker.artifact_path("policy_input_audit.json")
            proposals_path = tracker.artifact_path("logs/policy_shadow.jsonl")
            detections_path = tracker.artifact_path("logs/detections.jsonl")
            summary_path = tracker.artifact_path("summary.json")
            _write_json(audit_path, audit)
            _write_jsonl(proposals_path, records)
            _write_jsonl(detections_path, detections)
            _write_json(
                summary_path,
                {
                    "run_id": tracker.run_id,
                    "status": "success",
                    "runtime_sensor_contract": "front_monocular_rgb_only",
                    "vision_shadow_enabled": True,
                    "vision_shadow_actuation_authorized": False,
                    "vision_shadow_stats": stats,
                },
            )
            tracker.register_artifact(
                audit_path,
                role="vision_policy_input_audit",
            )
            tracker.register_artifact(
                proposals_path,
                role="vision_shadow_proposals",
            )
            tracker.register_artifact(
                detections_path,
                role="detections_jsonl",
            )
            tracker.register_artifact(summary_path, role="run_summary")
        return tracker.run_dir

    def test_shadow_run_semantics_and_generic_deep_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = self._fixture(Path(temporary_directory))
            verified = load_verified_vision_shadow_run(run_dir)
            self.assertEqual(verified.run_id, "shadow-test-v1")
            self.assertEqual(len(verified.proposal_records), 2)

            generic = verify_research_object(
                run_dir,
                reject_unregistered=True,
            )
            self.assertEqual(generic.deep_verification["kind"], "vision_shadow_run")
            self.assertFalse(generic.deep_verification["actuation_applied"])

    def test_rehashed_actuation_tampering_fails_semantic_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = self._fixture(Path(temporary_directory))
            path = run_dir / "logs" / "policy_shadow.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows[0]["actuation_applied"] = True
            _write_jsonl(path, rows)
            _rehash_manifest_artifact(run_dir, "logs/policy_shadow.jsonl")

            shallow = verify_research_object(
                run_dir,
                deep=False,
                reject_unregistered=True,
            )
            self.assertEqual(shallow.deep_verification["kind"], "skipped")
            with self.assertRaisesRegex(ArtifactIntegrityError, "malformed"):
                load_verified_vision_shadow_run(run_dir)


if __name__ == "__main__":
    unittest.main()
