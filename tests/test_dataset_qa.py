from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from test_dataset_writer import synchronized_pair

from carla_vision.dataset import DatasetAuditError, DatasetWriter, audit_dataset


class DatasetAuditTests(unittest.TestCase):
    def create_dataset(self, root: Path) -> Path:
        with DatasetWriter(
            root / "datasets",
            dataset_id="ds-audit-source",
            carla_endpoint={"host": "localhost", "port": 2000},
            carla_version="0.9.16",
            carla_map="Town10HD_Opt",
            repository_root=root,
            minimum_pixels=4,
        ) as writer:
            pair, _, _ = synchronized_pair()
            writer.add_pair(
                pair,
                split="train",
                scenario_id="scn-clear",
                episode_id="ep-001",
            )
        return root / "datasets" / "ds-audit-source"

    def test_audit_reproduces_labels_exports_and_creates_hashed_plots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_dir = self.create_dataset(root)

            result = audit_dataset(
                dataset_dir,
                runs_root=root / "runs",
                run_id="dataset-audit",
                montage_count=1,
                repository_root=root,
            )

            summary = result["summary"]
            self.assertTrue(summary["valid"])
            self.assertEqual(summary["samples"], 1)
            self.assertEqual(summary["annotations"], 1)
            self.assertEqual(summary["checks"]["teacher_masks_reproduced"], 1)
            self.assertEqual(summary["checks"]["yolo_exports_reproduced"], 1)
            self.assertEqual(summary["checks"]["checksum_entries_verified"], 7)
            self.assertEqual(summary["class_frequency"], {"car": 1})
            run_dir = Path(result["run_dir"])
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "success")
            self.assertEqual(len(manifest["artifacts"]), 5)
            for name in (
                "plots/class_frequency.png",
                "plots/bbox_area_distribution.png",
                "plots/qa_montage.png",
            ):
                self.assertEqual(
                    (run_dir / name).read_bytes()[:8],
                    b"\x89PNG\r\n\x1a\n",
                )

    def test_checksum_tampering_is_rejected_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_dir = self.create_dataset(root)
            image = dataset_dir / "images" / "train" / "00000001.png"
            image.write_bytes(image.read_bytes() + b"tamper")
            runs_root = root / "runs"

            with self.assertRaisesRegex(DatasetAuditError, "checksum mismatch"):
                audit_dataset(
                    dataset_dir,
                    runs_root=runs_root,
                    run_id="must-not-exist",
                    repository_root=root,
                )

            self.assertFalse(runs_root.exists())


if __name__ == "__main__":
    unittest.main()
