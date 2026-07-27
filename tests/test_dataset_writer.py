from __future__ import annotations

import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from carla_vision.bridge import CarlaImageFrame
from carla_vision.dataset import (
    CarlaSemanticTag,
    DatasetWriter,
    SynchronizedFramePair,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def camera_frame(
    sequence: int,
    carla_frame: int,
    pixels: np.ndarray,
    *,
    timestamp: float = 10.0,
) -> CarlaImageFrame:
    height, width = pixels.shape[:2]
    return CarlaImageFrame(
        sequence=sequence,
        sensor_type=1,
        frame=carla_frame,
        timestamp=timestamp,
        transform=(1.0, 2.0, 3.0, 0.0, 90.0, 0.0),
        width=width,
        height=height,
        fov=90.0,
        bgra=pixels.tobytes(),
        received_monotonic=time.monotonic(),
    )


def synchronized_pair(
    *,
    carla_frame: int = 200,
    timestamp: float = 10.0,
) -> tuple[SynchronizedFramePair, np.ndarray, np.ndarray]:
    rgb = np.zeros((6, 8, 4), dtype=np.uint8)
    rgb[:, :, :3] = (11, 22, 33)
    rgb[:, :, 3] = 255
    teacher = np.zeros_like(rgb)
    actor_id = 0x1234
    teacher[1:5, 2:7, 0] = (actor_id >> 8) & 0xFF
    teacher[1:5, 2:7, 1] = actor_id & 0xFF
    teacher[1:5, 2:7, 2] = int(CarlaSemanticTag.CAR)
    teacher[1:5, 2:7, 3] = 255
    return (
        SynchronizedFramePair(
            rgb=camera_frame(
                1,
                carla_frame,
                rgb,
                timestamp=timestamp,
            ),
            teacher=camera_frame(
                1,
                carla_frame,
                teacher,
                timestamp=timestamp,
            ),
            rgb_skipped=2,
            teacher_skipped=1,
        ),
        rgb,
        teacher,
    )


class DatasetWriterTests(unittest.TestCase):
    def test_release_contains_lossless_sources_coco_yolo_and_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pair, rgb, teacher = synchronized_pair()
            with DatasetWriter(
                root / "datasets",
                dataset_id="ds-pilot-v001",
                carla_endpoint={"host": "172.20.10.7", "port": 2000},
                carla_version="0.9.16",
                carla_map="Town10HD_Opt",
                repository_root=root,
                minimum_pixels=4,
            ) as writer:
                sample = writer.add_pair(
                    pair,
                    split="train",
                    scenario_id="scn-town10-clear",
                    episode_id="ep-001",
                    context={"weather_recipe": "clear-noon"},
                )
                writer.add_auxiliary_json(
                    "episodes/ep-001.json",
                    {"episode_id": "ep-001", "actors": [24, 25]},
                    role="episode_provenance",
                    metadata={"episode_id": "ep-001"},
                )

            dataset_dir = root / "datasets" / "ds-pilot-v001"
            self.assertEqual(sample.sample_id, "00000001")
            self.assertEqual(sample.annotation_count, 1)
            saved_rgb = cv2.imread(
                str(dataset_dir / sample.rgb.path),
                cv2.IMREAD_UNCHANGED,
            )
            saved_teacher = cv2.imread(
                str(dataset_dir / sample.instance_mask.path),
                cv2.IMREAD_UNCHANGED,
            )
            self.assertTrue(np.array_equal(saved_rgb, rgb[:, :, :3]))
            self.assertTrue(np.array_equal(saved_teacher, teacher))

            coco = json.loads((dataset_dir / "annotations" / "instances.coco.json").read_text())
            self.assertEqual(coco["images"][0]["carla_frame"], 200)
            self.assertEqual(coco["categories"][4]["name"], "car")
            self.assertEqual(
                coco["annotations"][0],
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 5,
                    "bbox": [2, 1, 5, 4],
                    "area": 20,
                    "iscrowd": 0,
                    "attributes": {
                        "carla_actor_id": 0x1234,
                        "carla_semantic_tag": 14,
                        "visible_area_pixels": 20,
                        "bbox_area_pixels": 20,
                        "mask_fill_ratio": 1.0,
                        "truncated": False,
                        "teacher_method": "instance_segmentation_visible_mask",
                    },
                },
            )
            yolo = (dataset_dir / sample.yolo_label.path).read_text().strip()
            self.assertEqual(yolo, "4 0.56250000 0.50000000 0.62500000 0.66666667")

            metadata = json.loads((dataset_dir / sample.metadata.path).read_text())
            self.assertTrue(metadata["teacher"]["privileged"])
            self.assertTrue(metadata["synchronization"]["exact_carla_frame_match"])
            self.assertEqual(
                metadata["teacher"]["encoding"]["actor_id_low_byte"],
                "G",
            )

            checksum_entries = {}
            for line in (dataset_dir / "checksums.sha256").read_text().splitlines():
                checksum, relative_path = line.split("  ", maxsplit=1)
                checksum_entries[relative_path] = checksum
            self.assertEqual(len(checksum_entries), 8)
            self.assertIn("episodes/ep-001.json", checksum_entries)
            for relative_path, expected in checksum_entries.items():
                self.assertEqual(digest(dataset_dir / relative_path), expected)

            dataset = json.loads((dataset_dir / "dataset.json").read_text())
            self.assertEqual(dataset["status"], "complete")
            self.assertEqual(dataset["sample_count"], 1)
            self.assertEqual(dataset["annotation_count"], 1)
            self.assertEqual(dataset["split_counts"], {"train": 1})
            self.assertEqual(dataset["category_counts"]["4"], 1)
            self.assertEqual(
                dataset["checksum_index"]["sha256"],
                digest(dataset_dir / "checksums.sha256"),
            )

            manifest = json.loads((dataset_dir / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "success")
            self.assertEqual(
                {artifact["path"] for artifact in manifest["artifacts"]},
                {
                    "annotations/instances.coco.json",
                    "checksums.sha256",
                    "data.yaml",
                    "dataset.json",
                    "episodes/ep-001.json",
                    "ontology.json",
                },
            )
            for artifact in manifest["artifacts"]:
                path = dataset_dir / artifact["path"]
                self.assertEqual(artifact["sha256"], digest(path))

    def test_pair_and_episode_leakage_guards_fail_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pair, _, _ = synchronized_pair()
            with self.assertRaisesRegex(ValueError, "same CARLA frame"):
                bad_teacher = camera_frame(
                    1,
                    201,
                    pair.teacher.bgra_array(),
                    timestamp=10.0,
                )
                with DatasetWriter(
                    root / "bad",
                    dataset_id="bad-pair",
                    carla_endpoint={"host": "localhost", "port": 2000},
                    carla_version="0.9.16",
                    carla_map="Town10HD_Opt",
                    repository_root=root,
                ) as writer:
                    writer.add_pair(
                        SynchronizedFramePair(
                            rgb=pair.rgb,
                            teacher=bad_teacher,
                            rgb_skipped=0,
                            teacher_skipped=0,
                        ),
                        split="train",
                        scenario_id="scn",
                        episode_id="ep",
                    )

            first, _, _ = synchronized_pair(carla_frame=300, timestamp=20.0)
            second, _, _ = synchronized_pair(carla_frame=301, timestamp=20.1)
            with self.assertRaisesRegex(ValueError, "cannot cross dataset splits"):
                with DatasetWriter(
                    root / "leakage",
                    dataset_id="split-leak",
                    carla_endpoint={"host": "localhost", "port": 2000},
                    carla_version="0.9.16",
                    carla_map="Town10HD_Opt",
                    repository_root=root,
                    minimum_pixels=4,
                ) as writer:
                    writer.add_pair(
                        first,
                        split="train",
                        scenario_id="scn",
                        episode_id="ep",
                    )
                    writer.add_pair(
                        second,
                        split="test",
                        scenario_id="scn",
                        episode_id="ep",
                    )

            leakage_manifest = json.loads(
                (root / "leakage" / "split-leak" / "manifest.json").read_text()
            )
            self.assertEqual(leakage_manifest["status"], "failed")

    def test_frame_ids_may_repeat_after_world_reload_in_distinct_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first, _, _ = synchronized_pair(carla_frame=10, timestamp=0.5)
            second, _, _ = synchronized_pair(carla_frame=10, timestamp=0.5)
            with DatasetWriter(
                root / "datasets",
                dataset_id="multi-episode",
                carla_endpoint={"host": "localhost", "port": 2000},
                carla_version="0.9.16",
                carla_map="multi-map",
                repository_root=root,
                minimum_pixels=4,
            ) as writer:
                writer.add_pair(
                    first,
                    split="train",
                    scenario_id="scn-a",
                    episode_id="ep-a",
                )
                writer.add_pair(
                    second,
                    split="test_map_ood",
                    scenario_id="scn-b",
                    episode_id="ep-b",
                )

            dataset_dir = root / "datasets" / "multi-episode"
            dataset = json.loads((dataset_dir / "dataset.json").read_text())
            self.assertEqual(
                dataset["split_counts"],
                {"test_map_ood": 1, "train": 1},
            )
            data_yaml = (dataset_dir / "data.yaml").read_text()
            self.assertIn("  - images/train", data_yaml)
            self.assertIn("  - images/test_map_ood", data_yaml)


if __name__ == "__main__":
    unittest.main()
