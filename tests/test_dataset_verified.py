from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from carla_vision.bridge import CarlaImageFrame
from carla_vision.dataset.sync import SynchronizedFramePair
from carla_vision.dataset.verified import DatasetIntegrityError, load_verified_dataset
from carla_vision.dataset.writer import DatasetWriter


def frame(
    sequence: int,
    carla_frame: int,
    pixels: np.ndarray,
    *,
    timestamp: float,
) -> CarlaImageFrame:
    height, width, _ = pixels.shape
    return CarlaImageFrame(
        sequence=sequence,
        sensor_type=0,
        frame=carla_frame,
        timestamp=timestamp,
        transform=(1.0, 2.0, 3.0, 0.0, 10.0, 0.0),
        width=width,
        height=height,
        fov=90.0,
        bgra=pixels.tobytes(),
        received_monotonic=timestamp,
    )


def pair(
    sequence: int,
    carla_frame: int,
    *,
    rgb_value: int,
) -> SynchronizedFramePair:
    rgb = np.full((6, 8, 4), rgb_value, dtype=np.uint8)
    rgb[:, :, 3] = 255
    teacher = np.zeros((6, 8, 4), dtype=np.uint8)
    teacher[1:5, 2:7, 2] = 0x12
    teacher[1:5, 2:7, 1] = 0x34
    teacher[1:5, 2:7, 0] = 14
    teacher[:, :, 3] = 255
    timestamp = carla_frame / 20.0
    return SynchronizedFramePair(
        rgb=frame(sequence, carla_frame, rgb, timestamp=timestamp),
        teacher=frame(sequence, carla_frame, teacher, timestamp=timestamp),
        rgb_skipped=0,
        teacher_skipped=0,
    )


def build_dataset(root: Path, *, duplicate_rgb: bool = False) -> Path:
    datasets_root = root / "datasets"
    with DatasetWriter(
        datasets_root,
        dataset_id="ds-verified",
        carla_endpoint={"host": "localhost", "port": 2000},
        carla_version="0.9.16",
        carla_map="multi-map",
        repository_root=root,
        minimum_pixels=4,
    ) as writer:
        writer.add_pair(
            pair(1, 10, rgb_value=20),
            split="train",
            scenario_id="scn-train",
            episode_id="ep-train",
        )
        writer.add_pair(
            pair(2, 10, rgb_value=20 if duplicate_rgb else 30),
            split="val_seen",
            scenario_id="scn-val",
            episode_id="ep-val",
        )
    return datasets_root / "ds-verified"


class VerifiedDatasetTests(unittest.TestCase):
    def test_native_release_requires_declared_teacher_control_on_every_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with DatasetWriter(
                root,
                dataset_id="ds-native-missing-control",
                carla_endpoint={"host": "localhost", "port": 2000},
                carla_version="0.9.16",
                carla_map="Town10HD_Opt",
                repository_root=root,
                minimum_pixels=4,
            ) as writer:
                writer.add_pair(
                    pair(1, 10, rgb_value=20),
                    split="train",
                    scenario_id="scn",
                    episode_id="ep",
                )
                writer.set_release_metadata(
                    {
                        "collector": "native_official_pythonapi",
                        "privileged_teacher_control": {
                            "schema_version": "1.0",
                            "available_for_every_sample": True,
                            "source": "carla.Vehicle.get_control",
                            "sample_alignment": (
                                "queried_after_exact_sensor_frame_before_sample_write"
                            ),
                            "runtime_model_input": False,
                        },
                    }
                )

            with self.assertRaisesRegex(
                DatasetIntegrityError,
                "teacher control fields",
            ):
                load_verified_dataset(root / "ds-native-missing-control")

    def test_complete_release_verifies_and_resolves_training_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_dir = build_dataset(Path(temporary))
            verified = load_verified_dataset(dataset_dir)
            self.assertEqual(verified.dataset_id, "ds-verified")
            self.assertEqual(verified.sample_count, 2)
            self.assertEqual(verified.partitions, {"train": 1, "val_seen": 1})
            self.assertEqual(
                verified.resolve_training_partitions(),
                (("train",), ("val_seen",)),
            )
            self.assertEqual(verified.categories[4], "car")
            self.assertEqual(len(verified.reference["manifest"]["sha256"]), 64)

    def test_checksum_verification_detects_changed_sample_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_dir = build_dataset(Path(temporary))
            image_path = dataset_dir / "images" / "train" / "00000001.png"
            image_path.write_bytes(image_path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(DatasetIntegrityError, "checksum mismatch"):
                load_verified_dataset(dataset_dir)

    def test_exact_rgb_content_cannot_cross_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_dir = build_dataset(Path(temporary), duplicate_rgb=True)
            with self.assertRaisesRegex(DatasetIntegrityError, "cross dataset partitions"):
                load_verified_dataset(dataset_dir)

    def test_unassigned_release_cannot_resolve_a_training_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with DatasetWriter(
                root,
                dataset_id="ds-unassigned",
                carla_endpoint={"host": "localhost", "port": 2000},
                carla_version="0.9.16",
                carla_map="Town10HD_Opt",
                repository_root=root,
                minimum_pixels=4,
            ) as writer:
                writer.add_pair(
                    pair(1, 10, rgb_value=20),
                    split="unassigned",
                    scenario_id="scn",
                    episode_id="ep",
                )
            verified = load_verified_dataset(root / "ds-unassigned")
            with self.assertRaisesRegex(DatasetIntegrityError, "training partitions"):
                verified.resolve_training_partitions()


if __name__ == "__main__":
    unittest.main()
