from __future__ import annotations

import io
import unittest
from unittest.mock import Mock, patch

from carla_vision.dataset.collector import _safe_destroy, parse_args


class DatasetCollectorParsingTests(unittest.TestCase):
    def test_defaults_preserve_unassigned_pilot_and_rgb_only_runtime_boundary(
        self,
    ) -> None:
        args = parse_args(["--dataset-id", "ds-pilot"])

        self.assertEqual(args.dataset_id, "ds-pilot")
        self.assertEqual(args.resolution, (1280, 720))
        self.assertEqual(args.control, "none")
        self.assertEqual(args.split, "unassigned")
        self.assertEqual(args.samples, 100)

    def test_explicit_teacher_and_capture_contract_parse(self) -> None:
        args = parse_args(
            [
                "--dataset-id",
                "ds-teacher",
                "--resolution",
                "640x384",
                "--samples",
                "5",
                "--sample-every",
                "3",
                "--control",
                "teacher",
                "--split",
                "train",
                "--scenario-id",
                "scn-001",
                "--episode-id",
                "ep-001",
            ]
        )

        self.assertEqual(args.resolution, (640, 384))
        self.assertEqual(args.samples, 5)
        self.assertEqual(args.sample_every, 3)
        self.assertEqual(args.control, "teacher")
        self.assertEqual(args.split, "train")

    def test_invalid_collection_limits_are_rejected(self) -> None:
        for arguments in (
            ["--dataset-id", "ds", "--samples", "0"],
            ["--dataset-id", "ds", "--sample-every", "0"],
            ["--dataset-id", "ds", "--pair-timeout", "0"],
            ["--dataset-id", "ds", "--minimum-pixels", "0"],
            ["--dataset-id", "ds", "--cruise-speed", "6"],
        ):
            with self.subTest(arguments=arguments):
                with (
                    self.assertRaises(SystemExit),
                    patch("sys.stderr", new=io.StringIO()),
                ):
                    parse_args(arguments)


class CleanupTests(unittest.TestCase):
    def test_destroy_records_failure_without_masking_other_cleanup(self) -> None:
        rpc = Mock()
        rpc.destroy_actor.side_effect = RuntimeError("actor busy")
        errors: list[dict[str, object]] = []

        _safe_destroy(rpc, 42, errors)
        _safe_destroy(rpc, None, errors)

        rpc.destroy_actor.assert_called_once_with(42)
        self.assertEqual(errors[0]["operation"], "destroy_actor")
        self.assertEqual(errors[0]["actor_id"], 42)
        self.assertIn("actor busy", str(errors[0]["message"]))


if __name__ == "__main__":
    unittest.main()
