"""Regression gates for research validity; fixtures are NOT CARLA evidence."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from test_dataset_writer import synchronized_pair
from test_imitation_dataset import _build_dataset

from carla_vision.dataset.sync import ExactFramePairer
from carla_vision.imitation import runner
from carla_vision.imitation.dataset import BehaviorImitationDataset


def source_dataset(tmp_path, splits):
    root = _build_dataset(tmp_path / "source", groups=len(splits))
    path = root / "dataset.json"
    raw = json.loads(path.read_text())
    for sample, split in zip(raw["samples"], splits, strict=True):
        sample["split"] = split
    path.write_text(json.dumps(raw))
    return root


def test_f01_source_test_partition_cannot_be_rehashed_into_training(tmp_path):
    root = source_dataset(tmp_path, ["train", "test_map_ood", "val_seen"])
    for seed in range(8):
        data = BehaviorImitationDataset(
            [root],
            partition="train",
            split_seed=seed,
            val_fraction=0,
            test_fraction=0,
            verify=False,
        )
        assert [sample.sample_id for sample in data.samples] == ["00000001"]


def test_f01_unassigned_requires_explicit_derived_release(tmp_path):
    root = source_dataset(tmp_path, ["unassigned"])
    with pytest.raises(ValueError, match="unassigned|partition|split"):
        BehaviorImitationDataset([root], partition="train", verify=False)


def test_f01_skip_unselected_targets_before_opening_metadata(tmp_path, monkeypatch):
    root = source_dataset(tmp_path, ["train", "test_seen"])
    (root / "metadata" / "00000002.json").unlink()
    data = BehaviorImitationDataset([root], partition="train", verify=False)
    assert len(data) == 1


@pytest.mark.parametrize("with_dataset", [False, True])
def test_f05_dry_run_never_optimizes_or_saves(tmp_path, monkeypatch, with_dataset):
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run performed training or checkpoint I/O")

    monkeypatch.setattr(runner.torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(runner.torch, "save", forbidden)
    argv = ["--dry-run", "--output", str(tmp_path / "out"), "--image-size", "16x24"]
    if with_dataset:
        root = source_dataset(tmp_path, ["train", "val_seen"])
        argv += ["--dataset", str(root), "--skip-dataset-verification"]
    result = runner.run(runner.build_parser().parse_args(argv))
    assert result["dry_run"] is True
    assert not (tmp_path / "out").exists()


def test_f06_no_validation_cannot_be_called_validation_loss(tmp_path):
    root = source_dataset(tmp_path, ["train"])
    args = runner.build_parser().parse_args(
        [
            "--dataset",
            str(root),
            "--output",
            str(tmp_path / "out"),
            "--skip-dataset-verification",
            "--epochs",
            "1",
            "--image-size",
            "16x24",
        ]
    )
    with pytest.raises((ValueError, RuntimeError), match="validation"):
        runner.run(args)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_f17_pairer_rejects_nonfinite_pose_and_tolerances(bad):
    pair, _, _ = synchronized_pair()
    invalid = replace(pair.rgb, transform=(bad, *pair.rgb.transform[1:]))
    with pytest.raises(ValueError, match="finite|pose|transform"):
        ExactFramePairer()._validate_pair(invalid, pair.teacher)
    with pytest.raises(ValueError, match="finite"):
        ExactFramePairer(transform_tolerance=bad)


def test_f17_pairer_requires_six_pose_coordinates():
    pair, _, _ = synchronized_pair()
    with pytest.raises(ValueError, match="six|transform|pose"):
        ExactFramePairer()._validate_pair(
            replace(pair.rgb, transform=(0.0,)), replace(pair.teacher, transform=(0.0,))
        )


def test_f17_equivalent_wrapped_angles_are_compatible():
    pair, _, _ = synchronized_pair()
    left = replace(pair.rgb, transform=(0.0, 0.0, 0.0, 0.0, 180.0, 0.0))
    right = replace(pair.teacher, transform=(0.0, 0.0, 0.0, 0.0, -180.0, 0.0))
    ExactFramePairer()._validate_pair(left, right)


def test_f04_legacy_actions_are_never_silently_shifted(tmp_path):
    root = source_dataset(tmp_path, ["train"])
    path = root / "metadata" / "00000001.json"
    raw = json.loads(path.read_text())
    raw["context"].pop("teacher_transition")
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="timing.*unknown"):
        BehaviorImitationDataset([root], partition="train", verify=False)
    assert (
        len(
            BehaviorImitationDataset(
                [root], partition="train", verify=False, allow_legacy_action_targets=True
            )
        )
        == 1
    )


def test_f04_future_decision_is_rejected():
    from carla_vision.native.teacher_timing import ObservationActionTransition

    with pytest.raises(ValueError, match="same state frame"):
        ObservationActionTransition(10, 11, 12, "teacher")
    assert ObservationActionTransition(10, 10, 11, "teacher").resulting_frame is None


def test_f16_epoch_resume_matches_continuous_training(tmp_path):
    import torch

    root = source_dataset(tmp_path, ["train", "train", "val_seen", "train"])
    common = [
        "--dataset",
        str(root),
        "--skip-dataset-verification",
        "--image-size",
        "32x48",
        "--encoder-channels",
        "4",
        "--hidden-dim",
        "8",
        "--batch-size",
        "3",
        "--device",
        "cpu",
    ]

    def execute(name, epochs, resume=None):
        argv = common + ["--output", str(tmp_path / name), "--epochs", str(epochs)]
        if resume:
            argv += ["--resume", str(resume)]
        runner.run(runner.build_parser().parse_args(argv))
        return torch.load(
            tmp_path / name / "checkpoint_last.pt", weights_only=True, map_location="cpu"
        )

    continuous = execute("continuous", 2)
    execute("first", 1)
    resumed = execute("resumed", 2, tmp_path / "first" / "checkpoint_last.pt")
    for key, value in continuous["state_dict"].items():
        assert torch.equal(value, resumed["state_dict"][key]), key
    assert resumed["training_state"]["best_epoch"] == continuous["training_state"]["best_epoch"]
    assert (tmp_path / "resumed" / "manifest.json").exists()
    summary = json.loads((tmp_path / "resumed" / "summary.json").read_text())
    assert summary["test_evaluated"] is False


def test_source_root_duplication_cannot_silently_double_training_weight(tmp_path):
    root = source_dataset(tmp_path, ["train"])
    with pytest.raises(ValueError, match="duplicate source"):
        BehaviorImitationDataset([root, root], partition="train", verify=False)


def test_epoch_augmentation_is_repeatable_but_not_fixed_across_epochs(tmp_path):
    root = source_dataset(tmp_path, ["train"])
    dataset = BehaviorImitationDataset([root], partition="train", verify=False, augment=True)
    dataset.set_epoch(1)
    first = dataset._augmentation_rng(dataset.samples[0]).random()
    assert dataset._augmentation_rng(dataset.samples[0]).random() == first
    dataset.set_epoch(2)
    assert dataset._augmentation_rng(dataset.samples[0]).random() != first


def test_external_test_protocol_binds_model_and_rejects_group_overlap(tmp_path):
    from types import SimpleNamespace

    from carla_vision.artifacts import fingerprint_file
    from carla_vision.imitation.evaluate import validate_external_protocol

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test fixture only; never deserialize")
    test_manifest = tmp_path / "test-manifest.json"
    test_manifest.write_text("{}")
    test_ref = fingerprint_file(test_manifest)
    model = {
        "source_references": [{"sha256": "train-sha"}],
        "split_summary": {
            "group_ids_by_partition": {"train": ["train-group"]},
            "rgb_hashes_by_partition": {"train": ["train-image"]},
        },
    }
    data = SimpleNamespace(
        source_references=[test_ref],
        samples=[SimpleNamespace(route_group_key="test-group", rgb_sha256="test-image")],
    )
    path = tmp_path / "protocol.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "checkpoint": fingerprint_file(checkpoint),
                "training_manifest_sha256": ["train-sha"],
                "test_manifests": [test_ref],
                "group_policy": "disjoint_scene_roots_and_exact_rgb",
            }
        )
    )
    assert validate_external_protocol(
        path, checkpoint=checkpoint, payload=model, data=data
    ) == fingerprint_file(path)
    data.samples[0].route_group_key = "train-group"
    with pytest.raises(ValueError, match="overlaps"):
        validate_external_protocol(path, checkpoint=checkpoint, payload=model, data=data)
