"""Experimental voxel consumers may not silently select on train or evaluate test."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("flow", [False, True])
def test_voxel_real_training_requires_validation_without_opening_test(tmp_path, monkeypatch, flow):
    from carla_vision.voxel.training import flow_runner, runner

    module = flow_runner if flow else runner
    seen = []

    class Data:
        def __init__(self, *args, partition, **kwargs):
            seen.append(partition)
            assert partition != "test", "training opened the locked test"
            self.partition = partition

        def __len__(self):
            return 1 if self.partition == "train" else 0

    monkeypatch.setattr(
        module, "TemporalVoxelFlowDataset" if flow else "TemporalVoxelDataset", Data
    )
    args = module.build_parser().parse_args(["--dataset", str(tmp_path), "--device", "cpu"])
    with pytest.raises(RuntimeError, match="validation"):
        module.run(args)
    assert seen == ["train", "val"]


def test_voxel_configuration_serializes_nested_paths_and_rejects_nan(tmp_path):
    from carla_vision.voxel.training.runner import _write_json

    path = tmp_path / "config.json"
    _write_json(path, {"dataset": [tmp_path / "data"]})
    assert json.loads(path.read_text())["dataset"] == [str(tmp_path / "data")]
    with pytest.raises(ValueError):
        _write_json(tmp_path / "invalid.json", {"rate": float("nan")})


def test_flow_evaluates_selected_best_model_on_validation_only(tmp_path, monkeypatch):
    from carla_vision.voxel.contracts import VoxelGridSpec
    from carla_vision.voxel.training import flow_runner as module

    spec = VoxelGridSpec(x_min=0, x_max=2, y_min=0, y_max=2, z_min=0, z_max=2, resolution=1)
    seen, evaluated = [], []

    class Data:
        def __init__(self, *args, partition, **kwargs):
            seen.append(partition)
            assert partition != "test"
            self.partition, self.spec = partition, spec

        def __len__(self):
            return 1

    class Tiny(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.value = torch.nn.Parameter(torch.tensor(0.0))
            self.parameter_count = 1

    def epoch(model, loader, *, optimizer, **kwargs):
        if optimizer is not None:
            with torch.no_grad():
                model.value.add_(1)
        else:
            evaluated.append((loader.partition, model.value.item()))
        return {"mean_dynamic_iou": 1.0 if model.value.item() == 1 else 0.0}

    monkeypatch.setattr(module, "TemporalVoxelFlowDataset", Data)
    monkeypatch.setattr(module, "TemporalVoxelFlowNet", Tiny)
    monkeypatch.setattr(
        module, "_loader", lambda data, **kw: SimpleNamespace(partition=data.partition)
    )
    monkeypatch.setattr(module, "_run_epoch", epoch)
    monkeypatch.setattr(module, "_inference_latency", lambda *a, **kw: {"synthetic_test": True})
    monkeypatch.setattr(module, "split_summary", lambda *a, **kw: {"synthetic_test": True})
    args = module.build_parser().parse_args(
        [
            "--dataset",
            str(tmp_path),
            "--device",
            "cpu",
            "--epochs",
            "2",
            "--output",
            str(tmp_path / "output"),
        ]
    )
    result = module.run(args)
    assert seen == ["train", "val"]
    assert evaluated == [("val", 1.0), ("val", 2.0), ("val", 1.0)]
    assert result["evaluation_partition"] == "validation"
    assert result["test_evaluation_completed"] is False
    assert result["best_epoch"] == 1
    assert (Path(result["output"]) / "config.json").exists()


@pytest.mark.parametrize("flag,value", [("--learning-rate", "nan"), ("--positive-weight", "inf")])
def test_voxel_nonfinite_settings_fail_before_model_execution(flag, value):
    from carla_vision.voxel.training.runner import build_parser, run

    args = build_parser().parse_args(["--dry-run", "--device", "cpu", flag, value])
    with pytest.raises(ValueError, match="finite"):
        run(args)
