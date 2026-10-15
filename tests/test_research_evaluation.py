"""Metric and frozen-operating-point regression tests, all with synthetic inputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from research_fixtures import rehash_dataset
from test_dataset_writer import synchronized_pair
from test_thresholds import _dataset, _evaluation_config, _ScoredDetector, _selection_config

from carla_vision.contracts import DetectorConfig
from carla_vision.dataset.instance_labels import extract_instance_labels
from carla_vision.dataset.verified import DatasetIntegrityError, load_verified_dataset
from carla_vision.dataset.writer import DatasetWriter
from carla_vision.evaluation.runner import run_evaluation
from carla_vision.evaluation.timing import measured_inference
from carla_vision.evaluation.verified import load_verified_evaluation
from carla_vision.imitation.runner import _run_epoch
from carla_vision.replay.runner import run_replay
from carla_vision.thresholds.selector import run_threshold_selection


def test_golden_mask_expected_box_and_area_are_hand_derived():
    mask = np.zeros((6, 9, 4), dtype=np.uint8)
    mask[1:3, 2:5, 2] = 14  # CARLA car tag, six visible pixels
    mask[1:3, 2:5, 1] = 7
    labels = extract_instance_labels(
        mask, minimum_pixels=1, minimum_box_width=1, minimum_box_height=1
    )
    assert len(labels) == 1
    assert labels[0].bbox_xyxy == (2, 1, 5, 3)
    assert labels[0].visible_area_pixels == 6
    assert labels[0].render_instance_id == 7
    assert labels[0].category_name == "car"


@pytest.mark.parametrize("fault", ["box", "category", "area", "timestamp", "sync_flag"])
def test_semantic_faults_rejected_even_after_all_hashes_are_recomputed(tmp_path, fault):
    with DatasetWriter(
        tmp_path,
        dataset_id="ds",
        carla_endpoint={},
        carla_version="fixture",
        carla_map="fixture",
        minimum_pixels=1,
    ) as writer:
        pair, _, _ = synchronized_pair()
        writer.add_pair(pair, split="train", scenario_id="s", episode_id="e")
    root = tmp_path / "ds"
    coco_path = root / "annotations" / "instances.coco.json"
    coco = json.loads(coco_path.read_text())
    metadata_path = root / "metadata" / "00000001.json"
    meta = json.loads(metadata_path.read_text())
    if fault == "box":
        coco["annotations"][0]["bbox"][0] += 1
    elif fault == "category":
        coco["annotations"][0]["category_id"] = 10
    elif fault == "area":
        coco["annotations"][0]["area"] += 1
    elif fault == "timestamp":
        meta["simulation_timestamp_seconds"] += 1
    else:
        meta["synchronization"]["exact_carla_frame_match"] = False
    coco_path.write_text(json.dumps(coco))
    metadata_path.write_text(json.dumps(meta))
    rehash_dataset(root)
    with pytest.raises(DatasetIntegrityError):
        load_verified_dataset(root)


def test_device_barriers_bracket_exact_adapter_call(monkeypatch):
    from carla_vision.evaluation import timing

    events = []
    monkeypatch.setattr(
        timing, "synchronize_device", lambda device: events.append(("sync", device))
    )
    result, seconds = measured_inference(
        lambda: events.append(("infer", None)) or "result", device="cuda:0"
    )
    assert result == "result" and seconds >= 0
    assert events == [("sync", "cuda:0"), ("infer", None), ("sync", "cuda:0")]


def test_epoch_rmse_and_brake_ratios_are_not_averages_of_batch_metrics():
    class Fixed(torch.nn.Module):
        def forward(self, image, speed):
            return {"steer": image[:, 0, 0, 0], "longitudinal": image[:, 1, 0, 0]}

    images = torch.zeros(3, 3, 1, 1)
    images[:, 0, 0, 0] = torch.tensor([0.0, 0.0, 0.9])
    images[:, 1, 0, 0] = torch.tensor([-0.1, -0.1, 0.5])
    targets = torch.tensor([[0.0, -0.1], [0.0, 0.5], [0.0, 0.5]])

    def batches(n):
        return [
            {
                "image": images[i : i + n],
                "speed": torch.zeros(len(images[i : i + n]), 1),
                "target": targets[i : i + n],
            }
            for i in range(0, 3, n)
        ]

    kwargs = dict(
        device=torch.device("cpu"),
        optimizer=None,
        steer_weight=2.0,
        longitudinal_weight=1.0,
        high_steer_boost=1.5,
        brake_boost=2.0,
        max_batches=None,
    )
    one = _run_epoch(Fixed(), batches(1), **kwargs)
    two = _run_epoch(Fixed(), batches(2), **kwargs)
    for metric in ("steer_rmse", "brake_precision", "brake_recall"):
        assert one[metric] == pytest.approx(two[metric])
    assert one["steer_rmse"] == pytest.approx((0.81 / 3) ** 0.5)
    assert one["brake_precision"] == 0.5


class SeparatedDetector(_ScoredDetector):
    correct = dict.fromkeys((20, 40, 60, 80), 0.9)
    false = dict.fromkeys((20, 40, 60, 80), 0.6)


def test_replay_consumes_each_models_verified_operating_point(tmp_path):
    dataset = _dataset(tmp_path)
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(_evaluation_config()))
    specs = []
    for name, detector_type in [("a", _ScoredDetector), ("b", SeparatedDetector)]:
        cfg = DetectorConfig(backend="custom", factory=f"unused:{name}", image_size=640)
        evaluation = run_evaluation(
            config_path=evaluation_path,
            dataset_path=dataset,
            detector_config=cfg,
            runs_root=tmp_path / "runs",
            run_id=f"eval-{name}",
            detector_override=detector_type(),
        )
        sel_path = tmp_path / f"selection-{name}.json"
        sel_path.write_text(json.dumps(_selection_config() | {"selection_id": f"selection-{name}"}))
        selection = run_threshold_selection(
            config_path=sel_path, evaluation_run=evaluation["run_dir"], runs_root=tmp_path / "runs"
        )
        specs.append(
            {
                "model_id": name,
                "label": name,
                "device": "cpu",
                "model_package": None,
                "threshold_selection": selection["run_dir"],
                "detector": {
                    "backend": "custom",
                    "factory": f"unused:{name}",
                    "image_size": 640,
                    "weights": None,
                    "options": {},
                },
            }
        )
    replay_cfg = {
        "schema_version": "1.0",
        "replay_id": "paired",
        "title": "two frozen thresholds",
        "purpose": "development",
        "master_seed": 1,
        "reference_model_id": "a",
        "bootstrap_replicates": 10,
        "bootstrap_confidence": 0.95,
        "latency_exclude_first_images": 1,
        "montage_count": 2,
        "models": specs,
    }
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(replay_cfg))
    result = run_replay(
        config_path=path,
        dataset_path=dataset,
        evaluation_config_path=evaluation_path,
        runs_root=tmp_path / "runs",
        detector_overrides={"a": _ScoredDetector(), "b": SeparatedDetector()},
    )
    b = load_verified_evaluation(tmp_path / "runs" / "paired--b")
    assert b.config.operating_confidence == pytest.approx(0.9)
    with (Path(result["run_dir"]) / "tables" / "per_image_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert sum(int(row["fp"]) for row in rows if row["model_id"] == "b") == 0
    with pytest.raises(ValueError, match="identity differs"):
        run_evaluation(
            config_path=evaluation_path,
            dataset_path=dataset,
            detector_config=DetectorConfig(backend="custom", factory="unused:wrong"),
            threshold_selection_path=specs[0]["threshold_selection"],
            runs_root=tmp_path / "runs",
            run_id="reject",
            detector_override=Mock(),
        )
    assert not (tmp_path / "runs" / "reject").exists()


def test_acquisition_pool_integrates_evaluation_audit_and_source_partition(tmp_path):
    from test_thresholds import _pair

    from carla_vision.artifacts import fingerprint_file
    from carla_vision.dataset.lineage import scene_group_key
    from carla_vision.failure_mining.acquisition_pool import prepare_pool

    with DatasetWriter(
        tmp_path / "datasets",
        dataset_id="acquire",
        carla_endpoint={},
        carla_version="fixture",
        carla_map="fixture",
        minimum_pixels=4,
        config={"synthetic": True},
    ) as writer:
        for i, value in enumerate((20, 40, 60, 80), start=1):
            writer.add_pair(
                _pair(i, value),
                split="acquisition_pool",
                scenario_id="s",
                episode_id=f"e{i}",
                context={"light": "night" if i % 2 else "day"},
            )
    root = tmp_path / "datasets" / "acquire"
    source = load_verified_dataset(root)
    config_path = tmp_path / "acq-eval.json"
    config_path.write_text(json.dumps(_evaluation_config() | {"partitions": ["acquisition_pool"]}))
    evaluation = run_evaluation(
        config_path=config_path,
        dataset_path=root,
        detector_config=DetectorConfig(backend="custom", factory="unused:acq"),
        runs_root=tmp_path / "runs",
        run_id="acq-evaluation",
        detector_override=_ScoredDetector(),
    )
    evidence_path = tmp_path / "review-evidence.json"
    evidence_path.write_text(json.dumps({"synthetic": True, "reviewer": "fixture"}))
    records = []
    for sample in source.dataset["samples"]:
        metadata = json.loads((root / sample["metadata"]["path"]).read_text())
        records.append(
            {
                "sample_id": sample["sample_id"],
                "group_id": scene_group_key(metadata["context"], episode_id=sample["episode_id"]),
                "reviewer_id": "fixture",
                "decision": "valid",
                "evidence_ref": "audit",
            }
        )
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_manifest": source.reference["manifest"],
                "records": records,
                "evidence": {"audit": fingerprint_file(evidence_path)},
            }
        )
    )
    pool = prepare_pool(
        evaluation["run_dir"],
        review,
        estimated_action_cost=2.0,
        runs_root=tmp_path / "runs",
        run_id="pool",
    )
    rows = json.loads((Path(pool["run_dir"]) / "candidates.json").read_text())["candidates"]
    assert len(rows) == 4 and all(row["partition"] == "acquisition_pool" for row in rows)
    assert any(row["action"] == "NEW_SCENE" for row in rows)


def test_acquisition_pool_refuses_validation_before_reading_review(tmp_path):
    from carla_vision.failure_mining.acquisition_pool import prepare_pool

    root = _dataset(tmp_path)
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(_evaluation_config()))
    result = run_evaluation(
        config_path=path,
        dataset_path=root,
        detector_config=DetectorConfig(backend="custom", factory="unused:a"),
        runs_root=tmp_path / "runs",
        run_id="val",
        detector_override=_ScoredDetector(),
    )
    with pytest.raises(ValueError, match="train/acquisition_pool"):
        prepare_pool(
            result["run_dir"], tmp_path / "nonexistent-review.json", estimated_action_cost=1.0
        )


def test_writer_capacity_aborts_before_another_sample_is_written(tmp_path):
    with pytest.raises(RuntimeError, match="capacity"):
        with DatasetWriter(
            tmp_path,
            dataset_id="bounded",
            carla_endpoint={},
            carla_version="fixture",
            carla_map="fixture",
            maximum_samples=1,
        ) as writer:
            pair, _, _ = synchronized_pair()
            writer.add_pair(pair, split="train", scenario_id="s", episode_id="first")
            writer.add_pair(pair, split="train", scenario_id="s", episode_id="second")
    assert len(list((tmp_path / "bounded" / "metadata").glob("*.json"))) == 1
    assert json.loads((tmp_path / "bounded" / "manifest.json").read_text())["status"] == "failed"
