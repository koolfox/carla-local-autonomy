"""Real pinned-framework smoke and fail-closed consumption-contract regressions.

Tiny random-weight fixtures test execution boundaries, not ADAS model quality.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_training import training_config

from carla_vision.training.contracts import TrainingConfig


def config(**changes):
    base = training_config(
        backend="yolo",
        custom_factory=None,
        epochs=1,
        image_size=64,
        batch_size=2,
        patience=0,
        warmup_epochs=0.0,
        extra_options={"mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0},
    )
    base.update(changes)
    return TrainingConfig.from_mapping(base)


@pytest.mark.parametrize(
    "override",
    [
        {"workers": 1},
        {"amp": True},
        {"cache": "ram"},
        {"patience": 1},
        {"warmup_epochs": 1.0},
        {"device": "0,1"},
        {"extra_options": {}},
        {"extra_options": {"mosaic": 1.0}},
        {"extra_options": {"trainer": "external"}},
        {"extra_options": {"fraction": 0.5}},
    ],
)
def test_unsupported_audited_profile_fails_before_model_loading(override):
    from carla_vision.training.ultralytics_consumption import validate_profile

    with pytest.raises(ValueError, match="audited|consumption"):
        validate_profile(config(**override), version="8.4.137")


def test_wrong_framework_version_rejected():
    from carla_vision.training.ultralytics_consumption import validate_profile

    with pytest.raises(ValueError, match="8.4.137"):
        validate_profile(config(), version="8.4.138")


def test_pinned_profile_and_final_validation_declared():
    from carla_vision.training.backends import UltralyticsTrainingBackend
    from carla_vision.training.ultralytics_consumption import validate_profile

    validate_profile(config(), version="8.4.137")
    backend = UltralyticsTrainingBackend("yolo")
    assert backend.consumption_contract_version == "1.0"
    assert backend.consumption_final_validation is True


def dataset(tmp_path):
    from test_training import sample_pair

    from carla_vision.dataset.writer import DatasetWriter

    with DatasetWriter(
        tmp_path / "data",
        dataset_id="fixture",
        minimum_pixels=4,
        carla_version="0.9.16",
        carla_map="multi-map",
        carla_endpoint={"host": "fixture", "port": 2000},
    ) as writer:
        for i, partition in enumerate(("train", "train", "val_seen", "val_seen"), 1):
            writer.add_pair(
                sample_pair(i, i, 20 + i * 20),
                split=partition,
                scenario_id=f"scene-{i}",
                episode_id=f"episode-{i}",
            )
    return tmp_path / "data" / "fixture"


def hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def tiny_weights(tmp_path, architecture):
    import yaml
    from ultralytics import RTDETR, YOLO

    model_path = tmp_path / f"tiny-{architecture}.yaml"
    spec = {
        "nc": 10,
        "backbone": [
            [-1, 1, "Conv", [8, 3, 2]],
            [-1, 1, "Conv", [16, 3, 2]],
            [-1, 1, "Conv", [32, 3, 2]],
            [-1, 1, "Conv", [32, 3, 2]],
            [-1, 1, "Conv", [32, 3, 2]],
        ],
        "head": [[[2, 3, 4], 1, "Detect", [10]]]
        if architecture == "yolo"
        else [[[2, 3, 4], 1, "RTDETRDecoder", [10, 32, 8, 2, 4, 1, 64, 0.0, None, -1, 4]]],
    }
    model_path.write_text(yaml.safe_dump(spec))
    model = (YOLO if architecture == "yolo" else RTDETR)(model_path)
    weights = tmp_path / f"tiny-{architecture}.pt"
    model.save(weights)
    return weights


@pytest.mark.parametrize("architecture", ["yolo", "rtdetr"])
def test_actual_pinned_trainer_records_optimizer_validation_and_best(
    tmp_path, monkeypatch, architecture
):
    pytest.importorskip("ultralytics")
    import torch
    from ultralytics.data import utils as data_utils
    from ultralytics.utils import checks

    from carla_vision.reproduction.consumption import verify_consumption
    from carla_vision.reproduction.study_links import freeze_split
    from carla_vision.training.runner import run_training

    # Suppress only update/font network lookups, not the trainer/model/optimizer/validator.
    monkeypatch.setattr(checks, "check_pip_update_available", lambda: None)
    monkeypatch.setattr(data_utils, "check_font", lambda *a, **k: None)
    torch.set_num_threads(2)
    root = dataset(tmp_path)
    before = hashes(root)
    split = tmp_path / "split.json"
    freeze_split([root], split)
    weights = tiny_weights(tmp_path, architecture)
    cfg = config(backend=architecture)
    source = tmp_path / "config.json"
    source.write_text(json.dumps(cfg.as_dict()))
    result = run_training(
        config_path=source,
        dataset_path=root,
        pretrained_weights=weights,
        split_inventory_path=split,
        runs_root=tmp_path / "runs",
        run_id="train",
    )
    run = Path(result["run_dir"])
    receipt = verify_consumption(run)
    assert [p["pass_id"] for p in receipt["passes"]] == [
        "train/1",
        "validation/1",
        "validation/final",
    ]
    assert receipt["processed_samples"] == 6
    assert hashes(root) == before  # No framework .cache/.npy or relabeling in the source.
    train_rows = receipt["passes"][0]["outputs"]
    assert all(r["boundary"] == "optimizer_step_completed" for r in train_rows)
    assert all(r["optimizer_step"] == 1 for r in train_rows)
    assert all(
        r["boundary"] == "validator_metrics_updated" for r in receipt["passes"][1]["outputs"]
    )
    final_rows = receipt["passes"][-1]["outputs"]
    assert all(r["weights_sha256"] == receipt["selected_weights"]["sha256"] for r in final_rows)
    assert (
        result["summary"]["backend_metadata"]["consumption_profile"]
        == "ultralytics-8.4.137-fixed-pass-v1"
    )


def view_request(tmp_path):
    from types import SimpleNamespace

    from carla_vision.dataset.verified import load_verified_dataset
    from carla_vision.reproduction.consumption import make_pass
    from carla_vision.training.contracts import TrainingRequest

    root = dataset(tmp_path)
    data = load_verified_dataset(root)
    specs = [
        make_pass("train/1", "train", ["train"], [data], ordered=False),
        make_pass("validation/1", "validation", ["val_seen"], [data], ordered=False),
    ]
    audit = SimpleNamespace(specs={s["pass_id"]: s for s in specs})
    output = tmp_path / "backend"
    output.mkdir()
    return TrainingRequest(
        config(), tmp_path / "weights.pt", root, root / "data.yaml", output, 1, audit
    )


@pytest.mark.parametrize("artifact", ["image", "label"])
def test_source_loader_rejects_changed_rgb_or_target_bytes(tmp_path, artifact):
    from carla_vision.training.ultralytics_consumption import SourceView

    request = view_request(tmp_path)
    view = SourceView(request)
    row = next(iter(view.by_path.values()))
    p = Path(row[artifact])
    p.write_bytes(p.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="bytes"):
        view.image(row["image"])


def test_source_loader_does_not_discover_unregistered_directory_neighbors(tmp_path):
    from carla_vision.training.ultralytics_consumption import SourceView

    request = view_request(tmp_path)
    view = SourceView(request)
    known = Path(view.by_role["train"][0])
    extra = known.with_name("unregistered.png")
    extra.write_bytes(known.read_bytes())
    assert str(extra) not in view.by_path
    with pytest.raises(ValueError, match="unregistered"):
        view.row(str(extra))


def test_source_view_wont_overwrite_an_existing_view(tmp_path):
    from carla_vision.training.ultralytics_consumption import SourceView

    request = view_request(tmp_path)
    SourceView(request)
    with pytest.raises(FileExistsError):
        SourceView(request)


def test_source_view_modification_is_rejected_at_finish(tmp_path):
    from carla_vision.training.ultralytics_consumption import SourceView

    view = SourceView(view_request(tmp_path))
    view.lists["train"].write_text("different image\n")
    with pytest.raises(ValueError, match="view changed"):
        view.check()


@pytest.mark.parametrize("name,value", [("WORLD_SIZE", "2"), ("RANK", "0"), ("LOCAL_RANK", "0")])
def test_distributed_environment_cannot_self_report_single_process(monkeypatch, name, value):
    from carla_vision.training.ultralytics_consumption import validate_profile

    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="distributed"):
        validate_profile(config(), version="8.4.137")


def test_input_stamp_hash_binds_shape_dtype_and_rejects_nonfinite():
    torch = pytest.importorskip("torch")
    from carla_vision.training.ultralytics_consumption import tensor_digest

    data = torch.arange(12, dtype=torch.float32)
    assert tensor_digest(data) != tensor_digest(data.reshape(3, 4))
    assert tensor_digest(data) != tensor_digest(data.double())
    with pytest.raises(ValueError, match="nonfinite"):
        tensor_digest(torch.tensor([float("nan")]))


def run_actual(tmp_path, monkeypatch, *, shard=False, epochs=1, partition_count=2):
    pytest.importorskip("ultralytics")
    import torch
    from test_training import sample_pair
    from ultralytics.data import utils as data_utils
    from ultralytics.utils import checks

    from carla_vision.dataset.shards import ShardedDatasetWriter
    from carla_vision.dataset.writer import DatasetWriter
    from carla_vision.reproduction.study_links import freeze_split
    from carla_vision.training.runner import run_training

    monkeypatch.setattr(checks, "check_pip_update_available", lambda: None)
    monkeypatch.setattr(data_utils, "check_font", lambda *a, **k: None)
    torch.set_num_threads(2)
    writer_class = ShardedDatasetWriter if shard else DatasetWriter
    with writer_class(
        tmp_path / "data",
        dataset_id="fixture",
        minimum_pixels=4,
        carla_version="0.9.16",
        carla_map="fixture",
        carla_endpoint={},
        **({"shard_samples": 1} if shard else {}),
    ) as writer:
        for i in range(partition_count + 2):
            writer.add_pair(
                sample_pair(i + 1, i + 1, 10 + i * 15),
                split="train" if i < partition_count else "val_seen",
                scenario_id=f"scene-{i}",
                episode_id=f"episode-{i}",
            )
    root = tmp_path / "data/fixture"
    before = hashes(root)
    split = tmp_path / "split.json"
    freeze_split([root], split)
    weights = tiny_weights(tmp_path, "yolo")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config(epochs=epochs).as_dict()))
    result = run_training(
        config_path=path,
        dataset_path=root,
        pretrained_weights=weights,
        split_inventory_path=split,
        runs_root=tmp_path / "runs",
        run_id="train",
    )
    assert hashes(root) == before
    return Path(result["run_dir"]), split


def test_actual_sharded_training_consumes_unique_manifest_sample_pairs(tmp_path, monkeypatch):
    from carla_vision.reproduction.consumption import verify_consumption

    root, split = run_actual(tmp_path, monkeypatch, shard=True)
    receipt = verify_consumption(root, split_path=split)
    assert receipt["processed_samples"] == 6
    keys = receipt["passes"][0]["keys"]
    assert len(set(keys)) == 2 and len({k.split(":")[0] for k in keys}) == 2


def test_actual_two_epoch_training_retains_partial_final_batch(tmp_path, monkeypatch):
    from carla_vision.reproduction.consumption import verify_consumption

    root, split = run_actual(tmp_path, monkeypatch, epochs=2, partition_count=3)
    receipt = verify_consumption(root, split_path=split)
    assert receipt["processed_samples"] == 12
    assert len(receipt["passes"]) == 5
    assert [r["optimizer_step"] for r in receipt["passes"][0]["outputs"]] == [1, 1, 2]
    assert [r["optimizer_step"] for r in receipt["passes"][2]["outputs"]] == [3, 3, 4]


def test_actual_failed_optimizer_batch_never_records_success(tmp_path, monkeypatch):
    pytest.importorskip("ultralytics")
    from ultralytics.engine.trainer import BaseTrainer

    def broken_step(self):
        raise RuntimeError("injected optimizer failure")

    monkeypatch.setattr(BaseTrainer, "optimizer_step", broken_step)
    with pytest.raises(RuntimeError, match="injected optimizer"):
        run_actual(tmp_path, monkeypatch)
    summary = json.loads((tmp_path / "runs/train/consumption/summary.json").read_text())
    assert summary["status"] == "failed" and summary["processed_samples"] == 0
    assert len(summary["omissions"]) == 6


def test_actual_callback_failure_retains_uncertified_not_unprocessed_distinction(
    tmp_path, monkeypatch
):
    pytest.importorskip("ultralytics")
    from ultralytics.engine.trainer import BaseTrainer

    original = BaseTrainer.run_callbacks

    def callback(self, event):
        original(self, event)
        if event == "on_train_batch_end":
            raise RuntimeError("injected post-step callback failure")

    monkeypatch.setattr(BaseTrainer, "run_callbacks", callback)
    with pytest.raises(RuntimeError, match="post-step"):
        run_actual(tmp_path, monkeypatch)
    summary = json.loads((tmp_path / "runs/train/consumption/summary.json").read_text())
    assert summary["status"] == "failed" and summary["processed_samples"] == 0


@pytest.mark.parametrize(
    "change", ["final_weights", "source", "targets", "predictions", "optimizer"]
)
def test_rehashed_bad_framework_receipt_fails_semantic_verification(tmp_path, monkeypatch, change):
    from test_consumed_samples import rehash_run

    from carla_vision.artifacts import fingerprint_file
    from carla_vision.dataset.lineage import canonical_digest
    from carla_vision.reproduction.consumption import verify_consumption

    root, split = run_actual(tmp_path, monkeypatch)
    trace = root / "consumption/batches.jsonl"
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    if change == "final_weights":
        rows[-1]["outputs"][0]["weights_sha256"] = "f" * 64
    elif change == "source":
        rows[0]["outputs"][0]["rgb_sha256"] = "f" * 64
    elif change == "targets":
        rows[0]["outputs"][0]["label_sha256"] = "f" * 64
    elif change == "predictions":
        rows[-1]["outputs"][0]["predictions"] = {"bboxes": [], "cls": [], "conf": [0.9]}
    else:
        rows[0]["outputs"][0]["optimizer_step"] = 99
    for row in rows:
        row["outputs_sha256"] = canonical_digest(row["outputs"])
    trace.write_text("".join(json.dumps(r) + "\n" for r in rows))
    path = root / "consumption/summary.json"
    summary = json.loads(path.read_text())
    summary["batches"] = fingerprint_file(trace)
    path.write_text(json.dumps(summary))
    rehash_run(root)  # test-only: bypass byte tamper checks to exercise meaning checks
    with pytest.raises(ValueError, match="Ultralytics"):
        verify_consumption(root, split_path=split)


def test_source_tensor_identity_and_count_must_match_loaded_batch(tmp_path):
    torch = pytest.importorskip("torch")
    from carla_vision.training.ultralytics_consumption import SourceView

    view = SourceView(view_request(tmp_path))
    paths = view.by_role["train"]
    keys = [view.row(p)["key"] for p in paths]
    batch = {"im_file": paths, "consumption_key": keys[::-1], "img": torch.zeros(2, 3, 64, 64)}
    with pytest.raises(ValueError, match="identity"):
        view.inputs(batch)
    batch.update(consumption_key=keys, img=torch.zeros(1, 3, 64, 64))
    with pytest.raises(ValueError, match="count"):
        view.inputs(batch)


def test_actual_oom_is_not_retried_with_reduced_batch(tmp_path, monkeypatch):
    pytest.importorskip("ultralytics")
    from ultralytics.nn.tasks import DetectionModel

    calls = []

    def fail_loss(self, batch, preds=None):
        calls.append(len(batch["img"]))
        raise RuntimeError("CUDA out of memory: injected failure before backward")

    monkeypatch.setattr(DetectionModel, "loss", fail_loss)
    with pytest.raises(RuntimeError, match="injected failure"):
        run_actual(tmp_path, monkeypatch)
    assert calls == [2]
    summary = json.loads((tmp_path / "runs/train/consumption/summary.json").read_text())
    assert summary["processed_samples"] == 0 and summary["status"] == "failed"
