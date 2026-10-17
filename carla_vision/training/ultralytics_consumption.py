"""Version-bound, fixed-pass observation of real Ultralytics detection training.

The upstream optimizer/validator remain the owners of ML work. This adapter
observes their completed boundaries and substitutes only a read-only canonical
source loader. See ADR 0005 for the restricted profile and evidence limits.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import re
from copy import copy
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..artifacts import fingerprint_file
from ..dataset.lineage import canonical_digest
from ..dataset.shards import expand_dataset_roots
from ..dataset.verified import load_verified_dataset
from ..reproduction.consumption import decode_rgb
from .contracts import TrainingConfig, TrainingRequest

VERSION = "8.4.137"
PROFILE = f"ultralytics-{VERSION}-fixed-pass-v1"
_COMPOSITE = {"mosaic", "mixup", "cutmix", "copy_paste"}
# Extra options that cannot alter membership, execution ownership or checkpoints.
_ALLOWED = _COMPOSITE | {
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "flipud",
    "fliplr",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "bgr",
    "conf",
    "iou",
    "max_det",
    "momentum",
    "box",
    "cls",
    "dfl",
    "plots",
    "verbose",
}


def validate_profile(config: TrainingConfig, *, version: str | None = None) -> None:
    """Reject unsupported requests before weights or an optimizer are loaded."""
    version = version or importlib.metadata.version("ultralytics")
    if version != VERSION:
        raise ValueError(f"audited consumption requires Ultralytics {VERSION}, got {version}")
    if (
        config.workers != 0
        or config.amp is not False
        or config.cache is not False
        or config.patience != 0
        or config.warmup_epochs != 0.0
    ):
        raise ValueError(
            "audited consumption requires workers=0, amp/cache=false, patience/warmup=0"
        )
    if not re.fullmatch(r"cpu|mps|(?:cuda:)?[0-9]+", config.device):
        raise ValueError("audited consumption requires an explicit single CPU/MPS/CUDA device")
    if os.environ.get("WORLD_SIZE", "1") != "1" or any(
        os.environ.get(name, "-1") != "-1" for name in ("RANK", "LOCAL_RANK")
    ):
        raise ValueError("audited consumption does not support distributed execution")
    extra = dict(config.extra_options)
    if set(extra) - _ALLOWED:
        raise ValueError(
            "unsupported audited consumption options: " + ", ".join(sorted(set(extra) - _ALLOWED))
        )
    if any(type(extra.get(k)) not in (int, float) or extra[k] != 0 for k in _COMPOSITE):
        raise ValueError("audited consumption requires explicit mosaic/mixup/cutmix/copy_paste=0")
    for k, v in extra.items():
        if k not in {"plots", "verbose"} and (type(v) not in (int, float) or not math.isfinite(v)):
            raise ValueError(f"audited consumption option {k} must be a finite number")
    for key in ("plots", "verbose"):
        if key in extra and type(extra[key]) is not bool:
            raise ValueError(f"audited consumption option {key} must be boolean")
    if "max_det" in extra and (
        type(extra["max_det"]) is not int or not 1 <= extra["max_det"] <= 1000
    ):
        raise ValueError("audited consumption max_det must be an integer in [1,1000]")


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _checked_bytes(path: Path, digest: str) -> bytes:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("audited source bytes changed: " + str(path))
    return raw


def tensor_digest(value: Any) -> str:
    """Hash the materialized tensor entering the model, including dtype/shape."""
    tensor = value.detach().cpu().contiguous()
    if not bool(tensor.isfinite().all()):
        raise ValueError("nonfinite audited tensor")
    prefix = json.dumps([str(tensor.dtype), list(tensor.shape)]).encode()
    return hashlib.sha256(prefix + tensor.numpy().tobytes()).hexdigest()


class SourceView:
    """No image discovery, label cache, NPY fallback or writes in source releases."""

    def __init__(self, request: TrainingRequest):
        self.audit = request.sample_consumption
        self.config = request.config
        self.by_path: dict[str, dict[str, Any]] = {}
        self.by_role: dict[str, list[str]] = {"train": [], "val": []}
        self.names: dict[int, str] | None = None
        train_keys = set().union(
            *(set(s["expected_keys"]) for s in self.audit.specs.values() if s["role"] == "train")
        )
        validation_keys = set().union(
            *(
                set(s["expected_keys"])
                for s in self.audit.specs.values()
                if s["role"] == "validation"
            )
        )
        if train_keys & validation_keys:
            raise ValueError("audited source view overlaps train and validation")
        allowed = set().union(*(set(s["expected_keys"]) for s in self.audit.specs.values()))
        for root in expand_dataset_roots([request.dataset_root]):
            data = load_verified_dataset(root)
            manifest_sha = fingerprint_file(root / "manifest.json")["sha256"]
            if self.names is not None and dict(data.categories) != self.names:
                raise ValueError("audited shards have different ontologies")
            self.names = dict(data.categories)
            for sample in data.dataset["samples"]:
                key = f"{manifest_sha}:{sample['sample_id']}"
                if key not in allowed:
                    continue
                role = "train" if key in train_keys else "val"
                image = (root / sample["rgb"]["path"]).resolve(strict=True)
                label = (root / sample["yolo_label"]["path"]).resolve(strict=True)
                row = {
                    "key": key,
                    "rgb_sha256": sample["rgb"]["sha256"],
                    "label_sha256": sample["yolo_label"]["sha256"],
                    "image": str(image),
                    "label": str(label),
                    "role": role,
                }
                if str(image) in self.by_path:
                    raise ValueError("ambiguous audited image path")
                self.by_path[str(image)] = row
                self.by_role[role].append(str(image))
        if {r["key"] for r in self.by_path.values()} != allowed:
            raise ValueError("audited source view differs from fixed pass plan")
        self.root = request.output_root / "consumption_view"
        self.root.mkdir(exist_ok=False)
        self.lists = {role: self.root / f"{role}.txt" for role in self.by_role}
        for role, paths in self.by_role.items():
            if not paths:
                raise ValueError("audited profile needs nonempty train and validation")
            self.lists[role].write_text("\n".join(paths) + "\n", encoding="utf-8")
        self.yaml = self.root / "data.yaml"
        # JSON is also valid YAML; all source paths are explicit and no download script exists.
        _write_json(
            self.yaml,
            {
                "path": str(self.root),
                "train": str(self.lists["train"]),
                "val": str(self.lists["val"]),
                "test": [],
                "names": self.names,
            },
        )
        _write_json(self.root / "sources.json", list(self.by_path.values()))
        self.file_refs = [
            fingerprint_file(p)
            for p in (*self.lists.values(), self.yaml, self.root / "sources.json")
        ]

    def row(self, image: str) -> dict[str, Any]:
        try:
            return self.by_path[str(Path(image).resolve(strict=True))]
        except KeyError as error:
            raise ValueError("unregistered image at audited execution boundary") from error

    def labels(self, image: str) -> np.ndarray:
        row = self.row(image)
        raw = _checked_bytes(Path(row["label"]), row["label_sha256"])
        records = [line.split() for line in raw.decode("utf-8").splitlines() if line.strip()]
        if not records:
            return np.empty((0, 5), dtype=np.float32)
        values = np.asarray(records, dtype=np.float32)
        if (
            values.ndim != 2
            or values.shape[1] != 5
            or not np.isfinite(values).all()
            or np.any(values[:, 0] != np.floor(values[:, 0]))
            or not set(values[:, 0].astype(int)) <= set(self.names)
            or np.any(values[:, 1:] < 0)
            or np.any(values[:, 1:] > 1)
        ):
            raise ValueError("invalid audited YOLO targets")
        return values

    def image(self, image: str) -> np.ndarray:
        row = self.row(image)
        self.labels(image)  # catch post-initialization target mutation too
        return decode_rgb(image, row["rgb_sha256"])

    def check(self) -> None:
        for ref in self.file_refs:
            if fingerprint_file(ref["path"]) != ref:
                raise ValueError("audited source view changed")
        for row in self.by_path.values():
            _checked_bytes(Path(row["image"]), row["rgb_sha256"])
            _checked_bytes(Path(row["label"]), row["label_sha256"])

    def inputs(self, batch: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
        paths = batch["im_file"]
        keys = [self.row(p)["key"] for p in paths]
        if tuple(keys) != tuple(batch.get("consumption_key", ())):
            raise ValueError("tensor source identity differs from loaded image identity")
        if len(keys) != len(batch["img"]):
            raise ValueError("audited batch image count mismatch")
        records = []
        for i, path in enumerate(paths):
            row = self.row(path)
            idx = batch["batch_idx"] == i
            records.append(
                {
                    "input_sha256": tensor_digest(batch["img"][i]),
                    "input_shape": list(batch["img"][i].shape),
                    "targets_sha256": canonical_digest(
                        {
                            "cls": batch["cls"][idx].detach().cpu().tolist(),
                            "bboxes": batch["bboxes"][idx].detach().cpu().tolist(),
                        }
                    ),
                    "rgb_sha256": row["rgb_sha256"],
                    "label_sha256": row["label_sha256"],
                }
            )
        return keys, records


def instrumented_trainer(request: TrainingRequest, architecture: str):
    """Return a per-run trainer class; no global monkeypatches or copied ML loop."""
    import torch
    from ultralytics.data.base import BaseDataset
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.models.rtdetr.train import RTDETRTrainer
    from ultralytics.models.rtdetr.val import RTDETRDataset, RTDETRValidator
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils import LOCAL_RANK, RANK

    validate_profile(request.config)
    if RANK != -1 or LOCAL_RANK != -1:
        raise ValueError("audited consumption requires an unranked single-process runtime")
    view = SourceView(request)
    audit = request.sample_consumption
    base_trainer = RTDETRTrainer if architecture == "rtdetr" else DetectionTrainer
    base_validator = RTDETRValidator if architecture == "rtdetr" else DetectionValidator
    base_dataset = RTDETRDataset if architecture == "rtdetr" else YOLODataset

    class AuditedDataset(base_dataset):
        def __init__(self, *args, receipt_role, **kwargs):
            self.receipt_role = receipt_role
            super().__init__(*args, **kwargs)

        def get_img_files(self, img_path):
            if str(img_path) != str(view.lists[self.receipt_role]):
                raise ValueError("audited dataset path differs from declared source view")
            return list(view.by_role[self.receipt_role])

        def get_labels(self):
            # Canonical labels are already independently verified. Do not allow the
            # framework's cache, duplicate removal or image-repair code to rewrite them.
            labels = []
            for path in self.im_files:
                array = view.labels(path)
                labels.append(
                    {
                        "im_file": path,
                        "shape": view.image(path).shape[:2],
                        "cls": array[:, :1].copy(),
                        "bboxes": array[:, 1:].copy(),
                        "segments": [],
                        "keypoints": None,
                        "normalized": True,
                        "bbox_format": "xywh",
                    }
                )
            return labels

        def load_image(self, i, rect_mode=None, resize_short=False):
            if resize_short:
                raise ValueError("short-side resize outside audited detection profile")
            image = view.image(self.im_files[i])
            h, w = image.shape[:2]
            rectangular = architecture == "yolo" if rect_mode is None else rect_mode
            if rectangular:
                ratio = self.imgsz / max(h, w)
                size = (
                    min(math.ceil(w * ratio), self.imgsz),
                    min(math.ceil(h * ratio), self.imgsz),
                )
            else:
                size = (self.imgsz, self.imgsz)
            if size != (w, h):
                image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
            return image, (h, w), image.shape[:2]

        def __getitem__(self, index):
            result = super().__getitem__(index)
            result["consumption_key"] = view.row(self.im_files[index])["key"]
            return result

    class AuditedValidator(base_validator):
        def __call__(self, trainer=None, model=None):
            self.receipt_pass = (
                f"validation/{trainer.epoch + 1}" if trainer is not None else "validation/final"
            )
            self.receipt_weights = None if trainer is not None else fingerprint_file(model)
            result = super().__call__(trainer=trainer, model=model)
            if self.receipt_weights is not None:
                if fingerprint_file(model) != self.receipt_weights:
                    raise ValueError("selected weights changed during final validation")
                view.final_weights = self.receipt_weights
            return result

        def update_metrics(self, preds, batch):
            keys, rows = view.inputs(batch)
            if len(preds) != len(keys):
                raise ValueError("audited validation prediction count differs from batch")
            with audit.batch(self.receipt_pass, keys) as observed:
                super().update_metrics(preds, batch)
                for row, pred in zip(rows, preds, strict=True):
                    row.update(
                        {
                            "boundary": "validator_metrics_updated",
                            "weights_sha256": None
                            if self.receipt_weights is None
                            else self.receipt_weights["sha256"],
                            "prediction_coordinates": "preprocessed_image_xyxy",
                            "predictions": {
                                k: pred[k].detach().cpu().tolist()
                                for k in ("bboxes", "conf", "cls")
                            },
                        }
                    )
                observed["outputs"] = rows

    class AuditedTrainer(base_trainer):
        def __init__(self, *args, **kwargs):
            self.receipt_pending = None
            self.receipt_steps = 0
            self.receipt_pipelines = 0
            super().__init__(*args, **kwargs)
            if self.world_size > 1:
                raise ValueError("distributed audited consumption is unsupported")
            self.check_receipt_profile()

        def check_receipt_profile(self):
            fixed = {
                "epochs": request.config.epochs,
                "batch": request.config.batch_size,
                "workers": 0,
                "amp": False,
                "cache": False,
                "patience": 0,
                "warmup_epochs": 0.0,
                "nbs": request.config.batch_size,
                "time": None,
                "compile": False,
                "fraction": 1.0,
                "single_cls": False,
                "classes": None,
                "multi_scale": 0.0,
                "distill_model": None,
                "resume": False,
                "val": True,
                "save": True,
                "task": "detect",
                "close_mosaic": 0,
                **{k: 0.0 for k in _COMPOSITE},
            }
            for key, value in fixed.items():
                if getattr(self.args, key, object()) != value:
                    raise ValueError(f"resolved audited consumption profile changed: {key}")

        def build_dataset(self, img_path, mode="train", batch=None):
            return AuditedDataset(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch,
                augment=mode == "train",
                hyp=self.args,
                rect=mode == "val" and architecture == "yolo",
                cache=False,
                single_cls=False,
                stride=max(int(self.model.stride.max()), 32),
                pad=0.0 if mode == "train" else 0.5,
                data=self.data,
                task="detect",
                fraction=1.0,
                receipt_role=mode,
            )

        def _build_train_pipeline(self):
            self.check_receipt_profile()
            self.receipt_pipelines += 1
            if self.receipt_pipelines != 1:
                raise ValueError("audited pipeline rebuild/retry is unsupported")
            super()._build_train_pipeline()
            if self.accumulate != 1 or self.train_loader.drop_last or self.train_loader.num_workers:
                raise ValueError("audited profile requires one optimizer step per complete batch")

        def get_validator(self):
            return AuditedValidator(
                self.test_loader,
                save_dir=self.save_dir,
                args=copy(self.args),
                _callbacks=self.callbacks,
            )

        def preprocess_batch(self, batch):
            if self.receipt_pending is not None:
                raise ValueError("previous audited batch has not completed")
            self.check_receipt_profile()
            result = super().preprocess_batch(batch)
            keys, rows = view.inputs(result)
            manager = audit.batch(f"train/{self.epoch + 1}", keys)
            observation = manager.__enter__()
            self.receipt_pending = (manager, observation, rows, self.receipt_steps)
            return result

        def optimizer_step(self):
            for parameter in self.model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise ValueError("nonfinite gradients at audited optimizer boundary")
            super().optimizer_step()
            if not all(torch.isfinite(p).all() for p in self.model.parameters()):
                raise ValueError("nonfinite weights after audited optimizer step")
            self.receipt_steps += 1

        def run_callbacks(self, event):
            super().run_callbacks(event)
            if event == "on_train_epoch_start":
                # Upstream resets this immediately before the epoch loop, so set
                # the no-retry guard here rather than in __init__ or on_train_start.
                self._oom_retries = 3
            if event == "on_train_batch_end":
                if self.receipt_pending is None:
                    raise ValueError("audited completion callback without a pending batch")
                manager, observation, rows, previous = self.receipt_pending
                if self.receipt_steps != previous + 1:
                    raise ValueError("audited optimizer did not complete exactly one step")
                for row in rows:
                    row.update(
                        {
                            "boundary": "optimizer_step_completed",
                            "optimizer_step": self.receipt_steps,
                            "batch_loss": float(self.loss.detach().cpu()),
                        }
                    )
                observation["outputs"] = rows
                self.receipt_pending = None
                manager.__exit__(None, None, None)

        def _handle_nan_recovery(self, epoch):
            if (
                self.fitness is None
                or not math.isfinite(float(self.fitness))
                or not all(math.isfinite(float(v)) for v in self.metrics.values())
            ):
                raise ValueError(
                    "nonfinite audited validation metrics; automatic recovery forbidden"
                )
            return False

        def train(self):
            try:
                result = super().train()
                if self.receipt_pending is not None:
                    raise ValueError("audited training exited with an unfinished batch")
                view.check()
                if getattr(view, "final_weights", None) != fingerprint_file(self.best):
                    raise ValueError("audited training lacks final selected-weight validation")
                return result
            except BaseException as error:
                if self.receipt_pending is not None:
                    manager, _, _, _ = self.receipt_pending
                    self.receipt_pending = None
                    manager.__exit__(type(error), error, error.__traceback__)
                raise
            finally:
                for name in ("train_loader", "test_loader"):
                    loader = getattr(self, name, None)
                    if hasattr(loader, "close"):
                        loader.close()

    source_files = sorted(
        {
            str(Path(inspect.getfile(c)).resolve())
            for c in (
                base_trainer,
                base_validator,
                base_dataset,
                BaseTrainer,
                BaseValidator,
                BaseDataset,
                YOLODataset,
            )
        }
    )
    view.framework_sources = [fingerprint_file(p) for p in source_files]
    return AuditedTrainer, view


def verify_receipt_outputs(
    training_summary: dict[str, Any],
    report: dict[str, Any],
    inventory: dict[str, Any],
    *,
    epochs: int,
    batches_path: Path,
) -> None:
    """Validate the additional pinned-framework observation semantics, without ML imports.

    This checks recorded relationships, not independent replay of optimization.
    """
    metadata = training_summary.get("backend_metadata", {})
    if (
        metadata.get("consumption_profile") != PROFILE
        or metadata.get("ultralytics_version") != VERSION
        or metadata.get("final_validation_weights") != report["selected_weights"]
    ):
        raise ValueError("Ultralytics consumption profile or final selected weights mismatch")
    expected_names = [f"{r}/{e}" for e in range(1, epochs + 1) for r in ("train", "validation")] + [
        "validation/final"
    ]
    if [p["pass_id"] for p in report["passes"]] != expected_names:
        raise ValueError("Ultralytics consumption pass schedule differs from configured epochs")
    frozen = {f"{r['manifest_sha256']}:{r['sample_id']}": r for r in inventory["samples"]}
    label_hashes = {}
    for source in inventory["dataset_manifests"]:
        data = load_verified_dataset(Path(source["path"]).parent)
        if fingerprint_file(data.root / "manifest.json")["sha256"] != source["sha256"]:
            raise ValueError("Ultralytics source changed during receipt verification")
        label_hashes.update(
            {
                f"{source['sha256']}:{sample['sample_id']}": sample["yolo_label"]["sha256"]
                for sample in data.dataset["samples"]
            }
        )
    previous_step = 0
    for entry in report["passes"]:
        steps = []
        for key, row in zip(entry["keys"], entry["outputs"], strict=True):
            if not isinstance(row, dict) or row.get("rgb_sha256") != frozen[key]["rgb_sha256"]:
                raise ValueError("Ultralytics consumed source differs from frozen RGB")
            for name in ("input_sha256", "targets_sha256", "label_sha256"):
                if not isinstance(row.get(name), str) or not re.fullmatch(
                    "[a-f0-9]{64}", row[name]
                ):
                    raise ValueError("invalid Ultralytics input/target hash")
            if row["label_sha256"] != label_hashes[key]:
                raise ValueError("Ultralytics consumed targets differ from source labels")
            shape = row.get("input_shape")
            if (
                not isinstance(shape, list)
                or len(shape) != 3
                or shape[0] != 3
                or any(type(v) is not int or v <= 0 for v in shape)
            ):
                raise ValueError("invalid Ultralytics image tensor shape")
            if entry["role"] == "train":
                step = row.get("optimizer_step")
                loss = row.get("batch_loss")
                if (
                    row.get("boundary") != "optimizer_step_completed"
                    or type(step) is not int
                    or type(loss) not in (int, float)
                    or not math.isfinite(loss)
                ):
                    raise ValueError("invalid Ultralytics optimizer observation")
                steps.append(step)
            else:
                expected = (
                    report["selected_weights"]["sha256"]
                    if entry["pass_id"] == "validation/final"
                    else None
                )
                if (
                    row.get("boundary") != "validator_metrics_updated"
                    or row.get("weights_sha256") != expected
                ):
                    raise ValueError("Ultralytics validation weights or observation mismatch")
                predictions = row.get("predictions")
                if not isinstance(predictions, dict) or set(predictions) != {
                    "bboxes",
                    "conf",
                    "cls",
                }:
                    raise ValueError("invalid Ultralytics per-sample predictions")
                if (
                    any(not isinstance(v, list) for v in predictions.values())
                    or len({len(v) for v in predictions.values()}) != 1
                ):
                    raise ValueError("Ultralytics prediction length mismatch")
                for box, conf, cls in zip(
                    predictions["bboxes"], predictions["conf"], predictions["cls"], strict=True
                ):
                    if (
                        not isinstance(box, list)
                        or len(box) != 4
                        or any(
                            type(v) not in (int, float) or not math.isfinite(v)
                            for v in [*box, conf, cls]
                        )
                        or not 0 <= conf <= 1
                        or cls < 0
                        or int(cls) != cls
                    ):
                        raise ValueError("invalid Ultralytics prediction values")
        for step in dict.fromkeys(steps):
            if step != previous_step + 1:
                raise ValueError("Ultralytics optimizer-step order mismatch")
            previous_step = step
        if steps != sorted(steps):
            raise ValueError("Ultralytics optimizer observations were reordered")

    # A per-sample copy of a step ID must also agree with real recorded batch boundaries.
    step = 0
    with batches_path.open(encoding="utf-8") as stream:
        for line in stream:
            batch = json.loads(line)
            if batch["pass_id"].startswith("train/"):
                step += 1
                if (
                    any(r["optimizer_step"] != step for r in batch["outputs"])
                    or len({r["batch_loss"] for r in batch["outputs"]}) != 1
                ):
                    raise ValueError("Ultralytics optimizer steps disagree with batch boundaries")
