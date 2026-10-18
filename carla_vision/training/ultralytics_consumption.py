"""Opt-in single-process Ultralytics 8.4.137 consumption instrumentation.

Hook order was inspected in the wheel pinned by uv.lock. No upstream training
loop is copied or replaced: preprocess, optimizer_step, validator.update_metrics
and callbacks supply the actual model-work boundary. Multi-image augmentation,
DDP, partial epochs, AMP skips and opaque custom transforms are intentionally
unsupported here; they require a different lineage protocol.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import math
import os
import tempfile
from pathlib import Path
from types import MethodType

import cv2

from ..artifacts import fingerprint_file
from ..reproduction.consumption import checked_image, tensor_digest
from .consumption import project_inputs

SUPPORTED_VERSION = "8.4.137"
# Hashes of the unmodified wheel recorded in uv.lock; upstream code is not vendored.
HOOK_SOURCE_HASHES = {
    "engine/trainer.py": "d09f028072cc6111261cc3f08632956ce982749ad4879ac04899470eecaa57e2",
    "engine/validator.py": "9bc5c98e493c2242499d0df474977466918bab5169e53b8b6b6e280145efa920",
    "engine/model.py": "42ea8f5222be0c7014cf35a7ea63590bba6903a3b5c4d9cc8907003f0833a740",
    "data/base.py": "2b6f0d4b34726796835dbd248b636ac50d8acf169dc50ca89781b7501b7a4ade",
    "data/dataset.py": "f58e5ac031fed589364aadf6795d6fef5135f9c681e43db83723fcbd68695af0",
    "data/build.py": "cc58a5b3a8303c9f13cbb64142cbeaf9d0942a5762d5fd500478d8a38674da05",
    "data/augment.py": "83e653664fa459beb66b9f245d90ca50ea25bc3c83caabfa9fab541d4abcb4b3",
    "models/yolo/detect/train.py": "d68cd131e759f92cfb556b9991f62f48adcb925ad1e51e89986489f7b8cac914",
    "models/yolo/detect/val.py": "10e7944ede74be2c40c176fea65b81956cbe3837fdff1e4a361f2b57f6483e60",
    "models/rtdetr/train.py": "6f5aa25d616baf40723d08b0b769c12910aff908f523ba97ff943bad4d50e7ea",
    "models/rtdetr/val.py": "4754071a389ff7796e16d559681fd40dfc0b9498097568ddd4379777f9f566e5",
    "nn/tasks.py": "595753e67711f1c9157b39c2a351feabcc9493f374852c349d15c6f5270f0ab4",
}


def check_hook_sources():
    import ultralytics

    root = Path(ultralytics.__file__).parent
    if any(
        hashlib.sha256((root / path).read_bytes()).hexdigest() != digest
        for path, digest in HOOK_SOURCE_HASHES.items()
    ):
        raise ValueError("installed trainer hook source differs from the reviewed locked wheel")


def validate_settings(config):
    if importlib.metadata.version("ultralytics") != SUPPORTED_VERSION:
        raise ValueError("consumption hooks support exactly the uv.lock Ultralytics version")
    check_hook_sources()
    allowed = {
        "mosaic",
        "mixup",
        "cutmix",
        "copy_paste",
        "close_mosaic",
        "nbs",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "bgr",
        "max_det",
        "conf",
        "iou",
        "plots",
        "verbose",
        "momentum",
        "box",
        "cls",
        "dfl",
    }
    if set(config.extra_options) - allowed:
        raise ValueError("unsupported extra options in consumed-sample training")
    if (
        config.workers != 0
        or config.amp is not False
        or config.cache is not False
        or config.batch_size < 1
        or config.patience != 0
    ):
        raise ValueError(
            "attested training requires workers=0, amp=false, cache=false, positive batch and patience=0"
        )
    for name, default in (("mosaic", 1.0), ("mixup", 0.0), ("cutmix", 0.0), ("copy_paste", 0.0)):
        if config.extra_options.get(name, default) != 0:
            raise ValueError(f"{name} must be explicitly disabled for single-source receipts")
    if config.extra_options.get("nbs", 64) != config.batch_size:
        raise ValueError("nbs must equal batch_size: every recorded batch needs an optimizer step")
    if (
        "," in config.device
        or int(os.environ.get("WORLD_SIZE", "1")) != 1
        or int(os.environ.get("RANK", "-1")) != -1
    ):
        raise ValueError("distributed sampling is outside this receipt protocol")


def train_instrumented(model, arguments, request):
    import torch

    validate_settings(request.config)
    boundary = request.consumption
    recorder = boundary.recorder
    trainer_type = model._smart_load("trainer")
    # Keep upstream caches and temporary input copies outside both run and source.
    with tempfile.TemporaryDirectory(prefix="carlapyvision-consumed-input-") as temp:
        yaml_path, path_keys, label_hashes = project_inputs(recorder, Path(temp) / "projection")
        boundary.image_paths = path_keys
        read_hashes = {}
        final_models = []

        def load_checked(dataset, index, rect_mode=True, resize_short=False):
            path = Path(dataset.im_files[index]).resolve(strict=True)
            key = path_keys[path]
            image, hashed = checked_image(path, recorder.rows[key]["rgb_sha256"])
            label = path.parents[2] / "labels" / path.parent.name / (path.stem + ".txt")
            if hashlib.sha256(label.read_bytes()).hexdigest() != label_hashes[label.resolve()]:
                raise ValueError("projected supervision changed before actual data consumption")
            h0, w0 = image.shape[:2]
            if rect_mode:
                ratio = dataset.imgsz / (min(h0, w0) if resize_short else max(h0, w0))
                if ratio != 1:
                    w, h = math.ceil(w0 * ratio), math.ceil(h0 * ratio)
                    if not resize_short:
                        w, h = min(w, dataset.imgsz), min(h, dataset.imgsz)
                    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            elif (h0, w0) != (dataset.imgsz, dataset.imgsz):
                image = cv2.resize(
                    image, (dataset.imgsz, dataset.imgsz), interpolation=cv2.INTER_LINEAR
                )
            read_hashes[path] = hashed
            return image, (h0, w0), image.shape[:2]

        def prepare(batch):
            paths = [Path(p).resolve(strict=True) for p in batch["im_file"]]
            if any(p not in read_hashes for p in paths):
                raise ValueError("batch was not decoded through the verified input boundary")
            return boundary.prepare(
                image_paths=paths,
                images=batch["img"],
                targets=[batch[k] for k in ("cls", "bboxes", "batch_idx")],
            )

        class ReceiptTrainer(trainer_type):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if (
                    self.world_size > 1
                    or self.args.compile
                    or self.args.distill_model
                    or self.args.time
                    or self.args.fraction != 1.0
                ):
                    raise ValueError("unsupported effective trainer configuration")
                self.receipt_steps = 0
                self.receipt_pending = None
                self.add_callback(
                    "on_train_epoch_start", lambda t: boundary.begin("train", t.epoch + 1)
                )
                self.add_callback("on_train_batch_end", self.receipt_batch_end)
                self.add_callback("on_train_epoch_end", lambda t: boundary.end())

            def build_dataset(self, *args, **kwargs):
                dataset = super().build_dataset(*args, **kwargs)
                if any(Path(p).resolve(strict=True) not in path_keys for p in dataset.im_files):
                    raise ValueError("trainer selected an unregistered input image")
                dataset.load_image = MethodType(load_checked, dataset)
                return dataset

            def preprocess_batch(self, batch):
                batch = super().preprocess_batch(batch)
                self.receipt_pending = prepare(batch)
                return batch

            def optimizer_step(self):
                if any(
                    p.grad is not None and not bool(torch.isfinite(p.grad).all())
                    for p in self.model.parameters()
                ):
                    raise ValueError(
                        "nonfinite gradients cannot establish a successful optimizer step"
                    )
                super().optimizer_step()
                self.receipt_steps += 1

            def receipt_batch_end(self, _trainer):
                boundary.commit(
                    self.receipt_pending,
                    phase="train",
                    loss=float(self.loss.detach().cpu()),
                    optimizer_step=self.receipt_steps,
                )
                self.receipt_pending = None

            def get_validator(self):
                validator = super().get_validator()
                owner = self
                base_validator = type(validator)

                class ReceiptValidator(base_validator):
                    def __call__(self, trainer=None, model=None):
                        phase = "val" if trainer is not None else "val_final"
                        self.receipt_phase = phase
                        checkpoint_before = None
                        if trainer is None:
                            checkpoint_before = fingerprint_file(model)
                            final_models.append(checkpoint_before)
                        boundary.begin(phase, owner.epoch + 1)
                        result = super().__call__(trainer=trainer, model=model)
                        boundary.end()
                        if (
                            checkpoint_before is not None
                            and fingerprint_file(model) != checkpoint_before
                        ):
                            raise ValueError(
                                "selected final-validation weights changed during inference"
                            )
                        return result

                    def preprocess(self, batch):
                        batch = super().preprocess(batch)
                        self.receipt_pending = prepare(batch)
                        return batch

                    def update_metrics(self, predictions, batch):
                        super().update_metrics(predictions, batch)
                        outputs = [
                            {
                                "prediction_sha256": tensor_digest(*[p[k] for k in sorted(p)]),
                                "prediction_count": len(p["conf"]),
                                "metrics_updated": True,
                            }
                            for p in predictions
                        ]
                        boundary.commit(
                            self.receipt_pending, phase=self.receipt_phase, outputs=outputs
                        )
                        self.receipt_pending = None

                validator.__class__ = ReceiptValidator
                return validator

        patched = {**arguments, "data": str(yaml_path), "trainer": ReceiptTrainer}
        receipt_source = {
            "ultralytics_version": SUPPORTED_VERSION,
            "trainer_source": fingerprint_file(inspect.getfile(trainer_type)),
            "scope": "single_source_images_single_process_full_epoch_no_amp_no_accumulation",
        }
        metrics = model.train(**patched)
        # Re-read all projection labels, including files cached during setup.
        for path, expected in label_hashes.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("trainer modified projected supervision")
        receipt_source["reviewed_hook_source_sha256"] = HOOK_SOURCE_HASHES
        receipt_source["final_validation_models"] = final_models
        return metrics, receipt_source
