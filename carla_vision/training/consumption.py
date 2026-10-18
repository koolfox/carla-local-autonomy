"""Training receipt boundary and canonical read-only input projection.

The adapter must signal successful work inside its loop. Returning checkpoint
paths does not establish batch consumption. Opaque legacy backends stay usable
without this opt-in, but cannot satisfy the reviewer consumption gate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..artifacts import fingerprint_file
from ..dataset.lineage import canonical_digest
from ..dataset.shards import expand_dataset_roots
from ..dataset.verified import load_verified_dataset
from ..reproduction.consumption import _artifact, checked_image, sample_key, tensor_digest
from ..reproduction.review_contracts import digest, fields, integer, number, read_json


def project_inputs(recorder, destination):
    """Copy only selected verified bytes; library cache files never touch releases."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    path_keys, label_hashes, names = {}, {}, None
    selected = {k for values in recorder.expected.values() for k in values}
    train_keys = set(recorder.expected["train"])
    projected = set()
    byte_count = 0
    for root in expand_dataset_roots(
        [Path(r["path"]).parent for r in recorder.plan["source_manifests"]]
    ):
        data = load_verified_dataset(root)
        if names is None:
            names = dict(data.categories)
        elif names != dict(data.categories):
            raise ValueError("projected sources have inconsistent ontology")
        source = fingerprint_file(root / "manifest.json")["sha256"]
        for sample in data.dataset["samples"]:
            key = sample_key(source, sample["sample_id"])
            if key not in selected:
                continue
            phase = "train" if key in train_keys else "val"
            basename = hashlib.sha256(key.encode()).hexdigest()
            rgb_path = root / sample["rgb"]["path"]
            label_path = root / "labels" / sample["split"] / (rgb_path.stem + ".txt")
            # The canonical verifier already binds labels to mask and COCO.
            for source_path, target in (
                (rgb_path, destination / "images" / phase / (basename + ".png")),
                (label_path, destination / "labels" / phase / (basename + ".txt")),
            ):
                maximum = min(128 * 1024 * 1024, 8 * 1024**3 - byte_count)
                if source_path.is_symlink() or source_path.stat().st_size > maximum:
                    raise ValueError("unsafe or over-budget attested projection source")
                with source_path.open("rb") as stream:
                    payload = stream.read(maximum + 1)
                if len(payload) > maximum:
                    raise ValueError("projection source changed beyond its read budget")
                byte_count += len(payload)
                if byte_count > 8 * 1024**3:
                    raise ValueError("attested input projection exceeds 8 GiB capacity")
                if (
                    source_path == rgb_path
                    and hashlib.sha256(payload).hexdigest() != recorder.rows[key]["rgb_sha256"]
                ):
                    raise ValueError("source RGB changed during projection")
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(payload)
                if source_path == rgb_path:
                    path_keys[target.resolve()] = key
                else:
                    label_hashes[target.resolve()] = hashlib.sha256(payload).hexdigest()
            projected.add(key)
    if projected != selected:
        raise ValueError("projection omitted frozen samples")
    path = destination / "data.yaml"
    path.write_text(
        "path: "
        + json.dumps(str(destination.resolve()))
        + "\ntrain: images/train\nval: images/val\ntest: []\nnames:\n"
        + "".join(f"  {i}: {json.dumps(n)}\n" for i, n in names.items()),
        encoding="utf-8",
    )
    return path, path_keys, label_hashes


class TrainingConsumption:
    """Minimal explicit protocol exposed to backend implementations (schema 1.0)."""

    schema_version = "1.0"

    def __init__(self, recorder, *, image_paths=None):
        self.recorder = recorder
        self.image_paths = image_paths if image_paths is not None else recorder.image_paths

    def begin(self, phase, epoch):
        self.recorder.begin(phase, epoch)

    def prepare(self, *, image_paths, images, targets=()):
        paths = [Path(p).resolve(strict=True) for p in image_paths]
        keys = [self.image_paths[p] for p in paths]
        hashes = [
            checked_image(p, self.recorder.rows[k]["rgb_sha256"])[1]
            for p, k in zip(paths, keys, strict=True)
        ]
        if len(images) != len(keys):
            raise ValueError("model tensor and source batch sizes differ")
        return self.recorder.prepare(keys, hashes, input_sha256=tensor_digest(images, *targets))

    def commit(self, token, *, phase, loss=None, optimizer_step=None, outputs=None):
        if phase != self.recorder.active["phase"]:
            raise ValueError("backend committed a different phase")
        if phase == "train":
            number(loss, "completed batch loss")
            if type(optimizer_step) is not int or optimizer_step < 1:
                raise ValueError("training receipt requires a completed optimizer step")
            record = {
                "loss": float(loss),
                "optimizer_step": optimizer_step,
                "backward_completed": True,
            }
            outputs = [record.copy() for _ in token["keys"]]
        elif outputs is None:
            raise ValueError("validation receipt requires per-sample output evidence")
        self.recorder.commit(token, outputs)

    def end(self):
        self.recorder.end()


def verify_training_receipts(root, manifest, trace):
    resolved = read_json(_artifact(root, manifest, "resolved_training_configuration"))
    cfg = resolved["config"]
    if trace["plan"]["model"] != resolved["pretrained_weights"]:
        raise ValueError("training initialization differs from consumption plan")
    expected = [
        (phase, epoch) for epoch in range(1, cfg["epochs"] + 1) for phase in ("train", "val")
    ]
    actual = [(p["phase"], p["epoch"]) for p in trace["passes"]]
    if actual not in (expected, expected + [("val_final", cfg["epochs"])]):
        raise ValueError("declared training epochs differ from consumed passes")
    if trace["plan"]["phases"]["train"] != list(resolved["resolved_partitions"]["train"]) or trace[
        "plan"
    ]["phases"]["val"] != list(resolved["resolved_partitions"]["validation"]):
        raise ValueError("backend training/validation partitions differ")
    steps = 0
    for record in trace["passes"]:
        for batch in record["batches"]:
            if record["phase"] == "train":
                steps += 1
                for output in batch["outputs"]:
                    if (
                        set(output) != {"loss", "optimizer_step", "backward_completed"}
                        or output["optimizer_step"] != steps
                        or type(output["optimizer_step"]) is not int
                        or output["backward_completed"] is not True
                    ):
                        raise ValueError("missing/repeated optimizer receipt")
                    number(output["loss"], "consumed training loss")
            else:
                for output in batch["outputs"]:
                    fields(output, {"prediction_sha256", "prediction_count", "metrics_updated"})
                    digest(output["prediction_sha256"])
                    integer(output["prediction_count"], "validation prediction count")
                    if output["metrics_updated"] is not True:
                        raise ValueError("validation metrics were not updated by the backend")
    refs = {
        name: fingerprint_file(_artifact(root, manifest, name + "_checkpoint"))
        for name in ("best", "last")
    }
    if trace["checkpoints"] != refs:
        raise ValueError("backend returned different selected weights than recorded")
    summary = read_json(_artifact(root, manifest, "training_summary"))
    if summary["training_executed"] is not True:
        raise ValueError("planned training is not consumed training")
    for role in ("best", "last"):
        if {k: summary[role + "_checkpoint"][k] for k in refs[role]} != refs[role]:
            raise ValueError("training summary checkpoint differs from receipt")
    if canonical_digest(summary["pretrained_weights"]) != canonical_digest(trace["plan"]["model"]):
        raise ValueError("training summary initialization changed")
    instrument = summary.get("backend_metadata", {}).get("consumption_instrumentation")
    if cfg["backend"] in {"yolo", "ultralytics-yolo", "rtdetr", "ultralytics-rtdetr"}:
        from .ultralytics_consumption import HOOK_SOURCE_HASHES, SUPPORTED_VERSION

        if (
            not isinstance(instrument, dict)
            or instrument.get("ultralytics_version") != SUPPORTED_VERSION
            or instrument.get("reviewed_hook_source_sha256") != HOOK_SOURCE_HASHES
            or instrument.get("final_validation_models") != [refs["best"]]
        ):
            raise ValueError(
                "training final validation does not bind the reviewed hooks and selected weights"
            )
    return []
