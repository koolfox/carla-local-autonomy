"""Producer-time sample receipts. Software observations, not signed execution proofs.

Only a completed model/optimizer batch is recorded as processed. A plan, prefetch,
zero detections, or an opaque trainer's declared directory is not such a batch.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..artifacts import fingerprint_file
from ..dataset.lineage import canonical_digest, partition_role
from .review_contracts import fields, integer, parse_json, read_json, relative_path, text
from .study_links import _reference, verify_split

SCHEMA = "1.0"
PRODUCERS = {
    "detector_evaluation": "offline_detector_evaluation",
    "detector_training": "detector_training",
    "imitation_training": "imitation_training",
    "imitation_evaluation": "imitation_locked_evaluation",
}
MAX_TRACE_BYTES = 256 * 1024 * 1024


def sample_key(row: Mapping[str, Any]) -> str:
    return f"{row['manifest_sha256']}:{row['sample_id']}"


def make_pass(pass_id, role, partitions, datasets, *, ordered=True):
    """Use verified source order, not filenames discovered by an ML framework."""
    sources, keys = [], []
    for dataset in datasets:
        source = fingerprint_file(dataset.root / "manifest.json")["sha256"]
        sources.append(source)
        keys.extend(
            f"{source}:{s['sample_id']}"
            for s in dataset.dataset["samples"]
            if s["split"] in partitions
        )
    return {
        "pass_id": pass_id,
        "role": role,
        "partitions": list(partitions),
        "source_manifests": sources,
        "expected_keys": keys,
        "ordered": ordered,
    }


def _plan_passes(specs, inventory):
    if not isinstance(specs, list) or not specs or len(specs) > 10_000:
        raise ValueError("consumption requires a bounded nonempty pass schedule")
    rows = {sample_key(r): r for r in inventory["samples"]}
    sources = {r["sha256"] for r in inventory["dataset_manifests"]}
    ids = set()
    for spec in specs:
        fields(
            spec, {"pass_id", "role", "partitions", "source_manifests", "expected_keys", "ordered"}
        )
        name = text(spec["pass_id"], "pass identity")
        if name in ids:
            raise ValueError("duplicate consumption pass")
        ids.add(name)
        if spec["role"] not in {"train", "validation", "evaluation"}:
            raise ValueError("invalid consumption role")
        if type(spec["ordered"]) is not bool:
            raise ValueError("ordered must be boolean")
        for key in ("partitions", "source_manifests", "expected_keys"):
            values = spec[key]
            if not isinstance(values, list) or not values or len(set(values)) != len(values):
                raise ValueError(f"nonempty unique {key} required")
            for value in values:
                text(value, key)
        if not set(spec["source_manifests"]) <= sources:
            raise ValueError("consumption source outside frozen inventory")
        roles = {partition_role(p) for p in spec["partitions"]}
        if "unassigned" in roles or (
            spec["role"] in {"train", "validation"}
            and roles != {"val" if spec["role"] == "validation" else "train"}
        ):
            raise ValueError("forbidden consumption partition for role")
        expected = {
            k
            for k, r in rows.items()
            if r["manifest_sha256"] in spec["source_manifests"]
            and r["partition"] in spec["partitions"]
        }
        if set(spec["expected_keys"]) != expected:
            raise ValueError("pass omits or substitutes frozen source membership")
    return rows


def _json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def decode_rgb(path, expected_sha256):
    """Hash and decode the SAME bytes; post-read file hashes are insufficient."""
    import cv2
    import numpy as np

    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("consumed RGB bytes differ from frozen source")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("consumed RGB cannot be decoded")
    return image


class ConsumptionRecorder:
    def __init__(self, tracker, split_path, specs, producer, initial_weights):
        if producer not in PRODUCERS:
            raise ValueError("unknown consumption producer")
        self.tracker = tracker
        self.split = Path(split_path).resolve(strict=True)
        self.split_ref = fingerprint_file(self.split)
        inventory = verify_split(self.split)
        # Freeze the schedule before handing the recorder to a backend.
        specs = copy.deepcopy(specs)
        self.rows = _plan_passes(specs, inventory)
        self.specs = {s["pass_id"]: s for s in specs}
        self.seen = {name: [] for name in self.specs}
        self.seen_sets = {name: set() for name in self.specs}
        self.expected_sets = {n: set(s["expected_keys"]) for n, s in self.specs.items()}
        self.pass_order = list(self.specs)
        self.busy = False
        self.active = None
        self.closed_passes = set()
        self.initial = fingerprint_file(initial_weights) if initial_weights else None
        if producer.endswith("evaluation") and self.initial is None:
            raise ValueError("attested evaluation needs actual immutable weights")
        self.producer = producer
        self.sealed = False
        self.events = 0
        self.root = tracker.artifact_path("consumption")
        self.root.mkdir(exist_ok=False)
        self.plan = {
            "schema_version": SCHEMA,
            "run_id": tracker.run_id,
            "producer": producer,
            "split_inventory": self.split_ref,
            "passes": specs,
            "initial_weights": self.initial,
            "boundary": "completed_model_batch_raw_source_identity_before_declared_transforms",
        }
        _json(self.root / "plan.json", self.plan)
        tracker.register_artifact(self.root / "plan.json", role="consumption_plan")
        tracker.add_input_reference(self.split_ref)
        self.stream = (self.root / "batches.jsonl").open("x", encoding="utf-8")

    @contextmanager
    def batch(self, pass_id: str, keys: Sequence[str]):
        if self.busy or self.sealed or pass_id not in self.specs or pass_id in self.closed_passes:
            raise ValueError("unknown, closed, or sealed consumption pass")
        keys = list(keys)
        spec = self.specs[pass_id]
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("empty or duplicate consumed batch")
        if not set(keys) <= self.expected_sets[pass_id] or set(keys) & self.seen_sets[pass_id]:
            raise ValueError("consumed batch is substituted, duplicated or in wrong partition")
        position = len(self.seen[pass_id])
        if spec["ordered"] and keys != spec["expected_keys"][position : position + len(keys)]:
            raise ValueError("consumed sample order differs from the declared pass")
        if self.active != pass_id:
            index = 0 if self.active is None else self.pass_order.index(self.active) + 1
            if (
                index >= len(self.pass_order)
                or self.pass_order[index] != pass_id
                or (
                    self.active is not None
                    and self.seen_sets[self.active] != self.expected_sets[self.active]
                )
            ):
                raise ValueError("consumption pass schedule or prior coverage is incomplete")
            if self.active is not None:
                self.closed_passes.add(self.active)
        self.active = pass_id
        observed = {}
        # Exceptions in model execution never reach the successful append below.
        self.busy = True
        try:
            yield observed
        finally:
            self.busy = False
        outputs = observed.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != len(keys):
            raise ValueError("one explicit output record per consumed sample required")
        record = {
            "sequence": self.events + 1,
            "pass_id": pass_id,
            "position": position,
            "keys": keys,
            "outputs": outputs,
            "outputs_sha256": canonical_digest(outputs),
        }
        encoded = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        if self.stream.tell() + len(encoded.encode()) > MAX_TRACE_BYTES:
            raise ValueError("consumption trace budget exceeded")
        self.stream.write(encoded)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.events += 1
        self.seen[pass_id].extend(keys)
        self.seen_sets[pass_id].update(keys)

    def _summary(self, status, weights=None, reason=None):
        return {
            "schema_version": SCHEMA,
            "run_id": self.tracker.run_id,
            "producer": self.producer,
            "status": status,
            "plan": fingerprint_file(self.root / "plan.json"),
            "batches": fingerprint_file(self.root / "batches.jsonl"),
            "selected_weights": weights,
            "processed_samples": sum(map(len, self.seen.values())),
            "batch_count": self.events,
            "passes": [
                {
                    "pass_id": name,
                    "processed_count": len(self.seen[name]),
                    "order_sha256": canonical_digest(self.seen[name]),
                }
                for name in self.specs
            ],
            "omissions": [
                {"pass_id": name, "key": k, "reason": reason or "not_processed"}
                for name, spec in self.specs.items()
                for k in spec["expected_keys"]
                if k not in self.seen_sets[name]
            ],
            "scope": "instrumented_software_receipt_not_signed_or_independent_execution_proof",
        }

    def seal(self, weights):
        if self.sealed or self.busy:
            raise ValueError("consumption already sealed or batch is still executing")
        if canonical_digest(read_json(self.root / "plan.json")) != canonical_digest(self.plan):
            raise ValueError("consumption plan changed during execution")
        if fingerprint_file(self.split) != self.split_ref:
            raise ValueError("split inventory changed during consumption")
        verify_split(self.split)
        if self.initial and fingerprint_file(self.initial["path"]) != self.initial:
            raise ValueError("initial weights changed during consumption")
        selected = fingerprint_file(weights)
        if self.producer.endswith("evaluation") and selected != self.initial:
            raise ValueError("evaluated weights differ from loaded weights")
        summary = self._summary("complete", selected)
        if summary["omissions"]:
            raise ValueError("incomplete consumed-sample coverage; missing inputs")
        self.stream.close()
        _json(self.root / "summary.json", summary)
        self._register()
        self.tracker.add_model_reference(selected)
        self.sealed = True

    def _register(self):
        self.tracker.register_artifact(self.root / "batches.jsonl", role="consumed_sample_batches")
        self.tracker.register_artifact(self.root / "summary.json", role="sample_consumption")

    def abort(self):
        if self.sealed:
            return
        self.stream.close()
        _json(
            self.root / "summary.json",
            self._summary("failed", reason="run_failed_before_completion"),
        )
        self._register()


@contextmanager
def consumption_session(tracker, split_path, passes, *, producer, initial_weights=None):
    if split_path is None:
        yield None
        return
    audit = ConsumptionRecorder(tracker, split_path, passes, producer, initial_weights)
    try:
        yield audit
        if not audit.sealed:
            raise ValueError("consumption run did not seal completed coverage")
    except BaseException as error:
        try:
            audit.abort()
        except BaseException as cleanup:
            error.add_note(f"consumption diagnostics failed: {type(cleanup).__name__}")
        raise


def verify_consumption(root, *, split_path=None, require_complete=True):
    """Reopen the producer, frozen sources, actual trace, and chosen weights."""
    from ..verification import verify_research_object

    root = Path(root).resolve(strict=True)
    verify_research_object(root, deep=False, verify_references=True, reject_unregistered=True)
    manifest = read_json(root / "manifest.json")

    def artifact(role):
        rows = [r for r in manifest["artifacts"] if r["role"] == role]
        if len(rows) != 1:
            raise ValueError(f"producer needs exactly one consumption artifact: {role}")
        return relative_path(root, rows[0]["path"])

    plan_path = artifact("consumption_plan")
    trace = artifact("consumed_sample_batches")
    summary_path = artifact("sample_consumption")
    if any(p.stat().st_size > MAX_TRACE_BYTES for p in (plan_path, summary_path)):
        raise ValueError("consumption envelope exceeds inspection budget")
    plan, summary = read_json(plan_path), read_json(summary_path)
    fields(
        plan,
        {
            "schema_version",
            "run_id",
            "producer",
            "split_inventory",
            "passes",
            "initial_weights",
            "boundary",
        },
    )
    fields(
        summary,
        {
            "schema_version",
            "run_id",
            "producer",
            "status",
            "plan",
            "batches",
            "selected_weights",
            "processed_samples",
            "batch_count",
            "passes",
            "omissions",
            "scope",
        },
    )
    if plan["schema_version"] != SCHEMA or summary["schema_version"] != SCHEMA:
        raise ValueError("unsupported consumption schema")
    if plan["run_id"] != manifest["run_id"] or summary["run_id"] != manifest["run_id"]:
        raise ValueError("consumption bound to another producer run")
    if plan["producer"] not in PRODUCERS or summary["producer"] != plan["producer"]:
        raise ValueError("consumption producer mismatch")
    if manifest["invocation"]["config"].get("object_type") != PRODUCERS[plan["producer"]]:
        raise ValueError("consumption does not belong to the declared producer type")
    integer(summary["processed_samples"], "processed count")
    integer(summary["batch_count"], "batch count")
    if plan["producer"].endswith("training"):
        selected = artifact("best_checkpoint")
        if fingerprint_file(selected) != summary["selected_weights"]:
            raise ValueError("consumption is not bound to the selected best checkpoint")

    for value, expected in ((summary["plan"], plan_path), (summary["batches"], trace)):
        if _reference(root, value) != expected:
            raise ValueError("consumption reference points outside registered artifacts")
    split = _reference(root, plan["split_inventory"])
    if split_path and fingerprint_file(split)["sha256"] != fingerprint_file(split_path)["sha256"]:
        raise ValueError("consumption belongs to a different frozen split")
    inventory = verify_split(split)
    _plan_passes(plan["passes"], inventory)
    if plan["initial_weights"] is not None:
        _reference(root, plan["initial_weights"])
    if summary["selected_weights"] is None:
        raise ValueError("consumption lacks selected weights")
    _reference(root, summary["selected_weights"])
    if (
        plan["producer"].endswith("evaluation")
        and summary["selected_weights"] != plan["initial_weights"]
    ):
        raise ValueError("evaluated and selected weights differ")
    specs = {s["pass_id"]: s for s in plan["passes"]}
    seen, outputs = {p: [] for p in specs}, {p: [] for p in specs}
    seen_sets = {n: set() for n in specs}
    expected_sets = {n: set(s["expected_keys"]) for n, s in specs.items()}
    pass_order = list(specs)
    events, active, closed = 0, None, set()
    if trace.stat().st_size > MAX_TRACE_BYTES:
        raise ValueError("consumption trace exceeds inspection budget")
    with trace.open(encoding="utf-8") as stream:
        for line in stream:
            r = parse_json(line)
            fields(r, {"sequence", "pass_id", "position", "keys", "outputs", "outputs_sha256"})
            events += 1
            if integer(r["sequence"], "sequence", minimum=1) != events:
                raise ValueError("consumption sequence mismatch")
            name, keys = r["pass_id"], r["keys"]
            if name not in specs or name in closed:
                raise ValueError("invalid consumption pass order")
            if active != name:
                index = 0 if active is None else pass_order.index(active) + 1
                if (
                    index >= len(pass_order)
                    or pass_order[index] != name
                    or (active is not None and seen_sets[active] != expected_sets[active])
                ):
                    raise ValueError("consumption pass schedule differs from observed execution")
                if active is not None:
                    closed.add(active)
            active = name
            if not isinstance(keys, list) or not keys or len(set(keys)) != len(keys):
                raise ValueError("invalid consumption keys")
            spec, position = specs[name], len(seen[name])
            if integer(r["position"], "position") != position or set(keys) & seen_sets[name]:
                raise ValueError("duplicate or shifted consumption position")
            if not set(keys) <= expected_sets[name]:
                raise ValueError("substituted consumption inputs")
            if spec["ordered"] and keys != spec["expected_keys"][position : position + len(keys)]:
                raise ValueError("consumption order mismatch")
            if not isinstance(r["outputs"], list) or len(r["outputs"]) != len(keys):
                raise ValueError("consumption output count mismatch")
            if canonical_digest(r["outputs"]) != r["outputs_sha256"]:
                raise ValueError("consumption outputs changed")
            seen[name].extend(keys)
            seen_sets[name].update(keys)
            outputs[name].extend(r["outputs"])
    if summary["batch_count"] != events or summary["processed_samples"] != sum(
        map(len, seen.values())
    ):
        raise ValueError("consumption counts inconsistent")
    passes = [
        {"pass_id": n, "processed_count": len(k), "order_sha256": canonical_digest(k)}
        for n, k in seen.items()
    ]
    if canonical_digest(summary["passes"]) != canonical_digest(passes):
        raise ValueError("consumption pass summary differs from observed order")
    missing = any(set(specs[n]["expected_keys"]) != set(keys) for n, keys in seen.items())
    if require_complete and (summary["status"] != "complete" or summary["omissions"] or missing):
        raise ValueError("incomplete consumption coverage")
    result = {
        "processed_samples": summary["processed_samples"],
        "producer": plan["producer"],
        "selected_weights": summary["selected_weights"],
        "split_inventory": plan["split_inventory"],
        "passes": [{**specs[n], "keys": seen[n], "outputs": outputs[n]} for n in specs],
    }

    config = manifest["invocation"]["config"]
    if plan["producer"] == "detector_training" and config.get("config", {}).get("backend") in {
        "yolo",
        "rtdetr",
        "ultralytics-yolo",
        "ultralytics-rtdetr",
    }:
        from ..training.ultralytics_consumption import verify_receipt_outputs

        verify_receipt_outputs(
            read_json(artifact("training_summary")),
            result,
            inventory,
            epochs=config["config"]["epochs"],
            batches_path=trace,
        )
    return result
