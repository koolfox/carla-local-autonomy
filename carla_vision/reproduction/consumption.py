"""Producer-side sample receipts. A successful model call is not a planned file list.

These are inspectable software records, not trusted-hardware attestations. A caller
that fabricates events cannot be detected without independent execution. Incomplete
runs retain their journal but never obtain a complete report. Prefetch is not logged
as consumption. One recorder owns one process and one pending batch at a time.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

from ..artifacts import fingerprint_file
from ..dataset.lineage import canonical_digest, partition_role
from ..dataset.shards import expand_dataset_roots
from ..dataset.verified import load_verified_dataset
from .review_contracts import digest, fields, integer, parse_json, read_json, relative_path, text
from .study_links import _reference, verify_split

SCHEMA = "1.0"
MAX_EVENTS = 1_000_000
MAX_BYTES = 512 * 1024 * 1024
TASKS = {"detector_training", "detector_evaluation", "imitation_training", "imitation_evaluation"}


def implementation_files(task):
    shared = ["carla_vision/reproduction/consumption.py", "carla_vision/dataset/verified.py"]
    components = {
        "pooled_detector_evaluation": [
            "evaluation/pool.py",
            "evaluation/coco.py",
            "evaluation/metrics.py",
            "reproduction/pooled_consumption.py",
        ],
        "detector_evaluation": [
            "evaluation/runner.py",
            "evaluation/metrics.py",
            "evaluation/coco.py",
            "detectors/factory.py",
        ],
        "detector_training": [
            "training/runner.py",
            "training/backends.py",
            "training/consumption.py",
            "training/ultralytics_consumption.py",
        ],
        "imitation_training": [
            "imitation/runner.py",
            "imitation/dataset.py",
            "imitation/model.py",
            "imitation/objectives.py",
            "imitation/checkpoint.py",
        ],
        "imitation_evaluation": [
            "imitation/evaluate.py",
            "imitation/runner.py",
            "imitation/dataset.py",
            "imitation/model.py",
            "imitation/objectives.py",
        ],
    }
    return sorted(shared + ["carla_vision/" + p for p in components[task]])


def implementation_digest(task):
    """Record the imported package's producer file bytes, not an arbitrary cwd marker."""
    root = Path(__file__).resolve().parents[2]
    return {p: fingerprint_file(root / p)["sha256"] for p in implementation_files(task)}


def sample_key(manifest_sha256: str, sample_id: str) -> str:
    return f"{digest(manifest_sha256)}:{text(sample_id, 'sample identity')}"


def checked_image(path: str | Path, expected_sha256: str) -> tuple[np.ndarray, str]:
    """Hash and decode one immutable byte buffer, avoiding a hash/decode race."""
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("unsafe or oversized RGB file")
    with path.open("rb") as stream:
        data = stream.read(128 * 1024 * 1024 + 1)
    if len(data) > 128 * 1024 * 1024:
        raise ValueError("consumed RGB exceeded the read budget")
    actual = hashlib.sha256(data).hexdigest()
    if actual != digest(expected_sha256):
        raise ValueError("consumed RGB bytes differ from the verified source")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("consumed RGB cannot be decoded")
    return image, actual


def tensor_digest(*values) -> str:
    """Fingerprint tensors at the input boundary (including shape and dtype)."""
    hashed = hashlib.sha256()
    for value in values:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject or not np.isfinite(array).all():
            raise ValueError("nonfinite or object tensor cannot be consumption evidence")
        header = canonical_digest({"shape": list(array.shape), "dtype": str(array.dtype)})
        hashed.update(header.encode())
        hashed.update(array.tobytes())
    return hashed.hexdigest()


def _selection(inventory, sources, phases):
    rows = {sample_key(s["manifest_sha256"], s["sample_id"]): s for s in inventory["samples"]}
    if not isinstance(phases, Mapping) or not phases:
        raise ValueError("consumption needs nonempty phase partitions")
    for phase, partitions in phases.items():
        if phase not in {"train", "val", "val_final", "test", "evaluation"}:
            raise ValueError("unknown consumption phase")
        if (
            not isinstance(partitions, list)
            or not partitions
            or len(set(partitions)) != len(partitions)
        ):
            raise ValueError("phase partition list must be nonempty and unique")
        roles = {partition_role(p) for p in partitions}
        if "unassigned" in roles or "acquisition_pool" in roles:
            raise ValueError("unfrozen partition in consumption plan")
        if (
            (phase == "train" and roles != {"train"})
            or (phase in {"val", "val_final"} and roles != {"val"})
            or (phase == "test" and roles != {"test"})
        ):
            raise ValueError("consumption phase violates source partition role")
    expected = {phase: [] for phase in phases}
    manifests, image_paths, seen = [], {}, set()
    for root in expand_dataset_roots(sources):
        dataset = load_verified_dataset(root)
        ref = fingerprint_file(root / "manifest.json")
        if ref["sha256"] in seen:
            raise ValueError("duplicate canonical consumption source")
        seen.add(ref["sha256"])
        manifests.append(ref)
        for sample in dataset.dataset["samples"]:
            key = sample_key(ref["sha256"], sample["sample_id"])
            if key not in rows or rows[key]["partition"] != sample["split"]:
                raise ValueError("consumption source outside frozen sample inventory")
            image_paths[(root / sample["rgb"]["path"]).resolve()] = key
            for phase, partitions in phases.items():
                if sample["split"] in partitions:
                    expected[phase].append(key)
    if any(not values for values in expected.values()):
        raise ValueError("empty consumption phase")
    return rows, expected, manifests, image_paths


class ConsumptionRecorder:
    def __init__(self, tracker, inventory_path, sources, *, phases, task, model):
        if task not in TASKS:
            raise ValueError("unsupported consumption task")
        self.tracker = tracker
        if isinstance(model, Mapping) and "path" in model:
            _reference(tracker.run_dir, {k: model[k] for k in ("path", "sha256", "size_bytes")})
        inventory_path = Path(inventory_path).resolve(strict=True)
        self.inventory_ref = fingerprint_file(inventory_path)
        inventory = verify_split(inventory_path)
        self.rows, self.expected, refs, self.image_paths = _selection(inventory, sources, phases)
        self.plan = {
            "schema_version": SCHEMA,
            "run_id": tracker.run_id,
            "task": task,
            "inventory": self.inventory_ref,
            "source_manifests": refs,
            "phases": dict(phases),
            "expected": self.expected,
            "model": model,
            "scope": "single_process_post_success_software_receipts",
            "implementation": implementation_digest(task),
        }
        self.plan_path = tracker.artifact_path("consumption/plan.json")
        with self.plan_path.open("x", encoding="utf-8") as stream:
            json.dump(self.plan, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
        self.events_path = tracker.artifact_path("consumption/events.jsonl")
        self.stream = self.events_path.open("x", encoding="utf-8")
        self.pid, self.previous, self.sequence = os.getpid(), "0" * 64, 0
        self.active, self.pending = None, None
        self.seen, self.passes = [], []
        self.seen_set = set()
        self.expected_sets = {k: set(v) for k, v in self.expected.items()}
        self.finished = False

    def __enter__(self):
        return self

    def _event(self, kind, **payload):
        if os.getpid() != self.pid or self.finished:
            raise ValueError("closed or cross-process consumption recorder")
        row = {"sequence": self.sequence, "previous": self.previous, "kind": kind, **payload}
        row["sha256"] = canonical_digest(row)
        data = json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
        if (
            len(data.encode()) > 16 * 1024 * 1024
            or self.sequence >= MAX_EVENTS
            or self.stream.tell() + len(data.encode()) > MAX_BYTES
        ):
            raise ValueError("consumption journal budget exceeded")
        self.stream.write(data)
        self.stream.flush()
        self.previous = row["sha256"]
        self.sequence += 1

    def begin(self, phase: str, epoch: int, *, ordered: bool = False):
        integer(epoch, "epoch")
        if self.active is not None or phase not in self.expected or type(ordered) is not bool:
            raise ValueError("invalid or overlapping consumption pass")
        if any(p["phase"] == phase and p["epoch"] == epoch for p in self.passes):
            raise ValueError("duplicate consumption pass")
        self.active = {"phase": phase, "epoch": epoch, "ordered": ordered}
        self.seen, self.pending = [], None
        self.seen_set = set()
        self._event("begin", **self.active)

    def prepare(self, keys, rgb_sha256, *, input_sha256):
        """Validate before model work; this does NOT count as successful consumption."""
        if self.active is None or self.pending is not None:
            raise ValueError("no active pass or unacknowledged prefetched batch")
        keys, hashes = list(keys), list(rgb_sha256)
        if not keys or len(keys) != len(hashes) or len(set(keys)) != len(keys):
            raise ValueError("invalid or duplicate batch identities")
        expected = self.expected[self.active["phase"]]
        if set(keys) - self.expected_sets[self.active["phase"]] or set(keys) & self.seen_set:
            raise ValueError("duplicate/substituted/wrong-partition consumed sample")
        if self.active["ordered"] and keys != expected[len(self.seen) : len(self.seen) + len(keys)]:
            raise ValueError("consumed samples violate declared evaluation order")
        for key, value in zip(keys, hashes, strict=True):
            if digest(value) != self.rows[key]["rgb_sha256"]:
                raise ValueError("consumed RGB hash mismatch")
        self.pending = {"keys": keys, "rgb_sha256": hashes, "input_sha256": digest(input_sha256)}
        return self.pending

    def commit(self, token, outputs):
        """Call only after forward/loss/backward (and any scheduled optimizer step) succeeds."""
        if token is not self.pending or token is None:
            raise ValueError("unknown or already committed consumption batch")
        if not isinstance(outputs, list) or len(outputs) != len(token["keys"]):
            raise ValueError("per-sample outputs do not align with consumed batch")
        # Serialization rejects NaN before the event is considered committed.
        self._event("batch", **token, outputs=outputs)
        self.seen.extend(token["keys"])
        self.seen_set.update(token["keys"])
        self.pending = None

    def end(self):
        if self.active is None or self.pending is not None:
            raise ValueError("unacknowledged batch or missing pass")
        expected = self.expected[self.active["phase"]]
        if len(self.seen) != len(expected) or set(self.seen) != set(expected):
            raise ValueError("missing consumed samples; pass cannot be certified")
        record = {
            **self.active,
            "sample_count": len(self.seen),
            "order_sha256": canonical_digest(self.seen),
        }
        self._event("end", **record)
        self.passes.append(record)
        self.active = None
        os.fsync(self.stream.fileno())

    def finish(self, checkpoints):
        if self.active is not None or not self.passes or self.finished:
            raise ValueError("incomplete consumption run")
        if set(self.expected) - {p["phase"] for p in self.passes}:
            raise ValueError("required consumption phase was never executed")
        if not isinstance(checkpoints, Mapping) or not checkpoints:
            raise ValueError("consumption must bind actual selected/evaluated weights")
        for ref in checkpoints.values():
            _reference(self.tracker.run_dir, ref)
        model = self.plan["model"]
        if isinstance(model, Mapping) and "path" in model:
            _reference(
                self.tracker.run_dir, {k: model[k] for k in ("path", "sha256", "size_bytes")}
            )
        # Detect source or inventory changes during execution. This is not a filesystem sandbox.
        _reference(self.tracker.run_dir, self.inventory_ref)
        inventory = verify_split(self.inventory_ref["path"])
        _, expected, refs, _ = _selection(
            inventory,
            [Path(r["path"]).parent for r in self.plan["source_manifests"]],
            self.plan["phases"],
        )
        if expected != self.expected or refs != self.plan["source_manifests"]:
            raise ValueError("source selection changed during consumption")
        if implementation_digest(self.plan["task"]) != self.plan["implementation"]:
            raise ValueError("producer source changed during recorded execution")
        self._event("complete", checkpoints=dict(checkpoints))
        os.fsync(self.stream.fileno())
        path = self.tracker.artifact_path("consumption/report.json")
        report = {
            "schema_version": SCHEMA,
            "run_id": self.tracker.run_id,
            "status": "complete",
            "plan": fingerprint_file(self.plan_path),
            "events": fingerprint_file(self.events_path),
            "passes": self.passes,
            "checkpoints": dict(checkpoints),
            "event_count": self.sequence,
            "last_event_sha256": self.previous,
        }
        with path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
        self.tracker.register_artifact(path, role="consumed_sample_report")
        self.finished = True
        return report

    def __exit__(self, exc_type, exc, tb):
        try:
            if not self.finished:
                self._event(
                    "aborted",
                    reason=str(exc or "producer did not finalize")[:1000],
                    completed_samples_in_pass=len(self.seen),
                    expected_samples_in_pass=len(self.expected[self.active["phase"]])
                    if self.active
                    else 0,
                )
            self.stream.close()
            self.tracker.register_artifact(self.plan_path, role="consumed_sample_plan")
            self.tracker.register_artifact(self.events_path, role="consumed_sample_events")
        except BaseException as secondary:
            self.stream.close()
            if exc is None:
                raise
            exc.add_note(f"consumption journal finalization failed: {secondary}")
        if exc is None and not self.finished:
            raise ValueError("producer left consumption unfinalized")
        return False


def _artifact(root, manifest, role):
    candidates = [a for a in manifest["artifacts"] if a["role"] == role]
    if len(candidates) != 1:
        raise ValueError(f"missing/ambiguous producer {role}; legacy evidence is not attested")
    entry = candidates[0]
    path = relative_path(root, entry["path"])
    actual = fingerprint_file(path)
    if any(actual[k] != entry[k] for k in ("sha256", "size_bytes")):
        raise ValueError("registered consumption bytes changed")
    return path


def verify_consumption(root, *, inventory_path=None):
    """Replay actual event coverage against the verified frozen sample inventory."""
    from ..verification import verify_research_object

    root = Path(root).resolve(strict=True)
    verify_research_object(root, verify_references=True, deep=False, reject_unregistered=True)
    manifest = read_json(root / "manifest.json")
    plan_path = _artifact(root, manifest, "consumed_sample_plan")
    events_path = _artifact(root, manifest, "consumed_sample_events")
    report_path = _artifact(root, manifest, "consumed_sample_report")
    plan, report = read_json(plan_path), read_json(report_path)
    fields(
        plan,
        {
            "schema_version",
            "run_id",
            "task",
            "inventory",
            "source_manifests",
            "phases",
            "expected",
            "model",
            "scope",
            "implementation",
        },
    )
    fields(
        report,
        {
            "schema_version",
            "run_id",
            "status",
            "plan",
            "events",
            "passes",
            "checkpoints",
            "event_count",
            "last_event_sha256",
        },
    )
    if (
        plan["schema_version"] != SCHEMA
        or report["schema_version"] != SCHEMA
        or plan["run_id"] != manifest["run_id"]
        or report["run_id"] != manifest["run_id"]
        or report["status"] != "complete"
        or plan["task"] not in TASKS
        or plan["scope"] != "single_process_post_success_software_receipts"
    ):
        raise ValueError("consumption envelope/producer identity mismatch")
    if not isinstance(plan["implementation"], dict) or set(plan["implementation"]) != set(
        implementation_files(plan["task"])
    ):
        raise ValueError("consumption implementation inventory is incomplete")
    for hashed in plan["implementation"].values():
        digest(hashed)
    if (
        _reference(root, report["plan"]) != plan_path
        or _reference(root, report["events"]) != events_path
    ):
        raise ValueError("consumption report refers to different registered files")
    inv_path = _reference(root, plan["inventory"])
    if inventory_path and fingerprint_file(inventory_path)["sha256"] != plan["inventory"]["sha256"]:
        raise ValueError("consumption belongs to a different split inventory")
    inventory = verify_split(inv_path)
    roots = [_reference(root, ref).parent for ref in plan["source_manifests"]]
    rows, expected, refs, _ = _selection(inventory, roots, plan["phases"])
    if refs != plan["source_manifests"] or expected != plan["expected"]:
        raise ValueError("declared consumption plan differs from frozen source selection")
    if isinstance(plan["model"], Mapping) and "path" in plan["model"]:
        _reference(root, {k: plan["model"][k] for k in ("path", "sha256", "size_bytes")})
    if events_path.stat().st_size > MAX_BYTES:
        raise ValueError("consumption journal exceeds inspection budget")
    active, seen, passes, batches = None, [], [], []
    seen_set = set()
    expected_sets = {k: set(v) for k, v in expected.items()}
    previous, complete, count = "0" * 64, False, 0
    checkpoints = None
    with events_path.open(encoding="utf-8") as stream:
        for line in stream:
            if count >= MAX_EVENTS or len(line) > 16 * 1024 * 1024 or complete:
                raise ValueError("oversized or trailing consumption event")
            row = parse_json(line)
            hashed = row.pop("sha256")
            if (
                row["sequence"] != count
                or type(row["sequence"]) is not int
                or row["previous"] != previous
                or canonical_digest(row) != hashed
            ):
                raise ValueError("consumption journal sequence/hash broken")
            count += 1
            previous = hashed
            kind = row["kind"]
            base = {"sequence", "previous", "kind"}
            if kind == "begin":
                fields(row, base | {"phase", "epoch", "ordered"})
                integer(row["epoch"], "epoch")
                if (
                    active is not None
                    or row["phase"] not in expected
                    or type(row["ordered"]) is not bool
                ):
                    raise ValueError("invalid consumption pass")
                if any(p["phase"] == row["phase"] and p["epoch"] == row["epoch"] for p in passes):
                    raise ValueError("duplicate consumption pass")
                active = {k: row[k] for k in ("phase", "epoch", "ordered")}
                seen, batches = [], []
                seen_set = set()
            elif kind == "batch":
                fields(row, base | {"keys", "rgb_sha256", "input_sha256", "outputs"})
                digest(row["input_sha256"])
                keys, hashes, outputs = row["keys"], row["rgb_sha256"], row["outputs"]
                if (
                    active is None
                    or not isinstance(keys, list)
                    or not keys
                    or not isinstance(hashes, list)
                    or not isinstance(outputs, list)
                    or len(keys) != len(hashes)
                    or len(keys) != len(outputs)
                ):
                    raise ValueError("batch is not bound to a successful pass")
                phase_expected = expected[active["phase"]]
                if (
                    len(set(keys)) != len(keys)
                    or set(keys) & seen_set
                    or set(keys) - expected_sets[active["phase"]]
                ):
                    raise ValueError("duplicate, extra or substituted consumed samples")
                if active["ordered"] and keys != phase_expected[len(seen) : len(seen) + len(keys)]:
                    raise ValueError("consumption order differs")
                if any(
                    rows[k]["rgb_sha256"] != digest(h) for k, h in zip(keys, hashes, strict=True)
                ):
                    raise ValueError("consumed source hash differs")
                seen.extend(keys)
                seen_set.update(keys)
                batches.append({"keys": keys, "outputs": outputs})
            elif kind == "end":
                fields(row, base | {"phase", "epoch", "ordered", "sample_count", "order_sha256"})
                if (
                    active is None
                    or any(row[k] != active[k] for k in active)
                    or set(seen) != set(expected[active["phase"]])
                ):
                    raise ValueError("missing consumed samples or mismatched pass end")
                integer(row["sample_count"], "sample count", minimum=1)
                if row["sample_count"] != len(seen) or row["order_sha256"] != canonical_digest(
                    seen
                ):
                    raise ValueError("consumption count/order inconsistent")
                passes.append(
                    {
                        **active,
                        "sample_count": len(seen),
                        "order_sha256": row["order_sha256"],
                        "keys": seen,
                        "batches": batches,
                    }
                )
                active = None
            elif kind == "complete":
                fields(row, base | {"checkpoints"})
                if active is not None or not passes or set(expected) - {p["phase"] for p in passes}:
                    raise ValueError("incomplete consumption report")
                checkpoints = row["checkpoints"]
                if not isinstance(checkpoints, Mapping) or not checkpoints:
                    raise ValueError("missing consumed model weights")
                for ref in checkpoints.values():
                    _reference(root, ref)
                complete = True
            else:
                raise ValueError("aborted or unknown consumption event is not complete evidence")
    integer(report["event_count"], "event count", minimum=1)
    summaries = [{k: v for k, v in p.items() if k not in {"keys", "batches"}} for p in passes]
    if (
        not complete
        or summaries != report["passes"]
        or checkpoints != report["checkpoints"]
        or count != report["event_count"]
        or previous != report["last_event_sha256"]
    ):
        raise ValueError("consumption report differs from actual event journal")
    return {
        "status": "passed",
        "plan": plan,
        "expected": expected,
        "passes": passes,
        "checkpoints": checkpoints,
        "rows": rows,
        "report": fingerprint_file(report_path),
    }
