"""Durable, restartable research cycle journal, owned by the existing job process.

This coordinates evidence states, not concurrent model weight updates or vehicle
actuation. Stage success requires producer artifacts; predictions alone are not
a dataset, a trained model, or a successful driving trial.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..artifacts import fingerprint_file
from ..dataset.lineage import canonical_digest, require_train_parent
from ..dataset.verified import load_verified_dataset
from ..verification import verify_research_object

STAGES = (
    "created",
    "planned",
    "probed",
    "audited",
    "selected",
    "collected",
    "verified",
    "frozen",
    "trained",
    "evaluated",
    "complete",
)
REQUIRED_ROLE = dict(
    zip(
        STAGES[1:],
        (
            "plan",
            "probe",
            "audit",
            "selection",
            "collection",
            "dataset",
            "training_dataset",
            "model_training",
            "evaluation",
            "report",
        ),
        strict=True,
    )
)


def _validate_evidence(stage: str, artifacts: Mapping[str, str | Path]) -> dict[str, Any]:
    if REQUIRED_ROLE.get(stage) not in artifacts:
        raise ValueError(f"{stage} requires {REQUIRED_ROLE.get(stage)} evidence")
    refs = {role: fingerprint_file(path) for role, path in artifacts.items()}
    if stage in {"verified", "frozen"}:
        role = REQUIRED_ROLE[stage]
        dataset = load_verified_dataset(Path(artifacts[role]).parent)
        if Path(artifacts[role]).name != "manifest.json":
            raise ValueError("dataset evidence must reference its tracker manifest")
        for partition in dataset.partitions:
            require_train_parent(partition)
            if stage == "frozen" and partition != "train":
                raise ValueError("frozen training release must contain only train samples")
    if stage in {"trained", "evaluated"}:
        path = Path(artifacts[REQUIRED_ROLE[stage]])
        if path.name != "manifest.json":
            raise ValueError("model/evaluation evidence must name the producer tracker")
        verify_research_object(
            path.parent, verify_references=True, deep=False, reject_unregistered=True
        )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        roles = {item["role"] for item in manifest["artifacts"]}
        expected = (
            {"training_summary", "imitation_training_summary"}
            if stage == "trained"
            else {"evaluation_summary", "imitation_test_summary"}
        )
        if not roles & expected:
            raise ValueError("producer artifact roles do not establish the requested cycle stage")
        config = manifest["invocation"]["config"]
        if config.get("synthetic") is True or config.get("dry_run") is True:
            raise ValueError("synthetic/dry-run output cannot advance an empirical cycle")
    return refs


class CycleJournal:
    def __init__(
        self,
        path: str | Path,
        *,
        cycle_id: str,
        partition: str,
        protocol_ref: str | Path,
        resume: bool = False,
    ):
        require_train_parent(partition)
        if not cycle_id or not isinstance(cycle_id, str):
            raise ValueError("cycle ID must be a non-empty string")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        identity = {
            "cycle_id": cycle_id,
            "partition": partition,
            "protocol": fingerprint_file(protocol_ref),
        }
        if resume:
            report = verify_cycle(self.path)
            if report["identity"] != identity or report["stage"] in {"complete", "failed"}:
                raise ValueError("cycle identity changed or cycle is terminal")
            self.stage, self.previous = report["stage"], report["last_sha256"]
            self.sequence = report["event_count"]
            self._stream = self.path.open("a", encoding="utf-8")
        else:
            self._stream = self.path.open("x", encoding="utf-8")
            self.stage, self.previous, self.sequence = "created", "0" * 64, 0
            self._append("created", {"protocol": identity["protocol"]}, identity=identity)
        self.identity = identity

    def _append(self, stage, evidence, **extra):
        row = {
            "schema_version": "1.0",
            "sequence": self.sequence,
            "stage": stage,
            "evidence": evidence,
            "previous_sha256": self.previous,
            **extra,
        }
        row["sha256"] = canonical_digest(row)
        self._stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self.stage, self.previous, self.sequence = stage, row["sha256"], self.sequence + 1
        return row

    def advance(self, stage: str, artifacts: Mapping[str, str | Path]):
        if (
            self.stage in {"failed", "complete"}
            or stage not in STAGES
            or STAGES.index(stage) != STAGES.index(self.stage) + 1
        ):
            raise ValueError("cycle stages cannot be skipped, repeated or reopened")
        return self._append(stage, _validate_evidence(stage, artifacts))

    def fail(self, reason: str):
        if self.stage in {"failed", "complete"} or not reason.strip():
            raise ValueError("failure requires a non-terminal cycle and reason")
        return self._append("failed", {}, reason=reason)

    def close(self):
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, error, _traceback):
        try:
            if error is not None and self.stage not in {"failed", "complete"}:
                try:
                    self.fail(type(error).__name__)  # no secrets in error text
                except BaseException as journal_error:
                    error.add_note(
                        "Cycle journal finalization failed: " + type(journal_error).__name__
                    )
        finally:
            self.close()


def verify_cycle(path: str | Path) -> dict[str, Any]:
    previous, stage, identity, count = "0" * 64, None, None, 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        digest = row.pop("sha256")
        if (
            canonical_digest(row) != digest
            or row["previous_sha256"] != previous
            or row["sequence"] != count
        ):
            raise ValueError("cycle hash chain/order mismatch")
        next_stage = row["stage"]
        if count == 0:
            identity = row["identity"]
            require_train_parent(identity["partition"])
            if next_stage != "created":
                raise ValueError("cycle must begin at created")
        elif stage in {"failed", "complete"} or (
            next_stage != "failed"
            and (next_stage not in STAGES or STAGES.index(next_stage) != STAGES.index(stage) + 1)
        ):
            raise ValueError("invalid cycle transition")
        for ref in row["evidence"].values():
            if fingerprint_file(ref["path"]) != ref:
                raise ValueError("cycle source artifact changed")
        if count and next_stage not in {"failed"}:
            _validate_evidence(
                next_stage, {role: ref["path"] for role, ref in row["evidence"].items()}
            )
        previous, stage, count = digest, next_stage, count + 1
    if not count:
        raise ValueError("empty cycle journal")
    return {"identity": identity, "stage": stage, "event_count": count, "last_sha256": previous}
