"""Append-only hash-linked cost journal. Failures/rejections are included in totals.

Integrity is not authenticity: publish the final digest with the study manifest.
Resource seconds are measured separately; they are never added to storage bytes.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..dataset.lineage import canonical_digest
from ..reproduction.review_contracts import digest, fields, number, parse_json, text

KINDS = {"render", "probe", "score", "audit", "repair", "storage", "train", "evaluate"}
OUTCOMES = {"success", "failed", "rejected", "cancelled"}
UNITS = {
    "wall_seconds",
    "simulation_seconds",
    "cpu_seconds",
    "gpu_seconds",
    "audit_seconds",
    "storage_bytes",
}


class CostLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8")
        self._previous = "0" * 64
        self._ids: set[str] = set()

    def record(
        self,
        event_id: str,
        *,
        kind: str,
        outcome: str,
        resources: Mapping[str, float],
        evidence_ref: str,
    ) -> dict[str, Any]:
        text(event_id, "cost event ID")
        text(evidence_ref, "cost evidence reference")
        if event_id in self._ids:
            raise ValueError("cost event needs a unique ID and an evidence reference")
        if kind not in KINDS or outcome not in OUTCOMES or not resources or set(resources) - UNITS:
            raise ValueError("invalid cost kind, outcome or resource units")
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
            for v in resources.values()
        ):
            raise ValueError("resource costs must be finite and non-negative")
        payload = {
            "schema_version": "1.0",
            "event_id": event_id,
            "kind": kind,
            "outcome": outcome,
            "resources": dict(resources),
            "evidence_ref": evidence_ref,
            "previous_sha256": self._previous,
        }
        payload["sha256"] = canonical_digest(payload)
        self._stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._previous = payload["sha256"]
        self._ids.add(event_id)
        return payload

    @contextmanager
    def measure(self, event_id: str, *, kind: str, evidence_ref: str):
        started = time.monotonic()
        outcome = "success"
        primary_error = None
        try:
            yield
        except BaseException as error:
            primary_error = error
            outcome = (
                "cancelled"
                if isinstance(error, (KeyboardInterrupt, InterruptedError))
                else "failed"
            )
            raise
        finally:
            try:
                self.record(
                    event_id,
                    kind=kind,
                    outcome=outcome,
                    resources={"wall_seconds": time.monotonic() - started},
                    evidence_ref=evidence_ref,
                )
            except BaseException as ledger_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    "Cost journal finalization also failed: " + type(ledger_error).__name__
                )

    def close(self) -> None:
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def verify_cost_ledger(path: str | Path) -> dict[str, Any]:
    previous, ids, totals, counts = "0" * 64, set(), dict.fromkeys(sorted(UNITS), 0.0), {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = parse_json(line)
        fields(
            row,
            {
                "schema_version",
                "event_id",
                "kind",
                "outcome",
                "resources",
                "evidence_ref",
                "previous_sha256",
                "sha256",
            },
        )
        current = digest(row.pop("sha256"))
        if (
            row["schema_version"] != "1.0"
            or row["previous_sha256"] != previous
            or canonical_digest(row) != current
        ):
            raise ValueError("cost ledger hash chain/schema mismatch")
        text(row["event_id"], "cost event ID")
        text(row["evidence_ref"], "cost evidence reference")
        if row["event_id"] in ids or row["kind"] not in KINDS or row["outcome"] not in OUTCOMES:
            raise ValueError("invalid or duplicated cost event")
        if not isinstance(row["resources"], dict) or not row["resources"]:
            raise ValueError("empty or invalid cost resource map")
        ids.add(row["event_id"])
        for name, value in row["resources"].items():
            if name not in UNITS:
                raise ValueError("invalid cost resource")
            totals[name] += number(value, "cost resource", minimum=0)
            if not math.isfinite(totals[name]):
                raise ValueError("cost totals overflow")
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
        previous = current
    if not ids:
        raise ValueError("empty cost ledger is not execution evidence")
    return {"event_count": len(ids), "totals": totals, "outcomes": counts, "last_sha256": previous}
