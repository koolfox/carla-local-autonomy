"""Advance one research evidence stage through the existing JobManager.

No second scheduler, service, retry loop, shell executor, or automatic holdout
access is introduced. Human-audit stages still need independent audit artifacts.
Receipts make the submit/crash ambiguity visible: a started or uncertain receipt
is reconciled, never silently submitted twice.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from ..artifacts import fingerprint_file
from ..reproduction.review_contracts import number, read_json, relative_path
from ..scenarios.cycle import REQUIRED_ROLE, STAGES, CycleJournal
from ..verification import verify_research_object
from .commands import build_command_plan
from .contracts import OperatorJobRequest
from .jobs import JobManager, _atomic_json

STAGE_KINDS = {
    "planned": {"scenario_plan", "acquisition_plan"},
    "selected": {"acquisition_pool", "coreset_select"},
    "collected": {"native_capture"},
    "verified": {"dataset_qa"},
    "frozen": {"dataset_derive"},
    "trained": {"train"},
}


def execute_cycle_job(
    cycle: CycleJournal,
    manager: JobManager,
    request: OperatorJobRequest,
    *,
    stage: str,
    receipt_path: str | Path,
    evidence_relative="manifest.json",
    timeout_seconds=600.0,
    executable=sys.executable,
):
    number(timeout_seconds, "timeout_seconds", minimum=0)
    if timeout_seconds <= 0:
        raise ValueError("positive bounded job wait required")
    if (
        stage not in STAGE_KINDS
        or request.kind not in STAGE_KINDS[stage]
        or cycle.stage not in STAGES
        or STAGES.index(stage) != STAGES.index(cycle.stage) + 1
    ):
        raise ValueError("job kind does not establish the next registered evidence stage")
    if request.parameters.get("dry_run") is True:
        raise ValueError("dry-run cannot advance an executed cycle stage")
    # Offline derivation cannot be renamed to a physical recapture.
    if stage == "frozen" and request.parameters.get("action") != "FREEZE":
        raise ValueError("frozen stage requires explicit FREEZE action")
    if stage == "collected" and request.parameters.get("partition") not in {
        "train",
        "acquisition_pool",
    }:
        raise ValueError("adaptive collection cannot open validation or test")
    plan = build_command_plan(request, workspace=manager.workspace, executable=executable)
    if plan.expected_output is None:
        raise ValueError("cycle job requires a producer output")
    receipt = Path(receipt_path).absolute()
    receipt.relative_to(manager.workspace)
    relative_path(
        manager.workspace, receipt.relative_to(manager.workspace).as_posix(), must_exist=False
    )
    # Exclusive creation before submit is an at-most-once *attempt*, not a
    # distributed transaction. Any ambiguous submission requires reconciliation.
    receipt.parent.mkdir(parents=True, exist_ok=True)
    initial = {
        "schema_version": "1.0",
        "status": "submitting",
        "stage": stage,
        "cycle_id": cycle.identity["cycle_id"],
        "request": request.as_dict(),
        "job_id": None,
        "journal_previous_sha256": cycle.previous,
    }
    import json
    import os

    with receipt.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(initial, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        job = manager.submit(request, plan)
        initial.update(status="submitted", job_id=job["job_id"])
        _atomic_json(receipt, initial)
        deadline = time.monotonic() + timeout_seconds
        while True:
            state = manager.get(job["job_id"])
            if state["status"] in {"success", "failed", "stopped"}:
                break
            if time.monotonic() >= deadline:
                try:
                    manager.stop(job["job_id"])
                except RuntimeError:
                    pass  # It may have just become terminal.
                initial["status"] = "timeout_reconciliation_required"
                _atomic_json(receipt, initial)
                raise TimeoutError("job outcome requires reconciliation; no resubmission")
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        if state["status"] != "success":
            initial.update(status="failed", job_status=state["status"])
            _atomic_json(receipt, initial)
            raise RuntimeError("research job failed; cycle stage not advanced")
        # Dataset QA emits a QA run; the verified stage needs the original
        # dataset manifest as well, not a QA JSON masquerading as a dataset.
        output = plan.expected_output
        if stage == "verified":
            output = (manager.workspace / str(request.parameters["dataset"])).resolve(strict=True)
            relative = output.relative_to(manager.workspace).as_posix()
            output = relative_path(manager.workspace, relative)
        evidence = relative_path(output, evidence_relative)
        # Every receipt is tied to a real retained JobManager session.
        session = Path(state["session_path"]) / "manifest.json"
        verify_research_object(
            session.parent, verify_references=True, deep=False, reject_unregistered=True
        )
        event = cycle.advance(stage, {REQUIRED_ROLE[stage]: evidence, "operator_session": session})
        initial.update(
            status="recorded",
            producer_evidence=fingerprint_file(evidence),
            operator_session=fingerprint_file(session),
            journal_event_sha256=event["sha256"],
        )
        _atomic_json(receipt, initial)
        return initial
    except BaseException as error:
        if initial["status"] in {"submitting", "submitted"}:
            initial["status"] = "reconciliation_required"
            try:
                _atomic_json(receipt, initial)
            except BaseException as secondary:
                error.add_note(type(secondary).__name__)
        raise


def inspect_cycle_receipt(path, manager: JobManager):
    """Read-only reconciliation; deliberately never retry or advance the cycle."""
    receipt = read_json(path)
    result = {"receipt": receipt, "resubmitted": False, "journal_advanced": False}
    if receipt.get("job_id"):
        result["job"] = manager.get(receipt["job_id"])
    else:
        result["ambiguity"] = "submission may have occurred before durable job-ID recording"
    return result
