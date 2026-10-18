"""Verify reviewer evidence completeness, not scientific truth or journal acceptance.

The fixed role catalog cannot be supplied or weakened by a submission manifest.
Run `python -m carla_vision.reproduction.reviewer init --output submission.json`
and replace null entries only with actual artifact fingerprints.
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from ..artifacts import fingerprint_file
from ..dataset.leakage import audit_corridors
from ..evaluation.study_statistics import agreement, validate_run_matrix
from ..scenarios.claims import check_claims
from ..scenarios.costs import verify_cost_ledger
from .closure import build_closure, verify_closure
from .review_contracts import digest, fields, integer, number, read_json, relative_path, text

DOCUMENTS = {
    "dataset_card": {"unit", "ontology", "splits", "label_definition", "exclusions", "limitations"},
    "model_card": {
        "input_contract",
        "architecture",
        "initial_weights",
        "training_protocol",
        "limitations",
    },
    "environment": {"python", "platform", "packages", "hardware"},
    "baseline_registry": {"methods", "citations", "implementation_scope", "fairness_controls"},
    "manuscript": {"title", "claims", "tables", "limitations", "references", "author_approval"},
    "independent_reproduction": {
        "reviewer",
        "affiliation",
        "independent",
        "source_commit",
        "result",
        "execution_log",
    },
}
ROLES = tuple(
    sorted(
        (
            *DOCUMENTS,
            "source_snapshot",
            "dependency_lock",
            "tests",
            "protocol",
            "geometry",
            "label_audit",
            "results",
            "costs",
            "claim_registry",
            "rights",
        )
    )
)
EMPIRICAL_EXTRA = {
    "results",
    "label_audit",
    "costs",
    "claim_registry",
    "rights",
    "independent_reproduction",
}


def template() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "study_id": "replace-with-registered-study-id",
        "mode": "empirical_simulation",
        "source_commit": None,
        "artifacts": dict.fromkeys(ROLES),
        "limitations": [],
    }


def _nonempty(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return value is not None


def _ref(base: Path, value: Any) -> Path:
    fields(value, {"path", "sha256", "size_bytes"})
    digest(value["sha256"])
    integer(value["size_bytes"], "artifact size", minimum=1)
    raw = Path(text(value["path"], "artifact path"))
    path = raw if raw.is_absolute() else relative_path(base, value["path"])
    if path.is_symlink():
        raise ValueError("evidence symlink is not allowed")
    actual = fingerprint_file(path)
    if any(actual[k] != value[k] for k in ("sha256", "size_bytes")):
        raise ValueError("reviewer evidence digest/size changed")
    return path.resolve(strict=True)


def _source_archive(path: Path, commit: str) -> None:
    # Inspect only; never extract or execute a reviewer-supplied source archive.
    from .source_snapshot import verify_source_objects

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = [x.filename for x in entries]
            if sum(x.file_size for x in entries) > 512 * 1024 * 1024:
                raise ValueError("source archive exceeds inspection budget")
            if any((x.external_attr >> 16) & 0o170000 == 0o120000 for x in entries):
                raise ValueError("source archive symlink")
            recorded = archive.read("SOURCE_COMMIT.txt").decode().strip()
            verify_source_objects(names, archive.read, commit)
    else:
        with tarfile.open(path, "r:*") as archive:
            entries = archive.getmembers()
            names = [x.name for x in entries]
            if sum(x.size for x in entries) > 512 * 1024 * 1024:
                raise ValueError("source archive exceeds inspection budget")
            if any(not (x.isfile() or x.isdir()) for x in entries):
                raise ValueError("source archive special/link entries")
            stream = archive.extractfile("SOURCE_COMMIT.txt")
            if stream is None:
                raise ValueError("missing source identity")
            with stream:
                recorded = stream.read(128).decode().strip()

            def read_member(name):
                member = archive.extractfile(name)
                if member is None:
                    raise ValueError("source proof member is not a file")
                with member:
                    return member.read()

            verify_source_objects(names, read_member, commit)
    if recorded != commit:
        raise ValueError("source archive does not bind the selected commit")
    if len(set(names)) != len(names):
        raise ValueError("duplicate source archive entries")
    for name in names:
        pieces = name.rstrip("/").split("/")
        if (
            name.startswith("/")
            or "\\" in name
            or ":" in name
            or any(p in {"", ".", ".."} for p in pieces)
        ):
            raise ValueError("unsafe source archive path")
    if not any(n.startswith("carla_vision/") and n.endswith(".py") for n in names):
        raise ValueError("archive has no project source")
    if not any(n.startswith("tests/") and n.endswith(".py") for n in names):
        raise ValueError("archive has no tests")


def _check_role(role: str, path: Path, manifest: dict, *, empirical: bool) -> dict:
    if role == "dependency_lock":
        if path.stat().st_size < 32:
            raise ValueError("dependency lock is empty/trivial")
        return {"status": "present", "verification": "fingerprinted_lock_bytes"}
    if role == "source_snapshot":
        _source_archive(path, manifest["source_commit"])
        return {"status": "passed", "verification": "source_archive_identity_and_safe_paths"}
    if role == "tests" and (empirical or path.suffix == ".json"):
        from .test_execution import verify_test_execution

        return verify_test_execution(path, source_commit=manifest["source_commit"])
    if role == "tests":
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("JUnit exceeds parsing budget")
        raw = path.read_bytes()
        if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
            raise ValueError("JUnit external entities are not allowed")
        xml = ElementTree.fromstring(raw)
        cases = list(xml.iter("testcase"))
        if not cases or any(list(c.iter("failure")) or list(c.iter("error")) for c in cases):
            raise ValueError("JUnit is empty or contains failed tests")
        skipped = sum(bool(list(c.iter("skipped"))) for c in cases)
        if skipped:
            raise ValueError("reviewer validation suite contains unaccounted skipped tests")
        return {"status": "passed", "cases": len(cases), "empirical_validation": False}
    payload = read_json(path)
    if role in DOCUMENTS:
        fields(payload, DOCUMENTS[role], optional={"schema_version"})
        if any(not _nonempty(payload[k]) for k in DOCUMENTS[role]):
            raise ValueError(f"incomplete {role}")
        if role == "independent_reproduction":
            if (
                payload["independent"] is not True
                or payload["result"] != "passed"
                or payload["source_commit"] != manifest["source_commit"]
            ):
                raise ValueError("independent reproduction does not match this source")
            _ref(path.parent, payload["execution_log"])
        if role == "manuscript" and payload["author_approval"] is not True:
            raise ValueError("manuscript authors have not approved the claim/table inventory")
        return {
            "status": "present",
            "verification": "structure_and_fingerprint_not_content_peer_review",
        }
    if role == "protocol":
        from ..scenarios.study import validate_protocol

        validate_protocol(payload)
        if payload["study_id"] != manifest["study_id"]:
            raise ValueError("protocol study identity differs from submission")
        if empirical and payload["purpose"] != "confirmatory":
            raise ValueError("empirical review requires the registered confirmatory protocol")
    elif role == "geometry":
        from ..dataset.trajectory_audit import captured_routes

        if payload.get("source_kind") == "verified_capture_trajectories":
            fields(
                payload,
                {
                    "schema_version",
                    "source_kind",
                    "routes",
                    "clearance_m",
                    "maximum_step_m",
                    "maximum_segment_checks",
                    "source_artifacts",
                    "datasets",
                    "map_documents",
                    "result",
                    "empirical_model_improvement",
                },
            )
            map_paths = [_ref(path.parent, r) for r in payload["map_documents"]]
            for reference in payload["source_artifacts"]:
                _ref(path.parent, reference)
            computed, refs = captured_routes(
                payload["datasets"], map_paths, maximum_step_m=payload["maximum_step_m"]
            )
            if computed != payload["routes"] or refs != payload["source_artifacts"]:
                raise ValueError("geometry differs from recorded source trajectories")
        else:
            if empirical:
                raise ValueError(
                    "empirical geometric separation must be derived from recorded source trajectories"
                )
            fields(payload, {"routes", "clearance_m", "maximum_segment_checks", "source_artifacts"})
            if not payload["source_artifacts"]:
                raise ValueError("geometry requires recorded sources")
            for reference in payload["source_artifacts"]:
                _ref(path.parent, reference)
        result = audit_corridors(
            payload["routes"],
            clearance_m=payload["clearance_m"],
            maximum_segment_checks=payload["maximum_segment_checks"],
        )
        if result["status"] != "passed":
            raise ValueError("route corridors cross preregistered partitions")
        if "result" in payload and payload["result"] != result:
            raise ValueError("stored geometry audit differs from recomputation")
        return result
    elif role == "label_audit":
        fields(payload, {"synthetic", "blind_to_model", "records", "sampling_protocol"})
        if empirical and payload["synthetic"] is not False:
            raise ValueError("human QA requires non-synthetic review records")
        if payload["blind_to_model"] is not True or not payload["records"]:
            raise ValueError("audit must contain independent blinded reviews")
        _ref(path.parent, payload["sampling_protocol"])
        left, right, ids = [], [], set()
        for row in payload["records"]:
            fields(
                row, {"sample_id", "group_id", "reviewer_a", "reviewer_b", "a", "b", "evidence"},
                optional={"dataset_manifest_sha256"},
            )
            anchor = row.get("dataset_manifest_sha256")
            if anchor is not None:
                digest(anchor)
            identity = (anchor, text(row["sample_id"], "sample ID"))
            if identity in ids or row["reviewer_a"] == row["reviewer_b"]:
                raise ValueError("duplicate sample or non-independent reviewers")
            ids.add(identity)
            for name in ("group_id", "reviewer_a", "reviewer_b"):
                text(row[name], name)
            if row["a"] not in {"valid", "invalid", "ambiguous"} or row["b"] not in {
                "valid",
                "invalid",
                "ambiguous",
            }:
                raise ValueError("invalid audit decision")
            _ref(path.parent, row["evidence"])
            left.append(row["a"])
            right.append(row["b"])
        return {
            "status": "passed",
            "agreement": agreement(left, right),
            "correctness_certified": False,
        }
    elif role == "results":
        fields(payload, {"protocol", "rows", "synthetic", "producers"})
        if empirical and payload["synthetic"] is not False:
            raise ValueError("synthetic results cannot establish an empirical paper")
        matrix_path = _ref(path.parent, payload["protocol"])
        matrix = read_json(matrix_path)
        result = validate_run_matrix(matrix, payload["rows"])
        if not result["complete"]:
            raise ValueError(
                "registered run matrix is incomplete/failed; retain all attempted runs"
            )
        weights = matrix.get("cost_weights")
        from ..scenarios.costs import UNITS

        if not isinstance(weights, dict) or not weights or set(weights) - UNITS:
            raise ValueError("matrix requires explicit cost unit/conversion weights")
        for name, value in weights.items():
            number(value, f"cost weight {name}", minimum=0)
        if not any(weights.values()):
            raise ValueError("cost weights cannot all be zero")
        if set(payload["producers"]) != {r["run_id"] for r in payload["rows"]}:
            raise ValueError("every result row needs exactly one producer binding")
        from ..verification import verify_research_object
        from .review_contracts import parse_json

        ledgers, event_ids = set(), set()
        for row in payload["rows"]:
            binding = payload["producers"][row["run_id"]]
            fields(binding, {"manifest", "summary", "metric_field", "cost_ledger"})
            producer = _ref(path.parent, binding["manifest"])
            summary = _ref(path.parent, binding["summary"])
            ledger_path = _ref(path.parent, binding["cost_ledger"])
            if producer.name != "manifest.json":
                raise ValueError("result requires the actual producer manifest")
            verify_research_object(
                producer.parent, verify_references=True, deep=False, reject_unregistered=True
            )
            source = read_json(producer)
            if source["run_id"] != row["run_id"]:
                raise ValueError("row run identity differs from producer")
            matches = [
                a
                for a in source["artifacts"]
                if relative_path(producer.parent, a["path"]) == summary
            ]
            if len(matches) != 1 or matches[0]["role"] not in {
                "evaluation_summary",
                "empirical_study_result",
                "real_world_result",
                "simulator_driving_result",
                "registered_experiment_result",
            }:
                raise ValueError("result summary has no registered result role")
            measured = read_json(summary)
            value = measured
            for part in text(binding["metric_field"], "metric field").split("."):
                value = value[part]
            if value != row["metric"] or isinstance(value, bool):
                raise ValueError("table metric differs from registered result")
            config = source["invocation"]["config"]
            registered = config.get("review_binding")
            fields(
                registered,
                {"matrix_sha256", "method", "seed", "holdout_sha256", "cost_ledger_sha256"},
            )
            expected = {
                "matrix_sha256": payload["protocol"]["sha256"],
                "method": row["method"],
                "seed": row["seed"],
                "holdout_sha256": row["holdout_sha256"],
                "cost_ledger_sha256": binding["cost_ledger"]["sha256"],
            }
            if registered != expected or type(registered["seed"]) is not int:
                raise ValueError("table method/seed/holdout/cost differs from registered run")
            ledger_hash = binding["cost_ledger"]["sha256"]
            if ledger_hash in ledgers:
                raise ValueError("same cost ledger counted in multiple result rows")
            ledgers.add(ledger_hash)
            costs = verify_cost_ledger(ledger_path)
            records = [parse_json(line) for line in ledger_path.read_text().splitlines()]
            for record in records:
                if record["event_id"] in event_ids:
                    raise ValueError("cost event counted more than once across runs")
                event_ids.add(record["event_id"])
            observed_units = set().union(*(set(r["resources"]) for r in records))
            if not set(weights) <= observed_units:
                raise ValueError("cost ledger omits a preregistered resource unit")
            total = sum(costs["totals"][unit] * weight for unit, weight in weights.items())
            import math

            if not math.isfinite(total) or not math.isclose(
                total, row["cost"], rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError("table cost differs from all-attempt cost ledger")
            if empirical and (
                config.get("synthetic") is not False
                or config.get("dry_run") is True
                or source.get("git", {}).get("commit") != manifest["source_commit"]
                or source.get("git", {}).get("dirty") is not False
            ):
                raise ValueError(
                    "empirical producer needs non-synthetic data and the clean selected source"
                )
        result["cost_scope"] = "registered_conversion_of_all_recorded_attempts_not_unrecorded_work"
        return result
    elif role == "costs":
        raise ValueError("internal: costs use a JSONL verifier")
    elif role == "claim_registry":
        result = check_claims(path)
        if not result["submission_evidence_ready"]:
            raise ValueError("scientific claims or redistribution evidence are incomplete")
        scopes = {claim["scope"] for claim in result["claims"] if claim["ready"]}
        if empirical and not scopes & {"training_utility", "simulator_driving", "real_world"}:
            raise ValueError("empirical submission has no empirical claim evidence")
        if manifest["mode"] == "empirical_real_world" and "real_world" not in scopes:
            raise ValueError("real-world submission needs real-world claim evidence")
        return result
    elif role == "rights":
        fields(payload, {"approved", "reviewer", "inventory", "license"})
        if payload["approved"] is not True:
            raise ValueError("redistribution has no owner-reviewed approval")
        text(payload["reviewer"], "rights reviewer")
        _ref(path.parent, payload["inventory"])
        _ref(path.parent, payload["license"])
    return {"status": "passed", "verification": "role_contract"}


def review_submission(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    document = read_json(path)
    fields(
        document,
        {"schema_version", "study_id", "mode", "source_commit", "artifacts", "limitations"},
    )
    if document["schema_version"] != "1.0" or document["mode"] not in {
        "offline_fixture",
        "empirical_simulation",
        "empirical_real_world",
    }:
        raise ValueError("unsupported reviewer schema/mode")
    text(document["study_id"], "study_id")
    fields(document["artifacts"], set(ROLES))
    if not isinstance(document["limitations"], list):
        raise ValueError("limitations must be an explicit array")
    empirical = document["mode"] != "offline_fixture"
    required = set(ROLES) if empirical else set(ROLES) - EMPIRICAL_EXTRA
    missing, errors, checks = [], {}, {}
    evidence_paths = {}
    commit = document["source_commit"]
    if not isinstance(commit, str) or re.fullmatch(r"[a-f0-9]{40}", commit) is None:
        missing.append("source_commit")
    for role in sorted(required):
        ref = document["artifacts"][role]
        if ref is None:
            missing.append(role)
            continue
        try:
            evidence_path = _ref(path.parent, ref)
            evidence_paths[role] = evidence_path
            if role == "costs":
                checks[role] = verify_cost_ledger(evidence_path)
            else:
                checks[role] = _check_role(role, evidence_path, document, empirical=empirical)
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            RuntimeError,
            ElementTree.ParseError,
            tarfile.TarError,
            zipfile.BadZipFile,
        ) as error:
            errors[role] = str(error)
    if "source_snapshot" in checks and "dependency_lock" in checks:
        try:
            archive_path = _ref(path.parent, document["artifacts"]["source_snapshot"])
            lock_path = _ref(path.parent, document["artifacts"]["dependency_lock"])
            if zipfile.is_zipfile(archive_path):
                with zipfile.ZipFile(archive_path) as archive:
                    frozen_lock = archive.read("uv.lock")
            else:
                with tarfile.open(archive_path, "r:*") as archive:
                    stream = archive.extractfile("uv.lock")
                    if stream is None:
                        raise ValueError("source snapshot lacks its dependency lock")
                    with stream:
                        frozen_lock = stream.read()
            if frozen_lock != lock_path.read_bytes():
                raise ValueError("dependency lock differs from the selected source tree")
        except (OSError, KeyError, ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
            errors["source_lock_binding"] = str(error)
    if empirical:
        from .study_links import LINK_ROLES, check_study_links

        if LINK_ROLES <= evidence_paths.keys():
            try:
                checks["study_linkage"] = check_study_links(
                    evidence_paths, study_id=document["study_id"]
                )
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
                errors["study_linkage"] = str(error)
        else:
            missing.append("study_linkage")
    if empirical:
        try:
            from .consumption_verify import check_study_consumption

            checks["sample_execution"] = check_study_consumption(
                evidence_paths, source_commit=document["source_commit"]
            )
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, zipfile.BadZipFile) as error:
            errors["sample_execution"] = str(error)
    ready = not missing and not errors
    return {
        "schema_version": "1.0",
        "study_id": document["study_id"],
        "mode": document["mode"],
        "ready": ready,
        "empirical_submission_ready": ready and empirical,
        "missing": missing,
        "errors": errors,
        "checks": checks,
        "acceptance_guaranteed": False,
        "scope": "reviewer_artifact_completeness_not_novelty_or_truth_certification",
        "limitations": document["limitations"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check")
    check.add_argument("submission", type=Path)
    pack = commands.add_parser("pack")
    pack.add_argument("submission", type=Path)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--maximum-files", type=int, default=100_000)
    pack.add_argument("--maximum-bytes", type=int, default=1024**3)
    verify = commands.add_parser("verify")
    verify.add_argument("bundle", type=Path)
    verify.add_argument("--sha256")
    freeze = commands.add_parser("freeze-split", help="freeze verified source sample membership")
    freeze.add_argument("--dataset", type=Path, action="append", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--maximum-samples", type=int, default=100_000)
    args = parser.parse_args(argv)
    if args.command == "freeze-split":
        from .study_links import freeze_split

        result = freeze_split(args.dataset, args.output, maximum_samples=args.maximum_samples)
        print(json.dumps({"sample_count": len(result["samples"]),
                          "holdout_count": result["holdout_count"],
                          "split": fingerprint_file(args.output)}, indent=2))
        return 0
    if args.command == "init":
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(template(), stream, indent=2, sort_keys=True)
            stream.write("\n")
        return 0
    if args.command == "verify":
        result = verify_closure(args.bundle, expected_sha256=args.sha256)
    else:
        result = review_submission(args.submission)
        if args.command == "pack":
            # Incomplete packs are useful for review, but their missing gates stay
            # explicit; no manifest is altered and no public upload is performed.
            result["closure"] = build_closure(
                [args.submission],
                args.output,
                maximum_files=args.maximum_files,
                maximum_bytes=args.maximum_bytes,
            )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("ready", result.get("status") == "passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
