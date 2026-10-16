"""Fail-closed evidence gates for publication claims; hashes do not prove truth."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..artifacts import fingerprint_file
from ..reproduction.review_contracts import digest, fields, integer, read_json, relative_path, text
from ..verification import verify_research_object

SCOPES = {
    "implementation",
    "offline_contract",
    "training_utility",
    "simulator_driving",
    "real_world",
}
EMPIRICAL_ROLES = {
    "training_utility": "empirical_study_result",
    "simulator_driving": "simulator_driving_result",
    "real_world": "real_world_result",
}


def _reference(reference: Any, base: Path) -> Path:
    fields(reference, {"path", "sha256", "size_bytes"})
    digest(reference["sha256"])
    integer(reference["size_bytes"], "size_bytes")
    path = Path(text(reference["path"], "reference path"))
    if not path.is_absolute():
        path = relative_path(base, reference["path"])
    actual = fingerprint_file(path)
    if actual["sha256"] != reference["sha256"] or actual["size_bytes"] != reference["size_bytes"]:
        raise ValueError("evidence fingerprint changed")
    return path.resolve(strict=True)


def _empirical_result(wrapper: Mapping, scope: str, base: Path) -> Mapping:
    # The external wrapper avoids a circular digest: a registered result cannot
    # fingerprint the manifest that in turn fingerprints the result itself.
    producer_path = _reference(wrapper.get("producer_manifest"), base)
    result_path = _reference(wrapper.get("result_artifact"), base)
    verify_research_object(
        producer_path.parent, verify_references=True, deep=False, reject_unregistered=True
    )
    manifest = read_json(producer_path)
    if producer_path.name != "manifest.json":
        raise ValueError("claim must use the verified producer manifest, not another JSON")
    config = manifest.get("invocation", {}).get("config", {})
    if (
        config.get("synthetic") is not False
        or config.get("evidence_kind") != "empirical"
        or config.get("dry_run") is True
    ):
        raise ValueError("producer does not declare a non-synthetic empirical run")
    registered = []
    for artifact in manifest.get("artifacts", []):
        if artifact.get("role") == EMPIRICAL_ROLES[scope]:
            candidate = relative_path(producer_path.parent, artifact["path"])
            if candidate == result_path:
                registered.append(artifact)
    if len(registered) != 1:
        raise ValueError("empirical result is not registered in the cited producer")
    actual = fingerprint_file(result_path)
    if any(registered[0].get(k) != actual[k] for k in ("sha256", "size_bytes")):
        raise ValueError("registered result identity differs")
    payload = read_json(result_path)
    if (
        not isinstance(payload, Mapping)
        or payload.get("scope") != scope
        or payload.get("synthetic") is not False
        or payload.get("empirical_results_available") is not True
    ):
        raise ValueError("registered result does not establish the empirical scope")
    return payload


def check_claims(registry_path: str | Path) -> dict[str, Any]:
    path = Path(registry_path).resolve(strict=True)
    raw = read_json(path)
    fields(raw, {"schema_version", "claims", "redistribution"})
    if raw["schema_version"] != "1.0" or not isinstance(raw["claims"], list):
        raise ValueError("claim registry schema/fields are invalid")
    reports, identities = [], set()
    for claim in raw["claims"]:
        fields(claim, {"claim_id", "scope", "statement", "evidence", "required_fields"})
        identity = text(claim["claim_id"], "claim ID")
        if identity in identities or claim["scope"] not in SCOPES:
            raise ValueError("duplicate claim ID or invalid scope")
        identities.add(identity)
        text(claim["statement"], "claim statement")
        if not isinstance(claim["evidence"], list) or not isinstance(
            claim["required_fields"], dict
        ):
            raise ValueError("claim evidence/required fields must be an array/object")
        errors = []
        if not claim["evidence"]:
            errors.append("missing_evidence")
        for reference in claim["evidence"]:
            try:
                evidence_path = _reference(reference, path.parent)
                payload = read_json(evidence_path)
                if not isinstance(payload, Mapping):
                    raise ValueError("claim evidence must be a JSON object")
                if claim["scope"] in EMPIRICAL_ROLES:
                    payload = _empirical_result(payload, claim["scope"], evidence_path.parent)
                for field, expected in claim["required_fields"].items():
                    value = payload
                    for key in text(field, "claim field").split("."):
                        value = value[key]
                    if type(value) is not type(expected) or value != expected:
                        raise ValueError(f"claim field not established: {field}")
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
                errors.append(str(error))
        reports.append(
            {"claim_id": identity, "scope": claim["scope"], "ready": not errors, "errors": errors}
        )
    rights_ready = False
    try:
        rights = raw["redistribution"]
        fields(rights, {"license_file", "owner_clearance"})
        license_path = _reference(rights["license_file"], path.parent)
        approval_path = _reference(rights["owner_clearance"], path.parent)
        approval = read_json(approval_path)
        if approval.get("approved") is not True:
            raise ValueError("owner approval absent")
        text(approval.get("reviewer"), "rights reviewer")
        inventory_path = _reference(approval.get("asset_inventory"), approval_path.parent)
        inventory = read_json(inventory_path)
        if not isinstance(inventory, list) or not inventory:
            raise ValueError("empty rights inventory")
        for asset in inventory:
            fields(asset, {"artifact", "license", "redistribution_approved"})
            _reference(asset["artifact"], inventory_path.parent)
            _reference(asset["license"], inventory_path.parent)
            if asset["redistribution_approved"] is not True:
                raise ValueError("asset redistribution not approved")
        if not license_path.read_text(encoding="utf-8").strip():
            raise ValueError("empty license")
        rights_ready = True
    except (KeyError, TypeError, ValueError, OSError):
        rights_ready = False
    return {
        "schema_version": "1.0",
        "claims": reports,
        "redistribution_ready": rights_ready,
        "submission_evidence_ready": bool(reports)
        and all(r["ready"] for r in reports)
        and rights_ready,
        "acceptance_guaranteed": False,
        "gate_scope": "artifact_consistency_not_scientific_or_legal_certification",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry")
    result = check_claims(parser.parse_args(argv).registry)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["submission_evidence_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
