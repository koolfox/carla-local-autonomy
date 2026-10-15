"""Fail-closed evidence gates for publication claims; hashes do not prove truth."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..artifacts import fingerprint_file
from ..verification import verify_research_object

SCOPES = {
    "implementation",
    "offline_contract",
    "training_utility",
    "simulator_driving",
    "real_world",
}


def check_claims(registry_path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    if set(raw) != {"schema_version", "claims", "redistribution"} or raw["schema_version"] != "1.0":
        raise ValueError("claim registry schema/fields are invalid")
    claims = raw["claims"]
    if not isinstance(claims, list) or len({c["claim_id"] for c in claims}) != len(claims):
        raise ValueError("claim IDs must be unique")
    reports = []
    for claim in claims:
        if (
            set(claim) != {"claim_id", "scope", "statement", "evidence", "required_fields"}
            or claim["scope"] not in SCOPES
        ):
            raise ValueError("invalid claim fields/scope")
        errors = []
        if not claim["evidence"]:
            errors.append("missing_evidence")
        for reference in claim["evidence"]:
            try:
                if fingerprint_file(reference["path"]) != reference:
                    raise ValueError("evidence fingerprint changed")
                payload = json.loads(Path(reference["path"]).read_text(encoding="utf-8"))
                if claim["scope"] not in {"implementation", "offline_contract"}:
                    if (
                        payload.get("synthetic") is not False
                        or payload.get("empirical_results_available") is not True
                    ):
                        raise ValueError(
                            "empirical claim requires explicit non-synthetic empirical evidence"
                        )
                    producer = payload.get("producer_manifest")
                    if (
                        not isinstance(producer, dict)
                        or fingerprint_file(producer["path"]) != producer
                    ):
                        raise ValueError("missing/changed empirical producer manifest")
                    verify_research_object(
                        Path(producer["path"]).parent,
                        verify_references=True,
                        deep=False,
                        reject_unregistered=True,
                    )
                    if payload.get("scope") != claim["scope"]:
                        raise ValueError("evidence scope differs from claim scope")
                for field, expected in claim["required_fields"].items():
                    value = payload
                    for key in field.split("."):
                        value = value[key]
                    if value != expected:
                        raise ValueError(f"claim field not established: {field}")
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
                errors.append(str(error))
        reports.append(
            {
                "claim_id": claim["claim_id"],
                "scope": claim["scope"],
                "ready": not errors,
                "errors": errors,
            }
        )
    rights = raw["redistribution"]
    rights_ready = False
    try:
        license_ref, approval = rights["license_file"], rights["owner_clearance"]
        if (
            fingerprint_file(license_ref["path"]) != license_ref
            or fingerprint_file(approval["path"]) != approval
        ):
            raise ValueError("rights review reference changed")
        clearance = json.loads(Path(approval["path"]).read_text(encoding="utf-8"))
        rights_ready = (
            clearance.get("approved") is True
            and bool(clearance.get("reviewer"))
            and bool(clearance.get("asset_inventory"))
        )
    except (KeyError, TypeError, ValueError, OSError):
        rights_ready = False
    return {
        "schema_version": "1.0",
        "claims": reports,
        "redistribution_ready": bool(rights_ready),
        "submission_evidence_ready": bool(reports)
        and all(r["ready"] for r in reports)
        and bool(rights_ready),
        "acceptance_guaranteed": False,
        "gate_scope": "artifact_consistency_not_scientific_or_legal_certification",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry")
    args = parser.parse_args(argv)
    result = check_claims(args.registry)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["submission_evidence_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
