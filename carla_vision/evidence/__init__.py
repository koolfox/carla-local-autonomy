"""Read-only evidence contracts with lazy report-generation imports."""

from typing import Any

from .contracts import EvidenceRegistryConfig

__all__ = [
    "EvidenceRegistryConfig",
    "EvidenceRegistryIntegrityError",
    "VerifiedEvidenceRegistry",
    "build_evidence_registry",
    "main",
    "parse_args",
    "verify_evidence_registry",
]


def __getattr__(name: str) -> Any:
    """Load plotting-heavy builder functions only when explicitly requested."""

    if name in {"build_evidence_registry", "main", "parse_args"}:
        from . import builder

        return getattr(builder, name)
    if name in {
        "EvidenceRegistryIntegrityError",
        "VerifiedEvidenceRegistry",
        "verify_evidence_registry",
    }:
        from . import verified

        return getattr(verified, name)
    raise AttributeError(name)
