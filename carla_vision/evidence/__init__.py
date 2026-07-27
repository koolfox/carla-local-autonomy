"""Read-only evidence discovery, indexing, and semantic verification."""

from .builder import build_evidence_registry, main, parse_args
from .contracts import EvidenceRegistryConfig
from .verified import (
    EvidenceRegistryIntegrityError,
    VerifiedEvidenceRegistry,
    verify_evidence_registry,
)

__all__ = [
    "EvidenceRegistryConfig",
    "EvidenceRegistryIntegrityError",
    "VerifiedEvidenceRegistry",
    "build_evidence_registry",
    "main",
    "parse_args",
    "verify_evidence_registry",
]
