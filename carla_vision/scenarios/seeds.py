"""Versioned hierarchical seed derivation for simulation and training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

DERIVATION_VERSION = "carla-vision-v1"
SEED_MODULUS = (2**31) - 1
REQUIRED_SEED_NAMESPACES = (
    "python",
    "numpy",
    "torch",
    "model_init",
    "data_order",
    "augmentation",
    "world",
    "traffic_manager",
    "vehicles",
    "walkers",
    "walker_crossing",
    "props",
    "weather",
    "route",
    "camera_jitter",
    "failure_mining",
    "bootstrap",
)


def _validate_master_seed(master_seed: int) -> int:
    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise TypeError("master_seed must be an integer")
    if master_seed < 0 or master_seed >= 2**63:
        raise ValueError("master_seed must be in [0, 2^63)")
    return master_seed


def _validate_namespace(namespace: str) -> str:
    if not isinstance(namespace, str):
        raise TypeError("namespace must be a string")
    normalized = namespace.strip()
    if not normalized:
        raise ValueError("namespace must not be empty")
    if "\0" in normalized:
        raise ValueError("namespace must not contain NUL")
    return normalized


def derive_seed(master_seed: int, namespace: str) -> int:
    """Derive one deterministic non-negative seed from the protocol formula."""

    validated_seed = _validate_master_seed(master_seed)
    validated_namespace = _validate_namespace(namespace)
    payload = f"{DERIVATION_VERSION}\0{validated_seed}\0{validated_namespace}".encode("utf-8")
    prefix = hashlib.sha256(payload).digest()[:8]
    return int.from_bytes(prefix, byteorder="big", signed=False) % SEED_MODULUS


@dataclass(frozen=True)
class SeedBundle:
    """All required independent seeds for one experiment or episode."""

    master_seed: int
    namespace_prefix: str | None
    values: dict[str, int]
    derivation_version: str = DERIVATION_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "derivation_version": self.derivation_version,
            "master_seed": self.master_seed,
            "namespace_prefix": self.namespace_prefix,
            "values": dict(sorted(self.values.items())),
        }


def derive_seed_bundle(
    master_seed: int,
    *,
    namespace_prefix: str | None = None,
) -> SeedBundle:
    """Derive the required seed family, optionally scoped to one episode."""

    validated_seed = _validate_master_seed(master_seed)
    prefix = None
    if namespace_prefix is not None:
        prefix = _validate_namespace(namespace_prefix).rstrip("/")
    values = {
        name: derive_seed(
            validated_seed,
            name if prefix is None else f"{prefix}/{name}",
        )
        for name in REQUIRED_SEED_NAMESPACES
    }
    if len(set(values.values())) != len(values):
        raise RuntimeError("derived seed collision within one seed bundle")
    return SeedBundle(
        master_seed=validated_seed,
        namespace_prefix=prefix,
        values=values,
    )


__all__ = [
    "DERIVATION_VERSION",
    "REQUIRED_SEED_NAMESPACES",
    "SEED_MODULUS",
    "SeedBundle",
    "derive_seed",
    "derive_seed_bundle",
]
