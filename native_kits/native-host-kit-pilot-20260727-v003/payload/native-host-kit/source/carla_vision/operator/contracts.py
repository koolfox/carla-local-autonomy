"""Strict JSON contracts accepted by the local operator UI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

OPERATOR_SCHEMA_VERSION = "1.0"
OPERATOR_JOB_KINDS = frozenset(
    {
        "analyze",
        "live",
        "native_capture",
        "native_preflight",
        "replay",
        "scenario_plan",
        "shadow_matrix",
        "train",
        "verify",
    }
)


def _strict_keys(raw: Mapping[str, Any], required: set[str], name: str) -> None:
    keys = {str(key) for key in raw}
    missing = sorted(required - keys)
    unknown = sorted(keys - required)
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(unknown)}")


@dataclass(frozen=True)
class OperatorJobRequest:
    """One tokenized, allow-listed workflow launch request."""

    schema_version: str
    kind: str
    parameters: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        _strict_keys(raw, {"schema_version", "kind", "parameters"}, "operator job request")
        schema_version = str(raw["schema_version"])
        if schema_version != OPERATOR_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported operator schema {schema_version!r}; "
                f"expected {OPERATOR_SCHEMA_VERSION!r}"
            )
        kind = str(raw["kind"]).strip().lower()
        if kind not in OPERATOR_JOB_KINDS:
            raise ValueError(f"unsupported operator job kind {kind!r}")
        parameters = raw["parameters"]
        if not isinstance(parameters, Mapping):
            raise TypeError("operator job parameters must be an object")
        return cls(
            schema_version=schema_version,
            kind=kind,
            parameters=dict(parameters),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "parameters": dict(self.parameters),
        }


__all__ = [
    "OPERATOR_JOB_KINDS",
    "OPERATOR_SCHEMA_VERSION",
    "OperatorJobRequest",
]
