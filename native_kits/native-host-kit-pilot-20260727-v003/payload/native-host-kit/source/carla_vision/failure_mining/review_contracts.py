"""Strict configuration contract for finalizing human failure reviews."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

FAILURE_REVIEW_CONFIG_SCHEMA_VERSION = "1.0"
REVIEW_DECISIONS = (
    "confirmed",
    "rejected",
    "label_issue",
    "duplicate",
    "needs_more_context",
)
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]{0,127}$")


def _strict_keys(
    raw: Mapping[str, Any],
    required: set[str],
    name: str,
) -> None:
    keys = {str(key) for key in raw}
    missing = sorted(required - keys)
    unknown = sorted(keys - required)
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(unknown)}")


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    result = tuple(dict.fromkeys(str(item).strip() for item in value))
    if not result or any(not item for item in result):
        raise ValueError(f"{name} must contain non-empty unique strings")
    return result


@dataclass(frozen=True)
class FailureReviewConfig:
    schema_version: str
    review_id: str
    title: str
    purpose: str
    reviewers: tuple[str, ...]
    require_complete: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        _strict_keys(
            raw,
            {
                "schema_version",
                "review_id",
                "title",
                "purpose",
                "reviewers",
                "require_complete",
            },
            "failure review config",
        )
        schema_version = str(raw["schema_version"])
        if schema_version != FAILURE_REVIEW_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported failure review schema {schema_version!r}; "
                f"expected {FAILURE_REVIEW_CONFIG_SCHEMA_VERSION!r}"
            )
        review_id = str(raw["review_id"]).strip()
        if not _ID_PATTERN.fullmatch(review_id):
            raise ValueError("review_id must be lowercase and use letters, digits, '.' or '-'")
        title = str(raw["title"]).strip()
        if not title:
            raise ValueError("title must not be empty")
        purpose = str(raw["purpose"]).strip().lower()
        if purpose != "development":
            raise ValueError(
                "failure review is a development workflow; purpose must be development"
            )
        if not isinstance(raw["require_complete"], bool):
            raise TypeError("require_complete must be boolean")
        if raw["require_complete"] is not True:
            raise ValueError("require_complete must be true for finalized review releases")
        return cls(
            schema_version=schema_version,
            review_id=review_id,
            title=title,
            purpose=purpose,
            reviewers=_strings(raw["reviewers"], "reviewers"),
            require_complete=True,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_failure_review_config(path: str | Path) -> FailureReviewConfig:
    resolved = Path(path).expanduser().resolve(strict=True)
    with resolved.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("failure review config must contain a JSON object")
    return FailureReviewConfig.from_mapping(payload)


__all__ = [
    "FAILURE_REVIEW_CONFIG_SCHEMA_VERSION",
    "REVIEW_DECISIONS",
    "FailureReviewConfig",
    "load_failure_review_config",
]
