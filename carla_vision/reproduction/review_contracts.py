"""Small strict primitives for untrusted, relocatable reviewer evidence."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

SHA256 = re.compile(r"^[0-9a-f]{64}$")


def fields(value: Any, required: set[str], *, optional: set[str] | None = None) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError("expected a JSON object")
    missing = required - value.keys()
    unknown = value.keys() - required - (optional or set())
    if missing or unknown:
        raise ValueError(f"invalid fields: missing={sorted(missing)}, unknown={sorted(unknown)}")
    return value


def number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return float(value)


def integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be nonempty text")
    return value


def digest(value: Any) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError("invalid SHA-256 digest")
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def parse_json(value: str) -> Any:
    return json.loads(value, object_pairs_hook=_pairs, parse_constant=_constant)


def read_json(path: str | Path) -> Any:
    return parse_json(Path(path).read_text(encoding="utf-8"))


def relative_path(root: Path, value: Any, *, must_exist: bool = True) -> Path:
    name = text(value, "path")
    p = PurePosixPath(name)
    if (
        p.is_absolute()
        or "\\" in name
        or ":" in name
        or any(x in {".", "..", ""} for x in name.split("/"))
    ):
        raise ValueError("unsafe relative path")
    root = root.resolve(strict=True)
    candidate = root
    for part in p.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("symlink path is not portable evidence")
    candidate = candidate.resolve(strict=must_exist)
    candidate.relative_to(root)
    return candidate
