#!/usr/bin/env python3
"""Verify every immutable file in an extracted native-host payload."""
from __future__ import annotations

import hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
index = root / "checksums.sha256"
entries = 0
for line_number, line in enumerate(index.read_text(encoding="utf-8").splitlines(), 1):
    digest, separator, relative = line.partition("  ")
    raw = Path(relative)
    if (
        not separator
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or raw.is_absolute()
        or any(part in {"", ".", ".."} for part in raw.parts)
    ):
        raise SystemExit(f"invalid checksum entry at line {line_number}")
    path = (root / raw).resolve(strict=True)
    path.relative_to(root)
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"unsafe payload file: {relative}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit(f"payload checksum mismatch: {relative}")
    entries += 1
print(f"native host payload verified: {entries} files")
