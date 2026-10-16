"""Content-addressed JSON reference closure, separate from source-code bundling.

Original evidence bytes are never rewritten. Absolute historical paths are merely
provenance after packing: verification resolves by digest, not by the old path.
No archived code is executed and no permission to redistribute is inferred.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..artifacts import fingerprint_file
from .review_contracts import digest, fields, integer, parse_json, read_json, relative_path


def _references(value: Any):
    if isinstance(value, dict):
        if {"path", "sha256", "size_bytes"} <= value.keys():
            digest(value["sha256"])
            integer(value["size_bytes"], "reference size")
            if not isinstance(value["path"], str) or not value["path"]:
                raise ValueError("invalid reference path")
            yield value
        for nested in value.values():
            yield from _references(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _references(nested)


def _document(path: Path, *, maximum_json_bytes: int = 64 * 1024 * 1024):
    with path.open("rb") as stream:
        prefix = stream.read(4096).lstrip()
    if not prefix.startswith((b"{", b"[")):
        return None
    if path.stat().st_size > maximum_json_bytes:
        raise ValueError("JSON evidence exceeds parsing budget")
    try:
        return read_json(path)
    except json.JSONDecodeError as error:
        if error.msg != "Extra data":
            raise
        # Hash-named objects have no suffix; JSONL is recognized structurally.
        rows = [
            parse_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows or any(not isinstance(row, dict) for row in rows):
            raise ValueError("invalid JSONL evidence") from error
        return rows


def _no_symlink(path: Path):
    for p in (path, *path.parents):
        if p.is_symlink():
            raise ValueError("symlink evidence source is not allowed")
    if path.name.startswith(".env") or path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}:
        raise ValueError("credential material cannot be packed")


def build_closure(
    roots, output: str | Path, *, maximum_files: int = 100_000, maximum_bytes: int = 1024**3
) -> dict[str, Any]:
    integer(maximum_files, "maximum_files", minimum=1)
    integer(maximum_bytes, "maximum_bytes", minimum=1)
    if not roots:
        raise ValueError("empty evidence closure")
    destination = Path(output).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".evidence-", dir=destination.parent))
    (temporary / "objects").mkdir()
    files, visiting, total = {}, set(), 0
    known_paths = {}

    def visit(raw_path):
        nonlocal total
        raw_path = Path(raw_path).absolute()
        _no_symlink(raw_path)
        source = raw_path.resolve(strict=True)
        ref = fingerprint_file(source)
        key = ref["sha256"]
        if key in visiting:
            raise ValueError("cyclic fingerprint references cannot form a research DAG")
        if key in files:
            return key
        if len(files) + len(visiting) >= maximum_files or total + ref["size_bytes"] > maximum_bytes:
            raise ValueError("evidence closure resource budget exceeded")
        visiting.add(key)
        total += ref["size_bytes"]
        stored = temporary / "objects" / key
        shutil.copyfile(source, stored)
        copied = fingerprint_file(stored)
        if (copied["sha256"], copied["size_bytes"]) != (key, ref["size_bytes"]):
            raise ValueError("source changed while copying evidence")
        payload = _document(stored)
        dependencies = []
        children = list(_references(payload))
        # A tracker inventories every relative artifact before descending. Nested
        # sample JSON often uses dataset-root-relative paths, not metadata-dir
        # paths. Resolve these only against already fingerprinted inventory entries.
        for child in children:
            p = Path(child["path"])
            if not p.is_absolute():
                p = relative_path(source.parent, child["path"], must_exist=False)
            _no_symlink(p)
            if p.is_file():
                actual = fingerprint_file(p)
                if (
                    actual["sha256"] != child["sha256"]
                    or actual["size_bytes"] != child["size_bytes"]
                ):
                    raise ValueError("source reference digest/size changed")
                known_paths[child["sha256"]] = p
        for child in children:
            target = known_paths.get(child["sha256"])
            if target is None:
                raise ValueError("unresolved reference in registered evidence closure")
            actual = fingerprint_file(target)
            if actual["size_bytes"] != child["size_bytes"]:
                raise ValueError("source reference size changed")
            dependencies.append(visit(target))
        files[key] = {
            "sha256": key,
            "size_bytes": ref["size_bytes"],
            "stored_path": f"objects/{key}",
            "json_document": payload is not None,
            "dependencies": sorted(set(dependencies)),
        }
        visiting.remove(key)
        return key

    try:
        root_ids = sorted({visit(path) for path in roots})
        index = {
            "schema_version": "1.0",
            "roots": root_ids,
            "files": [files[k] for k in sorted(files)],
            "contains_original_paths": True,
            "redistribution_cleared": False,
        }
        (temporary / "closure.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
        verify_closure(temporary)
        # Fresh directory only: no replacement of a previous bundle.
        if destination.exists():
            raise FileExistsError(destination)
        temporary.rename(destination)
        return {
            "root": str(destination),
            "index": fingerprint_file(destination / "closure.json"),
            "file_count": len(files),
            "total_bytes": total,
            "scientific_validity_established": False,
        }
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_closure(root: str | Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    root = Path(root).resolve(strict=True)
    index_path = relative_path(root, "closure.json")
    index_ref = fingerprint_file(index_path)
    if expected_sha256 is not None and index_ref["sha256"] != digest(expected_sha256):
        raise ValueError("closure index digest differs from the published anchor")
    index = fields(
        read_json(index_path),
        {"schema_version", "roots", "files", "contains_original_paths", "redistribution_cleared"},
    )
    if index["schema_version"] != "1.0" or not index["roots"] or not index["files"]:
        raise ValueError("empty/unsupported closure inventory")
    if index["contains_original_paths"] is not True or index["redistribution_cleared"] is not False:
        raise ValueError("closure cannot certify redistribution or anonymization")
    nodes, total, actual_files = {}, 0, {"closure.json"}
    for row in index["files"]:
        fields(row, {"sha256", "size_bytes", "stored_path", "json_document", "dependencies"})
        key = digest(row["sha256"])
        integer(row["size_bytes"], "size")
        if key in nodes or row["stored_path"] != f"objects/{key}":
            raise ValueError("duplicate or unsafe inventory path")
        path = relative_path(root, row["stored_path"])
        ref = fingerprint_file(path)
        if ref["sha256"] != key or ref["size_bytes"] != row["size_bytes"]:
            raise ValueError("evidence digest/size mismatch")
        payload = _document(path)
        if row["json_document"] is not (payload is not None):
            raise ValueError("JSON classification was changed")
        refs = list(_references(payload))
        dependencies = sorted({r["sha256"] for r in refs})
        if row["dependencies"] != dependencies:
            raise ValueError("dependency inventory differs from original evidence bytes")
        nodes[key] = (row, refs)
        total += ref["size_bytes"]
        actual_files.add(row["stored_path"])
    seen_files = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlink in evidence closure")
        if path.is_file():
            seen_files.add(path.relative_to(root).as_posix())
    if seen_files != actual_files:
        raise ValueError("unregistered files in evidence closure")
    for _row, refs in nodes.values():
        for ref in refs:
            if (
                ref["sha256"] not in nodes
                or nodes[ref["sha256"]][0]["size_bytes"] != ref["size_bytes"]
            ):
                raise ValueError("missing or size-inconsistent reference target")
    visited, active = set(), set()

    def walk(key):
        if key in active:
            raise ValueError("cyclic reference closure")
        if key in visited:
            return
        if key not in nodes:
            raise ValueError("missing root object")
        active.add(key)
        for child in nodes[key][0]["dependencies"]:
            walk(child)
        active.remove(key)
        visited.add(key)

    for key in index["roots"]:
        walk(digest(key))
    if visited != nodes.keys():
        raise ValueError("unreachable/unregistered evidence objects")
    return {
        "schema_version": "1.0",
        "status": "passed",
        "file_count": len(nodes),
        "total_bytes": total,
        "index_sha256": index_ref["sha256"],
        "externally_anchored": expected_sha256 is not None,
        "scope": "portable_byte_and_reference_integrity_not_empirical_truth",
    }
