"""Build a sealed, read-only registry of every immediate-child research object."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import stat
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402

from ..artifacts import RunArtifactTracker
from ..verification import verify_research_object
from .contracts import (
    CANONICAL_EVIDENCE_ROOTS,
    EVIDENCE_REGISTRY_OBJECT_TYPE,
    EVIDENCE_REGISTRY_SCHEMA_VERSION,
    EvidenceRegistryConfig,
)

EVIDENCE_REGISTRY_RELEASE_TYPE = "evidence_registry_release"

_OBJECT_FIELDS = (
    "root_kind",
    "relative_path",
    "object_id",
    "declared_status",
    "schema_version",
    "manifest_present",
    "manifest_sha256",
    "manifest_size_bytes",
    "reproducibility_present",
    "git_available",
    "git_commit",
    "git_dirty",
    "registered_artifact_count",
    "registered_artifact_bytes",
    "verification_ok",
    "verified_artifact_count",
    "verified_artifact_bytes",
    "checksum_index_count",
    "checksum_index_entries",
    "external_reference_count",
    "unregistered_file_count",
    "deep_kind",
    "failed_tree_entry_count",
    "failed_tree_file_count",
    "failed_tree_bytes",
    "failed_tree_sha256",
    "verification_error_type",
    "verification_error",
)

_ARTIFACT_FIELDS = (
    "root_kind",
    "source_relative_path",
    "source_id",
    "artifact_index",
    "path",
    "role",
    "sha256",
    "size_bytes",
    "mime_type",
    "entry_valid",
    "entry_json",
)

_FAILED_TREE_FIELDS = (
    "root_kind",
    "source_relative_path",
    "path",
    "entry_type",
    "sha256",
    "size_bytes",
    "link_target",
    "error_type",
    "error",
)


@dataclass(frozen=True)
class SourceCandidate:
    root_kind: str
    relative_path: str
    root: Path
    manifest_path: Path


@dataclass(frozen=True)
class Discovery:
    candidates: tuple[SourceCandidate, ...]
    excluded_registry_paths: tuple[str, ...]


@dataclass(frozen=True)
class ManifestRead:
    value: Mapping[str, Any] | None
    raw: bytes | None
    error_type: str
    error: str


class UnsafeSourcePathError(RuntimeError):
    """Raised when evidence discovery would have to follow a symbolic link."""


class SourceChangedDuringScanError(RuntimeError):
    """Raised when a source cannot be observed as one stable snapshot."""


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fields,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _digest(payload: bytes) -> tuple[str, int]:
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _payload_reference(relative_path: str, payload: bytes) -> dict[str, Any]:
    digest, size_bytes = _digest(payload)
    return {
        "path": relative_path,
        "sha256": digest,
        "size_bytes": size_bytes,
    }


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular_file_nofollow(path: Path) -> bytes:
    """Read one regular file without following its final path component."""

    try:
        before_path = os.lstat(path)
    except OSError:
        raise
    if stat.S_ISLNK(before_path.st_mode):
        raise UnsafeSourcePathError(f"symbolic-link file is not allowed: {path}")
    if not stat.S_ISREG(before_path.st_mode):
        raise UnsafeSourcePathError(f"path is not a regular file: {path}")

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before_fd = os.fstat(descriptor)
        if not stat.S_ISREG(before_fd.st_mode):
            raise UnsafeSourcePathError(f"path is not a regular file: {path}")
        if (before_fd.st_dev, before_fd.st_ino) != (
            before_path.st_dev,
            before_path.st_ino,
        ):
            raise SourceChangedDuringScanError(f"file changed before it could be read: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    try:
        after_path = os.lstat(path)
    except OSError as error:
        raise SourceChangedDuringScanError(f"file disappeared while being read: {path}") from error
    if _stat_identity(before_fd) != _stat_identity(after_fd) or _stat_identity(
        before_path
    ) != _stat_identity(after_path):
        raise SourceChangedDuringScanError(f"file changed while being read: {path}")
    return b"".join(chunks)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _strict_json_loads(raw: bytes, *, name: str) -> Any:
    try:
        text = raw.decode("utf-8")
        return json.loads(text, parse_constant=_reject_json_constant)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"could not parse {name}: {error}") from error


def _read_manifest(path: Path) -> ManifestRead:
    try:
        raw = _read_regular_file_nofollow(path)
    except FileNotFoundError:
        return ManifestRead(None, None, "FileNotFoundError", "manifest.json is missing")
    except (OSError, UnsafeSourcePathError, SourceChangedDuringScanError) as error:
        return ManifestRead(None, None, type(error).__name__, _clean_error(error))
    try:
        value = _strict_json_loads(raw, name="manifest.json")
    except ValueError as error:
        return ManifestRead(None, raw, "InvalidManifest", _clean_error(error))
    if not isinstance(value, Mapping):
        return ManifestRead(
            None,
            raw,
            "InvalidManifest",
            "manifest.json must contain an object",
        )
    return ManifestRead(value, raw, "", "")


def _validate_canonical_roots(workspace: Path) -> None:
    """Reject canonical-root symlinks before discovery, verification, or output."""

    for root_kind in CANONICAL_EVIDENCE_ROOTS:
        candidate = workspace / root_kind
        try:
            metadata = os.lstat(candidate)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise UnsafeSourcePathError(
                f"could not inspect canonical evidence root {candidate}: {error}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise UnsafeSourcePathError(
                f"canonical evidence root must not be a symbolic link: {candidate}"
            )


def _source_path_issue(
    candidate: SourceCandidate,
    *,
    workspace: Path,
) -> tuple[str, str]:
    expected = workspace / candidate.relative_path
    if (
        candidate.root != expected
        or candidate.manifest_path != expected / "manifest.json"
        or candidate.root_kind not in CANONICAL_EVIDENCE_ROOTS
        or expected.parent != workspace / candidate.root_kind
    ):
        return "UnsafeSourcePathError", "source path is outside its canonical root"
    try:
        metadata = os.lstat(candidate.root)
    except FileNotFoundError:
        return "FileNotFoundError", "source path is missing"
    except OSError as error:
        return type(error).__name__, _clean_error(error)
    if stat.S_ISLNK(metadata.st_mode):
        return "UnsafeSourcePathError", "source path is a symbolic link"
    if not stat.S_ISDIR(metadata.st_mode):
        return "UnsafeSourcePathError", "source path is not a directory"
    try:
        resolved = candidate.root.resolve(strict=True)
    except OSError as error:
        return type(error).__name__, _clean_error(error)
    if resolved != candidate.root:
        return "UnsafeSourcePathError", "source path does not remain in its canonical root"
    return "", ""


def _is_registry_manifest(manifest: Mapping[str, Any] | None) -> bool:
    if manifest is None:
        return False
    invocation = manifest.get("invocation")
    if isinstance(invocation, Mapping):
        config = invocation.get("config")
        if (
            isinstance(config, Mapping)
            and config.get("object_type") == EVIDENCE_REGISTRY_OBJECT_TYPE
        ):
            return True
    artifacts = manifest.get("artifacts")
    return isinstance(artifacts, list) and any(
        isinstance(entry, Mapping) and entry.get("role") == "evidence_registry_release_manifest"
        for entry in artifacts
    )


def _discover_sources(
    workspace: Path,
    *,
    current_output: Path | None = None,
) -> Discovery:
    _validate_canonical_roots(workspace)
    current = current_output if current_output is not None else None
    candidates: list[SourceCandidate] = []
    excluded: list[str] = []
    for root_kind in CANONICAL_EVIDENCE_ROOTS:
        parent = workspace / root_kind
        try:
            parent_metadata = os.lstat(parent)
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(parent_metadata.st_mode):
            continue
        children = sorted(parent.iterdir(), key=lambda path: path.name.encode("utf-8"))
        for child in children:
            try:
                child_metadata = os.lstat(child)
            except FileNotFoundError:
                continue
            is_symlink = stat.S_ISLNK(child_metadata.st_mode)
            if not is_symlink and not stat.S_ISDIR(child_metadata.st_mode):
                continue
            relative = child.relative_to(workspace).as_posix()
            if current is not None and child == current:
                excluded.append(relative)
                continue
            manifest_path = child / "manifest.json"
            manifest = None if is_symlink else _read_manifest(manifest_path).value
            if _is_registry_manifest(manifest):
                excluded.append(relative)
                continue
            candidates.append(
                SourceCandidate(
                    root_kind=root_kind,
                    relative_path=relative,
                    root=child,
                    manifest_path=manifest_path,
                )
            )
    candidates.sort(key=lambda source: source.relative_path.encode("utf-8"))
    excluded.sort(key=lambda value: value.encode("utf-8"))
    return Discovery(tuple(candidates), tuple(excluded))


def _clean_error(error: BaseException) -> str:
    return " ".join(str(error).split())[:2000]


def _tree_row(
    candidate: SourceCandidate,
    *,
    relative: str,
    entry_type: str,
    sha256: str = "",
    size_bytes: int | str = "",
    link_target: str = "",
    error_type: str = "",
    error: str = "",
) -> dict[str, Any]:
    return {
        "root_kind": candidate.root_kind,
        "source_relative_path": candidate.relative_path,
        "path": relative,
        "entry_type": entry_type,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "link_target": link_target,
        "error_type": error_type,
        "error": error,
    }


def _entry_type(mode: int) -> str:
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "regular_file"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISCHR(mode):
        return "character_device"
    if stat.S_ISBLK(mode):
        return "block_device"
    return "other"


def _read_regular_at(
    parent_descriptor: int,
    name: str,
    before: os.stat_result,
    *,
    display_path: str,
) -> bytes:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    try:
        before_fd = os.fstat(descriptor)
        if not stat.S_ISREG(before_fd.st_mode):
            raise UnsafeSourcePathError(f"tree entry is not a regular file: {display_path}")
        if (before_fd.st_dev, before_fd.st_ino) != (before.st_dev, before.st_ino):
            raise SourceChangedDuringScanError(
                f"tree entry changed before it could be read: {display_path}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as error:
        raise SourceChangedDuringScanError(
            f"tree entry disappeared while being read: {display_path}"
        ) from error
    if _stat_identity(before_fd) != _stat_identity(after_fd) or _stat_identity(
        before
    ) != _stat_identity(after):
        raise SourceChangedDuringScanError(f"tree entry changed while being read: {display_path}")
    return b"".join(chunks)


def _inventory_directory(
    candidate: SourceCandidate,
    descriptor: int,
    *,
    prefix: str,
    rows: list[dict[str, Any]],
) -> None:
    try:
        names = sorted(os.listdir(descriptor), key=lambda value: value.encode("utf-8"))
    except OSError as error:
        rows.append(
            _tree_row(
                candidate,
                relative=prefix or ".",
                entry_type="directory_error",
                error_type=type(error).__name__,
                error=_clean_error(error),
            )
        )
        return
    for name in names:
        relative = f"{prefix}/{name}" if prefix else name
        try:
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError as error:
            rows.append(
                _tree_row(
                    candidate,
                    relative=relative,
                    entry_type="unreadable",
                    error_type=type(error).__name__,
                    error=_clean_error(error),
                )
            )
            continue
        kind = _entry_type(metadata.st_mode)
        if kind == "symlink":
            try:
                target = os.readlink(name, dir_fd=descriptor)
                rows.append(
                    _tree_row(
                        candidate,
                        relative=relative,
                        entry_type=kind,
                        link_target=target,
                    )
                )
            except OSError as error:
                rows.append(
                    _tree_row(
                        candidate,
                        relative=relative,
                        entry_type=kind,
                        error_type=type(error).__name__,
                        error=_clean_error(error),
                    )
                )
            continue
        if kind == "regular_file":
            try:
                payload = _read_regular_at(
                    descriptor,
                    name,
                    metadata,
                    display_path=f"{candidate.relative_path}/{relative}",
                )
            except (OSError, UnsafeSourcePathError, SourceChangedDuringScanError) as error:
                rows.append(
                    _tree_row(
                        candidate,
                        relative=relative,
                        entry_type=kind,
                        error_type=type(error).__name__,
                        error=_clean_error(error),
                    )
                )
            else:
                digest, size_bytes = _digest(payload)
                rows.append(
                    _tree_row(
                        candidate,
                        relative=relative,
                        entry_type=kind,
                        sha256=digest,
                        size_bytes=size_bytes,
                    )
                )
            continue
        rows.append(_tree_row(candidate, relative=relative, entry_type=kind))
        if kind != "directory":
            continue
        flags = os.O_RDONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_DIRECTORY", 0)
        try:
            child_descriptor = os.open(name, flags, dir_fd=descriptor)
            opened = os.fstat(child_descriptor)
            if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise SourceChangedDuringScanError(
                    f"directory changed before traversal: {candidate.relative_path}/{relative}"
                )
        except (OSError, SourceChangedDuringScanError) as error:
            if "child_descriptor" in locals():
                os.close(child_descriptor)
                del child_descriptor
            rows.append(
                _tree_row(
                    candidate,
                    relative=relative,
                    entry_type="directory_error",
                    error_type=type(error).__name__,
                    error=_clean_error(error),
                )
            )
            continue
        try:
            _inventory_directory(
                candidate,
                child_descriptor,
                prefix=relative,
                rows=rows,
            )
        finally:
            os.close(child_descriptor)
            del child_descriptor


def _inventory_failed_source(candidate: SourceCandidate) -> list[dict[str, Any]]:
    """Inventory a failed source without dereferencing any symbolic link."""

    rows: list[dict[str, Any]] = []
    try:
        metadata = os.lstat(candidate.root)
    except OSError as error:
        return [
            _tree_row(
                candidate,
                relative=".",
                entry_type="missing",
                error_type=type(error).__name__,
                error=_clean_error(error),
            )
        ]
    kind = _entry_type(metadata.st_mode)
    if kind == "symlink":
        try:
            target = os.readlink(candidate.root)
            return [
                _tree_row(
                    candidate,
                    relative=".",
                    entry_type=kind,
                    link_target=target,
                )
            ]
        except OSError as error:
            return [
                _tree_row(
                    candidate,
                    relative=".",
                    entry_type=kind,
                    error_type=type(error).__name__,
                    error=_clean_error(error),
                )
            ]
    if kind == "regular_file":
        try:
            payload = _read_regular_file_nofollow(candidate.root)
        except (OSError, UnsafeSourcePathError, SourceChangedDuringScanError) as error:
            return [
                _tree_row(
                    candidate,
                    relative=".",
                    entry_type=kind,
                    error_type=type(error).__name__,
                    error=_clean_error(error),
                )
            ]
        digest, size_bytes = _digest(payload)
        return [
            _tree_row(
                candidate,
                relative=".",
                entry_type=kind,
                sha256=digest,
                size_bytes=size_bytes,
            )
        ]
    rows.append(_tree_row(candidate, relative=".", entry_type=kind))
    if kind != "directory":
        return rows
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(candidate.root, flags)
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise SourceChangedDuringScanError(
                f"source changed before tree traversal: {candidate.relative_path}"
            )
    except (OSError, SourceChangedDuringScanError) as error:
        if "descriptor" in locals():
            os.close(descriptor)
        rows.append(
            _tree_row(
                candidate,
                relative=".",
                entry_type="directory_error",
                error_type=type(error).__name__,
                error=_clean_error(error),
            )
        )
        return rows
    try:
        _inventory_directory(candidate, descriptor, prefix="", rows=rows)
    finally:
        os.close(descriptor)
    rows.sort(key=lambda row: str(row["path"]).encode("utf-8"))
    return rows


def _registered_artifacts(
    candidate: SourceCandidate,
    manifest: Mapping[str, Any] | None,
    *,
    source_id: str,
) -> tuple[list[dict[str, Any]], int]:
    raw_artifacts = manifest.get("artifacts") if manifest is not None else None
    if not isinstance(raw_artifacts, list):
        return [], 0
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    for index, raw in enumerate(raw_artifacts):
        entry = raw if isinstance(raw, Mapping) else {}
        size = entry.get("size_bytes")
        valid_size = isinstance(size, int) and not isinstance(size, bool) and size >= 0
        if valid_size:
            total_bytes += size
        entry_valid = (
            isinstance(raw, Mapping)
            and isinstance(entry.get("path"), str)
            and bool(entry.get("path"))
            and isinstance(entry.get("role"), str)
            and bool(entry.get("role"))
            and isinstance(entry.get("sha256"), str)
            and len(str(entry.get("sha256"))) == 64
            and valid_size
        )
        rows.append(
            {
                "root_kind": candidate.root_kind,
                "source_relative_path": candidate.relative_path,
                "source_id": source_id,
                "artifact_index": index,
                "path": str(entry.get("path", "")),
                "role": str(entry.get("role", "")),
                "sha256": str(entry.get("sha256", "")),
                "size_bytes": size if valid_size else "",
                "mime_type": str(entry.get("mime_type", "")),
                "entry_valid": entry_valid,
                "entry_json": json.dumps(
                    raw,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            str(row["source_relative_path"]).encode("utf-8"),
            str(row["path"]).encode("utf-8"),
            int(row["artifact_index"]),
        )
    )
    return rows, total_bytes


def _inspect_source_once(
    candidate: SourceCandidate,
    *,
    workspace: Path,
) -> dict[str, Any] | None:
    path_error_type, path_error = _source_path_issue(candidate, workspace=workspace)
    manifest_read = (
        ManifestRead(None, None, path_error_type, path_error)
        if path_error_type
        else _read_manifest(candidate.manifest_path)
    )
    manifest = manifest_read.value
    source_id = (
        str(manifest.get("run_id"))
        if manifest is not None and isinstance(manifest.get("run_id"), str)
        else candidate.root.name
    )
    artifact_rows, registered_bytes = _registered_artifacts(
        candidate,
        manifest,
        source_id=source_id,
    )
    portable_fingerprint: dict[str, Any] | None = None
    input_reference: dict[str, Any] | None = None
    if manifest_read.raw is not None:
        digest, size_bytes = _digest(manifest_read.raw)
        portable_fingerprint = {
            "path": f"{candidate.relative_path}/manifest.json",
            "sha256": digest,
            "size_bytes": size_bytes,
        }
        input_reference = {
            "kind": "evidence_registry_source_manifest",
            "root_kind": candidate.root_kind,
            "source_relative_path": candidate.relative_path,
            "source_id": source_id,
            "path": str(candidate.manifest_path),
            "sha256": digest,
            "size_bytes": size_bytes,
        }

    verification = None
    verification_error_type = manifest_read.error_type
    verification_error = manifest_read.error
    if not path_error_type and manifest is not None:
        try:
            verification = verify_research_object(
                candidate.root,
                verify_references=True,
                deep=True,
                reject_unregistered=True,
                allow_non_success=True,
            )
            verification_error_type = ""
            verification_error = ""
        except Exception as error:  # every discovered object remains in the registry
            verification_error_type = type(error).__name__
            verification_error = _clean_error(error)

    failed_tree_rows: list[dict[str, Any]] = []
    failed_tree_sha256 = ""
    failed_tree_file_count: int | str = ""
    failed_tree_bytes: int | str = ""
    if verification is None:
        first_tree = _inventory_failed_source(candidate)
        second_tree = _inventory_failed_source(candidate)
        if first_tree != second_tree or any(
            row["error_type"] == "SourceChangedDuringScanError" for row in first_tree
        ):
            return None
        failed_tree_rows = first_tree
        failed_tree_sha256, _ = _digest(_json_bytes(failed_tree_rows))
        failed_tree_file_count = sum(
            row["entry_type"] == "regular_file" and bool(row["sha256"]) for row in failed_tree_rows
        )
        failed_tree_bytes = sum(
            int(row["size_bytes"]) for row in failed_tree_rows if isinstance(row["size_bytes"], int)
        )

    after_path_error_type, after_path_error = _source_path_issue(
        candidate,
        workspace=workspace,
    )
    after_manifest_read = (
        ManifestRead(
            None,
            None,
            after_path_error_type,
            after_path_error,
        )
        if after_path_error_type
        else _read_manifest(candidate.manifest_path)
    )
    if (
        (path_error_type, path_error) != (after_path_error_type, after_path_error)
        or manifest_read != after_manifest_read
        or after_manifest_read.error_type == "SourceChangedDuringScanError"
    ):
        return None

    declared_git = manifest.get("git") if manifest is not None else None
    git = verification.git if verification is not None else declared_git
    git = git if isinstance(git, Mapping) else {}
    declared_artifacts = manifest.get("artifacts") if manifest is not None else None
    registered_count = len(declared_artifacts) if isinstance(declared_artifacts, list) else 0
    declared_status = str(manifest.get("status", "unknown")) if manifest is not None else "unknown"
    schema_version = str(manifest.get("schema_version", "")) if manifest is not None else ""
    row = {
        "root_kind": candidate.root_kind,
        "relative_path": candidate.relative_path,
        "object_id": source_id,
        "declared_status": declared_status,
        "schema_version": schema_version,
        "manifest_present": manifest_read.raw is not None,
        "manifest_sha256": (
            str(portable_fingerprint["sha256"]) if portable_fingerprint is not None else ""
        ),
        "manifest_size_bytes": (
            int(portable_fingerprint["size_bytes"]) if portable_fingerprint is not None else ""
        ),
        "reproducibility_present": (
            manifest is not None and isinstance(manifest.get("reproducibility"), Mapping)
        ),
        "git_available": bool(git.get("available", False)),
        "git_commit": str(git.get("commit") or ""),
        "git_dirty": git.get("dirty") if isinstance(git.get("dirty"), bool) else "",
        "registered_artifact_count": registered_count,
        "registered_artifact_bytes": registered_bytes,
        "verification_ok": verification is not None,
        "verified_artifact_count": (
            verification.artifact_count if verification is not None else ""
        ),
        "verified_artifact_bytes": (
            verification.artifact_bytes if verification is not None else ""
        ),
        "checksum_index_count": (
            verification.checksum_index_count if verification is not None else ""
        ),
        "checksum_index_entries": (
            verification.checksum_index_entries if verification is not None else ""
        ),
        "external_reference_count": (
            verification.external_reference_count if verification is not None else ""
        ),
        "unregistered_file_count": (
            verification.unregistered_file_count if verification is not None else ""
        ),
        "deep_kind": (
            str(verification.deep_verification.get("kind", "")) if verification is not None else ""
        ),
        "failed_tree_entry_count": len(failed_tree_rows) if verification is None else "",
        "failed_tree_file_count": failed_tree_file_count,
        "failed_tree_bytes": failed_tree_bytes,
        "failed_tree_sha256": failed_tree_sha256,
        "verification_error_type": verification_error_type,
        "verification_error": verification_error,
    }
    descriptor = {
        "root_kind": candidate.root_kind,
        "relative_path": candidate.relative_path,
        "object_id": source_id,
        "manifest": portable_fingerprint,
        "verification": {
            "ok": verification is not None,
            "status": verification.status if verification is not None else declared_status,
            "artifact_count": (
                verification.artifact_count if verification is not None else registered_count
            ),
            "artifact_bytes": (
                verification.artifact_bytes if verification is not None else registered_bytes
            ),
            "deep_kind": row["deep_kind"],
            "error_type": verification_error_type,
            "error": verification_error,
        },
        "failed_tree": (
            {
                "entry_count": len(failed_tree_rows),
                "regular_file_count": failed_tree_file_count,
                "regular_file_bytes": failed_tree_bytes,
                "sha256": failed_tree_sha256,
            }
            if verification is None
            else None
        ),
    }
    return {
        "row": row,
        "artifact_rows": artifact_rows,
        "failed_tree_rows": failed_tree_rows,
        "descriptor": descriptor,
        "input_reference": input_reference,
    }


def _inspect_source(
    candidate: SourceCandidate,
    *,
    workspace: Path,
) -> dict[str, Any]:
    previous: dict[str, Any] | None = None
    for _ in range(4):
        inspected = _inspect_source_once(candidate, workspace=workspace)
        if inspected is None:
            previous = None
            continue
        if inspected == previous:
            return inspected
        previous = inspected
    raise SourceChangedDuringScanError(
        f"source did not stabilize after four scans: {candidate.relative_path}"
    )


def _summary(
    *,
    registry_id: str,
    workspace: Path,
    rows: Sequence[Mapping[str, Any]],
    artifact_rows: Sequence[Mapping[str, Any]],
    failed_tree_rows: Sequence[Mapping[str, Any]],
    source_manifest_reference_count: int,
    excluded_registry_paths: Sequence[str],
) -> dict[str, Any]:
    root_counts = {root: 0 for root in CANONICAL_EVIDENCE_ROOTS}
    failed_root_counts = {root: 0 for root in CANONICAL_EVIDENCE_ROOTS}
    statuses: Counter[str] = Counter()
    id_paths: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        root_kind = str(row["root_kind"])
        root_counts[root_kind] += 1
        if row["verification_ok"] is not True:
            failed_root_counts[root_kind] += 1
        status = str(row["declared_status"] or "unknown")
        statuses[status] += 1
        id_paths[str(row["object_id"])].append(str(row["relative_path"]))
    duplicate_groups = {
        object_id: sorted(paths, key=lambda value: value.encode("utf-8"))
        for object_id, paths in sorted(id_paths.items())
        if len(paths) > 1
    }
    return {
        "schema_version": EVIDENCE_REGISTRY_SCHEMA_VERSION,
        "object_type": "evidence_registry_summary",
        "registry_id": registry_id,
        "workspace_root": str(workspace),
        "canonical_roots": list(CANONICAL_EVIDENCE_ROOTS),
        "source_count": len(rows),
        "source_manifest_count": source_manifest_reference_count,
        "regular_manifest_count": sum(row["manifest_present"] is True for row in rows),
        "verified_count": sum(row["verification_ok"] is True for row in rows),
        "verification_failed_count": sum(row["verification_ok"] is not True for row in rows),
        "declared_non_success_count": sum(row["declared_status"] != "success" for row in rows),
        "missing_manifest_count": sum(row["manifest_present"] is not True for row in rows),
        "legacy_reproducibility_count": sum(
            row["reproducibility_present"] is not True for row in rows
        ),
        "clean_git_count": sum(
            bool(row["git_commit"]) and row["git_dirty"] is False for row in rows
        ),
        "dirty_git_count": sum(row["git_dirty"] is True for row in rows),
        "registered_artifact_count": len(artifact_rows),
        "registered_artifact_bytes": sum(
            int(row["size_bytes"]) for row in artifact_rows if isinstance(row["size_bytes"], int)
        ),
        "failed_source_tree_entry_count": len(failed_tree_rows),
        "failed_source_tree_file_count": sum(
            row["entry_type"] == "regular_file" and bool(row["sha256"]) for row in failed_tree_rows
        ),
        "failed_source_tree_bytes": sum(
            int(row["size_bytes"]) for row in failed_tree_rows if isinstance(row["size_bytes"], int)
        ),
        "root_counts": root_counts,
        "root_verification_failed_counts": failed_root_counts,
        "status_counts": dict(sorted(statuses.items())),
        "duplicate_object_id_group_count": len(duplicate_groups),
        "duplicate_object_id_groups": duplicate_groups,
        "excluded_registry_count": len(excluded_registry_paths),
        "excluded_registry_paths": list(excluded_registry_paths),
        "verification_policy": {
            "allow_non_success": True,
            "deep": True,
            "reject_unregistered": True,
            "verify_references": True,
        },
        "generation": {
            "carla_contacted": False,
            "simulator_mutated": False,
            "source_objects_modified": False,
        },
    }


def _build_snapshot(
    *,
    workspace: Path,
    registry_id: str,
    discovery: Discovery,
) -> dict[str, Any]:
    _validate_canonical_roots(workspace)
    inspected = [
        _inspect_source(candidate, workspace=workspace) for candidate in discovery.candidates
    ]
    rows = [dict(item["row"]) for item in inspected]
    rows.sort(key=lambda row: str(row["relative_path"]).encode("utf-8"))
    artifact_rows = [dict(row) for item in inspected for row in item["artifact_rows"]]
    artifact_rows.sort(
        key=lambda row: (
            str(row["source_relative_path"]).encode("utf-8"),
            str(row["path"]).encode("utf-8"),
            int(row["artifact_index"]),
        )
    )
    failed_tree_rows = [dict(row) for item in inspected for row in item["failed_tree_rows"]]
    failed_tree_rows.sort(
        key=lambda row: (
            str(row["source_relative_path"]).encode("utf-8"),
            str(row["path"]).encode("utf-8"),
            str(row["entry_type"]).encode("utf-8"),
        )
    )
    descriptors = [dict(item["descriptor"]) for item in inspected]
    descriptors.sort(key=lambda row: str(row["relative_path"]).encode("utf-8"))
    input_references = [
        dict(item["input_reference"]) for item in inspected if item["input_reference"] is not None
    ]
    input_references.sort(key=lambda row: str(row["source_relative_path"]).encode("utf-8"))
    summary = _summary(
        registry_id=registry_id,
        workspace=workspace,
        rows=rows,
        artifact_rows=artifact_rows,
        failed_tree_rows=failed_tree_rows,
        source_manifest_reference_count=len(input_references),
        excluded_registry_paths=discovery.excluded_registry_paths,
    )
    return {
        "summary": summary,
        "object_rows": rows,
        "artifact_rows": artifact_rows,
        "failed_tree_rows": failed_tree_rows,
        "source_descriptors": descriptors,
        "input_references": input_references,
    }


def _plot_bytes(
    *,
    registry_id: str,
    summary: Mapping[str, Any],
    plot_kind: str,
    image_format: str,
) -> bytes:
    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.hashsalt": f"{registry_id}-{plot_kind}",
            "svg.fonttype": "none",
        }
    ):
        figure, axis = plt.subplots(figsize=(8.0, 4.2))
        if plot_kind == "root_inventory":
            labels = list(CANONICAL_EVIDENCE_ROOTS)
            totals = [int(summary["root_counts"][label]) for label in labels]
            failed = [int(summary["root_verification_failed_counts"][label]) for label in labels]
            positions = list(range(len(labels)))
            axis.bar(
                [position - 0.19 for position in positions],
                totals,
                width=0.38,
                label="discovered",
                color="#3178c6",
            )
            axis.bar(
                [position + 0.19 for position in positions],
                failed,
                width=0.38,
                label="verification failed",
                color="#d95f59",
            )
            axis.set_xticks(positions, labels, rotation=24, ha="right")
            axis.set_ylabel("research objects")
            axis.set_title("Evidence objects by canonical root")
            axis.legend()
        elif plot_kind == "verification_status":
            labels = (
                "discovered",
                "verified",
                "verification failed",
                "non-success status",
                "legacy envelope",
            )
            values = (
                int(summary["source_count"]),
                int(summary["verified_count"]),
                int(summary["verification_failed_count"]),
                int(summary["declared_non_success_count"]),
                int(summary["legacy_reproducibility_count"]),
            )
            colors = ("#4c78a8", "#59a14f", "#e15759", "#f28e2b", "#b07aa1")
            bars = axis.bar(range(len(labels)), values, color=colors)
            axis.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
            axis.set_ylabel("research objects")
            axis.set_title("Registry verification and provenance status")
            axis.bar_label(bars, padding=2)
        else:
            raise ValueError(f"unknown evidence registry plot kind {plot_kind!r}")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        buffer = io.BytesIO()
        metadata = (
            {"Date": None, "Creator": "CARLA Vision Research"}
            if image_format == "svg"
            else {"Software": "CARLA Vision Research"}
        )
        figure.savefig(
            buffer,
            format=image_format,
            dpi=150,
            metadata=metadata,
        )
        plt.close(figure)
        return buffer.getvalue()


def _markdown(
    config: EvidenceRegistryConfig,
    snapshot: Mapping[str, Any],
) -> str:
    summary = snapshot["summary"]
    rows = snapshot["object_rows"]

    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    object_lines = [
        "| path | id | status | verified | artifacts | git |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        git_state = (
            "clean"
            if row["git_commit"] and row["git_dirty"] is False
            else "dirty"
            if row["git_dirty"] is True
            else "uncommitted/unknown"
        )
        object_lines.append(
            "| "
            + " | ".join(
                (
                    cell(row["relative_path"]),
                    cell(row["object_id"]),
                    cell(row["declared_status"]),
                    "yes" if row["verification_ok"] else "no",
                    cell(row["registered_artifact_count"]),
                    git_state,
                )
            )
            + " |"
        )
    failed = [row for row in rows if row["verification_ok"] is not True]
    failure_lines = [
        f"- `{cell(row['relative_path'])}`: "
        f"{cell(row['verification_error_type'] or 'verification error')} — "
        f"{cell(row['verification_error'])}"
        for row in failed
    ] or ["- None."]
    excluded_lines = [f"- `{cell(path)}`" for path in config.excluded_registry_paths] or ["- None."]
    return "\n".join(
        [
            f"# Evidence Registry {config.registry_id}",
            "",
            "This is a point-in-time, read-only inventory of immediate-child research "
            "objects under the canonical workspace roots.",
            "",
            "## Summary",
            "",
            "| measure | value |",
            "|---|---:|",
            f"| discovered objects | {summary['source_count']} |",
            f"| verified objects | {summary['verified_count']} |",
            f"| verification failures retained | {summary['verification_failed_count']} |",
            f"| registered artifacts inventoried | {summary['registered_artifact_count']} |",
            f"| legacy provenance envelopes | {summary['legacy_reproducibility_count']} |",
            f"| clean Git objects | {summary['clean_git_count']} |",
            "",
            "The scan used the generic verifier with external references and semantic "
            "checks enabled, `allow_non_success=True`, and "
            "`reject_unregistered=True`.",
            "",
            "## Objects",
            "",
            *object_lines,
            "",
            "## Verification failures",
            "",
            *failure_lines,
            "",
            "## Excluded evidence registries",
            "",
            "Evidence-registry objects are excluded to prevent registry-of-registry "
            "cycles. The registry being built is also never scanned as transient input.",
            "",
            *excluded_lines,
            "",
            "## Safety boundary",
            "",
            "Building this object did not contact CARLA, mutate the simulator, or modify "
            "any source research object.",
            "",
        ]
    )


def _checksum_index(payloads: Mapping[str, bytes]) -> bytes:
    lines = []
    for relative in sorted(payloads, key=lambda value: value.encode("utf-8")):
        digest, _ = _digest(payloads[relative])
        lines.append(f"{digest}  {relative}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _expected_payloads(
    config: EvidenceRegistryConfig,
    snapshot: Mapping[str, Any],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    summary_payload = {
        "schema_version": EVIDENCE_REGISTRY_SCHEMA_VERSION,
        "object_type": "evidence_registry_summary",
        "registry_id": config.registry_id,
        "summary": snapshot["summary"],
        "objects": snapshot["object_rows"],
    }
    payloads: dict[str, bytes] = {
        "registry_config.json": _json_bytes(config.as_dict()),
        "summary.json": _json_bytes(summary_payload),
        "tables/objects.csv": _csv_bytes(snapshot["object_rows"], _OBJECT_FIELDS),
        "tables/artifacts.csv": _csv_bytes(snapshot["artifact_rows"], _ARTIFACT_FIELDS),
        "tables/failed_source_tree.csv": _csv_bytes(
            snapshot["failed_tree_rows"],
            _FAILED_TREE_FIELDS,
        ),
        "report.md": _markdown(config, snapshot).encode("utf-8"),
        "plots/root_inventory.png": _plot_bytes(
            registry_id=config.registry_id,
            summary=snapshot["summary"],
            plot_kind="root_inventory",
            image_format="png",
        ),
        "plots/root_inventory.svg": _plot_bytes(
            registry_id=config.registry_id,
            summary=snapshot["summary"],
            plot_kind="root_inventory",
            image_format="svg",
        ),
        "plots/verification_status.png": _plot_bytes(
            registry_id=config.registry_id,
            summary=snapshot["summary"],
            plot_kind="verification_status",
            image_format="png",
        ),
        "plots/verification_status.svg": _plot_bytes(
            registry_id=config.registry_id,
            summary=snapshot["summary"],
            plot_kind="verification_status",
            image_format="svg",
        ),
    }
    descriptor = {
        "schema_version": EVIDENCE_REGISTRY_SCHEMA_VERSION,
        "object_type": EVIDENCE_REGISTRY_RELEASE_TYPE,
        "status": "complete",
        "registry_id": config.registry_id,
        "workspace_root": config.workspace_root,
        "source_count": snapshot["summary"]["source_count"],
        "source_manifest_count": snapshot["summary"]["source_manifest_count"],
        "verified_count": snapshot["summary"]["verified_count"],
        "verification_failed_count": snapshot["summary"]["verification_failed_count"],
        "registered_artifact_count": snapshot["summary"]["registered_artifact_count"],
        "failed_source_tree_entry_count": snapshot["summary"]["failed_source_tree_entry_count"],
        "failed_source_tree_file_count": snapshot["summary"]["failed_source_tree_file_count"],
        "failed_source_tree_bytes": snapshot["summary"]["failed_source_tree_bytes"],
        "excluded_registry_count": snapshot["summary"]["excluded_registry_count"],
        "excluded_registry_paths": list(config.excluded_registry_paths),
        "sources": snapshot["source_descriptors"],
        "outputs": {
            "configuration": _payload_reference(
                "registry_config.json", payloads["registry_config.json"]
            ),
            "summary": _payload_reference("summary.json", payloads["summary.json"]),
            "object_table": _payload_reference(
                "tables/objects.csv", payloads["tables/objects.csv"]
            ),
            "artifact_table": _payload_reference(
                "tables/artifacts.csv", payloads["tables/artifacts.csv"]
            ),
            "failed_source_tree_table": _payload_reference(
                "tables/failed_source_tree.csv",
                payloads["tables/failed_source_tree.csv"],
            ),
            "report": _payload_reference("report.md", payloads["report.md"]),
            "plots": [
                _payload_reference(relative, payloads[relative])
                for relative in sorted(payloads)
                if relative.startswith("plots/")
            ],
        },
        "generation": {
            "canonical_immediate_children_only": True,
            "verification_attempted_for_every_source": True,
            "failed_sources_retained": True,
            "registry_sources_excluded": True,
            "carla_contacted": False,
            "simulator_mutated": False,
            "source_objects_modified": False,
        },
    }
    payloads["registry.json"] = _json_bytes(descriptor)
    payloads["checksums.sha256"] = _checksum_index(payloads)
    return payloads, descriptor


def _role_for(relative_path: str) -> str:
    exact = {
        "registry_config.json": "evidence_registry_configuration",
        "registry.json": "evidence_registry_release_manifest",
        "summary.json": "evidence_registry_summary",
        "tables/objects.csv": "evidence_registry_object_table",
        "tables/artifacts.csv": "evidence_registry_artifact_table",
        "tables/failed_source_tree.csv": "evidence_registry_failed_source_tree",
        "report.md": "evidence_registry_report",
        "checksums.sha256": "evidence_registry_checksum_index",
    }
    if relative_path.startswith("plots/"):
        return "evidence_registry_plot"
    return exact[relative_path]


def build_evidence_registry(
    *,
    workspace: str | Path,
    registry_id: str,
    cli_args: Sequence[str] | Mapping[str, Any] = (),
) -> dict[str, Any]:
    """Scan and seal one point-in-time evidence registry without contacting CARLA."""

    root = Path(workspace).expanduser().resolve(strict=True)
    _validate_canonical_roots(root)
    runs_root = root / "runs"
    output = runs_root / registry_id
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"evidence registry output already exists: {output}")
    discovery = _discover_sources(root, current_output=output)
    config = EvidenceRegistryConfig.from_mapping(
        {
            "schema_version": EVIDENCE_REGISTRY_SCHEMA_VERSION,
            "object_type": EVIDENCE_REGISTRY_OBJECT_TYPE,
            "registry_id": registry_id,
            "workspace_root": str(root),
            "canonical_roots": list(CANONICAL_EVIDENCE_ROOTS),
            "exclude_registry_objects": True,
            "source_paths": [candidate.relative_path for candidate in discovery.candidates],
            "excluded_registry_paths": list(discovery.excluded_registry_paths),
        }
    )
    snapshot = _build_snapshot(
        workspace=root,
        registry_id=config.registry_id,
        discovery=discovery,
    )
    payloads, descriptor = _expected_payloads(config, snapshot)
    _validate_canonical_roots(root)
    tracker = RunArtifactTracker(
        runs_root,
        run_id=config.registry_id,
        cli_args=cli_args,
        config={
            "schema_version": EVIDENCE_REGISTRY_SCHEMA_VERSION,
            "object_type": EVIDENCE_REGISTRY_OBJECT_TYPE,
            "registry": config.as_dict(),
            "generation": {
                "carla_contacted": False,
                "simulator_mutated": False,
                "source_objects_modified": False,
            },
        },
        repository_root=root,
        input_refs=snapshot["input_references"],
    )
    with tracker:
        for relative_path in sorted(payloads, key=lambda value: value.encode("utf-8")):
            path = tracker.artifact_path(relative_path)
            _atomic_write_bytes(path, payloads[relative_path])
            tracker.register_artifact(
                path,
                role=_role_for(relative_path),
                metadata={
                    "registry_id": config.registry_id,
                    "source_count": snapshot["summary"]["source_count"],
                },
            )
    return {
        "registry_id": config.registry_id,
        "registry_dir": str(tracker.run_dir),
        "manifest": str(tracker.manifest_path),
        "descriptor": descriptor,
        "summary": snapshot["summary"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a checksum-indexed, point-in-time registry of research objects "
            "without contacting CARLA"
        )
    )
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--registry-id", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_evidence_registry(
        workspace=args.workspace,
        registry_id=args.registry_id,
        cli_args=vars(args),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


__all__ = [
    "EVIDENCE_REGISTRY_RELEASE_TYPE",
    "build_evidence_registry",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    raise SystemExit(main())
