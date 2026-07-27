"""Build standalone, checksum-indexed reproduction bundles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import stat
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..artifacts import RunArtifactTracker, fingerprint_file
from ..verification import VerificationResult, verify_research_object
from .contracts import (
    ReproductionBundleConfig,
    ReproductionSource,
    load_reproduction_bundle_config,
)

REPRODUCTION_BUNDLE_SCHEMA_VERSION = "1.0"
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_MAX_SOURCE_FILE_BYTES = 25 * 1024 * 1024
_MAX_SOURCE_TOTAL_BYTES = 250 * 1024 * 1024
_FORBIDDEN_SOURCE_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
    }
)
_FORBIDDEN_SOURCE_SUFFIXES = frozenset(
    {
        ".der",
        ".key",
        ".p12",
        ".pfx",
        ".pem",
        ".pyc",
        ".pyo",
    }
)
_FORBIDDEN_SOURCE_NAMES = frozenset(
    {
        ".env",
        ".env.local",
        ".env.production",
        ".env.test",
        ".DS_Store",
        "credentials.json",
        "secrets.json",
    }
)


def _atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
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
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: Any) -> None:
    data = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, data)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(fields),
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _relative_reference(root: Path, path: Path) -> dict[str, Any]:
    reference = fingerprint_file(path)
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": reference["sha256"],
        "size_bytes": reference["size_bytes"],
    }


def _safe_source_file(repository_root: Path, candidate: Path) -> Path:
    if candidate.is_symlink():
        raise ValueError(f"source snapshot cannot contain symlinks: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(repository_root)
    except (OSError, ValueError) as error:
        raise ValueError(f"source snapshot path escapes repository: {candidate}") from error
    if not resolved.is_file():
        raise ValueError(f"source snapshot entry is not a regular file: {relative}")
    if (
        any(part in _FORBIDDEN_SOURCE_PARTS for part in relative.parts)
        or resolved.name in _FORBIDDEN_SOURCE_NAMES
        or resolved.suffix.casefold() in _FORBIDDEN_SOURCE_SUFFIXES
    ):
        raise ValueError(f"source snapshot includes a forbidden path: {relative}")
    if resolved.stat().st_size > _MAX_SOURCE_FILE_BYTES:
        raise ValueError(f"source snapshot file exceeds {_MAX_SOURCE_FILE_BYTES} bytes: {relative}")
    return resolved


def _collect_source_files(
    repository_root: Path,
    source_paths: Sequence[str],
) -> list[Path]:
    files: dict[str, Path] = {}
    for relative in source_paths:
        candidate = repository_root / relative
        if candidate.is_symlink():
            raise ValueError(f"source_paths cannot select a symlink: {relative}")
        if not candidate.exists():
            raise FileNotFoundError(f"source snapshot path is missing: {relative}")
        if candidate.is_file():
            resolved = _safe_source_file(repository_root, candidate)
            files[resolved.relative_to(repository_root).as_posix()] = resolved
            continue
        if not candidate.is_dir():
            raise ValueError(f"source snapshot path is neither file nor directory: {relative}")
        for descendant in candidate.rglob("*"):
            if descendant.is_symlink():
                raise ValueError(f"source snapshot cannot contain symlinks: {descendant}")
            if not descendant.is_file():
                continue
            descendant_relative = descendant.relative_to(repository_root)
            if (
                any(part in _FORBIDDEN_SOURCE_PARTS for part in descendant_relative.parts)
                or descendant.name in _FORBIDDEN_SOURCE_NAMES
                or descendant.suffix.casefold() in _FORBIDDEN_SOURCE_SUFFIXES
            ):
                continue
            resolved = _safe_source_file(repository_root, descendant)
            files[resolved.relative_to(repository_root).as_posix()] = resolved
    ordered = [files[key] for key in sorted(files, key=lambda item: item.encode("utf-8"))]
    if not ordered:
        raise ValueError("source snapshot selection contains no files")
    total = sum(path.stat().st_size for path in ordered)
    if total > _MAX_SOURCE_TOTAL_BYTES:
        raise ValueError(f"source snapshot exceeds {_MAX_SOURCE_TOTAL_BYTES} total bytes")
    return ordered


def _source_inventory(repository_root: Path, files: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in files:
        reference = fingerprint_file(path)
        executable = bool(path.stat().st_mode & stat.S_IXUSR)
        rows.append(
            {
                "path": path.relative_to(repository_root).as_posix(),
                "sha256": reference["sha256"],
                "size_bytes": reference["size_bytes"],
                "archive_mode": "0755" if executable else "0644",
            }
        )
    return rows


def _write_source_archive(
    path: Path,
    repository_root: Path,
    files: Sequence[Path],
    inventory: Sequence[Mapping[str, Any]],
) -> None:
    expected = {str(row["path"]): row for row in inventory}
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_STORED,
            strict_timestamps=True,
        ) as archive:
            for source in files:
                relative = source.relative_to(repository_root).as_posix()
                executable = bool(source.stat().st_mode & stat.S_IXUSR)
                payload = source.read_bytes()
                expected_row = expected.get(relative)
                if (
                    expected_row is None
                    or len(payload) != expected_row["size_bytes"]
                    or hashlib.sha256(payload).hexdigest() != expected_row["sha256"]
                ):
                    raise RuntimeError(f"source file changed while building archive: {relative}")
                info = zipfile.ZipInfo(relative, date_time=_ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = (0o100755 if executable else 0o100644) << 16
                info.flag_bits |= 0x800
                archive.writestr(info, payload)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def render_reproduction_script(commands: Sequence[Sequence[str]]) -> str:
    rendered_commands = "\n".join(
        " ".join(shlex.quote(str(token)) for token in command) for command in commands
    )
    return f"""#!/bin/sh
set -eu

BUNDLE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
WORK_DIR=${{1:-"$PWD/reproduced-workspace"}}

if [ -e "$WORK_DIR" ]; then
  echo "Refusing to overwrite existing reproduction directory: $WORK_DIR" >&2
  exit 2
fi

mkdir -p "$WORK_DIR"
unzip -q "$BUNDLE_DIR/source/source.zip" -d "$WORK_DIR"
cd "$WORK_DIR"

{rendered_commands}
"""


def _write_checksum_index(root: Path, paths: Sequence[Path], output: Path) -> None:
    entries: list[tuple[str, str]] = []
    for path in paths:
        reference = fingerprint_file(path)
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        entries.append((relative, str(reference["sha256"])))
    entries.sort(key=lambda item: item[0].encode("utf-8"))
    text = "".join(f"{digest}  {relative}\n" for relative, digest in entries)
    _atomic_write_bytes(output, text.encode("utf-8"))


def _resolve_from_config(config_path: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return candidate.resolve(strict=True)


def _source_verification_record(result: VerificationResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "status": result.status,
        "schema_version": result.schema_version,
        "artifact_count": result.artifact_count,
        "artifact_bytes": result.artifact_bytes,
        "role_counts": dict(result.role_counts),
        "checksum_index_count": result.checksum_index_count,
        "checksum_index_entries": result.checksum_index_entries,
        "external_reference_count": result.external_reference_count,
        "deep_verification": dict(result.deep_verification),
        "git": dict(result.git),
        "unregistered_file_count": result.unregistered_file_count,
    }


def _verify_source(
    config_path: Path,
    source: ReproductionSource,
    *,
    confirmatory: bool,
) -> tuple[Path, VerificationResult, dict[str, Any]]:
    root = _resolve_from_config(config_path, source.path)
    verification = verify_research_object(
        root,
        verify_references=True,
        deep=True,
        reject_unregistered=True,
        require_clean_git=confirmatory,
    )
    if verification.run_id != source.source_id:
        raise ValueError(f"source {source.source_id!r} resolves to run_id {verification.run_id!r}")
    manifest_path = Path(verification.root) / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise TypeError(f"source manifest is not an object: {manifest_path}")
    return manifest_path, verification, manifest


def _readme(
    config: ReproductionBundleConfig,
    *,
    source_count: int,
    file_count: int,
    lock_count: int,
) -> str:
    commands = "\n".join(
        f"{index}. `{' '.join(shlex.quote(token) for token in command)}`"
        for index, command in enumerate(config.commands, start=1)
    )
    limitations = "\n".join(f"- {value}" for value in config.limitations)
    return f"""# {config.title}

Bundle ID: `{config.bundle_id}`  
Purpose: `{config.purpose}`  
Sources: {source_count} verified research objects  
Source snapshot: {file_count} files  
Dependency locks: {lock_count}

This bundle archives the exact source tree selected by the frozen configuration,
the dependency lockfiles, verified source manifests, hardware/software metadata,
and tokenized reproduction commands. It does not silently fetch mutable model or
dataset aliases.

## Quick reproduction

Run `./reproduce.sh /path/to/new-empty-workspace`. The script refuses to
overwrite an existing directory, extracts the immutable source snapshot, and
executes these pre-registered commands:

{commands}

## Integrity

Before use, run:

```text
uv run carla-verify-reproduction {config.bundle_id}
uv run carla-verify {config.bundle_id} --reject-unregistered
```

## Declared limitations

{limitations}
"""


def build_reproduction_bundle(
    *,
    config_path: str | Path,
    bundles_root: str | Path = "bundles",
    cli_args: Sequence[str] | Mapping[str, Any] = (),
) -> dict[str, Any]:
    resolved_config_path = Path(config_path).expanduser().resolve(strict=True)
    config = load_reproduction_bundle_config(resolved_config_path)
    repository_root = _resolve_from_config(resolved_config_path, config.repository_root)
    if not repository_root.is_dir():
        raise ValueError("repository_root must resolve to a directory")

    source_files = _collect_source_files(repository_root, config.source_paths)
    inventory = _source_inventory(repository_root, source_files)
    source_records = [
        (
            source,
            *_verify_source(
                resolved_config_path,
                source,
                confirmatory=config.purpose == "confirmatory",
            ),
        )
        for source in config.sources
    ]

    input_refs: list[dict[str, Any]] = []
    for source, manifest_path, _verification, _manifest in source_records:
        reference = fingerprint_file(manifest_path)
        input_refs.append(
            {
                "kind": "archived_reproduction_source",
                "source_id": source.source_id,
                "source_role": source.role,
                "manifest_sha256": reference["sha256"],
                "manifest_size_bytes": reference["size_bytes"],
            }
        )

    tracker = RunArtifactTracker(
        bundles_root,
        run_id=config.bundle_id,
        cli_args=cli_args,
        config={
            "schema_version": REPRODUCTION_BUNDLE_SCHEMA_VERSION,
            "object_type": "reproduction_bundle",
            "bundle": config.as_dict(),
        },
        repository_root=repository_root,
        input_refs=input_refs,
    )
    with tracker:
        if config.purpose == "confirmatory":
            git = tracker.manifest["git"]
            if (
                git.get("available") is not True
                or git.get("dirty") is not False
                or not git.get("commit")
            ):
                raise RuntimeError(
                    "confirmatory reproduction bundle requires a clean committed repository"
                )

        manifest_snapshot = tracker.manifest
        dependency_locks = manifest_snapshot["reproducibility"]["dependency_locks"]
        if config.purpose == "confirmatory" and not dependency_locks:
            raise RuntimeError("confirmatory reproduction bundle requires a dependency lockfile")
        if config.purpose == "confirmatory" and not any(
            command[:2] == ("uv", "sync") and "--frozen" in command for command in config.commands
        ):
            raise RuntimeError("confirmatory reproduction commands must include 'uv sync --frozen'")

        config_output = tracker.artifact_path("bundle_config.json")
        descriptor_path = tracker.artifact_path("bundle.json")
        archive_path = tracker.artifact_path("source/source.zip")
        inventory_json_path = tracker.artifact_path("source/inventory.json")
        inventory_csv_path = tracker.artifact_path("source/inventory.csv")
        graph_path = tracker.artifact_path("sources/source_graph.json")
        environment_path = tracker.artifact_path("environment/environment.json")
        commands_path = tracker.artifact_path("commands.json")
        script_path = tracker.artifact_path("reproduce.sh")
        readme_path = tracker.artifact_path("README.md")
        checksum_path = tracker.artifact_path("checksums.sha256")

        _write_json(config_output, config.as_dict())
        _write_source_archive(
            archive_path,
            repository_root,
            source_files,
            inventory,
        )
        _write_json(
            inventory_json_path,
            {
                "schema_version": REPRODUCTION_BUNDLE_SCHEMA_VERSION,
                "file_count": len(inventory),
                "total_bytes": sum(int(row["size_bytes"]) for row in inventory),
                "files": inventory,
            },
        )
        _write_csv(
            inventory_csv_path,
            inventory,
            ("path", "sha256", "size_bytes", "archive_mode"),
        )

        graph_entries: list[dict[str, Any]] = []
        copied_source_manifests: list[Path] = []
        for source, manifest_path, verification, _manifest in source_records:
            copied = tracker.artifact_path(f"sources/{source.source_id}/manifest.json")
            _atomic_write_bytes(copied, manifest_path.read_bytes())
            copied_source_manifests.append(copied)
            graph_entries.append(
                {
                    "source_id": source.source_id,
                    "role": source.role,
                    "label": source.label,
                    "configured_locator": source.path,
                    "archived_manifest": _relative_reference(tracker.run_dir, copied),
                    "verification": _source_verification_record(verification),
                }
            )
        _write_json(
            graph_path,
            {
                "schema_version": REPRODUCTION_BUNDLE_SCHEMA_VERSION,
                "bundle_id": config.bundle_id,
                "source_count": len(graph_entries),
                "all_sources_verified_before_archival": True,
                "sources": graph_entries,
            },
        )

        copied_locks: list[Path] = []
        lock_descriptors: list[dict[str, Any]] = []
        for lock in dependency_locks:
            source_lock = repository_root / str(lock["relative_path"])
            copied_lock = tracker.artifact_path(
                f"environment/lockfiles/{Path(str(lock['relative_path'])).name}"
            )
            _atomic_write_bytes(copied_lock, source_lock.read_bytes())
            copied_locks.append(copied_lock)
            archived = _relative_reference(tracker.run_dir, copied_lock)
            if archived["sha256"] != lock["sha256"] or archived["size_bytes"] != lock["size_bytes"]:
                raise RuntimeError("dependency lock changed between provenance capture and copy")
            lock_descriptors.append(
                {
                    "repository_relative_path": lock["relative_path"],
                    "archived": archived,
                }
            )
        _write_json(
            environment_path,
            {
                "schema_version": REPRODUCTION_BUNDLE_SCHEMA_VERSION,
                "environment": manifest_snapshot["environment"],
                "hardware": manifest_snapshot["reproducibility"]["hardware"],
                "git": manifest_snapshot["git"],
                "dependency_locks": lock_descriptors,
            },
        )
        _write_json(
            commands_path,
            {
                "schema_version": REPRODUCTION_BUNDLE_SCHEMA_VERSION,
                "working_directory": "extracted_source_root",
                "commands": [list(command) for command in config.commands],
            },
        )
        _atomic_write_bytes(
            script_path,
            render_reproduction_script(config.commands).encode("utf-8"),
            mode=0o755,
        )
        _atomic_write_bytes(
            readme_path,
            _readme(
                config,
                source_count=len(graph_entries),
                file_count=len(inventory),
                lock_count=len(copied_locks),
            ).encode("utf-8"),
        )

        descriptor = {
            "schema_version": REPRODUCTION_BUNDLE_SCHEMA_VERSION,
            "object_type": "reproduction_bundle_release",
            "status": "complete",
            "bundle_id": config.bundle_id,
            "title": config.title,
            "authors": list(config.authors),
            "purpose": config.purpose,
            "source_count": len(graph_entries),
            "source_file_count": len(inventory),
            "source_total_bytes": sum(int(row["size_bytes"]) for row in inventory),
            "dependency_lock_count": len(copied_locks),
            "command_count": len(config.commands),
            "configuration": _relative_reference(tracker.run_dir, config_output),
            "source_archive": _relative_reference(tracker.run_dir, archive_path),
            "source_inventory_json": _relative_reference(
                tracker.run_dir,
                inventory_json_path,
            ),
            "source_inventory_csv": _relative_reference(
                tracker.run_dir,
                inventory_csv_path,
            ),
            "source_graph": _relative_reference(tracker.run_dir, graph_path),
            "environment": _relative_reference(tracker.run_dir, environment_path),
            "commands": _relative_reference(tracker.run_dir, commands_path),
            "reproduction_script": _relative_reference(tracker.run_dir, script_path),
            "readme": _relative_reference(tracker.run_dir, readme_path),
            "dependency_locks": lock_descriptors,
            "limitations": list(config.limitations),
            "generation": {
                "source_archive_format": "zip-stored-fixed-metadata-v1",
                "source_archive_timestamp": "1980-01-01T00:00:00Z",
                "source_symlinks_allowed": False,
                "secrets_from_environment_captured": False,
                "external_sources_required_for_bundle_verification": False,
                "confirmatory_clean_git_gate": config.purpose == "confirmatory",
            },
        }
        _write_json(descriptor_path, descriptor)

        payload_paths = [
            config_output,
            descriptor_path,
            archive_path,
            inventory_json_path,
            inventory_csv_path,
            graph_path,
            environment_path,
            commands_path,
            script_path,
            readme_path,
            *copied_source_manifests,
            *copied_locks,
        ]
        _write_checksum_index(tracker.run_dir, payload_paths, checksum_path)
        role_by_path = {
            config_output: "reproduction_bundle_configuration",
            descriptor_path: "reproduction_bundle_release_manifest",
            archive_path: "reproduction_source_archive",
            inventory_json_path: "reproduction_source_inventory_json",
            inventory_csv_path: "reproduction_source_inventory_csv",
            graph_path: "reproduction_source_graph",
            environment_path: "reproduction_environment_snapshot",
            commands_path: "reproduction_commands",
            script_path: "reproduction_script",
            readme_path: "reproduction_readme",
            checksum_path: "reproduction_checksum_index",
            **{path: "reproduction_source_manifest" for path in copied_source_manifests},
            **{path: "dependency_lockfile_snapshot" for path in copied_locks},
        }
        for path, role in role_by_path.items():
            tracker.register_artifact(
                path,
                role=role,
                metadata={
                    "bundle_id": config.bundle_id,
                    "purpose": config.purpose,
                },
            )

    return {
        "bundle_id": config.bundle_id,
        "bundle_dir": str(tracker.run_dir),
        "manifest": str(tracker.manifest_path),
        "descriptor": descriptor,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a standalone checksum-indexed reproduction bundle from "
            "verified research objects and an exact source snapshot"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--bundles-root", default="bundles")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_reproduction_bundle(
        config_path=args.config,
        bundles_root=args.bundles_root,
        cli_args=vars(args),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


__all__ = [
    "REPRODUCTION_BUNDLE_SCHEMA_VERSION",
    "build_reproduction_bundle",
    "main",
    "parse_args",
    "render_reproduction_script",
]
