"""Consumer-side semantic verification for standalone reproduction bundles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import stat
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import fingerprint_file
from ..verification import ArtifactIntegrityError, verify_research_object
from .builder import REPRODUCTION_BUNDLE_SCHEMA_VERSION, render_reproduction_script
from .contracts import ReproductionBundleConfig

_REQUIRED_ROLES = frozenset(
    {
        "reproduction_bundle_configuration",
        "reproduction_bundle_release_manifest",
        "reproduction_source_archive",
        "reproduction_source_inventory_json",
        "reproduction_source_inventory_csv",
        "reproduction_source_graph",
        "reproduction_environment_snapshot",
        "reproduction_commands",
        "reproduction_script",
        "reproduction_readme",
        "reproduction_checksum_index",
    }
)
_SHA256_LENGTH = 64
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class VerifiedReproductionBundle:
    root: Path
    bundle_id: str
    manifest: Mapping[str, Any]
    config: ReproductionBundleConfig
    descriptor: Mapping[str, Any]
    source_graph: Mapping[str, Any]
    source_inventory: Mapping[str, Any]
    environment: Mapping[str, Any]
    commands: Mapping[str, Any]


def _load_json(path: Path, name: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(f"could not read {name}: {error}") from error


def _object(path: Path, name: str) -> Mapping[str, Any]:
    value = _load_json(path, name)
    if not isinstance(value, Mapping):
        raise ArtifactIntegrityError(f"{name} must contain an object")
    return value


def _safe_file(root: Path, relative: str, *, context: str) -> Path:
    candidate = Path(relative)
    if (
        not relative
        or candidate.is_absolute()
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ArtifactIntegrityError(f"{context} has an unsafe relative path")
    current = root
    for part in candidate.parts:
        current = current / part
        if current.is_symlink():
            raise ArtifactIntegrityError(f"{context} uses a symlink")
    try:
        resolved = (root / candidate).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ArtifactIntegrityError(f"{context} is missing or escapes bundle") from error
    if not resolved.is_file():
        raise ArtifactIntegrityError(f"{context} is not a regular file")
    return resolved


def _roles(
    root: Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    raw = manifest.get("artifacts")
    if not isinstance(raw, list):
        raise ArtifactIntegrityError("bundle tracker artifacts must be an array")
    unique: dict[str, Path] = {}
    repeated: dict[str, list[Path]] = {}
    for artifact in raw:
        if not isinstance(artifact, Mapping):
            raise ArtifactIntegrityError("bundle tracker artifact is malformed")
        role = artifact.get("role")
        relative = artifact.get("path")
        if not isinstance(role, str) or not isinstance(relative, str):
            raise ArtifactIntegrityError("bundle tracker artifact role/path is malformed")
        path = _safe_file(root, relative, context=f"artifact role {role}")
        repeated.setdefault(role, []).append(path)
        if role in _REQUIRED_ROLES:
            if role in unique:
                raise ArtifactIntegrityError(f"bundle has duplicate required role {role}")
            unique[role] = path
    missing = sorted(_REQUIRED_ROLES - set(unique))
    if missing:
        raise ArtifactIntegrityError(
            f"bundle is missing required artifact roles: {', '.join(missing)}"
        )
    return unique, repeated


def _fingerprint_matches(path: Path, raw: Mapping[str, Any], *, context: str) -> None:
    reference = fingerprint_file(path)
    if raw.get("sha256") != reference["sha256"] or raw.get("size_bytes") != reference["size_bytes"]:
        raise ArtifactIntegrityError(f"{context} fingerprint does not match")


def _relative_ref_path(
    root: Path,
    raw: Any,
    *,
    context: str,
) -> Path:
    if not isinstance(raw, Mapping):
        raise ArtifactIntegrityError(f"{context} must be an object")
    if set(raw) != {"path", "sha256", "size_bytes"}:
        raise ArtifactIntegrityError(f"{context} fields differ from schema")
    relative = raw.get("path")
    digest = raw.get("sha256")
    size = raw.get("size_bytes")
    if (
        not isinstance(relative, str)
        or not isinstance(digest, str)
        or len(digest) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in digest)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
    ):
        raise ArtifactIntegrityError(f"{context} reference is malformed")
    path = _safe_file(root, relative, context=context)
    _fingerprint_matches(path, raw, context=context)
    return path


def _verify_inventory(
    *,
    root: Path,
    inventory_path: Path,
    inventory_csv_path: Path,
    archive_path: Path,
) -> Mapping[str, Any]:
    inventory = _object(inventory_path, "source inventory")
    if set(inventory) != {"schema_version", "file_count", "total_bytes", "files"}:
        raise ArtifactIntegrityError("source inventory fields differ from schema")
    if inventory.get("schema_version") != REPRODUCTION_BUNDLE_SCHEMA_VERSION:
        raise ArtifactIntegrityError("source inventory schema is unsupported")
    raw_files = inventory.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ArtifactIntegrityError("source inventory files must be a non-empty array")
    rows: list[dict[str, Any]] = []
    prior: bytes | None = None
    for index, raw in enumerate(raw_files):
        context = f"source inventory file {index}"
        if not isinstance(raw, Mapping):
            raise ArtifactIntegrityError(f"{context} must be an object")
        if set(raw) != {"path", "sha256", "size_bytes", "archive_mode"}:
            raise ArtifactIntegrityError(f"{context} fields differ from schema")
        relative = raw.get("path")
        digest = raw.get("sha256")
        size = raw.get("size_bytes")
        mode = raw.get("archive_mode")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in Path(relative).parts)
        ):
            raise ArtifactIntegrityError(f"{context} path is unsafe")
        encoded = relative.encode("utf-8")
        if prior is not None and encoded <= prior:
            raise ArtifactIntegrityError("source inventory paths are not uniquely sorted")
        if (
            not isinstance(digest, str)
            or len(digest) != _SHA256_LENGTH
            or any(character not in "0123456789abcdef" for character in digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or mode not in {"0644", "0755"}
        ):
            raise ArtifactIntegrityError(f"{context} fingerprint/mode is malformed")
        rows.append(dict(raw))
        prior = encoded
    if inventory.get("file_count") != len(rows):
        raise ArtifactIntegrityError("source inventory file_count is inconsistent")
    if inventory.get("total_bytes") != sum(int(row["size_bytes"]) for row in rows):
        raise ArtifactIntegrityError("source inventory total_bytes is inconsistent")

    try:
        with inventory_csv_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["path", "sha256", "size_bytes", "archive_mode"]:
                raise ArtifactIntegrityError("source inventory CSV header is invalid")
            csv_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ArtifactIntegrityError(f"could not read source inventory CSV: {error}") from error
    expected_csv = [
        {
            "path": str(row["path"]),
            "sha256": str(row["sha256"]),
            "size_bytes": str(row["size_bytes"]),
            "archive_mode": str(row["archive_mode"]),
        }
        for row in rows
    ]
    if csv_rows != expected_csv:
        raise ArtifactIntegrityError("source inventory CSV differs from canonical JSON")

    expected_by_path = {str(row["path"]): row for row in rows}
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if names != list(expected_by_path):
                raise ArtifactIntegrityError(
                    "source archive order/content differs from source inventory"
                )
            if archive.testzip() is not None:
                raise ArtifactIntegrityError("source archive CRC verification failed")
            for info in infos:
                row = expected_by_path[info.filename]
                unix_mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(unix_mode):
                    raise ArtifactIntegrityError("source archive contains a symlink")
                expected_mode = 0o100755 if row["archive_mode"] == "0755" else 0o100644
                if (
                    info.date_time != _ZIP_TIMESTAMP
                    or info.compress_type != zipfile.ZIP_STORED
                    or unix_mode != expected_mode
                ):
                    raise ArtifactIntegrityError(
                        f"source archive metadata differs from contract: {info.filename}"
                    )
                payload = archive.read(info)
                if (
                    len(payload) != row["size_bytes"]
                    or hashlib.sha256(payload).hexdigest() != row["sha256"]
                ):
                    raise ArtifactIntegrityError(
                        f"source archive entry differs from inventory: {info.filename}"
                    )
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        if isinstance(error, ArtifactIntegrityError):
            raise
        raise ArtifactIntegrityError(f"could not verify source archive: {error}") from error
    del root
    return inventory


def _verify_source_graph(
    *,
    root: Path,
    graph_path: Path,
    config: ReproductionBundleConfig,
    role_paths: Mapping[str, list[Path]],
) -> Mapping[str, Any]:
    graph = _object(graph_path, "source graph")
    if set(graph) != {
        "schema_version",
        "bundle_id",
        "source_count",
        "all_sources_verified_before_archival",
        "sources",
    }:
        raise ArtifactIntegrityError("source graph fields differ from schema")
    if (
        graph.get("schema_version") != REPRODUCTION_BUNDLE_SCHEMA_VERSION
        or graph.get("bundle_id") != config.bundle_id
        or graph.get("all_sources_verified_before_archival") is not True
    ):
        raise ArtifactIntegrityError("source graph envelope is invalid")
    sources = graph.get("sources")
    if not isinstance(sources, list) or len(sources) != len(config.sources):
        raise ArtifactIntegrityError("source graph source count is inconsistent")
    if graph.get("source_count") != len(sources):
        raise ArtifactIntegrityError("source graph source_count is inconsistent")
    registered_manifests = {
        path.relative_to(root).as_posix()
        for path in role_paths.get("reproduction_source_manifest", [])
    }
    seen: set[str] = set()
    for configured, raw in zip(config.sources, sources, strict=True):
        if not isinstance(raw, Mapping):
            raise ArtifactIntegrityError("source graph entry must be an object")
        if set(raw) != {
            "source_id",
            "role",
            "label",
            "configured_locator",
            "archived_manifest",
            "verification",
        }:
            raise ArtifactIntegrityError("source graph entry fields differ from schema")
        if (
            raw.get("source_id") != configured.source_id
            or raw.get("role") != configured.role
            or raw.get("label") != configured.label
            or raw.get("configured_locator") != configured.path
        ):
            raise ArtifactIntegrityError("source graph entry differs from bundle config")
        archived = _relative_ref_path(
            root,
            raw.get("archived_manifest"),
            context=f"archived source manifest {configured.source_id}",
        )
        archived_relative = archived.relative_to(root).as_posix()
        if archived_relative not in registered_manifests:
            raise ArtifactIntegrityError("archived source manifest is not registered")
        source_manifest = _object(archived, f"source manifest {configured.source_id}")
        verification = raw.get("verification")
        if not isinstance(verification, Mapping):
            raise ArtifactIntegrityError("source graph verification record is malformed")
        if (
            source_manifest.get("run_id") != configured.source_id
            or source_manifest.get("status") != "success"
            or verification.get("run_id") != configured.source_id
            or verification.get("status") != "success"
            or verification.get("unregistered_file_count") != 0
        ):
            raise ArtifactIntegrityError("archived source verification evidence is inconsistent")
        if configured.source_id in seen:
            raise ArtifactIntegrityError("source graph contains a duplicate source ID")
        seen.add(configured.source_id)
    if len(registered_manifests) != len(sources):
        raise ArtifactIntegrityError("bundle has unexpected archived source manifests")
    return graph


def _verify_environment(
    *,
    root: Path,
    path: Path,
    manifest: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    role_paths: Mapping[str, list[Path]],
) -> Mapping[str, Any]:
    environment = _object(path, "reproduction environment")
    if set(environment) != {
        "schema_version",
        "environment",
        "hardware",
        "git",
        "dependency_locks",
    }:
        raise ArtifactIntegrityError("reproduction environment fields differ from schema")
    reproducibility = manifest.get("reproducibility")
    if not isinstance(reproducibility, Mapping):
        raise ArtifactIntegrityError("bundle tracker lacks reproducibility metadata")
    if (
        environment.get("schema_version") != REPRODUCTION_BUNDLE_SCHEMA_VERSION
        or environment.get("environment") != manifest.get("environment")
        or environment.get("hardware") != reproducibility.get("hardware")
        or environment.get("git") != manifest.get("git")
        or environment.get("dependency_locks") != descriptor.get("dependency_locks")
    ):
        raise ArtifactIntegrityError("reproduction environment differs from tracker provenance")
    raw_locks = environment.get("dependency_locks")
    tracker_locks = reproducibility.get("dependency_locks")
    if not isinstance(raw_locks, list) or not isinstance(tracker_locks, list):
        raise ArtifactIntegrityError("dependency lock metadata must be arrays")
    if len(raw_locks) != len(tracker_locks):
        raise ArtifactIntegrityError("dependency lock counts are inconsistent")
    registered_locks = {
        item.relative_to(root).as_posix()
        for item in role_paths.get("dependency_lockfile_snapshot", [])
    }
    for raw, tracker_lock in zip(raw_locks, tracker_locks, strict=True):
        if not isinstance(raw, Mapping) or not isinstance(tracker_lock, Mapping):
            raise ArtifactIntegrityError("dependency lock record is malformed")
        if raw.get("repository_relative_path") != tracker_lock.get("relative_path"):
            raise ArtifactIntegrityError("dependency lock source path is inconsistent")
        archived = _relative_ref_path(
            root,
            raw.get("archived"),
            context="archived dependency lock",
        )
        if archived.relative_to(root).as_posix() not in registered_locks:
            raise ArtifactIntegrityError("archived dependency lock is not registered")
        reference = raw["archived"]
        if reference.get("sha256") != tracker_lock.get("sha256") or reference.get(
            "size_bytes"
        ) != tracker_lock.get("size_bytes"):
            raise ArtifactIntegrityError("archived dependency lock differs from provenance")
    if len(registered_locks) != len(raw_locks):
        raise ArtifactIntegrityError("bundle has unexpected dependency lock snapshots")
    return environment


def load_verified_reproduction_bundle(
    path: str | Path,
) -> VerifiedReproductionBundle:
    """Verify a reproduction bundle without requiring its original source objects."""

    generic = verify_research_object(
        path,
        verify_references=True,
        deep=False,
        reject_unregistered=True,
    )
    root = Path(generic.root)
    manifest = _object(root / "manifest.json", "bundle tracker manifest")
    unique_roles, role_paths = _roles(root, manifest)
    config_raw = _object(
        unique_roles["reproduction_bundle_configuration"],
        "bundle configuration",
    )
    try:
        config = ReproductionBundleConfig.from_mapping(config_raw)
    except (TypeError, ValueError) as error:
        raise ArtifactIntegrityError(f"bundle configuration is invalid: {error}") from error
    descriptor = _object(
        unique_roles["reproduction_bundle_release_manifest"],
        "bundle release manifest",
    )
    if (
        descriptor.get("schema_version") != REPRODUCTION_BUNDLE_SCHEMA_VERSION
        or descriptor.get("object_type") != "reproduction_bundle_release"
        or descriptor.get("status") != "complete"
        or descriptor.get("bundle_id") != config.bundle_id
        or generic.run_id != config.bundle_id
    ):
        raise ArtifactIntegrityError("reproduction bundle descriptor envelope is invalid")

    descriptor_roles = {
        "configuration": "reproduction_bundle_configuration",
        "source_archive": "reproduction_source_archive",
        "source_inventory_json": "reproduction_source_inventory_json",
        "source_inventory_csv": "reproduction_source_inventory_csv",
        "source_graph": "reproduction_source_graph",
        "environment": "reproduction_environment_snapshot",
        "commands": "reproduction_commands",
        "reproduction_script": "reproduction_script",
        "readme": "reproduction_readme",
    }
    for descriptor_key, role in descriptor_roles.items():
        referenced = _relative_ref_path(
            root,
            descriptor.get(descriptor_key),
            context=f"bundle descriptor {descriptor_key}",
        )
        if referenced != unique_roles[role]:
            raise ArtifactIntegrityError(
                f"bundle descriptor {descriptor_key} points to the wrong artifact"
            )

    inventory = _verify_inventory(
        root=root,
        inventory_path=unique_roles["reproduction_source_inventory_json"],
        inventory_csv_path=unique_roles["reproduction_source_inventory_csv"],
        archive_path=unique_roles["reproduction_source_archive"],
    )
    graph = _verify_source_graph(
        root=root,
        graph_path=unique_roles["reproduction_source_graph"],
        config=config,
        role_paths=role_paths,
    )
    environment = _verify_environment(
        root=root,
        path=unique_roles["reproduction_environment_snapshot"],
        manifest=manifest,
        descriptor=descriptor,
        role_paths=role_paths,
    )
    commands = _object(unique_roles["reproduction_commands"], "reproduction commands")
    expected_commands = [list(command) for command in config.commands]
    if commands != {
        "schema_version": REPRODUCTION_BUNDLE_SCHEMA_VERSION,
        "working_directory": "extracted_source_root",
        "commands": expected_commands,
    }:
        raise ArtifactIntegrityError("reproduction commands differ from bundle config")
    try:
        script = unique_roles["reproduction_script"].read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ArtifactIntegrityError(f"could not read reproduction script: {error}") from error
    if script != render_reproduction_script(config.commands):
        raise ArtifactIntegrityError(
            "reproduction script does not reproduce from tokenized commands"
        )
    if not unique_roles["reproduction_script"].stat().st_mode & stat.S_IXUSR:
        raise ArtifactIntegrityError("reproduction script is not executable")

    if (
        descriptor.get("source_count") != len(config.sources)
        or descriptor.get("source_file_count") != inventory.get("file_count")
        or descriptor.get("source_total_bytes") != inventory.get("total_bytes")
        or descriptor.get("dependency_lock_count") != len(environment.get("dependency_locks", []))
        or descriptor.get("command_count") != len(config.commands)
        or descriptor.get("limitations") != list(config.limitations)
    ):
        raise ArtifactIntegrityError("reproduction bundle descriptor counts are inconsistent")
    generation = descriptor.get("generation")
    if not isinstance(generation, Mapping):
        raise ArtifactIntegrityError("reproduction bundle generation record is missing")
    if (
        generation.get("source_archive_format") != "zip-stored-fixed-metadata-v1"
        or generation.get("source_archive_timestamp") != "1980-01-01T00:00:00Z"
        or generation.get("source_symlinks_allowed") is not False
        or generation.get("secrets_from_environment_captured") is not False
        or generation.get("external_sources_required_for_bundle_verification") is not False
    ):
        raise ArtifactIntegrityError("reproduction bundle generation contract is invalid")
    if config.purpose == "confirmatory":
        git = manifest.get("git")
        if (
            not isinstance(git, Mapping)
            or git.get("available") is not True
            or git.get("dirty") is not False
            or not git.get("commit")
            or not environment.get("dependency_locks")
            or not any(
                command[:2] == ("uv", "sync") and "--frozen" in command
                for command in config.commands
            )
        ):
            raise ArtifactIntegrityError("confirmatory reproduction gate is not satisfied")
        sources = graph.get("sources")
        if not isinstance(sources, Sequence) or any(
            not isinstance(source, Mapping)
            or not isinstance(source.get("verification"), Mapping)
            or source["verification"].get("git", {}).get("dirty") is not False
            for source in sources
        ):
            raise ArtifactIntegrityError("confirmatory sources do not retain clean-git evidence")

    return VerifiedReproductionBundle(
        root=root,
        bundle_id=config.bundle_id,
        manifest=manifest,
        config=config,
        descriptor=descriptor,
        source_graph=graph,
        source_inventory=inventory,
        environment=environment,
        commands=commands,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only semantic verification of a standalone CARLA research reproduction bundle"
        )
    )
    parser.add_argument("paths", nargs="+", help="bundle directories or manifest.json files")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results: list[dict[str, Any]] = []
    for raw_path in args.paths:
        bundle = load_verified_reproduction_bundle(raw_path)
        results.append(
            {
                "bundle_id": bundle.bundle_id,
                "root": str(bundle.root),
                "purpose": bundle.config.purpose,
                "source_count": bundle.descriptor["source_count"],
                "source_file_count": bundle.descriptor["source_file_count"],
                "dependency_lock_count": bundle.descriptor["dependency_lock_count"],
                "status": "verified",
            }
        )
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    return 0


__all__ = [
    "REPRODUCTION_BUNDLE_SCHEMA_VERSION",
    "VerifiedReproductionBundle",
    "load_verified_reproduction_bundle",
    "main",
    "parse_args",
]
