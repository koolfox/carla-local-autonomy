"""Semantic verification for portable native-host collection kits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import fingerprint_file
from ..scenarios.splits import canonical_map_family
from ..scenarios.verified_plan import load_verified_scenario_plan
from ..verification import ArtifactIntegrityError, verify_research_object
from .host_kit import (
    NATIVE_HOST_KIT_SCHEMA_VERSION,
    NativeHostKitConfig,
    _capture_count,
    _render_bootstrap_ps1,
    _render_bootstrap_sh,
    _render_collect_ps1,
    _render_collect_sh,
    _render_postrun_ps1,
    _render_postrun_sh,
    _render_preflight_ps1,
    _render_preflight_sh,
    _render_readme,
    _render_verify_payload,
    _validate_hashed_requirements,
)
from .worker import select_episodes

_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_REQUIRED_ROLES = frozenset(
    {
        "native_host_kit_configuration",
        "native_host_kit_release_manifest",
        "native_host_kit_payload_archive",
        "native_host_kit_payload_inventory_json",
        "native_host_kit_payload_inventory_csv",
        "native_host_kit_readme",
        "native_host_kit_checksum_index",
    }
)


@dataclass(frozen=True)
class VerifiedNativeHostKit:
    root: Path
    kit_id: str
    manifest: Mapping[str, Any]
    config: NativeHostKitConfig
    descriptor: Mapping[str, Any]
    inventory: Mapping[str, Any]
    kit_plan: Mapping[str, Any]


def _load_json(path: Path, name: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(f"could not read {name}: {error}") from error


def _object(path: Path, name: str) -> Mapping[str, Any]:
    value = _load_json(path, name)
    if not isinstance(value, Mapping):
        raise ArtifactIntegrityError(f"{name} must contain a JSON object")
    return value


def _safe_file(root: Path, relative: str, *, context: str) -> Path:
    raw = Path(relative)
    if (
        not relative
        or raw.is_absolute()
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in raw.parts)
    ):
        raise ArtifactIntegrityError(f"{context} path is unsafe")
    current = root
    for part in raw.parts:
        current = current / part
        if current.is_symlink():
            raise ArtifactIntegrityError(f"{context} uses a symlink")
    try:
        resolved = (root / raw).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ArtifactIntegrityError(f"{context} is missing or escapes kit root") from error
    if not resolved.is_file():
        raise ArtifactIntegrityError(f"{context} is not a regular file")
    return resolved


def _role_paths(
    root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Path]:
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, Sequence):
        raise ArtifactIntegrityError("native host kit artifacts must be an array")
    roles: dict[str, Path] = {}
    for index, artifact in enumerate(raw_artifacts):
        if not isinstance(artifact, Mapping):
            raise ArtifactIntegrityError(f"native host kit artifact {index} is malformed")
        role = artifact.get("role")
        relative = artifact.get("path")
        if not isinstance(role, str) or not isinstance(relative, str):
            raise ArtifactIntegrityError(f"native host kit artifact {index} is incomplete")
        if role in _REQUIRED_ROLES:
            if role in roles:
                raise ArtifactIntegrityError(f"native host kit duplicates role {role!r}")
            roles[role] = _safe_file(root, relative, context=f"artifact role {role}")
    missing = sorted(_REQUIRED_ROLES - roles.keys())
    if missing:
        raise ArtifactIntegrityError(
            "native host kit is missing required roles: " + ", ".join(missing)
        )
    return roles


def _reference_matches(root: Path, raw: Any, *, context: str) -> Path:
    if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256", "size_bytes"}:
        raise ArtifactIntegrityError(f"{context} reference schema is invalid")
    relative = raw.get("path")
    digest = raw.get("sha256")
    size = raw.get("size_bytes")
    if (
        not isinstance(relative, str)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
    ):
        raise ArtifactIntegrityError(f"{context} reference is malformed")
    path = _safe_file(root, relative, context=context)
    actual = fingerprint_file(path)
    if actual["sha256"] != digest or actual["size_bytes"] != size:
        raise ArtifactIntegrityError(f"{context} fingerprint differs")
    return path


def _inventory_rows(
    inventory: Mapping[str, Any],
    inventory_csv_path: Path,
) -> list[dict[str, Any]]:
    if set(inventory) != {
        "schema_version",
        "object_type",
        "file_count",
        "total_bytes",
        "files",
    }:
        raise ArtifactIntegrityError("native host payload inventory fields differ from schema")
    if (
        inventory.get("schema_version") != NATIVE_HOST_KIT_SCHEMA_VERSION
        or inventory.get("object_type") != "native_host_kit_payload_inventory"
    ):
        raise ArtifactIntegrityError("native host payload inventory identity is invalid")
    raw_files = inventory.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ArtifactIntegrityError("native host payload inventory must contain files")
    rows: list[dict[str, Any]] = []
    prior: bytes | None = None
    for index, raw in enumerate(raw_files):
        if not isinstance(raw, Mapping) or set(raw) != {
            "path",
            "sha256",
            "size_bytes",
            "archive_mode",
        }:
            raise ArtifactIntegrityError(f"native host payload inventory row {index} is invalid")
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
            raise ArtifactIntegrityError(f"payload inventory row {index} path is unsafe")
        encoded = relative.encode("utf-8")
        if prior is not None and encoded <= prior:
            raise ArtifactIntegrityError("payload inventory paths are not uniquely sorted")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or mode not in {"0644", "0755"}
        ):
            raise ArtifactIntegrityError(
                f"payload inventory row {index} fingerprint/mode is invalid"
            )
        rows.append(dict(raw))
        prior = encoded
    if inventory.get("file_count") != len(rows):
        raise ArtifactIntegrityError("payload inventory file_count is inconsistent")
    if inventory.get("total_bytes") != sum(int(row["size_bytes"]) for row in rows):
        raise ArtifactIntegrityError("payload inventory total_bytes is inconsistent")

    try:
        with inventory_csv_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["path", "sha256", "size_bytes", "archive_mode"]:
                raise ArtifactIntegrityError("payload inventory CSV header is invalid")
            csv_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ArtifactIntegrityError(f"could not read payload inventory CSV: {error}") from error
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
        raise ArtifactIntegrityError("payload inventory CSV differs from JSON")
    return rows


def _verify_archive(
    archive_path: Path,
    rows: Sequence[Mapping[str, Any]],
    extraction_root: Path,
) -> dict[str, bytes]:
    expected = {str(row["path"]): row for row in rows}
    payloads: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if names != list(expected):
                raise ArtifactIntegrityError(
                    "native host payload ZIP order/content differs from inventory"
                )
            for info in infos:
                row = expected[info.filename]
                if (
                    info.is_dir()
                    or info.filename.endswith("/")
                    or info.date_time != _ZIP_TIMESTAMP
                    or info.compress_type != zipfile.ZIP_STORED
                ):
                    raise ArtifactIntegrityError(
                        f"native host payload ZIP metadata is invalid: {info.filename}"
                    )
                mode = stat.S_IMODE(info.external_attr >> 16)
                expected_mode = int(str(row["archive_mode"]), 8)
                if mode != expected_mode:
                    raise ArtifactIntegrityError(
                        f"native host payload mode differs: {info.filename}"
                    )
                payload = archive.read(info)
                if (
                    len(payload) != row["size_bytes"]
                    or hashlib.sha256(payload).hexdigest() != row["sha256"]
                ):
                    raise ArtifactIntegrityError(
                        f"native host payload fingerprint differs: {info.filename}"
                    )
                target = extraction_root / Path(info.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                os.chmod(target, expected_mode)
                payloads[info.filename] = payload
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        if isinstance(error, ArtifactIntegrityError):
            raise
        raise ArtifactIntegrityError(
            f"could not verify native host payload ZIP: {error}"
        ) from error
    return payloads


def _verify_internal_checksums(payloads: Mapping[str, bytes]) -> None:
    try:
        text = payloads["checksums.sha256"].decode("utf-8")
    except (KeyError, UnicodeError) as error:
        raise ArtifactIntegrityError("payload checksum index is missing or invalid") from error
    observed: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative == "checksums.sha256"
            or relative not in payloads
        ):
            raise ArtifactIntegrityError(
                f"invalid native host payload checksum entry at line {line_number}"
            )
        if hashlib.sha256(payloads[relative]).hexdigest() != digest:
            raise ArtifactIntegrityError(f"payload checksum differs: {relative}")
        observed.append(relative)
    expected = sorted(
        set(payloads) - {"checksums.sha256"},
        key=lambda value: value.encode("utf-8"),
    )
    if observed != expected:
        raise ArtifactIntegrityError(
            "payload checksum index does not cover every immutable payload file exactly once"
        )


def _expected_kit_plan(
    *,
    config: NativeHostKitConfig,
    plan: Any,
    episodes: Sequence[Any],
    plan_manifest: Path,
) -> dict[str, Any]:
    reference = fingerprint_file(plan_manifest)
    capture_count = sum(_capture_count(episode) for episode in episodes)
    return {
        "schema_version": NATIVE_HOST_KIT_SCHEMA_VERSION,
        "object_type": "native_host_kit_payload",
        "kit_id": config.kit_id,
        "title": config.title,
        "target": {
            "python_implementation": "CPython",
            "python_version": config.python_version,
            "systems": ["Windows", "Linux"],
            "machine": "x86_64",
            "carla_version": config.carla_version,
        },
        "endpoint": {"host": config.host, "port": config.port},
        "dataset": {
            "dataset_id": config.dataset_id,
            "output_directory": f"datasets/{config.dataset_id}",
        },
        "scenario_plan": {
            "directory": f"scenario-plan/{plan.run_id}",
            "run_id": plan.run_id,
            "sha256": reference["sha256"],
            "size_bytes": reference["size_bytes"],
        },
        "selection": {
            "episode_ids": [episode.episode_id for episode in episodes],
            "partitions": list(config.partitions),
            "selected_episode_count": len(episodes),
            "planned_capture_count": capture_count,
            "map_families": sorted(
                {canonical_map_family(episode.recipe.map_name) for episode in episodes}
            ),
        },
        "requirements": {
            "windows": "requirements/windows-py312.txt",
            "linux": "requirements/linux-py312.txt",
            "package_count": 14,
            "carla_wheel_hash_count": 6,
            "all_packages_hash_pinned": True,
        },
        "safety": {
            "build_contacted_simulator": False,
            "build_mutated_simulator": False,
            "preflight_is_read_only": True,
            "collection_requires_verified_ready_preflight": True,
            "collection_requires_literal_confirmation_token": True,
            "map_reload_destroys_existing_world_actors": True,
            "exclusive_world_tick_owner_required": True,
            "full_thesis_plan_included": False,
        },
    }


def _verify_exact_scripts(
    *,
    payloads: Mapping[str, bytes],
    config: NativeHostKitConfig,
    scenario_run_id: str,
    planned_capture_count: int,
) -> None:
    default_preflight_run_id = f"native-preflight-{config.kit_id}"
    expected = {
        "scripts/verify_payload.py": _render_verify_payload(),
        "scripts/bootstrap.sh": _render_bootstrap_sh(),
        "scripts/bootstrap.ps1": _render_bootstrap_ps1(),
        "scripts/preflight.sh": _render_preflight_sh(
            default_run_id=default_preflight_run_id,
            scenario_run_id=scenario_run_id,
            dataset_id=config.dataset_id,
            host=config.host,
            port=config.port,
        ),
        "scripts/preflight.ps1": _render_preflight_ps1(
            default_run_id=default_preflight_run_id,
            scenario_run_id=scenario_run_id,
            dataset_id=config.dataset_id,
            host=config.host,
            port=config.port,
        ),
        "scripts/collect.sh": _render_collect_sh(
            scenario_run_id=scenario_run_id,
            dataset_id=config.dataset_id,
            host=config.host,
            port=config.port,
        ),
        "scripts/collect.ps1": _render_collect_ps1(
            scenario_run_id=scenario_run_id,
            dataset_id=config.dataset_id,
            host=config.host,
            port=config.port,
        ),
        "scripts/postrun.sh": _render_postrun_sh(config.dataset_id),
        "scripts/postrun.ps1": _render_postrun_ps1(config.dataset_id),
        "README.md": _render_readme(
            kit_id=config.kit_id,
            scenario_run_id=scenario_run_id,
            dataset_id=config.dataset_id,
            host=config.host,
            port=config.port,
            capture_count=planned_capture_count,
        ),
    }
    for relative, text in expected.items():
        if payloads.get(relative) != text.encode("utf-8"):
            raise ArtifactIntegrityError(
                f"native host generated payload differs from contract: {relative}"
            )
    required_token = b"I_CONFIRM_WORLD_RELOAD_AND_EXCLUSIVE_TICK"
    if payloads["scripts/collect.sh"].count(required_token) < 2:
        raise ArtifactIntegrityError(
            "native Bash collection script lacks explicit confirmation guard"
        )
    if (
        payloads["scripts/collect.ps1"].count(required_token) < 1
        or b"[ValidateSet(" not in payloads["scripts/collect.ps1"]
    ):
        raise ArtifactIntegrityError(
            "native PowerShell collection script lacks explicit confirmation guard"
        )


def load_verified_native_host_kit(root: str | Path) -> VerifiedNativeHostKit:
    resolved = Path(root).expanduser().resolve(strict=True)
    manifest = _object(resolved / "manifest.json", "native host kit manifest")
    if manifest.get("run_id") is None or manifest.get("status") != "success":
        raise ArtifactIntegrityError("native host kit tracker manifest is not successful")
    roles = _role_paths(resolved, manifest)
    config_raw = _object(
        roles["native_host_kit_configuration"],
        "native host kit configuration",
    )
    try:
        config = NativeHostKitConfig.from_mapping(config_raw)
    except (TypeError, ValueError) as error:
        raise ArtifactIntegrityError(
            f"native host kit configuration is invalid: {error}"
        ) from error
    descriptor = _object(
        roles["native_host_kit_release_manifest"],
        "native host kit descriptor",
    )
    if (
        descriptor.get("schema_version") != NATIVE_HOST_KIT_SCHEMA_VERSION
        or descriptor.get("object_type") != "native_host_kit_release"
        or descriptor.get("status") != "complete"
        or descriptor.get("kit_id") != config.kit_id
        or manifest.get("run_id") != config.kit_id
    ):
        raise ArtifactIntegrityError("native host kit identity/status is inconsistent")

    archive_path = _reference_matches(
        resolved,
        descriptor.get("payload_archive"),
        context="payload archive",
    )
    inventory_json_path = _reference_matches(
        resolved,
        descriptor.get("payload_inventory_json"),
        context="payload inventory JSON",
    )
    inventory_csv_path = _reference_matches(
        resolved,
        descriptor.get("payload_inventory_csv"),
        context="payload inventory CSV",
    )
    _reference_matches(
        resolved,
        descriptor.get("configuration"),
        context="kit configuration",
    )
    readme_path = _reference_matches(
        resolved,
        descriptor.get("readme"),
        context="kit README",
    )
    inventory = _object(inventory_json_path, "native host payload inventory")
    rows = _inventory_rows(inventory, inventory_csv_path)

    with tempfile.TemporaryDirectory(prefix="carla-native-host-kit-verify-") as temporary:
        extraction_root = Path(temporary)
        payloads = _verify_archive(archive_path, rows, extraction_root)
        _verify_internal_checksums(payloads)
        try:
            kit_plan = json.loads(payloads["kit-plan.json"].decode("utf-8"))
        except (KeyError, UnicodeError, json.JSONDecodeError) as error:
            raise ArtifactIntegrityError("payload kit-plan.json is invalid") from error
        if not isinstance(kit_plan, Mapping):
            raise ArtifactIntegrityError("payload kit plan must contain an object")

        scenario_dir = (
            extraction_root
            / "scenario-plan"
            / str(
                kit_plan.get("scenario_plan", {}).get("run_id", "")
                if isinstance(kit_plan.get("scenario_plan"), Mapping)
                else ""
            )
        )
        scenario_verification = verify_research_object(
            scenario_dir,
            verify_references=False,
            deep=True,
            reject_unregistered=True,
        )
        plan = load_verified_scenario_plan(scenario_dir)
        episodes = select_episodes(
            plan,
            partitions=config.partitions,
            max_episodes=config.max_episodes,
        )
        expected_plan = _expected_kit_plan(
            config=config,
            plan=plan,
            episodes=episodes,
            plan_manifest=scenario_dir / "manifest.json",
        )
        if dict(kit_plan) != expected_plan:
            raise ArtifactIntegrityError(
                "payload kit plan does not reproduce from config and scenario plan"
            )

        windows_lock = _validate_hashed_requirements(
            extraction_root / "requirements" / "windows-py312.txt"
        )
        linux_lock = _validate_hashed_requirements(
            extraction_root / "requirements" / "linux-py312.txt"
        )
        if (
            windows_lock != linux_lock
            or payloads["requirements/windows-py312.txt"]
            != payloads["requirements/linux-py312.txt"]
        ):
            raise ArtifactIntegrityError("portable Windows/Linux dependency locks differ")

        for relative, payload in payloads.items():
            if not relative.startswith("source/") or not relative.endswith(".py"):
                continue
            try:
                compile(payload, relative, "exec")
            except (SyntaxError, ValueError) as error:
                raise ArtifactIntegrityError(
                    f"portable Python source does not compile: {relative}: {error}"
                ) from error

        capture_count = int(expected_plan["selection"]["planned_capture_count"])
        _verify_exact_scripts(
            payloads=payloads,
            config=config,
            scenario_run_id=plan.run_id,
            planned_capture_count=capture_count,
        )
        if readme_path.read_bytes() != payloads["README.md"]:
            raise ArtifactIntegrityError("outer and payload native host README differ")

    if descriptor.get("payload_file_count") != len(rows):
        raise ArtifactIntegrityError("native host kit payload_file_count is inconsistent")
    if descriptor.get("payload_total_bytes") != sum(int(row["size_bytes"]) for row in rows):
        raise ArtifactIntegrityError("native host kit payload_total_bytes is inconsistent")
    source_count = sum(str(row["path"]).startswith("source/") for row in rows)
    if descriptor.get("payload_source_file_count") != source_count:
        raise ArtifactIntegrityError("native host kit source-file count is inconsistent")
    for section in ("target", "endpoint", "dataset", "selection", "requirements", "safety"):
        if descriptor.get(section) != kit_plan.get(section):
            raise ArtifactIntegrityError(f"descriptor {section} differs from payload kit plan")
    generation = descriptor.get("generation")
    if not isinstance(generation, Mapping) or generation != {
        "archive_format": "zip-stored-fixed-metadata-v1",
        "archive_timestamp": "1980-01-01T00:00:00Z",
        "payload_symlinks_allowed": False,
        "simulator_contacted": False,
        "simulator_mutated": False,
        "secrets_captured": False,
    }:
        raise ArtifactIntegrityError("native host kit generation/safety metadata is invalid")
    scenario_descriptor = descriptor.get("scenario_plan")
    if not isinstance(scenario_descriptor, Mapping):
        raise ArtifactIntegrityError("native host kit scenario descriptor is invalid")
    source_verification = scenario_descriptor.get("source_verification")
    if not isinstance(source_verification, Mapping):
        raise ArtifactIntegrityError("native host source verification record is missing")
    if (
        source_verification.get("status") != "success"
        or source_verification.get("unregistered_file_count") != 0
        or source_verification.get("deep_verification")
        != dict(scenario_verification.deep_verification)
    ):
        raise ArtifactIntegrityError(
            "native host source scenario verification record is inconsistent"
        )

    return VerifiedNativeHostKit(
        root=resolved,
        kit_id=config.kit_id,
        manifest=manifest,
        config=config,
        descriptor=descriptor,
        inventory=inventory,
        kit_plan=kit_plan,
    )


def verify_native_host_kit(path: str | Path) -> dict[str, Any]:
    verified = load_verified_native_host_kit(path)
    return {
        "status": "verified",
        "kit_id": verified.kit_id,
        "root": str(verified.root),
        "dataset_id": str(verified.descriptor["dataset"]["dataset_id"]),
        "planned_capture_count": int(verified.descriptor["selection"]["planned_capture_count"]),
        "payload_file_count": int(verified.descriptor["payload_file_count"]),
        "payload_source_file_count": int(verified.descriptor["payload_source_file_count"]),
        "requirements_package_count": int(verified.descriptor["requirements"]["package_count"]),
        "simulator_mutated": bool(verified.descriptor["generation"]["simulator_mutated"]),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a portable native CARLA host kit, its deterministic ZIP, "
            "dependency hashes, scenario plan, scripts, and safety gates"
        )
    )
    parser.add_argument("path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        json.dumps(
            verify_native_host_kit(args.path),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


__all__ = [
    "VerifiedNativeHostKit",
    "load_verified_native_host_kit",
    "main",
    "parse_args",
    "verify_native_host_kit",
]


if __name__ == "__main__":
    raise SystemExit(main())
