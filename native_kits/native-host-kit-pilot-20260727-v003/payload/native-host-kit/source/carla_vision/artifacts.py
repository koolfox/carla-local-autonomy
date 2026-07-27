"""Reproducible run manifests and artifact integrity tracking.

The tracker deliberately records a small, explicit environment allow-list.  It
never serializes ``os.environ`` and redacts common secret-bearing configuration
and command-line keys before writing the manifest.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import mimetypes
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

from carla_vision.reproducibility import (
    HardwareProbe,
    build_reproducibility_snapshot,
    canonical_experiment_id,
    default_hardware_probe,
)

SCHEMA_VERSION = "1.0"
DEFAULT_PACKAGE_NAMES = (
    "carla",
    "numpy",
    "opencv-python",
    "torch",
    "torchvision",
    "ultralytics",
)

Clock = Callable[[], datetime]
IdFactory = Callable[[], str]
GitProbe = Callable[[Path], Mapping[str, Any]]
EnvironmentProbe = Callable[[Sequence[str]], Mapping[str, Any]]

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_URL_CREDENTIALS_PATTERN = re.compile(r"(?i)([a-z][a-z0-9+.-]*://)([^/@\s]+)@")
_FREE_TEXT_SECRET_PATTERN = re.compile(
    r"(?i)\b(password|passwd|token|secret|api[_-]?key|access[_-]?key)"
    r"(\s*[=:]\s*)([^\s,;]+)"
)
_BEARER_PATTERN = re.compile(r"(?i)\b(bearer)(\s+)([^\s,;]+)")
_SENSITIVE_NORMALIZED_KEYS = frozenset(
    {
        "accesstoken",
        "accesskey",
        "apikey",
        "authorization",
        "clientsecret",
        "credential",
        "credentials",
        "password",
        "passwd",
        "privatekey",
        "refreshtoken",
        "secret",
        "token",
    }
)


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""

    return datetime.now(timezone.utc)


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    normalized = value.astimezone(timezone.utc)
    return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _is_sensitive_key(value: str) -> bool:
    normalized = _normalized_key(value)
    if normalized in _SENSITIVE_NORMALIZED_KEYS:
        return True
    return normalized.endswith(
        (
            "accesskey",
            "accesstoken",
            "apikey",
            "authorization",
            "clientsecret",
            "credentials",
            "password",
            "passwd",
            "privatekey",
            "refreshtoken",
            "secret",
            "token",
        )
    )


def _redact_free_text(value: str) -> str:
    redacted = _URL_CREDENTIALS_PATTERN.sub(r"\1[REDACTED]@", value)
    redacted = _FREE_TEXT_SECRET_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
        redacted,
    )
    return _BEARER_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
        redacted,
    )


def _sanitize_for_manifest(value: Any) -> Any:
    """Convert a value to a JSON-safe form while redacting named secrets."""

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("manifest values cannot contain NaN or infinity")
        return value
    if isinstance(value, str):
        return _redact_free_text(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime):
        return _iso_utc(value)
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            sanitized[key] = (
                "[REDACTED]" if _is_sensitive_key(key) else _sanitize_for_manifest(raw_value)
            )
        return sanitized
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_sanitize_for_manifest(item) for item in value]
    raise TypeError(
        f"unsupported manifest value {type(value).__module__}.{type(value).__qualname__}"
    )


def _sanitize_cli_args(cli_args: Sequence[str] | Mapping[str, Any]) -> Any:
    if isinstance(cli_args, Mapping):
        return _sanitize_for_manifest(cli_args)

    sanitized: list[str] = []
    redact_next = False
    for raw_argument in cli_args:
        argument = str(raw_argument)
        if redact_next:
            sanitized.append("[REDACTED]")
            redact_next = False
            continue

        option, separator, value = argument.partition("=")
        option_key = option.lstrip("-")
        if option.startswith("-") and _is_sensitive_key(option_key):
            if separator:
                sanitized.append(f"{option}=[REDACTED]")
            else:
                sanitized.append(option)
                redact_next = True
            continue
        sanitized.append(_redact_free_text(argument))
    return sanitized


def _package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _default_environment_probe(package_names: Sequence[str]) -> Mapping[str, Any]:
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {
            package_name: _package_version(package_name) for package_name in package_names
        },
    }


def _safe_environment_snapshot(
    package_names: Sequence[str],
    probe: EnvironmentProbe,
) -> dict[str, Any]:
    """Apply a strict allow-list even when a custom probe is injected."""

    raw = probe(package_names)
    raw_python = raw.get("python", {})
    raw_platform = raw.get("platform", {})
    raw_packages = raw.get("packages", {})

    python_data = raw_python if isinstance(raw_python, Mapping) else {}
    platform_data = raw_platform if isinstance(raw_platform, Mapping) else {}
    package_data = raw_packages if isinstance(raw_packages, Mapping) else {}

    snapshot = {
        "python": {
            "version": python_data.get("version"),
            "implementation": python_data.get("implementation"),
        },
        "platform": {
            "system": platform_data.get("system"),
            "release": platform_data.get("release"),
            "machine": platform_data.get("machine"),
        },
        "packages": {name: package_data.get(name) for name in package_names},
    }
    return _sanitize_for_manifest(snapshot)


def _run_git(repository_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", "-C", str(repository_root), *arguments),
        check=False,
        capture_output=True,
        text=True,
        timeout=3.0,
    )


def _default_git_probe(repository_root: Path) -> Mapping[str, Any]:
    try:
        repository = _run_git(repository_root, "rev-parse", "--show-toplevel")
        if repository.returncode != 0:
            return {"available": False, "commit": None, "dirty": None}

        head = _run_git(repository_root, "rev-parse", "HEAD")
        status = _run_git(
            repository_root,
            "status",
            "--porcelain",
            "--untracked-files=normal",
        )
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "commit": None, "dirty": None}

    commit = head.stdout.strip() if head.returncode == 0 else None
    dirty = bool(status.stdout) if status.returncode == 0 else None
    return {"available": True, "commit": commit, "dirty": dirty}


def _safe_git_snapshot(repository_root: Path, probe: GitProbe) -> dict[str, Any]:
    raw = probe(repository_root)
    available = bool(raw.get("available", False))
    commit = raw.get("commit")
    dirty = raw.get("dirty")
    return {
        "available": available,
        "commit": str(commit) if commit is not None else None,
        "dirty": bool(dirty) if dirty is not None else None,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _fingerprint(path: Path) -> tuple[str, int]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    before_signature = (before.st_size, before.st_mtime_ns)
    after_signature = (after.st_size, after.st_mtime_ns)
    if before_signature != after_signature:
        raise RuntimeError(f"artifact changed while hashing: {path}")
    return digest.hexdigest(), after.st_size


def fingerprint_file(path: str | Path) -> dict[str, Any]:
    """Return a stable external-file reference suitable for a run manifest."""

    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"reference is not a regular file: {resolved}")
    digest, size_bytes = _fingerprint(resolved)
    return {
        "path": str(resolved),
        "sha256": digest,
        "size_bytes": size_bytes,
    }


def _as_reference_list(
    references: Sequence[Mapping[str, Any] | str] | Mapping[str, Any] | str | None,
) -> list[Any]:
    if references is None:
        return []
    if isinstance(references, (str, Mapping)):
        values: Sequence[Mapping[str, Any] | str] = (references,)
    else:
        values = references
    return [_sanitize_for_manifest(value) for value in values]


class RunArtifactTracker:
    """Own a reproducible ``runs/<run_id>`` directory and its manifest.

    All registered artifacts must live inside :attr:`run_dir`.  Large or shared
    datasets and model checkpoints can instead be described through the model
    and dataset reference fields.
    """

    def __init__(
        self,
        runs_root: str | Path = "runs",
        *,
        run_id: str | None = None,
        cli_args: Sequence[str] | Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
        repository_root: str | Path | None = None,
        carla_endpoint: str | Mapping[str, Any] | None = None,
        carla_version: str | None = None,
        carla_map: str | None = None,
        model_refs: (Sequence[Mapping[str, Any] | str] | Mapping[str, Any] | str | None) = None,
        dataset_refs: (Sequence[Mapping[str, Any] | str] | Mapping[str, Any] | str | None) = None,
        input_refs: (Sequence[Mapping[str, Any] | str] | Mapping[str, Any] | str | None) = None,
        package_names: Sequence[str] = DEFAULT_PACKAGE_NAMES,
        clock: Clock = utc_now,
        id_factory: IdFactory | None = None,
        git_probe: GitProbe = _default_git_probe,
        environment_probe: EnvironmentProbe = _default_environment_probe,
        hardware_probe: HardwareProbe = default_hardware_probe,
        experiment_stage: str | None = None,
        experiment_model: str | None = None,
        master_seed: int | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._clock = clock
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self.runs_root = Path(runs_root).expanduser().resolve()
        self.repository_root = Path(repository_root or Path.cwd()).expanduser().resolve()

        created_at_value = self._clock()
        created_at = _iso_utc(created_at_value)
        sanitized_config = _sanitize_for_manifest(config or {})
        if not isinstance(sanitized_config, Mapping):
            raise TypeError("config must sanitize to an object")
        identity_fields = (experiment_stage, experiment_model, master_seed)
        has_identity = any(value is not None for value in identity_fields)
        if has_identity and not all(value is not None for value in identity_fields):
            raise ValueError(
                "experiment_stage, experiment_model, and master_seed must be provided together"
            )
        canonical_id: str | None = None
        if has_identity:
            canonical_id = canonical_experiment_id(
                started_at=created_at_value,
                stage=str(experiment_stage),
                model=str(experiment_model),
                config=sanitized_config,
                seed=int(master_seed),
            )
            if run_id is not None and run_id != canonical_id:
                raise ValueError(
                    f"explicit run_id {run_id!r} does not match canonical ID {canonical_id!r}"
                )
            run_id = canonical_id

        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.run_id, self.run_dir = self._create_run_directory(
            requested_run_id=run_id,
            created_at=created_at_value,
        )
        self.manifest_path = self.run_dir / "manifest.json"

        selected_packages = tuple(sorted({str(name) for name in package_names}))
        invocation_args = tuple(sys.argv[1:]) if cli_args is None else cli_args
        self._manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": "created",
            "timestamps": {
                "created_at": created_at,
                "started_at": None,
                "finished_at": None,
                "updated_at": created_at,
            },
            "git": _safe_git_snapshot(self.repository_root, git_probe),
            "invocation": {
                "cli_args": _sanitize_cli_args(invocation_args),
                "config": sanitized_config,
            },
            "experiment": (
                {
                    "canonical_id": canonical_id,
                    "stage": str(experiment_stage),
                    "model": str(experiment_model),
                    "master_seed": int(master_seed),
                }
                if canonical_id is not None
                else None
            ),
            "reproducibility": _sanitize_for_manifest(
                build_reproducibility_snapshot(
                    config=sanitized_config,
                    repository_root=self.repository_root,
                    hardware_probe=hardware_probe,
                )
            ),
            "environment": _safe_environment_snapshot(
                selected_packages,
                environment_probe,
            ),
            "carla": {
                "endpoint": _sanitize_for_manifest(carla_endpoint),
                "version": _sanitize_for_manifest(carla_version),
                "map": _sanitize_for_manifest(carla_map),
            },
            "references": {
                "models": _as_reference_list(model_refs),
                "datasets": _as_reference_list(dataset_refs),
                "inputs": _as_reference_list(input_refs),
            },
            "artifacts": [],
            "failure": None,
        }
        try:
            self._write_manifest()
        except BaseException:
            try:
                self.run_dir.rmdir()
            except OSError:
                pass
            raise

    def _create_run_directory(
        self,
        *,
        requested_run_id: str | None,
        created_at: datetime,
    ) -> tuple[str, Path]:
        if requested_run_id is not None:
            validated = self._validate_run_id(requested_run_id)
            directory = self.runs_root / validated
            directory.mkdir(mode=0o755, exist_ok=False)
            return validated, directory

        timestamp = created_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        for _ in range(32):
            token = re.sub(r"[^A-Za-z0-9]", "", str(self._id_factory()))[:16]
            if not token:
                raise ValueError("id_factory must return at least one alphanumeric character")
            candidate = self._validate_run_id(f"{timestamp}-{token}")
            directory = self.runs_root / candidate
            try:
                directory.mkdir(mode=0o755, exist_ok=False)
            except FileExistsError:
                continue
            return candidate, directory
        raise RuntimeError("could not allocate a unique run_id after 32 attempts")

    @staticmethod
    def _validate_run_id(run_id: str) -> str:
        if run_id in {".", ".."} or not _RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(
                "run_id must be 1-128 characters using letters, digits, '.', '_' or '-'"
            )
        return run_id

    @property
    def manifest(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._manifest)

    def read_manifest(self) -> dict[str, Any]:
        with self.manifest_path.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    def __enter__(self) -> "RunArtifactTracker":
        with self._lock:
            if self._manifest["status"] != "created":
                raise RuntimeError("run context can only be entered once")
            now = _iso_utc(self._clock())
            self._manifest["status"] = "running"
            self._manifest["timestamps"]["started_at"] = now
            self._manifest["timestamps"]["updated_at"] = now
            self._write_manifest()
            return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del traceback
        with self._lock:
            if self._manifest["status"] != "running":
                raise RuntimeError("run context is not active")
            now = _iso_utc(self._clock())
            self._manifest["timestamps"]["finished_at"] = now
            self._manifest["timestamps"]["updated_at"] = now
            if exception_type is None:
                self._manifest["status"] = "success"
                self._manifest["failure"] = None
            else:
                self._manifest["status"] = "failed"
                self._manifest["failure"] = {
                    "type": exception_type.__qualname__,
                    "module": exception_type.__module__,
                    "message": _redact_free_text(str(exception or ""))[:2000],
                }
            self._write_manifest()
        return False

    def artifact_path(
        self,
        relative_path: str | Path,
        *,
        create_parent: bool = True,
    ) -> Path:
        raw_path = Path(relative_path)
        if raw_path.is_absolute():
            raise ValueError("artifact_path requires a path relative to the run directory")
        candidate = (self.run_dir / raw_path).resolve(strict=False)
        self._relative_artifact_path(candidate)
        if create_parent:
            candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def register_artifact(
        self,
        path: str | Path,
        *,
        role: str,
        mime_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Hash and register a completed artifact inside this run directory."""

        normalized_role = role.strip()
        if not normalized_role:
            raise ValueError("artifact role cannot be empty")

        artifact_path = Path(path).expanduser().resolve(strict=True)
        relative_path = self._relative_artifact_path(artifact_path)
        if artifact_path == self.manifest_path.resolve():
            raise ValueError("manifest.json cannot register itself as an artifact")
        if not artifact_path.is_file():
            raise ValueError(f"artifact is not a regular file: {artifact_path}")

        digest, size_bytes = _fingerprint(artifact_path)
        detected_mime = mime_type or mimetypes.guess_type(artifact_path.name)[0]
        now = _iso_utc(self._clock())
        entry = {
            "path": relative_path.as_posix(),
            "role": normalized_role,
            "sha256": digest,
            "size_bytes": size_bytes,
            "mime_type": detected_mime or "application/octet-stream",
            "registered_at": now,
            "metadata": _sanitize_for_manifest(metadata or {}),
        }

        with self._lock:
            self._require_active()
            artifacts = self._manifest["artifacts"]
            artifacts[:] = [existing for existing in artifacts if existing["path"] != entry["path"]]
            artifacts.append(entry)
            artifacts.sort(key=lambda item: item["path"])
            self._manifest["timestamps"]["updated_at"] = now
            self._write_manifest()
        return copy.deepcopy(entry)

    def add_model_reference(self, reference: Mapping[str, Any] | str) -> None:
        self._add_reference("models", reference)

    def add_dataset_reference(self, reference: Mapping[str, Any] | str) -> None:
        self._add_reference("datasets", reference)

    def add_input_reference(self, reference: Mapping[str, Any] | str) -> None:
        self._add_reference("inputs", reference)

    def _add_reference(self, kind: str, reference: Mapping[str, Any] | str) -> None:
        with self._lock:
            self._require_active()
            now = _iso_utc(self._clock())
            self._manifest["references"][kind].append(_sanitize_for_manifest(reference))
            self._manifest["timestamps"]["updated_at"] = now
            self._write_manifest()

    def _relative_artifact_path(self, path: Path) -> Path:
        try:
            return path.relative_to(self.run_dir.resolve())
        except ValueError as error:
            raise ValueError(
                f"artifact must be located inside run directory {self.run_dir}"
            ) from error

    def _require_active(self) -> None:
        if self._manifest["status"] != "running":
            raise RuntimeError("artifacts and references can only be added to an active run")

    def _write_manifest(self) -> None:
        _atomic_write_json(self.manifest_path, self._manifest)


__all__ = [
    "DEFAULT_PACKAGE_NAMES",
    "RunArtifactTracker",
    "SCHEMA_VERSION",
    "fingerprint_file",
    "utc_now",
]
