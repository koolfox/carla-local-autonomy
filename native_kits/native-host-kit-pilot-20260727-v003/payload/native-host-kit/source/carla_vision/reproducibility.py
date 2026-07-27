"""Canonical experiment identity and privacy-safe reproducibility metadata.

The functions in this module deliberately avoid reading environment variables,
host names, user names, network interfaces, or repository remotes.  Only an
explicit allow-list of hardware and tool metadata is retained.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import platform
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPRODUCIBILITY_SCHEMA_VERSION = "1.0"
CANONICAL_JSON_ALGORITHM = "json-sort-keys-utf8-v1"
CONFIG_IDENTITY_EXCLUDED_TOP_LEVEL_KEYS = frozenset(
    {
        "created_at",
        "output_dir",
        "output_path",
        "run_id",
        "runs_root",
        "started_at",
    }
)
CONTROLLED_EXPERIMENT_STAGES = frozenset(
    {
        "analysis",
        "collect",
        "evaluation",
        "failure-mining",
        "live",
        "package",
        "replay",
        "report",
        "review",
        "test",
        "threshold",
        "train",
        "validation",
    }
)
DEFAULT_LOCKFILE_NAMES = (
    "uv.lock",
    "poetry.lock",
    "Pipfile.lock",
    "requirements.txt",
)

HardwareProbe = Callable[[], Mapping[str, Any]]

_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def _normalize_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON cannot contain NaN or infinity")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("canonical datetimes must be timezone-aware")
        return (
            value.astimezone(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace(
                "+00:00",
                "Z",
            )
        )
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if key in normalized:
                raise ValueError(f"canonical JSON has a duplicate string key: {key!r}")
            normalized[key] = _normalize_json_value(raw_value)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [_normalize_json_value(item) for item in value]
    raise TypeError(
        f"unsupported canonical JSON value {type(value).__module__}.{type(value).__qualname__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible data with the framework's frozen algorithm."""

    normalized = _normalize_json_value(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_record(payload: bytes) -> dict[str, Any]:
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def configuration_identity(
    config: Mapping[str, Any],
    *,
    excluded_top_level_keys: Sequence[str] = tuple(sorted(CONFIG_IDENTITY_EXCLUDED_TOP_LEVEL_KEYS)),
) -> dict[str, Any]:
    """Return exact and output-location-independent configuration identities."""

    normalized = _normalize_json_value(config)
    if not isinstance(normalized, dict):
        raise TypeError("configuration must normalize to an object")
    excluded = tuple(sorted({str(key) for key in excluded_top_level_keys}))
    identity_payload = {key: value for key, value in normalized.items() if key not in excluded}
    resolved = _sha256_record(canonical_json_bytes(normalized))
    identity = _sha256_record(canonical_json_bytes(identity_payload))
    return {
        "canonicalization": CANONICAL_JSON_ALGORITHM,
        "resolved_sha256": resolved["sha256"],
        "resolved_size_bytes": resolved["size_bytes"],
        "identity_sha256": identity["sha256"],
        "identity_size_bytes": identity["size_bytes"],
        "identity_excluded_top_level_keys": list(excluded),
        "sensitive_values_redacted_before_hashing": True,
    }


def _slug(value: str, *, field: str, maximum_length: int) -> str:
    slug = _SLUG_PATTERN.sub("-", value.strip().casefold()).strip("-")
    if not slug:
        raise ValueError(f"{field} must contain at least one letter or digit")
    return slug[:maximum_length].rstrip("-")


def canonical_experiment_id(
    *,
    started_at: datetime,
    stage: str,
    model: str,
    config: Mapping[str, Any],
    seed: int,
) -> str:
    """Build ``exp-<UTC>-<stage>-<model>-<cfg8>-s<seed>``."""

    if started_at.tzinfo is None or started_at.utcoffset() is None:
        raise ValueError("started_at must be timezone-aware")
    normalized_stage = _slug(stage, field="stage", maximum_length=24)
    if normalized_stage not in CONTROLLED_EXPERIMENT_STAGES:
        allowed = ", ".join(sorted(CONTROLLED_EXPERIMENT_STAGES))
        raise ValueError(f"unsupported experiment stage {stage!r}; expected one of {allowed}")
    normalized_model = _slug(model, field="model", maximum_length=40)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2^31)")
    timestamp = started_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = configuration_identity(config)["identity_sha256"]
    return f"exp-{timestamp}-{normalized_stage}-{normalized_model}-{digest[:8]}-s{seed}"


def _fingerprint(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"dependency lockfile changed while hashing: {path}")
    return {"sha256": digest.hexdigest(), "size_bytes": after.st_size}


def dependency_lock_snapshot(
    repository_root: str | Path,
    *,
    filenames: Sequence[str] = DEFAULT_LOCKFILE_NAMES,
) -> list[dict[str, Any]]:
    """Fingerprint known dependency locks without retaining host-specific paths."""

    root = Path(repository_root).expanduser().resolve()
    records: list[dict[str, Any]] = []
    for filename in sorted({str(value) for value in filenames}):
        candidate = root / filename
        if not candidate.is_file() or candidate.is_symlink():
            continue
        record = _fingerprint(candidate)
        records.append(
            {
                "relative_path": filename,
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
            }
        )
    return records


def _run_text(arguments: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            tuple(arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    line = completed.stdout.strip().splitlines()
    return line[0][:300] if line else None


def _total_memory_bytes() -> int | None:
    if platform.system() == "Darwin":
        raw = _run_text(("sysctl", "-n", "hw.memsize"))
        if raw is not None and raw.isdigit():
            return int(raw)
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
        return pages * page_size
    return None


def _cpu_model() -> str | None:
    for command in (
        ("sysctl", "-n", "machdep.cpu.brand_string"),
        ("sysctl", "-n", "hw.model"),
    ):
        value = _run_text(command)
        if value:
            return value
    value = platform.processor().strip()
    return value or None


def _torch_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": False,
        "version": None,
        "cuda": {
            "available": False,
            "runtime_version": None,
            "device_count": 0,
            "devices": [],
        },
        "cudnn": {"available": False, "version": None},
        "mps": {"available": False, "built": False},
        "rocm_version": None,
        "determinism": {
            "deterministic_algorithms": None,
            "cudnn_benchmark": None,
            "cudnn_deterministic": None,
        },
    }
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError, RuntimeError):
        return result

    result["available"] = True
    result["version"] = str(getattr(torch, "__version__", "")) or None
    version = getattr(torch, "version", None)
    cuda_runtime = getattr(version, "cuda", None)
    result["cuda"]["runtime_version"] = str(cuda_runtime) if cuda_runtime is not None else None
    hip_runtime = getattr(version, "hip", None)
    result["rocm_version"] = str(hip_runtime) if hip_runtime is not None else None

    cuda = getattr(torch, "cuda", None)
    try:
        cuda_available = bool(cuda is not None and cuda.is_available())
    except (AttributeError, RuntimeError):
        cuda_available = False
    result["cuda"]["available"] = cuda_available
    if cuda_available:
        try:
            count = int(cuda.device_count())
            result["cuda"]["device_count"] = count
            result["cuda"]["devices"] = [
                {"index": index, "name": str(cuda.get_device_name(index))} for index in range(count)
            ]
        except (AttributeError, RuntimeError):
            result["cuda"]["device_count"] = 0
            result["cuda"]["devices"] = []

    backends = getattr(torch, "backends", None)
    cudnn = getattr(backends, "cudnn", None)
    if cudnn is not None:
        try:
            cudnn_available = bool(cudnn.is_available())
        except (AttributeError, RuntimeError):
            cudnn_available = False
        result["cudnn"] = {
            "available": cudnn_available,
            "version": int(cudnn.version()) if cudnn_available and cudnn.version() else None,
        }
        result["determinism"]["cudnn_benchmark"] = bool(getattr(cudnn, "benchmark", False))
        result["determinism"]["cudnn_deterministic"] = bool(getattr(cudnn, "deterministic", False))

    mps = getattr(backends, "mps", None)
    if mps is not None:
        try:
            result["mps"] = {
                "available": bool(mps.is_available()),
                "built": bool(mps.is_built()),
            }
        except (AttributeError, RuntimeError):
            pass
    deterministic = getattr(torch, "are_deterministic_algorithms_enabled", None)
    if callable(deterministic):
        result["determinism"]["deterministic_algorithms"] = bool(deterministic())
    return result


def default_hardware_probe() -> Mapping[str, Any]:
    """Collect an allow-listed hardware/backend snapshot."""

    return {
        "cpu": {
            "architecture": platform.machine() or None,
            "model": _cpu_model(),
            "logical_count": os.cpu_count(),
        },
        "memory": {"total_bytes": _total_memory_bytes()},
        "accelerators": _torch_snapshot(),
        "tools": {
            "python_compiler": platform.python_compiler() or None,
            "git": _run_text(("git", "--version")),
            "uv": _run_text(("uv", "--version")),
        },
        "container": {
            "detected": Path("/.dockerenv").exists(),
            "image_digest": None,
        },
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def safe_hardware_snapshot(probe: HardwareProbe = default_hardware_probe) -> dict[str, Any]:
    """Apply a strict field allow-list to a hardware probe result."""

    raw = probe()
    cpu = _mapping(raw.get("cpu"))
    memory = _mapping(raw.get("memory"))
    accelerators = _mapping(raw.get("accelerators"))
    cuda = _mapping(accelerators.get("cuda"))
    cudnn = _mapping(accelerators.get("cudnn"))
    mps = _mapping(accelerators.get("mps"))
    determinism = _mapping(accelerators.get("determinism"))
    tools = _mapping(raw.get("tools"))
    container = _mapping(raw.get("container"))

    devices: list[dict[str, Any]] = []
    raw_devices = cuda.get("devices")
    if isinstance(raw_devices, Sequence) and not isinstance(
        raw_devices,
        (str, bytes, bytearray),
    ):
        for item in raw_devices:
            if isinstance(item, Mapping):
                devices.append({"index": item.get("index"), "name": item.get("name")})

    return {
        "cpu": {
            "architecture": cpu.get("architecture"),
            "model": cpu.get("model"),
            "logical_count": cpu.get("logical_count"),
        },
        "memory": {"total_bytes": memory.get("total_bytes")},
        "accelerators": {
            "available": accelerators.get("available"),
            "version": accelerators.get("version"),
            "cuda": {
                "available": cuda.get("available"),
                "runtime_version": cuda.get("runtime_version"),
                "device_count": cuda.get("device_count"),
                "devices": devices,
            },
            "cudnn": {
                "available": cudnn.get("available"),
                "version": cudnn.get("version"),
            },
            "mps": {
                "available": mps.get("available"),
                "built": mps.get("built"),
            },
            "rocm_version": accelerators.get("rocm_version"),
            "determinism": {
                "deterministic_algorithms": determinism.get("deterministic_algorithms"),
                "cudnn_benchmark": determinism.get("cudnn_benchmark"),
                "cudnn_deterministic": determinism.get("cudnn_deterministic"),
            },
        },
        "tools": {
            "python_compiler": tools.get("python_compiler"),
            "git": tools.get("git"),
            "uv": tools.get("uv"),
        },
        "container": {
            "detected": container.get("detected"),
            "image_digest": container.get("image_digest"),
        },
    }


def build_reproducibility_snapshot(
    *,
    config: Mapping[str, Any],
    repository_root: str | Path,
    hardware_probe: HardwareProbe = default_hardware_probe,
) -> dict[str, Any]:
    """Build the reproducibility section embedded in new tracker manifests."""

    return {
        "schema_version": REPRODUCIBILITY_SCHEMA_VERSION,
        "configuration": configuration_identity(config),
        "dependency_locks": dependency_lock_snapshot(repository_root),
        "hardware": safe_hardware_snapshot(hardware_probe),
    }


__all__ = [
    "CANONICAL_JSON_ALGORITHM",
    "CONFIG_IDENTITY_EXCLUDED_TOP_LEVEL_KEYS",
    "CONTROLLED_EXPERIMENT_STAGES",
    "DEFAULT_LOCKFILE_NAMES",
    "HardwareProbe",
    "REPRODUCIBILITY_SCHEMA_VERSION",
    "build_reproducibility_snapshot",
    "canonical_experiment_id",
    "canonical_json_bytes",
    "configuration_identity",
    "default_hardware_probe",
    "dependency_lock_snapshot",
    "safe_hardware_snapshot",
]
