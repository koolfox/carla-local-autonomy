"""Read-only project catalogue for operator form choices."""

from __future__ import annotations

import importlib.util
import json
import socket
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..evidence.contracts import CANONICAL_EVIDENCE_ROOTS

RESEARCH_ROOTS = CANONICAL_EVIDENCE_ROOTS


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _files(root: Path, pattern: str) -> list[str]:
    return sorted(
        (_relative(root, path) for path in root.glob(pattern) if path.is_file()),
        key=lambda value: value.encode("utf-8"),
    )


def _object_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for top_level in RESEARCH_ROOTS:
        parent = root / top_level
        if parent.is_symlink() or not parent.is_dir():
            continue
        for directory in parent.iterdir():
            manifest_path = directory / "manifest.json"
            if (
                not directory.is_dir()
                or directory.is_symlink()
                or not manifest_path.is_file()
                or manifest_path.is_symlink()
            ):
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(manifest, dict):
                continue
            artifact_entries = [
                entry
                for entry in manifest.get("artifacts", [])
                if isinstance(entry, Mapping)
                and isinstance(entry.get("path"), str)
                and isinstance(entry.get("role"), str)
            ]
            roles = sorted(
                {str(entry["role"]) for entry in artifact_entries if str(entry["role"]).strip()}
            )
            timestamps = manifest.get("timestamps")
            if not isinstance(timestamps, Mapping):
                timestamps = {}
            invocation = manifest.get("invocation")
            if not isinstance(invocation, Mapping):
                invocation = {}
            config = invocation.get("config")
            if not isinstance(config, Mapping):
                config = {}
            object_type = manifest.get("object_type") or config.get("object_type")
            declared_bytes = sum(
                int(entry["size_bytes"])
                for entry in artifact_entries
                if isinstance(entry.get("size_bytes"), int)
                and not isinstance(entry.get("size_bytes"), bool)
                and int(entry["size_bytes"]) >= 0
            )
            rows.append(
                {
                    "id": str(manifest.get("run_id", directory.name)),
                    "path": _relative(root, directory),
                    "root_kind": top_level,
                    "status": str(manifest.get("status", "unknown")),
                    "object_type": (
                        str(object_type)
                        if isinstance(object_type, str) and object_type.strip()
                        else "unknown"
                    ),
                    "created_at": timestamps.get("created_at"),
                    "finished_at": timestamps.get("finished_at"),
                    "artifact_count": len(artifact_entries),
                    "declared_artifact_bytes": declared_bytes,
                    "roles": roles,
                    "manifest_source": "declared",
                    "verification_status": "not_checked",
                }
            )
    return sorted(rows, key=lambda row: str(row["path"]).encode("utf-8"))


def _has_role(row: dict[str, Any], role: str) -> bool:
    return role in row["roles"]


def probe_endpoint(host: str, port: int, *, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def build_catalog(
    workspace: str | Path,
    *,
    carla_host: str,
    carla_port: int,
) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve(strict=True)
    objects = _object_rows(root)
    roots = Counter(str(row["root_kind"]) for row in objects)
    statuses = Counter(str(row["status"]) for row in objects)
    weights = _files(root, "*.pt")
    model_packages = [row["path"] for row in objects if row["root_kind"] == "models"]
    datasets = [row["path"] for row in objects if row["root_kind"] == "datasets"]
    scenario_plans = [row["path"] for row in objects if _has_role(row, "scenario_plan_summary")]
    runtime_runs = [row["path"] for row in objects if _has_role(row, "detections_jsonl")]
    native_host_kits = [
        row["path"] for row in objects if _has_role(row, "native_host_kit_release_manifest")
    ]
    return {
        "workspace": str(root),
        "defaults": {
            "carla_host": carla_host,
            "carla_port": carla_port,
            "vehicle_id": 24,
            "camera_id": 25,
            "map": "Town10HD_Opt",
        },
        "capabilities": {
            "carla_tcp_reachable": probe_endpoint(carla_host, carla_port),
            "native_pythonapi_importable": importlib.util.find_spec("carla") is not None,
            "local_only": True,
            "vision_control_enabled": False,
        },
        "weights": weights,
        "model_packages": model_packages,
        "datasets": datasets,
        "scenario_plans": scenario_plans,
        "runtime_runs": runtime_runs,
        "native_host_kits": native_host_kits,
        "training_configs": _files(root, "configs/training/*.json"),
        "replay_configs": _files(root, "configs/replay/*.json"),
        "evaluation_configs": _files(root, "configs/evaluation/*.json"),
        "shadow_configs": _files(root, "configs/shadow/*.json"),
        "situation_configs": _files(root, "operator_configs/situations/*.json"),
        "scenario_suites": _files(root, "configs/scenarios/*.json")
        + _files(root, "operator_configs/situations/*.json"),
        "split_plans": [
            value
            for value in _files(root, "configs/scenarios/*.json")
            if "split_plan" in Path(value).name
        ],
        "research_objects": objects,
        "research_object_counts": {
            "total": len(objects),
            "by_root": {name: roots.get(name, 0) for name in RESEARCH_ROOTS},
            "by_status": dict(sorted(statuses.items())),
        },
    }


__all__ = ["RESEARCH_ROOTS", "build_catalog", "probe_endpoint"]
