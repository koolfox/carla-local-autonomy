"""Read-only browsing of registered research artifacts, independent of HTTP and CARLA.

ArtifactStore lists manifest declarations and resolves allowed files. It does
not create runs, verify their hashes, transcode videos, or start background work.
"""

from __future__ import annotations

import json
import mimetypes
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote

from ..evidence.contracts import CANONICAL_EVIDENCE_ROOTS as RESEARCH_ROOTS

_INLINE_ARTIFACT_SUFFIXES = frozenset(
    {
        ".csv",
        ".jpeg",
        ".jpg",
        ".json",
        ".jsonl",
        ".log",
        ".md",
        ".mp4",
        ".png",
        ".svg",
        ".txt",
    }
)
_DOWNLOAD_ARTIFACT_SUFFIXES = frozenset(
    {
        ".bin",
        ".ckpt",
        ".engine",
        ".onnx",
        ".pt",
        ".pth",
        ".safetensors",
        ".sha256",
        ".yaml",
        ".yml",
        ".zip",
    }
)
_ARTIFACT_SUFFIXES = _INLINE_ARTIFACT_SUFFIXES | _DOWNLOAD_ARTIFACT_SUFFIXES
_IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".svg"})
_VIDEO_SUFFIXES = frozenset({".mp4"})


def _safe_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    explicit = {
        ".ckpt": "application/octet-stream",
        ".engine": "application/octet-stream",
        ".jsonl": "application/x-ndjson",
        ".onnx": "application/octet-stream",
        ".pt": "application/octet-stream",
        ".pth": "application/octet-stream",
        ".safetensors": "application/octet-stream",
        ".sha256": "text/plain",
        ".yaml": "application/yaml",
        ".yml": "application/yaml",
    }
    return explicit.get(suffix) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _path_parts(raw: str, name: str) -> tuple[str, ...]:
    value = unquote(str(raw)).strip()
    if not value:
        raise ValueError(f"{name} is required")
    if "\\" in value or "\x00" in value or "\r" in value or "\n" in value:
        raise ValueError(f"{name} must use a safe forward-slash relative path")
    if PurePosixPath(value).is_absolute():
        raise ValueError(f"{name} must be workspace-relative")
    parts = tuple(value.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{name} contains an unsafe path component")
    if any(":" in part for part in parts):
        raise ValueError(f"{name} must not contain drive or stream separators")
    return parts


def _walk_has_symlink(base: Path, parts: Sequence[str]) -> bool:
    current = base
    for part in parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


class ArtifactStore:
    """Browse one workspace without constructing the Operator or a simulator."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve(strict=True)

    def _load_research_object(
        self,
        raw: str,
    ) -> tuple[Path, str, str, dict[str, Any]]:
        parts = _path_parts(raw, "research object path")
        if len(parts) != 2 or parts[0] not in RESEARCH_ROOTS:
            raise ValueError("research object path must be <allowed-root>/<object-directory>")
        if _walk_has_symlink(self.workspace, parts):
            raise ValueError("research object path must not contain symlinks")
        directory = self.workspace.joinpath(*parts)
        if not directory.is_dir():
            raise FileNotFoundError("research object directory does not exist")
        resolved = directory.resolve(strict=True)
        root = (self.workspace / parts[0]).resolve(strict=True)
        if resolved.parent != root:
            raise ValueError("research object path escapes its allow-listed root")
        manifest_path = directory / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise FileNotFoundError("research object manifest does not exist")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("research object manifest is not valid UTF-8 JSON") from error
        if not isinstance(manifest, dict):
            raise ValueError("research object manifest must contain an object")
        return resolved, "/".join(parts), parts[0], manifest

    @staticmethod
    def _manifest_artifacts(
        directory: Path,
        manifest: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        raw_artifacts = manifest.get("artifacts", [])
        if not isinstance(raw_artifacts, list):
            raise ValueError("research object manifest artifacts must be a list")
        artifacts: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for entry in raw_artifacts:
            if not isinstance(entry, Mapping):
                continue
            raw_path = entry.get("path")
            raw_role = entry.get("role")
            if not isinstance(raw_path, str) or not isinstance(raw_role, str):
                continue
            relative_path = raw_path.strip()
            role = raw_role.strip()
            if not relative_path or not role:
                continue
            if relative_path in seen_paths:
                raise ValueError(
                    f"research object manifest registers duplicate artifact {relative_path!r}"
                )
            seen_paths.add(relative_path)

            availability = "available"
            available = False
            try:
                parts = _path_parts(relative_path, "registered artifact path")
                candidate = directory.joinpath(*parts)
                if _walk_has_symlink(directory, parts):
                    availability = "unsafe_symlink"
                elif not candidate.is_file():
                    availability = "missing"
                else:
                    resolved = candidate.resolve(strict=True)
                    resolved.relative_to(directory)
                    available = True
            except (OSError, ValueError):
                availability = "unsafe_path"

            suffix = PurePosixPath(relative_path).suffix.lower()
            declared_mime = entry.get("mime_type", entry.get("media_type"))
            mime_type = (
                str(declared_mime)
                if isinstance(declared_mime, str) and declared_mime.strip()
                else _safe_mime_type(Path(relative_path))
            )
            size_bytes = entry.get("size_bytes")
            if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
                size_bytes = None
            sha256 = entry.get("sha256")
            if not isinstance(sha256, str):
                sha256 = None
            metadata = entry.get("metadata")
            if not isinstance(metadata, Mapping):
                metadata = {}
            artifacts.append(
                {
                    "role": role,
                    "path": relative_path,
                    "mime_type": mime_type,
                    "size_bytes": size_bytes,
                    "sha256": sha256,
                    "metadata": dict(metadata),
                    "available": available,
                    "availability": availability,
                    "preview_kind": (
                        "image"
                        if suffix in _IMAGE_SUFFIXES
                        else "video"
                        if suffix in _VIDEO_SUFFIXES
                        else "text"
                        if suffix in _INLINE_ARTIFACT_SUFFIXES
                        else "download"
                    ),
                    "downloadable": available and suffix in _ARTIFACT_SUFFIXES,
                }
            )
        return artifacts

    def inspect_research_object(self, raw: str) -> dict[str, Any]:
        directory, relative, root_kind, manifest = self._load_research_object(raw)
        artifacts = self._manifest_artifacts(directory, manifest)
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
        return {
            "schema_version": "1.0",
            "source": "manifest_declarations",
            "verification": {
                "performed": False,
                "status": "not_checked",
                "note": (
                    "This view lists manifest declarations and file availability; "
                    "use Verify for hash and semantic validation."
                ),
            },
            "object": {
                "id": str(manifest.get("run_id", directory.name)),
                "path": relative,
                "root_kind": root_kind,
                "status": str(manifest.get("status", "unknown")),
                "object_type": (
                    str(object_type)
                    if isinstance(object_type, str) and object_type.strip()
                    else "unknown"
                ),
                "created_at": timestamps.get("created_at"),
                "finished_at": timestamps.get("finished_at"),
                "artifact_count": len(artifacts),
                "available_artifact_count": sum(
                    bool(artifact["available"]) for artifact in artifacts
                ),
                "declared_artifact_bytes": sum(
                    int(artifact["size_bytes"])
                    for artifact in artifacts
                    if isinstance(artifact["size_bytes"], int)
                ),
            },
            "artifacts": artifacts,
        }

    def artifact_path(self, object_raw: str, artifact_raw: str) -> tuple[Path, bool]:
        directory, _, _, manifest = self._load_research_object(object_raw)
        artifact_parts = _path_parts(artifact_raw, "registered artifact path")
        normalized = "/".join(artifact_parts)
        matches = [
            artifact
            for artifact in self._manifest_artifacts(directory, manifest)
            if artifact["path"] == normalized
        ]
        if not matches:
            raise KeyError("artifact is not registered by the selected object manifest")
        artifact = matches[0]
        if not artifact["available"]:
            if artifact["availability"] == "missing":
                raise FileNotFoundError("registered artifact file is missing")
            raise ValueError("registered artifact path is unsafe")
        candidate = directory.joinpath(*artifact_parts)
        if _walk_has_symlink(directory, artifact_parts):
            raise ValueError("registered artifact path must not contain symlinks")
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(directory)
        suffix = resolved.suffix.lower()
        if suffix not in _ARTIFACT_SUFFIXES:
            raise ValueError("registered artifact type is not available through the UI")
        return resolved, suffix in _DOWNLOAD_ARTIFACT_SUFFIXES
