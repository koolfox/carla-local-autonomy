"""Integrity-checked model hydration and official optional CARLA packages."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import threading
import time
import urllib.request
import uuid
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from .config import CARLA_VERSION, MODEL_SOURCE_COMMIT, LauncherError, atomic_json, native_platform

Report = Callable[[str], None]
CARLA_URLS = {
    "Windows": "https://downloads.carlasim.com/Windows/CARLA_0.9.16.zip",
    "Linux": "https://downloads.carlasim.com/Linux/CARLA_0.9.16.tar.gz",
}


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative).parts
    if not parts or PurePosixPath(relative).is_absolute() or any(
        part in {".", ".."} or "\\" in part or ":" in part for part in parts
    ):
        raise LauncherError("Archive or model path is not a safe relative path.")
    target = root.joinpath(*parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise LauncherError("Archive or model path escapes its destination.")
    return target


class HttpsRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlparse(newurl).scheme != "https":
            raise LauncherError("Download refused an insecure redirect.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(
    url: str,
    target: Path,
    *,
    expected_sha256: str | None,
    expected_size: int | None,
    report: Report,
    cancel: threading.Event,
) -> str:
    if urlparse(url).scheme != "https":
        raise LauncherError("Downloads require HTTPS.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".part")
    digest = hashlib.sha256()
    opener = urllib.request.build_opener(HttpsRedirects())
    request = urllib.request.Request(url, headers={"User-Agent": "CARLA-Vision-Installer/1"})
    try:
        with opener.open(request, timeout=30) as response, temporary.open("xb") as output:
            declared = response.headers.get("Content-Length")
            total = expected_size or (int(declared) if declared and declared.isdecimal() else None)
            if total and shutil.disk_usage(target.parent).free < total + 64 * 1024 * 1024:
                raise LauncherError(f"Not enough free disk space to download {target.name}.")
            size = 0
            last_report = 0.0
            while chunk := response.read(1024 * 1024):
                if cancel.is_set():
                    raise LauncherError("Installation cancelled; existing files were preserved.")
                size += len(chunk)
                if expected_size is not None and size > expected_size:
                    raise LauncherError(f"Download exceeded recorded size: {target.name}")
                digest.update(chunk)
                output.write(chunk)
                if time.monotonic() - last_report > 1:
                    suffix = f" / {total / 1024**2:.0f} MiB" if total else " MiB"
                    report(f"Downloading {target.name}: {size / 1024**2:.0f}{suffix}")
                    last_report = time.monotonic()
            output.flush()
        actual = digest.hexdigest()
        if expected_size is not None and size != expected_size:
            raise LauncherError(f"Download is incomplete: {target.name}")
        if expected_sha256 is not None and actual != expected_sha256:
            raise LauncherError(f"Checksum mismatch: {target.name}; existing file was preserved.")
        temporary.replace(target)
        return actual
    finally:
        temporary.unlink(missing_ok=True)


def model_entries(root: Path) -> list[dict]:
    manifest_path = root / "models/assets.json"
    if not manifest_path.is_file():
        raise LauncherError("The installer is missing its model inventory; use the full setup bundle.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise LauncherError("Model inventory is empty or invalid.")
    for entry in entries:
        if not isinstance(entry, dict) or type(entry.get("size_bytes")) is not int:
            raise LauncherError("Model inventory contains an invalid file declaration.")
        digest = entry.get("sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise LauncherError("Model inventory contains an invalid digest.")
        contained(root / "models", entry.get("path", ""))
    return entries


def model_status(root: Path) -> dict:
    entries = [entry for entry in model_entries(root) if entry.get("storage") == "git-lfs"]
    if not entries:
        raise LauncherError("Model inventory has no weights; repair the full installation bundle.")
    present = sum(
        (path := contained(root / "models", entry["path"])).is_file()
        and path.stat().st_size == entry["size_bytes"]
        for entry in entries
    )
    return {"present": present, "total": len(entries), "ready": present == len(entries)}


def hydrate_models(root: Path, report: Report, cancel: threading.Event) -> None:
    for entry in model_entries(root):
        target = contained(root / "models", entry["path"])
        if target.is_file() and target.stat().st_size == entry["size_bytes"]:
            if file_hash(target) == entry["sha256"]:
                continue
        if entry.get("storage") != "git-lfs":
            raise LauncherError(f"Bundled model metadata is missing or changed: {entry['path']}")
        if target.is_symlink():
            raise LauncherError(f"Refusing to replace a model symlink: {entry['path']}")
        backup = None
        if target.is_file() and target.stat().st_size > 1024:
            backup = root / ".launcher/backups" / (target.name + "." + uuid.uuid4().hex)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target, backup)
            report(f"Preserved the previous {target.name} in launcher backups.")
        url = (
            "https://media.githubusercontent.com/media/koolfox/carla-local-autonomy/"
            f"{MODEL_SOURCE_COMMIT}/models/{entry['path']}"
        )
        download(
            url, target, expected_sha256=entry["sha256"], expected_size=entry["size_bytes"],
            report=report, cancel=cancel,
        )
        report(f"Verified {target.name}")


def install_carla(root: Path, report: Report, cancel: threading.Event) -> Path:
    system = native_platform()
    destination = root / f"simulator/CARLA_{CARLA_VERSION}"
    executable_name = "CarlaUE4.exe" if system == "Windows" else "CarlaUE4.sh"
    if (destination / executable_name).is_file():
        return destination / executable_name
    if destination.exists():
        raise LauncherError("CARLA destination already exists without its launcher; select another folder.")
    if shutil.disk_usage(root).free < 35 * 1024**3:
        raise LauncherError("Automatic CARLA download/extraction needs at least 35 GiB free disk space.")
    archive = root / ".launcher/downloads" / ("CARLA.zip" if system == "Windows" else "CARLA.tar.gz")
    report("Downloading the official CARLA 0.9.16 package; this is a multi-gigabyte download.")
    digest = download(
        CARLA_URLS[system], archive, expected_sha256=None, expected_size=None,
        report=report, cancel=cancel,
    )
    stage = destination.with_name(destination.name + ".install-" + uuid.uuid4().hex)
    stage.mkdir(parents=True)
    try:
        report("Extracting CARLA 0.9.16")
        if system == "Windows":
            with zipfile.ZipFile(archive) as source:
                for entry in source.infolist():
                    contained(stage, entry.filename.rstrip("/"))
                    if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                        raise LauncherError("CARLA ZIP contains an unsupported symlink.")
                    if cancel.is_set():
                        raise LauncherError("CARLA installation cancelled.")
                    source.extract(entry, stage)
        else:
            with tarfile.open(archive, "r:gz") as source:
                for entry in source:
                    if cancel.is_set():
                        raise LauncherError("CARLA installation cancelled.")
                    source.extract(entry, stage, filter="data")
        candidates = list(stage.glob(executable_name)) + list(stage.glob("*/" + executable_name))
        if len(candidates) != 1:
            raise LauncherError("Official archive did not contain one recognizable CARLA launcher.")
        content_root = candidates[0].parent
        content_root.replace(destination)
        atomic_json(root / ".launcher/carla-download.json", {
            "version": CARLA_VERSION, "url": CARLA_URLS[system], "local_archive_sha256": digest,
            "publisher_checksum_available": False,
        })
        return destination / executable_name
    finally:
        # These are this operation's staging/archive files, never installed user data.
        if stage.exists():
            shutil.rmtree(stage)
        archive.unlink(missing_ok=True)
