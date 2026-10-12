"""Build reproducible, secret-free preview installers from a clean Git checkout."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import stat
import subprocess
import tarfile
import zipfile
from pathlib import Path

from .config import MODEL_SOURCE_COMMIT, LauncherError, source_root
from .downloads import model_entries

PREFIX = "CARLA-Vision-setup"
SOURCE_DIRECTORIES = (
    "carla_vision/", "configs/", "operator_configs/", "deploy/desktop/", "deploy/restore/",
    "models/", "samples/", "docs/",
)
SOURCE_FILES = {
    "pyproject.toml", "uv.lock", "README.md", "CONTRIBUTING.md", "PRODUCT.md", "DESIGN.md",
    "restore_project.py", "restore.sh", "Restore.cmd",
}
WINDOWS_ENTRY = (
    '@echo off\r\n'
    'setlocal\r\n'
    'title CARLA Vision Setup\r\n'
    'powershell.exe -NoProfile -ExecutionPolicy Bypass -File '
    '"%~dp0app\\deploy\\desktop\\install.ps1" %*\r\n'
    'set "SetupExit=%errorlevel%"\r\n'
    'if not "%SetupExit%"=="0" (\r\n  echo Setup failed. Read the error above and retry.\r\n  pause\r\n)\r\n'
    'exit /b %SetupExit%\r\n'
)
LINUX_ENTRY = (
    '#!/bin/sh\nset -eu\n'
    'InstallerDirectory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
    'exec sh "$InstallerDirectory/app/deploy/desktop/install.sh" "$@"\n'
)
INSTALL_README = """CARLA Vision - setup preview

Windows x86-64: Extract the ZIP completely, then double-click Install.cmd.
Linux x86-64: Extract the archive, then run: sh install.sh

No preinstalled Python, Node or Git is required by these setup bundles.
Setup creates a private Python 3.12 environment and the full-screen launcher.
In its Setup tab, choose your installed CARLA 0.9.16 folder and Install / Repair.
That installs the locked vision dependencies and downloads all ten model weights
(about 699 MiB) from a fixed GitHub source, checking each SHA-256.
An optional checkbox downloads the official CARLA server when it is missing.
The server download/extraction needs at least 35 GiB free.
Then use Start all. The existing web app opens after actual API readiness.

Desktop/application-menu shortcuts provide daily startup.
Setup preserves local recordings, settings and hydrated model files on upgrade.
Close the installed launcher and its services before reinstalling.

This is a preview pending Windows/Linux installation, visual and live-CARLA
acceptance. Do not interpret package integrity as runtime acceptance.
Linux needs curl, tar, coreutils and the OS libraries required by CARLA/OpenCV.
Both platforms need compatible graphics hardware and installed graphics drivers.
Internet is needed for the first dependency/model installation.
See app/docs/desktop_launcher.md for details, CLI usage and tomorrow's gates.
"""


def _git(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments], capture_output=True, check=False,
    )
    if result.returncode:
        raise LauncherError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def payload(root: Path) -> tuple[dict[str, tuple[bytes, int]], dict]:
    if _git(root, "status", "--porcelain").strip():
        raise LauncherError("Commit the source first; installer bundles require a clean Git checkout.")
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    weights = {
        "models/" + entry["path"]: entry
        for entry in model_entries(root) if entry.get("storage") == "git-lfs"
    }
    members: dict[str, tuple[bytes, int]] = {}
    inventory = []
    case_names: set[str] = set()
    # Read committed bytes so checkout line-ending settings cannot change a bundle.
    with tarfile.open(fileobj=io.BytesIO(_git(root, "archive", "--format=tar", "HEAD"))) as archive:
        committed = [
            (entry.name, entry.mode, entry.isfile(), archive.extractfile(entry).read() if entry.isfile() else b"")
            for entry in archive if not entry.isdir()
        ]
    for relative, mode, regular, original in committed:
        if relative not in SOURCE_FILES and not relative.startswith(SOURCE_DIRECTORIES):
            continue
        if not regular:
            raise LauncherError("Installer payload must contain only committed regular files.")
        if relative.casefold() in case_names:
            raise LauncherError("Installer paths collide on Windows: " + relative)
        case_names.add(relative.casefold())
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise LauncherError("Installer source is not a regular file: " + relative)
        if relative in weights:
            entry = weights[relative]
            data = (
                "version https://git-lfs.github.com/spec/v1\n"
                f"oid sha256:{entry['sha256']}\nsize {entry['size_bytes']}\n"
            ).encode()
        else:
            data = original
            if relative.endswith((".cmd", ".ps1")) and (relative.startswith("deploy/") or relative == "Restore.cmd"):
                data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        permissions = 0o755 if mode & 0o111 else 0o644
        members["app/" + relative] = data, permissions
        inventory.append({
            "path": relative, "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data), "mode": permissions, "model_weight": relative in weights,
        })
    required = {
        "app/pyproject.toml", "app/uv.lock", "app/models/assets.json",
        "app/carla_vision/operator/console_static/index.html",
        "app/deploy/desktop/install.ps1", "app/deploy/desktop/install.sh",
        "app/deploy/desktop/copy_install.py",
    }
    if required - members.keys() or not weights or not all("app/" + name in members for name in weights):
        raise LauncherError("The committed source lacks required installer assets.")
    manifest = {
        "schema_version": 1, "source_commit": commit, "preview": True,
        "native_validation": "deferred", "model_source_commit": MODEL_SOURCE_COMMIT,
        "model_weight_count": len(weights),
        "model_download_bytes": sum(entry["size_bytes"] for entry in weights.values()),
        "files": sorted(inventory, key=lambda item: item["path"]),
    }
    members["app/installer-files.json"] = (json.dumps(manifest, indent=2) + "\n").encode(), 0o644
    members["Install.cmd"] = WINDOWS_ENTRY.encode(), 0o644
    members["install.sh"] = LINUX_ENTRY.encode(), 0o755
    members["README.txt"] = INSTALL_README.encode(), 0o644
    return members, manifest


def build(root: Path, destination: Path) -> dict:
    members, inventory = payload(root)
    destination.mkdir(parents=True, exist_ok=True)
    zip_path = destination / "CARLA-Vision-setup-preview.zip"
    tar_path = destination / "CARLA-Vision-setup-preview.tar.gz"
    temporary_zip = zip_path.with_suffix(".zip.partial")
    temporary_tar = tar_path.with_suffix(".gz.partial")
    try:
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for relative, (data, mode) in sorted(members.items()):
                entry = zipfile.ZipInfo(PREFIX + "/" + relative, (1980, 1, 1, 0, 0, 0))
                entry.create_system = 3
                entry.external_attr = (stat.S_IFREG | mode) << 16
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, data)
        with temporary_tar.open("wb") as output, gzip.GzipFile(
            filename="", mode="wb", fileobj=output, mtime=0,
        ) as compressed, tarfile.open(
            mode="w", fileobj=compressed, format=tarfile.PAX_FORMAT,
        ) as archive:
            for relative, (data, mode) in sorted(members.items()):
                entry = tarfile.TarInfo(PREFIX + "/" + relative)
                entry.size, entry.mode, entry.mtime = len(data), mode, 0
                archive.addfile(entry, io.BytesIO(data))
        temporary_zip.replace(zip_path)
        temporary_tar.replace(tar_path)
    finally:
        temporary_zip.unlink(missing_ok=True)
        temporary_tar.unlink(missing_ok=True)
    artifacts = []
    for path in (zip_path, tar_path):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        artifacts.append({"path": path.name, "sha256": digest, "size_bytes": path.stat().st_size})
    manifest = {
        key: value for key, value in inventory.items() if key != "files"
    } | {"payload_files": len(members), "artifacts": artifacts}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (destination / "checksums.sha256").write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in artifacts), encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=source_root())
    parser.add_argument("--output", type=Path, default=Path("dist/desktop"))
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build(args.workspace.resolve(), args.output.resolve()), indent=2))
        return 0
    except (OSError, ValueError, LauncherError) as error:
        parser.exit(1, f"CARLA installer build: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
