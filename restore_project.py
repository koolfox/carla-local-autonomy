#!/usr/bin/env python3
"""Restore public project files/models and optionally import a private local backup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlparse

REPOSITORY = "koolfox/carla-local-autonomy"
DEFAULT_REF = "main"
UV_VERSION = "0.11.2"
UV_PACKAGES = {
    ("Windows", "amd64"): (
        "uv-x86_64-pc-windows-msvc.zip",
        "171b7ccda1bbd562da6babeffcf533a1c6cc7862cf998da826e1db534fc43e48",
    ),
    ("Linux", "x86_64"): (
        "uv-x86_64-unknown-linux-gnu.tar.gz",
        "7ac2ca0449c8d68dae9b99e635cd3bc9b22a4cb1de64b7c43716398447d42981",
    ),
    ("Darwin", "arm64"): (
        "uv-aarch64-apple-darwin.tar.gz",
        "4beaa9550f93ef7f0fc02f7c28c9c48cd61fe30db00f5ac8947e0a425c3fb282",
    ),
    ("Darwin", "x86_64"): (
        "uv-x86_64-apple-darwin.tar.gz",
        "a9c3653245031304c50dd60ac0301bf6c112e12c38c32302a71d4fa6a63ba2cb",
    ),
}
RUNTIME_EXTRAS = ["launcher", "vision", "m9", "deit64", "segmentation", "voxel"]


class RestoreError(RuntimeError):
    pass


class SecureRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlparse(newurl).scheme != "https":
            raise RestoreError("Refusing an insecure download redirect.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def safe_path(root: Path, relative: str, *, scratch_file: bool = False) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or PurePosixPath(relative).is_absolute()
        or any(value in relative for value in ("\\", "\0"))
        or (":" in relative and not (scratch_file and relative == ":memory:.ses"))
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise RestoreError("Invalid relative archive or inventory path.")
    target = root.joinpath(*PurePosixPath(relative).parts)
    parent = root
    for part in PurePosixPath(relative).parts:
        parent = parent / part
        if parent.is_symlink():
            raise RestoreError("Symbolic links are not supported in restoration paths.")
    if not target.resolve().is_relative_to(root.resolve()):
        raise RestoreError("An archive or inventory path escapes its destination.")
    return target


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
        stream.write("\n")
    try:
        temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def download(
    url: str, destination: Path, digest: str | None = None, size: int | None = None
) -> None:
    if urlparse(url).scheme != "https":
        raise RestoreError("Only HTTPS downloads are supported.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, suffix=".part", delete=False
        ) as output:
            temporary = Path(output.name)
            total, checksum = 0, hashlib.sha256()
            opener = urllib.request.build_opener(SecureRedirects())
            request = urllib.request.Request(url, headers={"User-Agent": "CARLA-Project-Restore/1"})
            with opener.open(request, timeout=30) as response:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > (size if size is not None else 2 * 1024**3):
                        raise RestoreError("Download exceeded its declared size or archive limit.")
                    if shutil.disk_usage(destination.parent).free < len(chunk) + 64 * 1024**2:
                        raise RestoreError("Not enough free disk space for restoration.")
                    checksum.update(chunk)
                    output.write(chunk)
            if size is not None and total != size:
                raise RestoreError("Download was incomplete.")
            if digest is not None and checksum.hexdigest() != digest:
                raise RestoreError("Download checksum did not match; existing file was preserved.")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def resolve_commit(reference: str) -> str:
    url = f"https://api.github.com/repos/{REPOSITORY}/commits/{quote(reference, safe='')}"
    request = urllib.request.Request(url, headers={"User-Agent": "CARLA-Project-Restore/1"})
    with urllib.request.build_opener(SecureRedirects()).open(request, timeout=20) as response:
        value = json.load(response)
    commit = value.get("sha")
    if not isinstance(commit, str) or not re.fullmatch("[a-f0-9]{40}", commit):
        raise RestoreError("GitHub did not return a valid source commit.")
    return commit


def extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as source:
        entries = source.infolist()
        if len(entries) > 100_000 or sum(entry.file_size for entry in entries) > 4 * 1024**3:
            raise RestoreError("Source archive is too large.")
        seen = set()
        for entry in entries:
            name = entry.filename.rstrip("/")
            safe_path(destination, name)
            kind = stat.S_IFMT(entry.external_attr >> 16)
            if kind not in {0, stat.S_IFREG, stat.S_IFDIR} or name.casefold() in seen:
                raise RestoreError("Source archive contains a link or colliding path.")
            seen.add(name.casefold())
        for entry in entries:
            source.extract(entry, destination)
            if not entry.is_dir() and entry.external_attr >> 16 & 0o111:
                safe_path(destination, entry.filename).chmod(0o755)


def obtain_source(destination: Path, commit: str, use_git: bool) -> None:
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise RestoreError(
            "Destination must be new or empty. Use --resume only for a prior restore."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".carla-restore-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        if use_git:
            if not shutil.which("git") or not shutil.which("git-lfs"):
                raise RestoreError(
                    "--git requires Git and Git LFS. Omit it for a source-archive restore."
                )
            payload = stage / "project"
            environment = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0")
            run(["git", "init", "-q", str(payload)], environment)
            run(
                [
                    "git",
                    "-C",
                    str(payload),
                    "remote",
                    "add",
                    "origin",
                    f"https://github.com/{REPOSITORY}.git",
                ],
                environment,
            )
            run(["git", "-C", str(payload), "fetch", "--depth", "1", "origin", commit], environment)
            run(
                ["git", "-C", str(payload), "checkout", "-q", "-b", "restored", "FETCH_HEAD"],
                environment,
            )
            run(["git", "-C", str(payload), "lfs", "install", "--local"], environment)
        else:
            archive = stage / "source.zip"
            print("Downloading the committed GitHub source archive...")
            download(f"https://api.github.com/repos/{REPOSITORY}/zipball/{commit}", archive)
            extracted = stage / "source"
            extracted.mkdir()
            extract_zip(archive, extracted)
            folders = list(extracted.iterdir())
            if len(folders) != 1 or not folders[0].is_dir():
                raise RestoreError("GitHub archive has an unexpected root.")
            payload = folders[0]
        required = (
            "pyproject.toml",
            "uv.lock",
            "models/assets.json",
            "samples/archive.json",
            "carla_vision/operator/console_static/index.html",
        )
        if any(not safe_path(payload, name).is_file() for name in required):
            raise RestoreError(
                "This GitHub ref lacks required model/sample assets; use main or a complete delivery commit."
            )
        if destination.exists():
            destination.rmdir()  # Only the verified empty destination.
        payload.rename(destination)


def hydrate_models(root: Path, commit: str) -> None:
    inventory = json.loads((root / "models/assets.json").read_text(encoding="utf-8"))
    entries = inventory["files"]
    if not isinstance(entries, list) or not entries:
        raise RestoreError("Model inventory is empty.")
    for entry in entries:
        if (
            not re.fullmatch("[a-f0-9]{64}", entry.get("sha256", ""))
            or type(entry.get("size_bytes")) is not int
            or entry["size_bytes"] < 0
        ):
            raise RestoreError("Invalid model inventory entry.")
        path = safe_path(root, "models/" + entry["path"])
        if (
            path.is_file()
            and path.stat().st_size == entry["size_bytes"]
            and file_hash(path) == entry["sha256"]
        ):
            continue
        if entry.get("storage") != "git-lfs":
            raise RestoreError(
                "A source model configuration or notice is missing/changed: " + entry["path"]
            )
        if path.exists():
            pointer = (
                "version https://git-lfs.github.com/spec/v1\n"
                f"oid sha256:{entry['sha256']}\nsize {entry['size_bytes']}\n"
            ).encode()
            if (
                not path.is_file()
                or path.stat().st_size > 1024
                or path.read_bytes().replace(b"\r\n", b"\n") != pointer
            ):
                raise RestoreError(
                    "A model weight or pointer changed; preserve it before retrying repair."
                )
        print("Downloading and verifying model: " + entry["path"])
        url = f"https://media.githubusercontent.com/media/{REPOSITORY}/{commit}/models/{quote(entry['path'], safe='/')}"
        download(url, path, entry["sha256"], entry["size_bytes"])
    print("All model weights, labels, processors and notices verified.")


def verify_samples(root: Path, copy_original_paths: bool) -> None:
    inventory = json.loads((root / "samples/archive.json").read_text(encoding="utf-8"))
    for entry in inventory["files"]:
        path = safe_path(root, "samples/" + entry["path"])
        if not path.is_file() or file_hash(path) != entry["sha256"]:
            raise RestoreError("Published sample checksum mismatch: " + entry["path"])
        if not copy_original_paths:
            continue
        relative = entry["source_path"]
        if not relative.startswith(("output/", "reports/", "datasets/", "models/")):
            raise RestoreError("Sample source path is outside the recorded data roots.")
        destination = safe_path(root, relative)
        if destination.exists():
            if not destination.is_file() or file_hash(destination) != entry["sha256"]:
                raise RestoreError(
                    "Sample restoration would overwrite a different file: " + relative
                )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    print(
        "Historical sample files verified; their original data paths restored."
        if copy_original_paths
        else "Historical sample files verified."
    )


def restore_backup(root: Path, archive: Path) -> None:
    allowed = (".env.local", ":memory:.ses")
    prefixes = ("operator_sessions/", "native_tasks/", "runs/", "dist/")
    private_settings = (
        ".launcher/config.json",
        ".launcher/worker.token",
        ".launcher/carla-download.json",
    )
    sidecar = archive.parent / "checksums.sha256"
    if sidecar.is_file():
        row = next(
            (
                line.split()
                for line in sidecar.read_text().splitlines()
                if line.endswith("  " + archive.name)
            ),
            None,
        )
        if row is None or row[0] != file_hash(archive):
            raise RestoreError("Local backup archive checksum did not match.")
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if len(members) > 100_000 or sum(member.file_size for member in members) > 4 * 1024**3:
            raise RestoreError("Private backup exceeds the supported archive size.")
        member_names = [member.filename for member in members]
        if (
            len(member_names) != len(set(member_names))
            or source.getinfo("manifest.json").file_size > 16 * 1024**2
        ):
            raise RestoreError("Private backup has duplicate entries or an oversized manifest.")
        manifest = json.loads(source.read("manifest.json"))
        if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
            raise RestoreError("Unsupported local backup format.")
        names = set()
        selected = []
        for entry in manifest["files"]:
            relative = entry["path"]
            # The original POSIX scratch filename is invalid on Windows; retain it in the backup.
            destination = (
                None
                if relative == ":memory:.ses" and os.name == "nt"
                else safe_path(root, relative, scratch_file=True)
            )
            if (
                not re.fullmatch("[a-f0-9]{64}", entry.get("sha256", ""))
                or type(entry.get("size_bytes")) is not int
                or not 0 <= entry["size_bytes"] <= 512 * 1024**2
                or type(entry.get("mode", 0o600)) is not int
            ):
                raise RestoreError("Invalid private backup inventory entry.")
            if relative.casefold() in names:
                raise RestoreError("Local backup contains colliding paths.")
            names.add(relative.casefold())
            member = source.getinfo("data/" + relative)
            if member.file_size != entry["size_bytes"] or stat.S_IFMT(
                member.external_attr >> 16
            ) not in {0, stat.S_IFREG}:
                raise RestoreError("Private backup contains an unexpected file type or size.")
            data = source.read("data/" + relative)
            if (
                len(data) != entry["size_bytes"]
                or hashlib.sha256(data).hexdigest() != entry["sha256"]
            ):
                raise RestoreError("Local backup file integrity failed.")
            permitted = (
                relative in allowed
                or relative.startswith(prefixes)
                or relative in private_settings
                or relative.startswith(".launcher/incoming-defaults/")
            )
            if "__pycache__" in PurePosixPath(relative).parts or relative.endswith(
                (".pyc", ".pyo")
            ):
                permitted = False
            if relative == ":memory:.ses" and os.name == "nt":
                permitted = False
            if not permitted:
                continue  # Never import process receipts, locks, control files or environment caches.
            assert destination is not None
            if destination.exists() and (
                not destination.is_file() or file_hash(destination) != entry["sha256"]
            ):
                raise RestoreError("Local backup would overwrite a different file: " + relative)
            selected.append((entry, destination))
        expected = {"manifest.json"} | {"data/" + entry["path"] for entry in manifest["files"]}
        if set(member_names) != expected:
            raise RestoreError("Private backup contains undeclared files.")
        for entry, destination in selected:
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
                    temporary = Path(output.name)
                    output.write(source.read("data/" + entry["path"]))
                mode = entry.get("mode", 0o600) & 0o777
                if entry["path"] in allowed or entry["path"].startswith(".launcher/"):
                    mode = 0o600
                temporary.chmod(mode)
                temporary.replace(destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    print(
        f"Private backup restored: {len(selected)} files; {len(manifest['files']) - len(selected)} runtime/cache files skipped."
    )


def run(command: list[str], environment: dict | None = None, cwd: Path | None = None) -> None:
    result = subprocess.run(command, env=environment, cwd=cwd, check=False)
    if result.returncode:
        raise RestoreError(f"{Path(command[0]).name} failed with exit code {result.returncode}.")


def uv_binary(root: Path, supplied: Path | None) -> Path:
    target = safe_path(root, ".launcher/uv/" + ("uv.exe" if os.name == "nt" else "uv"))
    target.parent.mkdir(parents=True, exist_ok=True)
    if supplied:
        if not supplied.is_file():
            raise RestoreError("The supplied uv executable was not found.")
        if supplied.resolve() != target.resolve():
            shutil.copyfile(supplied, target)
            target.chmod(0o755)
        return target
    system = shutil.which("uv")
    if system:
        if Path(system).resolve() != target.resolve():
            shutil.copyfile(system, target)
            target.chmod(0o755)
        return target
    key = platform.system(), platform.machine().lower()
    if key not in UV_PACKAGES:
        raise RestoreError("No pinned setup runtime is available for this platform.")
    filename, digest = UV_PACKAGES[key]
    with tempfile.TemporaryDirectory(prefix="uv-install-", dir=target.parent) as temporary:
        stage = Path(temporary)
        archive = stage / filename
        download(
            f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{filename}",
            archive,
            digest,
        )
        (stage / "payload").mkdir()
        if filename.endswith(".zip"):
            extract_zip(archive, stage / "payload")
        else:
            with tarfile.open(archive) as source:
                # Python 3.11 before the data-filter backport is also supported.
                for member in source.getmembers():
                    safe_path(stage / "payload", member.name.rstrip("/"))
                    if not (member.isfile() or member.isdir()):
                        raise RestoreError(
                            "The pinned runtime archive contains an unexpected link/type."
                        )
                options = {"filter": "data"} if sys.version_info >= (3, 12) else {}
                source.extractall(stage / "payload", **options)
        executables = list((stage / "payload").rglob(target.name))
        if len(executables) != 1:
            raise RestoreError("The pinned uv archive has an unexpected layout.")
        shutil.copyfile(executables[0], target)
        target.chmod(0o755)
    return target


@contextmanager
def restoration_lock(root: Path, name: str):
    private = safe_path(root, ".launcher")
    private.mkdir(parents=True, exist_ok=True)
    with safe_path(root, ".launcher/" + name).open("a+b") as stream:
        # Do not read a byte locked by another process on Windows.
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RestoreError(
                "Close the target launcher and finish other operations before resuming."
            ) from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def require_stopped(root: Path) -> None:
    receipt = safe_path(root, ".launcher/services.json")
    if receipt.exists() and json.loads(receipt.read_text(encoding="utf-8")).get("services"):
        raise RestoreError(
            "Stop the target workspace's launcher services before resuming restoration."
        )


def selected_role(role: str) -> str:
    native = platform.system() in {"Windows", "Linux"} and platform.machine().lower() in {
        "amd64",
        "x86_64",
    }
    if role == "auto":
        return "local" if native else "web"
    if role == "local" and not native:
        raise RestoreError(
            "CARLA local mode requires Windows/Linux x86-64. Use --role web on macOS."
        )
    return role


def finish_setup(root: Path, role: str, args) -> None:
    sys.path.insert(0, str(root))
    from carla_vision.launcher.install import prepare_ffmpeg

    if role == "local":
        # Existing native setup, without any Start or world/actor mutation.
        from dataclasses import replace

        from carla_vision.launcher.config import LauncherConfig
        from carla_vision.launcher.install import setup
        from carla_vision.launcher.services import LocalStack

        config = LauncherConfig.load(root)
        if args.research:
            config = replace(config, profile="research")
        if args.carla_root:
            config = replace(config, carla_root=args.carla_root)
        LocalStack(root).configure(config)
        setup(root, download_carla=args.download_carla)
    else:
        with restoration_lock(root, "operation.lock"):
            require_stopped(root)
            prepare_ffmpeg(root, print)
            create_web_settings(root)


def create_web_settings(root: Path) -> None:
    path = safe_path(root, ".env.local")
    if not path.exists():
        path.write_text(
            "# Configure the Windows/LAN Worker URL and matching token before Garage use.\n"
            "CARLA_HOST='auto'\nCARLA_PORT='2000'\n"
            "CARLA_OPERATOR_BIND='127.0.0.1'\nCARLA_OPERATOR_PORT='8765'\n"
            "CARLA_WORKSPACE='.'\nCARLA_ENABLE_EXPERIMENTAL='false'\n"
            "# CARLA_WORLD_WORKER_URL='http://<windows-lan-address>:8766'\n"
            "# CARLA_WORLD_WORKER_TOKEN='replace-with-the-worker-token'\n",
            encoding="utf-8",
        )
        path.chmod(0o600)


def install_environment(root: Path, role: str, args) -> dict:
    uv = uv_binary(root, args.uv)
    environment = dict(os.environ)
    environment.update(
        {
            "UV_PYTHON_INSTALL_DIR": str(root / ".launcher/python"),
            "UV_CACHE_DIR": str(root / ".launcher/cache"),
            "UV_PROJECT_ENVIRONMENT": str(root / ".venv"),
            "PYTHONUTF8": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )
    lean = args.lean or (platform.system() == "Darwin" and platform.machine().lower() == "x86_64")
    extras = ["launcher"] if lean else list(RUNTIME_EXTRAS)
    if lean:
        print(
            "Installing the web/recording profile; the locked full vision stack requires a supported vision host."
        )
    if role == "local":
        extras.append("native-host")
    if args.research:
        extras.extend(["research", "experimental"])
    command = [str(uv), "sync", "--python", "3.12", "--managed-python", "--frozen", "--inexact"]
    command += ["--group", "dev"] if args.dev else ["--no-dev"]
    for extra in extras:
        command += ["--extra", extra]
    print("Installing the private locked application environment...")
    run(command, environment, root)
    return environment


def complete_environment(root: Path, role: str, args, environment: dict) -> None:
    interpreter = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    command = [
        str(interpreter),
        str(Path(__file__).resolve()),
        "--finish-setup",
        str(root),
        "--role",
        role,
    ]
    if args.carla_root:
        command += ["--carla-root", args.carla_root]
    if args.download_carla:
        command.append("--download-carla")
    if args.research:
        command.append("--research")
    # The parent releases operation.lock before existing native setup acquires it.
    run(command, environment, root)


def main(argv: list[str] | None = None) -> int:
    working = Path.cwd()
    if (working / "pyproject.toml").is_file() and (working / "uv.lock").is_file():
        working = working.parent  # Keep the restored copy outside the original project.
    default = working / "carla-local-autonomy-restored"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=default)
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--role", choices=("auto", "local", "web"), default="auto")
    parser.add_argument(
        "--git", action="store_true", help="Create a shallow Git checkout; needs Git + Git LFS"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a prior restore at its recorded source commit",
    )
    parser.add_argument("--no-setup", action="store_true", help="Restore source/models/data only")
    parser.add_argument(
        "--no-sample-copies", action="store_true", help="Keep samples only under samples/"
    )
    parser.add_argument(
        "--local-backup", type=Path, help="Import local-state.zip saved outside the old project"
    )
    parser.add_argument("--research", action="store_true")
    parser.add_argument("--dev", action="store_true")
    parser.add_argument(
        "--lean",
        action="store_true",
        help="Web/recording dependencies only; all bundled files/models still restored",
    )
    parser.add_argument("--carla-root")
    parser.add_argument("--download-carla", action="store_true")
    parser.add_argument("--uv", type=Path)
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print the intended role/paths without downloading or installing",
    )
    parser.add_argument("--finish-setup", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        role = selected_role(args.role)
        if args.download_carla and role != "local":
            raise RestoreError("--download-carla is available only in local Windows/Linux mode.")
        lean = args.lean or (
            platform.system() == "Darwin" and platform.machine().lower() == "x86_64"
        )
        if lean and (role != "web" or args.research):
            raise RestoreError("The lean profile supports web mode without --research.")
        if (
            not args.no_setup
            and not args.plan
            and platform.system() == "Darwin"
            and int(platform.mac_ver()[0].split(".")[0]) < 14
        ):
            raise RestoreError(
                "The locked macOS environment requires macOS 14 or newer. --no-setup restores files only."
            )
        if args.finish_setup:
            finish_setup(args.finish_setup.resolve(), role, args)
            return 0
        root = args.destination.expanduser().absolute()
        if root.is_symlink():
            raise RestoreError("Destination must not be a symbolic link.")
        root = root.resolve()
        if args.plan:
            print(
                json.dumps(
                    {
                        "repository": REPOSITORY,
                        "ref": args.ref,
                        "destination": str(root),
                        "role": role,
                        "method": "git" if args.git else "archive",
                        "install_environment": not args.no_setup,
                        "local_backup": bool(args.local_backup),
                        "vision_dependencies": not lean,
                        "server_start": False,
                    },
                    indent=2,
                )
            )
            return 0
        marker = root / ".restore-state.json"
        if args.resume:
            state = json.loads(safe_path(root, ".restore-state.json").read_text())
            if (
                state.get("schema_version") != 1
                or state.get("repository") != REPOSITORY
                or state.get("method") not in {"git", "archive"}
                or not re.fullmatch("[a-f0-9]{40}", state.get("commit", ""))
            ):
                raise RestoreError("Destination is not a recorded project restore.")
            commit, use_git = state["commit"], state["method"] == "git"
            if file_hash(root / "models/assets.json") != state["model_inventory_sha256"]:
                raise RestoreError("The restored model inventory changed; use a new destination.")
        else:
            commit, use_git = resolve_commit(args.ref), args.git
            obtain_source(root, commit, use_git)
            state = {
                "schema_version": 1,
                "repository": REPOSITORY,
                "commit": commit,
                "ref": args.ref,
                "method": "git" if use_git else "archive",
                "role": role,
                "phase": "source",
                "model_inventory_sha256": file_hash(root / "models/assets.json"),
            }
            write_json(marker, state)
        with restoration_lock(root, "launcher.lock"):
            with restoration_lock(root, "operation.lock"):
                require_stopped(root)
                hydrate_models(root, commit)
                verify_samples(root, not args.no_sample_copies)
                if args.local_backup:
                    restore_backup(root, args.local_backup.expanduser().resolve())
                state["role"], state["phase"] = role, "assets"
                write_json(marker, state)
                if not args.no_setup:
                    environment = install_environment(root, role, args)
            if not args.no_setup:
                complete_environment(root, role, args, environment)
            state["phase"] = "ready" if not args.no_setup else "assets"
            write_json(marker, state)
        print(f"Restored {commit} to {root}. Role: {role}. No CARLA or web service was started.")
        if args.no_setup:
            print(
                "Files and assets are restored. Rerun with --resume without --no-setup to install the application environment."
            )
        elif role == "web":
            print(
                "Set the LAN Worker URL/token in .env.local. Start from the project root "
                "with the bundled encoder on PATH: "
                'PATH="$PWD/.launcher/bin:$PATH" .venv/bin/carla-operator-ui --open-browser '
                '(Windows PowerShell: $env:PATH="$PWD\\.launcher\\bin;$env:PATH"; '
                ".\\.venv\\Scripts\\carla-operator-ui.exe --open-browser)."
            )
        else:
            print(
                "Start the full-screen launcher: .venv/Scripts/carla-app.exe tui (Windows) or .venv/bin/carla-app tui (Linux)."
            )
        return 0
    except (RuntimeError, OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"Restore: {error}", file=sys.stderr)
        print(
            "Existing projects were preserved. A partial restore can be retried with --resume.",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("Restore interrupted. Retry the new destination with --resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
