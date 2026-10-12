"""Copy the indexed setup payload without touching local runtime data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path, PurePosixPath


def safe_path(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative).parts
    if (
        not parts or PurePosixPath(relative).is_absolute()
        or any(part in {".", ".."} or "\\" in part or ":" in part for part in parts)
    ):
        raise ValueError("Unsafe installer path")
    path = root.joinpath(*parts)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Installer path escapes its destination or is a symbolic link")
    return path


def require_closed(target: Path) -> None:
    private = target / ".launcher"
    if private.is_symlink():
        raise ValueError("The private launcher directory must not be a symbolic link")
    receipt = private / "services.json"
    if receipt.is_file():
        state = json.loads(receipt.read_text(encoding="utf-8"))
        boot = next(
            int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
            if line.startswith("btime ")
        )
        for entry in state.get("services", {}).values():
            for process in [entry, *entry.get("descendants", [])]:
                pid, created = process.get("pid"), process.get("create_time")
                if type(pid) is not int or pid <= 0 or type(created) not in {int, float}:
                    raise ValueError("Malformed service receipt; close the installed application first")
                try:
                    # Linux starttime is field 22; comm may itself contain spaces/parentheses.
                    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                except (FileNotFoundError, ProcessLookupError):
                    continue
                actual = boot + int(fields[19]) / os.sysconf("SC_CLK_TCK")
                if fields[0] != "Z" and abs(actual - created) <= 0.02:
                    raise ValueError("Close installed launcher-owned services before reinstalling")


@contextmanager
def installation_guard(target: Path):
    import fcntl

    private = target / ".launcher"
    if private.is_symlink():
        raise ValueError("The private launcher directory must not be a symbolic link")
    private.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for name in ("launcher.lock", "operation.lock"):
            path = private / name
            if path.is_symlink():
                raise ValueError("Installer lock files must not be symbolic links")
            stream = stack.enter_context(path.open("a+b"))
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise ValueError("Close the installed launcher and finish other operations first") from error
        yield


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_payload(source: Path, target: Path) -> None:
    require_closed(target)
    if source == target:
        return
    index_path = safe_path(source, "installer-files.json")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entries = index["files"]
    if index.get("schema_version") != 1 or not isinstance(entries, list) or not entries:
        raise ValueError("Invalid installer file inventory")
    previous_path = safe_path(target, "installer-files.json")
    previous = {}
    if previous_path.is_file():
        previous = {
            entry["path"]: entry for entry in json.loads(previous_path.read_text(encoding="utf-8"))["files"]
        }
    # Verify the whole source before updating any installed file.
    for entry in entries:
        path = safe_path(source, entry["path"])
        if not path.is_file():
            raise ValueError("Installer source is missing: " + entry["path"])
        if file_digest(path) != entry["sha256"]:
            raise ValueError("Installer source integrity failed: " + entry["path"])
        safe_path(target, entry["path"])
    target.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        origin = safe_path(source, entry["path"])
        destination = safe_path(target, entry["path"])
        if entry["model_weight"] and destination.is_file() and destination.stat().st_size > 1024:
            continue
        if entry["path"].startswith(("configs/", "operator_configs/")) and destination.is_file():
            original = previous.get(entry["path"])
            if original is None or file_digest(destination) != original["sha256"]:
                destination = safe_path(
                    target,
                    ".launcher/incoming-defaults/" + index["source_commit"] + "/" + entry["path"],
                )
                print("Preserved customized configuration: " + entry["path"])
                print("Incoming default: " + str(destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(dir=destination.parent, prefix=".setup-")
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as output, origin.open("rb") as input_stream:
                shutil.copyfileobj(input_stream, output)
            temporary.chmod(entry["mode"])
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    shutil.copyfile(index_path, target / "installer-files.json")


def desktop_shortcut(target: Path) -> None:
    data = Path(os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local/share"))
    applications = data / "applications"
    applications.mkdir(parents=True, exist_ok=True)
    launch = str(target / "deploy/desktop/launch.sh")
    escaped = launch.replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace(chr(96), "\\" + chr(96)).replace("$", "\\$")
    # Percent is a field-code escape in a freedesktop Exec value.
    escaped = escaped.replace("%", "%%")
    (applications / "carla-vision.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=CARLA Vision\n"
        "Comment=Install and run the local CARLA console\nTerminal=true\n"
        'Exec=sh "' + escaped + '"\nCategories=Science;Development;\n',
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--shortcut", action="store_true")
    parser.add_argument("--uv", type=Path, help="Synchronize the minimal launcher environment under the install guard")
    args = parser.parse_args()
    try:
        target = args.target.resolve()
        with installation_guard(target):
            copy_payload(args.source.resolve(), target)
            if args.uv:
                result = subprocess.run([
                    str(args.uv.resolve()), "sync", "--python", "3.12", "--managed-python",
                    "--extra", "launcher", "--no-dev", "--inexact", "--frozen",
                ], cwd=target, check=False)
                if result.returncode:
                    raise ValueError("Python/launcher installation failed; correct the error and rerun setup")
            if args.shortcut:
                desktop_shortcut(target)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"CARLA setup: {error}\n")


if __name__ == "__main__":
    main()
