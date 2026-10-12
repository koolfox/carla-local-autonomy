"""Install into the private environment; keep existing source and actor owners."""

from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path

from .config import LauncherError, find_carla, native_platform, operation_lock, worker_token
from .downloads import Report, hydrate_models, install_carla
from .services import LocalStack


def uv_binary(root: Path) -> str:
    local = root / ".launcher/uv" / ("uv.exe" if os.name == "nt" else "uv")
    found = str(local) if local.is_file() else shutil.which("uv")
    if not found:
        raise LauncherError("Run the platform installer first; its private uv executable is missing.")
    return found


def run_install_command(
    command: list[str], root: Path, report: Report, cancel: threading.Event
) -> None:
    environment = dict(os.environ)
    environment.update({
        "UV_COLOR": "never", "UV_PYTHON_INSTALL_DIR": str(root / ".launcher/python"),
        "UV_CACHE_DIR": str(root / ".launcher/cache"), "PYTHONUTF8": "1",
        "UV_PROJECT_ENVIRONMENT": str(root / ".venv"),
    })
    log = root / ".launcher/install.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                if cancel.is_set():
                    process.terminate()
                    raise LauncherError("Dependency installation cancelled; rerun Install / Repair.")
                stream.write(line)
                stream.flush()
                report(line.rstrip())
            if process.wait() != 0:
                raise LauncherError("Dependency installation failed; inspect the installation log.")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def prepare_ffmpeg(root: Path, report: Report) -> None:
    import imageio_ffmpeg

    source = Path(imageio_ffmpeg.get_ffmpeg_exe())
    target = root / ".launcher/bin" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        temporary = target.with_suffix(target.suffix + ".install")
        shutil.copyfile(source, temporary)
        temporary.chmod(0o755)
        temporary.replace(target)
    result = subprocess.run(
        [str(target), "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20
    )
    if result.returncode != 0 or "libx264" not in result.stdout:
        raise LauncherError("The bundled FFmpeg encoder is unavailable or lacks libx264.")
    report("Browser recording encoder is ready in the application's private bin folder.")


def setup(
    root: Path,
    *,
    download_carla: bool = False,
    report: Report = print,
    cancel: threading.Event | None = None,
) -> dict:
    native_platform()
    if sys.version_info[:2] != (3, 12):
        raise LauncherError("Run the platform installer to create the required Python 3.12 environment.")
    cancel = cancel or threading.Event()
    stack = LocalStack(root, report)
    with operation_lock(root):
        if stack.active():
            raise LauncherError("Stop launcher-owned services before Install / Repair.")
        if not (root / "pyproject.toml").is_file() or not (root / "uv.lock").is_file():
            raise LauncherError("Setup requires the full source installation bundle or repository checkout.")
        config = stack.config
        extras = ["launcher", "native-host", "vision", "m9", "deit64", "segmentation", "voxel"]
        if config.profile == "research":
            extras.extend(["research", "experimental"])
        command = [
            uv_binary(root), "sync", "--python", "3.12", "--managed-python",
            "--frozen", "--no-dev", "--inexact",
        ]
        for extra in extras:
            command.extend(["--extra", extra])
        report("Installing the locked Python environment and official CARLA PythonAPI.")
        run_install_command(command, root, report, cancel)
        if importlib.metadata.version("carla") != "0.9.16":
            raise LauncherError("The installed CARLA PythonAPI does not match 0.9.16.")
        prepare_ffmpeg(root, report)
        report("Checking model files and hydrating missing Git LFS payloads.")
        hydrate_models(root, report, cancel)
        executable = find_carla(root, config.carla_root)
        if executable is None and download_carla:
            executable = install_carla(root, report, cancel)
        if executable is not None:
            config = replace(config, carla_root=str(executable.parent))
        config.save(root)
        worker_token(root, create=True)
        report("Application environment and models are ready.")
        if executable is None:
            report("Select an installed CARLA folder or enable its download before starting the server.")
    return stack.snapshot()
