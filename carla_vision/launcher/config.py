"""Private launcher configuration for local and remote-web operation."""

from __future__ import annotations

import ipaddress
import json
import os
import platform
import secrets
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

CARLA_VERSION = "0.9.16"
MODEL_SOURCE_COMMIT = "2c600765e34862becc4f236d99291b295604f973"


class LauncherError(RuntimeError):
    """An actionable installation or lifecycle failure."""


def default_mode() -> str:
    return (
        "local"
        if platform.system() in {"Windows", "Linux"}
        and platform.machine().lower()
        in {
            "amd64",
            "x86_64",
        }
        else "web"
    )


def validate_worker_url(value: str) -> str:
    if len(value) > 2048 or any(character.isspace() for character in value):
        raise LauncherError("Worker URL must not contain whitespace.")
    parsed = urlparse(value)
    try:
        port = parsed.port
    except ValueError as error:
        raise LauncherError("Worker URL contains an invalid port.") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or port == 0
    ):
        raise LauncherError(
            "Use a Worker base URL such as http://192.168.1.100:8766, without credentials."
        )
    return value.rstrip("/")


def source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def native_platform() -> str:
    system = platform.system()
    if system not in {"Windows", "Linux"} or platform.machine().lower() not in {
        "amd64",
        "x86_64",
    }:
        raise LauncherError("Same-computer CARLA requires Windows or Linux on x86-64.")
    return system


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as stream:
        temporary = Path(stream.name)
        try:
            os.chmod(temporary, 0o600)
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    if path.is_symlink() or not path.is_file():
        raise LauncherError(f"Expected a regular launcher file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise LauncherError(f"Cannot read {path.name}; restore it or run setup again.") from error


@contextmanager
def _file_lock(root: Path, name: str, message: str):
    """Use an OS lock that releases on process exit, including crashes."""
    directory = root / ".launcher"
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise LauncherError("The launcher state directory must not be a symlink.")
    path = directory / name
    if path.is_symlink():
        raise LauncherError("Launcher lock files must not be symbolic links.")
    with path.open("a+b") as stream:
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
            raise LauncherError(message) from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def operation_lock(root: Path):
    """Prevent overlapping install/start/stop."""
    with _file_lock(
        root,
        "operation.lock",
        "Another launcher operation is active. Wait for it to finish.",
    ):
        yield


@contextmanager
def launcher_lock(root: Path):
    """Keep every full-screen entry point exclusive for the TUI's lifetime."""
    with _file_lock(
        root,
        "launcher.lock",
        "A full-screen launcher or installer is already open for this workspace. Close it first.",
    ):
        yield


@dataclass(frozen=True)
class LauncherConfig:
    carla_root: str = ""
    carla_port: int = 2000
    worker_port: int = 8766
    web_port: int = 8765
    traffic_manager_port: int = 8000
    quality: str = "Low"
    offscreen: bool = False
    profile: str = "vision"
    startup_timeout_seconds: int = 180
    enable_experimental: bool = False
    mode: str = "local"
    carla_host: str = "127.0.0.1"
    remote_worker_url: str = ""
    worker_bind: str = "127.0.0.1"
    share_worker: bool = False

    def __post_init__(self) -> None:
        ports = [self.carla_port, self.worker_port, self.web_port, self.traffic_manager_port]
        if any(type(p) is not int or not 1 <= p <= 65533 for p in ports):
            raise LauncherError("Ports must be integers between 1 and 65533.")
        if len(set(ports + [self.carla_port + 1, self.carla_port + 2])) != 6:
            raise LauncherError(
                "CARLA, streaming, Worker, web and Traffic Manager ports must differ."
            )
        if self.quality not in {"Low", "Epic"} or self.profile not in {
            "lean",
            "vision",
            "research",
        }:
            raise LauncherError(
                "Choose Low/Epic quality and lean/vision/research installation profile."
            )
        if type(self.offscreen) is not bool or type(self.enable_experimental) is not bool:
            raise LauncherError("Window and experimental options must be boolean.")
        if (
            type(self.startup_timeout_seconds) is not int
            or not 30 <= self.startup_timeout_seconds <= 900
        ):
            raise LauncherError("Startup timeout must be between 30 and 900 seconds.")
        if not isinstance(self.carla_root, str):
            raise LauncherError("CARLA folder must be a path string.")
        if self.mode not in {"local", "web"}:
            raise LauncherError("Choose local CARLA or web-only mode.")
        if (
            not isinstance(self.carla_host, str)
            or not self.carla_host
            or any(c.isspace() for c in self.carla_host)
            or "://" in self.carla_host
        ):
            raise LauncherError("CARLA host must be an address or 'auto', without a URL scheme.")
        if not isinstance(self.remote_worker_url, str):
            raise LauncherError("Worker URL must be a string.")
        if self.remote_worker_url:
            validate_worker_url(self.remote_worker_url)
        if type(self.share_worker) is not bool:
            raise LauncherError("Worker sharing must be boolean.")
        try:
            address = ipaddress.ip_address(self.worker_bind)
        except ValueError as error:
            raise LauncherError("Select a local IPv4 address for the Worker.") from error
        if (
            address.version != 4
            or not address.is_private
            or address.is_unspecified
            or address.is_multicast
        ):
            raise LauncherError("Worker binding must be loopback or a private LAN IPv4 address.")
        if self.share_worker and address.is_loopback:
            raise LauncherError(
                "Select this computer's LAN address before enabling Worker sharing."
            )

    @classmethod
    def load(cls, root: Path) -> LauncherConfig:
        path = root / ".launcher/config.json"
        raw = read_json(path, {})
        if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(cls)}:
            raise LauncherError("Launcher configuration has unknown or malformed fields.")
        if not path.exists():
            raw["mode"] = default_mode()
            if platform.system() == "Darwin" and platform.machine().lower() == "x86_64":
                raw["profile"] = "lean"
            # Preserve existing remote-console settings on first TUI launch.
            from ..operator.local_entrypoint import load_local_env

            values = load_local_env(root / ".env.local")
            if raw["mode"] == "web":
                raw["carla_host"] = values.get("CARLA_HOST", "127.0.0.1")
                for key, name in [
                    ("CARLA_PORT", "carla_port"),
                    ("CARLA_OPERATOR_PORT", "web_port"),
                ]:
                    if key in values:
                        raw[name] = int(values[key])
                url = values.get("CARLA_WORLD_WORKER_URL", "")
                if url and url != "auto":
                    raw["remote_worker_url"] = validate_worker_url(url)
        return cls(**raw)

    def save(self, root: Path) -> None:
        atomic_json(root / ".launcher/config.json", asdict(self))

    @property
    def web_url(self) -> str:
        return f"http://127.0.0.1:{self.web_port}/"

    @property
    def worker_url(self) -> str:
        if self.mode == "web":
            return self.remote_worker_url.rstrip("/")
        return f"http://{self.worker_bind if self.share_worker else '127.0.0.1'}:{self.worker_port}"


def remote_token(root: Path) -> str:
    path = root / ".launcher/remote-worker.json"
    if path.exists():
        stored = read_json(path, {})
        if not isinstance(stored, dict):
            raise LauncherError("Remote Worker credential file is malformed; paste the token again in Setup.")
        value = stored.get("token", "")
    else:
        from ..operator.local_entrypoint import load_local_env

        value = load_local_env(root / ".env.local").get("CARLA_WORLD_WORKER_TOKEN", "")
    if not isinstance(value, str) or any(c.isspace() for c in value) or len(value) > 4096:
        raise LauncherError("Remote Worker token is invalid; paste it again in Setup.")
    return value


def save_remote_token(root: Path, value: str) -> None:
    if not isinstance(value, str) or any(c.isspace() for c in value) or len(value) > 4096:
        raise LauncherError("Remote Worker token must not contain whitespace.")
    atomic_json(root / ".launcher/remote-worker.json", {"token": value})


def worker_token(root: Path, *, create: bool = False) -> str | None:
    path = root / ".launcher/worker.token"
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise LauncherError("Worker credential must be a regular private file.")
        value = path.read_text(encoding="utf-8").strip()
        if len(value) < 32 or any(c.isspace() for c in value):
            raise LauncherError("Worker credential is malformed; restore its private file.")
        return value
    if not create:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    value = secrets.token_urlsafe(48)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(value + "\n")
    return value


def find_carla(root: Path, configured: str = "") -> Path | None:
    names = ("CarlaUE4.exe",) if os.name == "nt" else ("CarlaUE4.sh",)
    candidates = [Path(configured).expanduser()] if configured else []
    if os.environ.get("CARLA_ROOT"):
        candidates.append(Path(os.environ["CARLA_ROOT"]).expanduser())
    candidates.extend([root / "simulator/CARLA_0.9.16", root.parent / "CARLA_0.9.16"])
    if os.name == "nt":
        candidates.extend([Path("C:/CARLA_0.9.16"), Path("C:/CARLA"), Path("D:/CARLA_0.9.16")])
    else:
        candidates.extend([Path.home() / "CARLA_0.9.16", Path("/opt/carla")])
    for candidate in candidates:
        for name in names:
            path = candidate if candidate.name == name else candidate / name
            if path.is_file():
                return path.resolve()
    return None
