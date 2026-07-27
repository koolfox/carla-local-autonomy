"""Build a small, checksum-verified native-collection kit for Windows/Linux."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import stat
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from ..artifacts import RunArtifactTracker, fingerprint_file
from ..scenarios.splits import canonical_map_family
from ..scenarios.verified_plan import load_verified_scenario_plan
from ..verification import verify_research_object
from .worker import select_episodes

NATIVE_HOST_KIT_SCHEMA_VERSION = "1.0"
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SUPPORTED_PYTHON = "3.12"
_SUPPORTED_CARLA = "0.9.16"
_EXPECTED_CARLA_WHEEL_HASHES = frozenset(
    {
        "52b1f2fafb0655e25954f9f6d1e97c211a4da404217fd1d0094b4b7350737c95",
        "5e7d46f13216f0c65ec5ccc6a4b922d0d939be4ef97e07167a4b5f4595865ca6",
        "9ca1e0793a5f453375a8bc93d85d43cc5629b3342bafe10ab8ebb1ef4fb111dc",
        "9d8bca9ea3a3bccd691536b17f6f278335ce9b6af7883612aa0800929b388ed1",
        "c323a7b1f8ad3ac81cfac79f5d96c9cac8afa6d0a768fa2fffa46bb336945c4f",
        "f1f22d153b1a43b579d90e816b7016ca19f1483ca64cbc5ebbcd4dbf325e3356",
    }
)


def _strict_keys(
    raw: Mapping[str, Any],
    required: set[str],
    *,
    context: str,
) -> None:
    actual = {str(key) for key in raw}
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing:
        raise ValueError(f"{context} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}")


def _identifier(raw: Any, *, field: str) -> str:
    value = str(raw)
    if value in {".", ".."} or not _ID_PATTERN.fullmatch(value):
        raise ValueError(f"{field} must be 1-128 characters using letters, digits, '.', '_' or '-'")
    return value


def _positive_integer(raw: Any, *, field: str, maximum: int = 65535) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or not 0 < raw <= maximum:
        raise ValueError(f"{field} must be an integer in [1, {maximum}]")
    return raw


def _string_array(raw: Any, *, field: str) -> tuple[str, ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(f"{field} must be an array")
    result = tuple(str(value) for value in raw)
    if not result or any(not value for value in result) or len(result) != len(set(result)):
        raise ValueError(f"{field} must contain unique non-empty strings")
    return result


@dataclass(frozen=True)
class NativeHostKitConfig:
    kit_id: str
    title: str
    scenario_plan: str
    dataset_id: str
    host: str
    port: int
    partitions: tuple[str, ...]
    max_episodes: int
    python_version: str
    carla_version: str
    requirements_windows: str
    requirements_linux: str
    schema_version: str = NATIVE_HOST_KIT_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        _strict_keys(
            raw,
            {
                "schema_version",
                "kit_id",
                "title",
                "scenario_plan",
                "dataset_id",
                "host",
                "port",
                "partitions",
                "max_episodes",
                "python_version",
                "carla_version",
                "requirements",
            },
            context="native host kit config",
        )
        if raw["schema_version"] != NATIVE_HOST_KIT_SCHEMA_VERSION:
            raise ValueError("native host kit schema version is unsupported")
        requirements = raw["requirements"]
        if not isinstance(requirements, Mapping):
            raise TypeError("native host kit requirements must be an object")
        _strict_keys(
            requirements,
            {"windows", "linux"},
            context="native host kit requirements",
        )
        title = str(raw["title"]).strip()
        host = str(raw["host"]).strip()
        if not title or not host:
            raise ValueError("native host kit title and host cannot be empty")
        python_version = str(raw["python_version"])
        carla_version = str(raw["carla_version"])
        partitions = _string_array(raw["partitions"], field="partitions")
        max_episodes = _positive_integer(
            raw["max_episodes"],
            field="max_episodes",
            maximum=10000,
        )
        if python_version != _SUPPORTED_PYTHON:
            raise ValueError(f"native host kit currently requires CPython {_SUPPORTED_PYTHON}")
        if carla_version != _SUPPORTED_CARLA:
            raise ValueError(f"native host kit currently requires CARLA {_SUPPORTED_CARLA}")
        if partitions != ("train",) or max_episodes != 1:
            raise ValueError(
                "the integration native host kit is intentionally limited to one train episode"
            )
        return cls(
            kit_id=_identifier(raw["kit_id"], field="kit_id"),
            title=title,
            scenario_plan=str(raw["scenario_plan"]),
            dataset_id=_identifier(raw["dataset_id"], field="dataset_id"),
            host=host,
            port=_positive_integer(raw["port"], field="port"),
            partitions=partitions,
            max_episodes=max_episodes,
            python_version=python_version,
            carla_version=carla_version,
            requirements_windows=str(requirements["windows"]),
            requirements_linux=str(requirements["linux"]),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kit_id": self.kit_id,
            "title": self.title,
            "scenario_plan": self.scenario_plan,
            "dataset_id": self.dataset_id,
            "host": self.host,
            "port": self.port,
            "partitions": list(self.partitions),
            "max_episodes": self.max_episodes,
            "python_version": self.python_version,
            "carla_version": self.carla_version,
            "requirements": {
                "windows": self.requirements_windows,
                "linux": self.requirements_linux,
            },
        }


def load_native_host_kit_config(path: str | Path) -> NativeHostKitConfig:
    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read native host kit config: {error}") from error
    if not isinstance(payload, Mapping):
        raise TypeError("native host kit config must contain a JSON object")
    return NativeHostKitConfig.from_mapping(payload)


def _resolve_from_config(config_path: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return candidate.resolve(strict=True)


def _atomic_bytes(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    _atomic_bytes(path, _json_bytes(payload))


def _csv_bytes(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(fields),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _relative_reference(root: Path, path: Path) -> dict[str, Any]:
    reference = fingerprint_file(path)
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": reference["sha256"],
        "size_bytes": reference["size_bytes"],
    }


def _validate_hashed_requirements(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    package_blocks: list[tuple[str, str]] = []
    current_name: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        if line and not line[0].isspace() and "==" in line:
            if current_name is not None:
                package_blocks.append((current_name, "\n".join(current_lines)))
            current_name = line.split("==", 1)[0].strip().casefold()
            current_lines = [line]
        elif current_name is not None:
            current_lines.append(line)
    if current_name is not None:
        package_blocks.append((current_name, "\n".join(current_lines)))
    if not package_blocks:
        raise ValueError(f"requirements lock contains no packages: {path}")
    package_names = [name for name, _block in package_blocks]
    if len(package_names) != len(set(package_names)):
        raise ValueError(f"requirements lock repeats a package: {path}")
    for name, block in package_blocks:
        if "--hash=sha256:" not in block:
            raise ValueError(f"requirements package {name!r} has no SHA-256: {path}")
    try:
        carla_block = next(block for name, block in package_blocks if name == "carla")
    except StopIteration as error:
        raise ValueError(f"requirements lock does not contain carla: {path}") from error
    carla_hashes = frozenset(re.findall(r"--hash=sha256:([0-9a-f]{64})", carla_block))
    if carla_hashes != _EXPECTED_CARLA_WHEEL_HASHES:
        raise ValueError(f"CARLA 0.9.16 wheel hashes differ from the frozen set: {path}")
    if not carla_block.startswith(f"carla=={_SUPPORTED_CARLA} "):
        raise ValueError(f"requirements lock does not pin CARLA {_SUPPORTED_CARLA}: {path}")
    return {
        "package_count": len(package_blocks),
        "packages": package_names,
        "carla_wheel_hash_count": len(carla_hashes),
    }


def _safe_payload_path(value: str) -> str:
    raw = Path(value)
    if (
        not value
        or raw.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in raw.parts)
    ):
        raise ValueError(f"unsafe payload path: {value!r}")
    return raw.as_posix()


def _add_payload(
    payloads: dict[str, tuple[bytes, int]],
    path: str,
    data: bytes,
    *,
    mode: int = 0o644,
) -> None:
    normalized = _safe_payload_path(path)
    if normalized in payloads:
        raise ValueError(f"duplicate native host kit payload path: {normalized}")
    if mode not in {0o644, 0o755}:
        raise ValueError(f"unsupported payload mode for {normalized}")
    payloads[normalized] = (data, mode)


def _source_payloads(repository_root: Path) -> dict[str, tuple[bytes, int]]:
    package_root = repository_root / "carla_vision"
    if not package_root.is_dir():
        raise FileNotFoundError(f"carla_vision package is missing from {repository_root}")
    payloads: dict[str, tuple[bytes, int]] = {}
    for source in sorted(
        package_root.rglob("*.py"),
        key=lambda item: item.relative_to(repository_root).as_posix().encode("utf-8"),
    ):
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"native host source cannot contain a symlink: {source}")
        relative = source.relative_to(repository_root).as_posix()
        _add_payload(payloads, f"source/{relative}", source.read_bytes())
    if not payloads:
        raise ValueError("native host source selection is empty")
    return payloads


def _scenario_payloads(plan_dir: Path, run_id: str) -> dict[str, tuple[bytes, int]]:
    payloads: dict[str, tuple[bytes, int]] = {}
    for source in sorted(
        plan_dir.rglob("*"),
        key=lambda item: item.relative_to(plan_dir).as_posix().encode("utf-8"),
    ):
        if source.is_symlink():
            raise ValueError(f"scenario plan cannot contain a symlink: {source}")
        if not source.is_file():
            continue
        relative = source.relative_to(plan_dir).as_posix()
        _add_payload(
            payloads,
            f"scenario-plan/{run_id}/{relative}",
            source.read_bytes(),
        )
    return payloads


def _capture_count(episode: Any) -> int:
    capture = episode.recipe.capture
    return 1 + (capture.duration_ticks - 1) // capture.capture_every_ticks


def _render_verify_payload() -> str:
    return '''#!/usr/bin/env python3
"""Verify every immutable file in an extracted native-host payload."""
from __future__ import annotations

import hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
index = root / "checksums.sha256"
entries = 0
for line_number, line in enumerate(index.read_text(encoding="utf-8").splitlines(), 1):
    digest, separator, relative = line.partition("  ")
    raw = Path(relative)
    if (
        not separator
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or raw.is_absolute()
        or any(part in {"", ".", ".."} for part in raw.parts)
    ):
        raise SystemExit(f"invalid checksum entry at line {line_number}")
    path = (root / raw).resolve(strict=True)
    path.relative_to(root)
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"unsafe payload file: {relative}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit(f"payload checksum mismatch: {relative}")
    entries += 1
print(f"native host payload verified: {entries} files")
'''


def _render_bootstrap_sh() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

KIT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$KIT_ROOT"
PYTHON_BIN=${PYTHON_BIN:-python3.12}

"$PYTHON_BIN" scripts/verify_payload.py
"$PYTHON_BIN" - <<'PY'
import platform
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit("native host kit requires CPython 3.12")
if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
    raise SystemExit("Linux native host kit requires x86_64 Linux")
print(platform.platform(), sys.version)
PY

if [ -e .venv ]; then
  echo "Refusing to overwrite existing .venv" >&2
  exit 2
fi
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements/linux-py312.txt
.venv/bin/python - <<'PY'
import carla
import cv2
import matplotlib
import msgpack
import numpy
print("carla", getattr(carla, "__version__", "unknown"), carla.__file__)
print("opencv", cv2.__version__)
print("matplotlib", matplotlib.__version__)
print("msgpack", msgpack.version)
print("numpy", numpy.__version__)
PY
"""


def _render_bootstrap_ps1() -> str:
    return """param()
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$KitRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $KitRoot

py -3.12 scripts\\verify_payload.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
py -3.12 -c "import platform,sys; assert sys.version_info[:2] == (3,12), 'native host kit requires CPython 3.12'; assert platform.system() == 'Windows' and platform.machine().lower() in {'amd64','x86_64'}, 'native host kit requires x86-64 Windows'; print(platform.platform(), sys.version)"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (Test-Path ".venv") {
    throw "Refusing to overwrite existing .venv"
}
py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\\.venv\\Scripts\\python.exe -m pip install --require-hashes -r requirements\\windows-py312.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\\.venv\\Scripts\\python.exe -c "import carla,cv2,matplotlib,msgpack,numpy; print('carla',getattr(carla,'__version__','unknown'),carla.__file__); print('opencv',cv2.__version__); print('matplotlib',matplotlib.__version__); print('msgpack',msgpack.version); print('numpy',numpy.__version__)"
exit $LASTEXITCODE
"""


def _render_preflight_sh(
    *,
    default_run_id: str,
    scenario_run_id: str,
    dataset_id: str,
    host: str,
    port: int,
) -> str:
    return f'''#!/usr/bin/env bash
set -euo pipefail

KIT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$KIT_ROOT"
RUN_ID=${{1:-{default_run_id}}}
if [ "$#" -gt 0 ]; then shift; fi
if [ -e "runs/$RUN_ID" ]; then
  echo "Refusing to overwrite existing preflight: runs/$RUN_ID" >&2
  exit 2
fi
PYTHONPATH="$KIT_ROOT/source" .venv/bin/python -m carla_vision.native.preflight \\
  --scenario-plan "scenario-plan/{scenario_run_id}" \\
  --dataset-id "{dataset_id}" \\
  --runs-root runs \\
  --run-id "$RUN_ID" \\
  --host "{host}" \\
  --port "{port}" \\
  --partition train \\
  --max-episodes 1 \\
  "$@"
'''


def _render_preflight_ps1(
    *,
    default_run_id: str,
    scenario_run_id: str,
    dataset_id: str,
    host: str,
    port: int,
) -> str:
    return f'''param(
    [string]$RunId = "{default_run_id}",
    [switch]$ConfirmWorldReload,
    [switch]$ConfirmExclusiveTickOwner
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$KitRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $KitRoot
if (Test-Path (Join-Path "runs" $RunId)) {{
    throw "Refusing to overwrite existing preflight: runs/$RunId"
}}
$env:PYTHONPATH = (Join-Path $KitRoot "source")
$Arguments = @(
    "-m", "carla_vision.native.preflight",
    "--scenario-plan", "scenario-plan/{scenario_run_id}",
    "--dataset-id", "{dataset_id}",
    "--runs-root", "runs",
    "--run-id", $RunId,
    "--host", "{host}",
    "--port", "{port}",
    "--partition", "train",
    "--max-episodes", "1"
)
if ($ConfirmWorldReload) {{ $Arguments += "--confirm-world-reload" }}
if ($ConfirmExclusiveTickOwner) {{ $Arguments += "--confirm-exclusive-tick-owner" }}
& .\\.venv\\Scripts\\python.exe @Arguments
exit $LASTEXITCODE
'''


def _render_collect_sh(
    *,
    scenario_run_id: str,
    dataset_id: str,
    host: str,
    port: int,
) -> str:
    return f'''#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 READY_PREFLIGHT_RUN_ID I_CONFIRM_WORLD_RELOAD_AND_EXCLUSIVE_TICK" >&2
  exit 2
fi
if [ "$2" != "I_CONFIRM_WORLD_RELOAD_AND_EXCLUSIVE_TICK" ]; then
  echo "explicit destructive confirmation token is required" >&2
  exit 3
fi
KIT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$KIT_ROOT"
if [ -e "datasets/{dataset_id}" ]; then
  echo "Refusing to overwrite existing dataset: datasets/{dataset_id}" >&2
  exit 4
fi
export PYTHONPATH="$KIT_ROOT/source"
.venv/bin/python -m carla_vision.native.host_gate \\
  --kit-plan kit-plan.json \\
  --preflight "runs/$1"
.venv/bin/python -m carla_vision.native.worker \\
  --scenario-plan "scenario-plan/{scenario_run_id}" \\
  --dataset-id "{dataset_id}" \\
  --datasets-root datasets \\
  --host "{host}" \\
  --port "{port}" \\
  --partition train \\
  --max-episodes 1 \\
  --timeout 30 \\
  --sensor-timeout 10 \\
  --acknowledge-exclusive-tick-owner
'''


def _render_collect_ps1(
    *,
    scenario_run_id: str,
    dataset_id: str,
    host: str,
    port: int,
) -> str:
    return f'''param(
    [Parameter(Mandatory=$true)][string]$ReadyPreflightRunId,
    [Parameter(Mandatory=$true)]
    [ValidateSet("I_CONFIRM_WORLD_RELOAD_AND_EXCLUSIVE_TICK")]
    [string]$Confirmation
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$KitRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $KitRoot
if (Test-Path "datasets\\{dataset_id}") {{
    throw "Refusing to overwrite existing dataset: datasets\\{dataset_id}"
}}
$env:PYTHONPATH = (Join-Path $KitRoot "source")
& .\\.venv\\Scripts\\python.exe -m carla_vision.native.host_gate `
    --kit-plan kit-plan.json `
    --preflight "runs\\$ReadyPreflightRunId"
if ($LASTEXITCODE -ne 0) {{ exit $LASTEXITCODE }}
& .\\.venv\\Scripts\\python.exe -m carla_vision.native.worker `
    --scenario-plan "scenario-plan\\{scenario_run_id}" `
    --dataset-id "{dataset_id}" `
    --datasets-root datasets `
    --host "{host}" `
    --port "{port}" `
    --partition train `
    --max-episodes 1 `
    --timeout 30 `
    --sensor-timeout 10 `
    --acknowledge-exclusive-tick-owner
exit $LASTEXITCODE
'''


def _render_postrun_sh(dataset_id: str) -> str:
    return f'''#!/usr/bin/env bash
set -euo pipefail

KIT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$KIT_ROOT"
QA_RUN_ID="{dataset_id}-qa"
if [ -e "runs/$QA_RUN_ID" ]; then
  echo "Refusing to overwrite existing QA run: runs/$QA_RUN_ID" >&2
  exit 2
fi
export PYTHONPATH="$KIT_ROOT/source"
.venv/bin/python -m carla_vision.verification \\
  "datasets/{dataset_id}" --reject-unregistered
.venv/bin/python -m carla_vision.dataset.qa \\
  --dataset "datasets/{dataset_id}" \\
  --runs-root runs \\
  --run-id "$QA_RUN_ID"
.venv/bin/python -m carla_vision.verification \\
  "runs/$QA_RUN_ID" --reject-unregistered
'''


def _render_postrun_ps1(dataset_id: str) -> str:
    return f'''param()
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$KitRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $KitRoot
$QaRunId = "{dataset_id}-qa"
if (Test-Path (Join-Path "runs" $QaRunId)) {{
    throw "Refusing to overwrite existing QA run: runs\\$QaRunId"
}}
$env:PYTHONPATH = (Join-Path $KitRoot "source")
& .\\.venv\\Scripts\\python.exe -m carla_vision.verification `
    "datasets\\{dataset_id}" --reject-unregistered
if ($LASTEXITCODE -ne 0) {{ exit $LASTEXITCODE }}
& .\\.venv\\Scripts\\python.exe -m carla_vision.dataset.qa `
    --dataset "datasets\\{dataset_id}" `
    --runs-root runs `
    --run-id $QaRunId
if ($LASTEXITCODE -ne 0) {{ exit $LASTEXITCODE }}
& .\\.venv\\Scripts\\python.exe -m carla_vision.verification `
    "runs\\$QaRunId" --reject-unregistered
exit $LASTEXITCODE
'''


def _render_readme(
    *,
    kit_id: str,
    scenario_run_id: str,
    dataset_id: str,
    host: str,
    port: int,
    capture_count: int,
) -> str:
    return f"""# Portable native CARLA collection kit

Kit ID: `{kit_id}`  
CARLA endpoint: `{host}:{port}`  
Scenario plan: `{scenario_run_id}`  
Dataset target: `{dataset_id}`  
Planned captures: `{capture_count}`

This payload is intentionally limited to one native 50-frame integration
episode. It contains the exact project Python sources needed for collection,
the verified scenario plan, hash-pinned CPython 3.12 dependencies for Windows
x86-64 and Linux x86-64, and guarded scripts. It contains no CARLA server,
model weight, secret, or previously collected dataset.

## 1. Verify and bootstrap

Extract the ZIP into a new directory. Do not run it from inside the archive.

Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\\scripts\\bootstrap.ps1
```

Ubuntu/Linux:

```bash
chmod +x scripts/*.sh scripts/verify_payload.py
./scripts/bootstrap.sh
```

Both bootstrap scripts first verify `checksums.sha256`, require CPython 3.12
on x86-64, create a new `.venv`, and install every dependency with
`--require-hashes`.

## 2. Read-only preflight

First run without manual confirmations:

```powershell
.\\scripts\\preflight.ps1 -RunId native-preflight-pilot-host-v1
```

```bash
./scripts/preflight.sh native-preflight-pilot-host-v1
```

Inspect `runs/native-preflight-pilot-host-v1/report.md`. Stop all other CARLA
clients, confirm that the current world is disposable, and confirm this worker
will be the only `world.tick()` owner. Then create a new preflight ID and
explicitly record both confirmations:

```powershell
.\\scripts\\preflight.ps1 -RunId native-preflight-pilot-host-ready-v1 `
  -ConfirmWorldReload -ConfirmExclusiveTickOwner
```

```bash
./scripts/preflight.sh native-preflight-pilot-host-ready-v1 \
  --confirm-world-reload --confirm-exclusive-tick-owner
```

Do not continue unless its summary says
`ready_for_native_execution: true`, `read_only: true`, and
`simulator_mutated: false`.

## 3. One destructive collection

The collection script accepts only a verified ready preflight that exactly
matches this kit. It also requires the literal confirmation token below.
Map loading/reloading destroys current world actors.

```powershell
.\\scripts\\collect.ps1 `
  -ReadyPreflightRunId native-preflight-pilot-host-ready-v1 `
  -Confirmation I_CONFIRM_WORLD_RELOAD_AND_EXCLUSIVE_TICK
```

```bash
./scripts/collect.sh native-preflight-pilot-host-ready-v1 \
  I_CONFIRM_WORLD_RELOAD_AND_EXCLUSIVE_TICK
```

## 4. Verify and audit before transfer

```powershell
.\\scripts\\postrun.ps1
```

```bash
./scripts/postrun.sh
```

Copy both `datasets/{dataset_id}` and `runs/{dataset_id}-qa` back to the main
research workspace. Verify them again after transfer. Do not train or scale
collection until the 50-frame QA montage, frame IDs, labels, actor cleanup,
and asynchronous-world restoration are accepted.
"""


def _payload_checksum_index(payloads: Mapping[str, tuple[bytes, int]]) -> bytes:
    return "".join(
        f"{hashlib.sha256(payloads[path][0]).hexdigest()}  {path}\n"
        for path in sorted(payloads, key=lambda item: item.encode("utf-8"))
    ).encode("utf-8")


def _payload_inventory(payloads: Mapping[str, tuple[bytes, int]]) -> list[dict[str, Any]]:
    return [
        {
            "path": path,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "archive_mode": "0755" if mode == 0o755 else "0644",
        }
        for path, (data, mode) in sorted(
            payloads.items(),
            key=lambda item: item[0].encode("utf-8"),
        )
    ]


def _write_payload_archive(
    path: Path,
    payloads: Mapping[str, tuple[bytes, int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_STORED,
            strict_timestamps=True,
        ) as archive:
            for relative, (payload, mode) in sorted(
                payloads.items(),
                key=lambda item: item[0].encode("utf-8"),
            ):
                info = zipfile.ZipInfo(relative, date_time=_ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | mode) << 16
                info.flag_bits |= 0x800
                archive.writestr(info, payload)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_checksum_index(root: Path, paths: Sequence[Path], output: Path) -> None:
    entries = []
    for path in paths:
        reference = fingerprint_file(path)
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        entries.append((relative, str(reference["sha256"])))
    entries.sort(key=lambda item: item[0].encode("utf-8"))
    _atomic_bytes(
        output,
        "".join(f"{digest}  {relative}\n" for relative, digest in entries).encode("utf-8"),
    )


def build_native_host_kit(
    *,
    config_path: str | Path,
    kits_root: str | Path = "native_kits",
    repository_root: str | Path | None = None,
    cli_args: Sequence[str] | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path).expanduser().resolve(strict=True)
    config = load_native_host_kit_config(config_file)
    repository = Path(repository_root or Path.cwd()).expanduser().resolve(strict=True)
    scenario_plan_path = _resolve_from_config(config_file, config.scenario_plan)
    windows_requirements = _resolve_from_config(
        config_file,
        config.requirements_windows,
    )
    linux_requirements = _resolve_from_config(
        config_file,
        config.requirements_linux,
    )
    windows_lock = _validate_hashed_requirements(windows_requirements)
    linux_lock = _validate_hashed_requirements(linux_requirements)
    if linux_lock != windows_lock:
        raise ValueError("Windows and Linux native dependency lock metadata differs")
    if windows_requirements.read_bytes() != linux_requirements.read_bytes():
        raise ValueError(
            "Windows and Linux native locks must resolve to identical pinned requirements"
        )

    plan_verification = verify_research_object(
        scenario_plan_path,
        verify_references=True,
        deep=True,
        reject_unregistered=True,
    )
    plan = load_verified_scenario_plan(scenario_plan_path)
    if plan.suite.carla_version != config.carla_version:
        raise ValueError("scenario plan CARLA version differs from the kit config")
    episodes = select_episodes(
        plan,
        partitions=config.partitions,
        max_episodes=config.max_episodes,
    )
    planned_capture_count = sum(_capture_count(episode) for episode in episodes)
    default_preflight_run_id = f"native-preflight-{config.kit_id}"

    manifest_reference = fingerprint_file(scenario_plan_path / "manifest.json")
    kit_plan = {
        "schema_version": NATIVE_HOST_KIT_SCHEMA_VERSION,
        "object_type": "native_host_kit_payload",
        "kit_id": config.kit_id,
        "title": config.title,
        "target": {
            "python_implementation": "CPython",
            "python_version": config.python_version,
            "systems": ["Windows", "Linux"],
            "machine": "x86_64",
            "carla_version": config.carla_version,
        },
        "endpoint": {"host": config.host, "port": config.port},
        "dataset": {
            "dataset_id": config.dataset_id,
            "output_directory": f"datasets/{config.dataset_id}",
        },
        "scenario_plan": {
            "directory": f"scenario-plan/{plan.run_id}",
            "run_id": plan.run_id,
            **{key: manifest_reference[key] for key in ("sha256", "size_bytes")},
        },
        "selection": {
            "episode_ids": [episode.episode_id for episode in episodes],
            "partitions": list(config.partitions),
            "selected_episode_count": len(episodes),
            "planned_capture_count": planned_capture_count,
            "map_families": sorted(
                {canonical_map_family(episode.recipe.map_name) for episode in episodes}
            ),
        },
        "requirements": {
            "windows": "requirements/windows-py312.txt",
            "linux": "requirements/linux-py312.txt",
            "package_count": windows_lock["package_count"],
            "carla_wheel_hash_count": windows_lock["carla_wheel_hash_count"],
            "all_packages_hash_pinned": True,
        },
        "safety": {
            "build_contacted_simulator": False,
            "build_mutated_simulator": False,
            "preflight_is_read_only": True,
            "collection_requires_verified_ready_preflight": True,
            "collection_requires_literal_confirmation_token": True,
            "map_reload_destroys_existing_world_actors": True,
            "exclusive_world_tick_owner_required": True,
            "full_thesis_plan_included": False,
        },
    }

    payloads = _source_payloads(repository)
    for relative, entry in _scenario_payloads(
        scenario_plan_path,
        plan.run_id,
    ).items():
        _add_payload(payloads, relative, entry[0], mode=entry[1])
    _add_payload(
        payloads,
        "requirements/windows-py312.txt",
        windows_requirements.read_bytes(),
    )
    _add_payload(
        payloads,
        "requirements/linux-py312.txt",
        linux_requirements.read_bytes(),
    )
    _add_payload(payloads, "kit-plan.json", _json_bytes(kit_plan))
    _add_payload(
        payloads,
        "scripts/verify_payload.py",
        _render_verify_payload().encode("utf-8"),
        mode=0o755,
    )
    _add_payload(
        payloads,
        "scripts/bootstrap.sh",
        _render_bootstrap_sh().encode("utf-8"),
        mode=0o755,
    )
    _add_payload(
        payloads,
        "scripts/bootstrap.ps1",
        _render_bootstrap_ps1().encode("utf-8"),
    )
    _add_payload(
        payloads,
        "scripts/preflight.sh",
        _render_preflight_sh(
            default_run_id=default_preflight_run_id,
            scenario_run_id=plan.run_id,
            dataset_id=config.dataset_id,
            host=config.host,
            port=config.port,
        ).encode("utf-8"),
        mode=0o755,
    )
    _add_payload(
        payloads,
        "scripts/preflight.ps1",
        _render_preflight_ps1(
            default_run_id=default_preflight_run_id,
            scenario_run_id=plan.run_id,
            dataset_id=config.dataset_id,
            host=config.host,
            port=config.port,
        ).encode("utf-8"),
    )
    _add_payload(
        payloads,
        "scripts/collect.sh",
        _render_collect_sh(
            scenario_run_id=plan.run_id,
            dataset_id=config.dataset_id,
            host=config.host,
            port=config.port,
        ).encode("utf-8"),
        mode=0o755,
    )
    _add_payload(
        payloads,
        "scripts/collect.ps1",
        _render_collect_ps1(
            scenario_run_id=plan.run_id,
            dataset_id=config.dataset_id,
            host=config.host,
            port=config.port,
        ).encode("utf-8"),
    )
    _add_payload(
        payloads,
        "scripts/postrun.sh",
        _render_postrun_sh(config.dataset_id).encode("utf-8"),
        mode=0o755,
    )
    _add_payload(
        payloads,
        "scripts/postrun.ps1",
        _render_postrun_ps1(config.dataset_id).encode("utf-8"),
    )
    readme = _render_readme(
        kit_id=config.kit_id,
        scenario_run_id=plan.run_id,
        dataset_id=config.dataset_id,
        host=config.host,
        port=config.port,
        capture_count=planned_capture_count,
    )
    _add_payload(payloads, "README.md", readme.encode("utf-8"))
    _add_payload(payloads, "checksums.sha256", _payload_checksum_index(payloads))
    inventory = _payload_inventory(payloads)

    input_refs = [
        {
            "kind": "verified_scenario_plan",
            "run_id": plan.run_id,
            "sha256": manifest_reference["sha256"],
            "size_bytes": manifest_reference["size_bytes"],
        },
        {
            "kind": "windows_dependency_lock",
            "sha256": fingerprint_file(windows_requirements)["sha256"],
            "size_bytes": fingerprint_file(windows_requirements)["size_bytes"],
        },
        {
            "kind": "linux_dependency_lock",
            "sha256": fingerprint_file(linux_requirements)["sha256"],
            "size_bytes": fingerprint_file(linux_requirements)["size_bytes"],
        },
    ]
    tracker = RunArtifactTracker(
        kits_root,
        run_id=config.kit_id,
        cli_args=cli_args,
        config={
            "schema_version": NATIVE_HOST_KIT_SCHEMA_VERSION,
            "object_type": "native_host_kit",
            "kit": config.as_dict(),
        },
        repository_root=repository,
        carla_endpoint={"host": config.host, "port": config.port},
        carla_version=config.carla_version,
        input_refs=input_refs,
        package_names=("numpy", "opencv-python", "msgpack", "matplotlib", "carla"),
    )
    with tracker:
        config_output = tracker.artifact_path("kit_config.json")
        descriptor_path = tracker.artifact_path("kit.json")
        archive_path = tracker.artifact_path("payload/native-host-kit.zip")
        inventory_json_path = tracker.artifact_path("payload/inventory.json")
        inventory_csv_path = tracker.artifact_path("payload/inventory.csv")
        readme_path = tracker.artifact_path("README.md")
        checksum_path = tracker.artifact_path("checksums.sha256")

        _write_json(config_output, config.as_dict())
        _write_payload_archive(archive_path, payloads)
        _write_json(
            inventory_json_path,
            {
                "schema_version": NATIVE_HOST_KIT_SCHEMA_VERSION,
                "object_type": "native_host_kit_payload_inventory",
                "file_count": len(inventory),
                "total_bytes": sum(int(row["size_bytes"]) for row in inventory),
                "files": inventory,
            },
        )
        _atomic_bytes(
            inventory_csv_path,
            _csv_bytes(
                inventory,
                ("path", "sha256", "size_bytes", "archive_mode"),
            ),
        )
        _atomic_bytes(readme_path, readme.encode("utf-8"))

        descriptor = {
            "schema_version": NATIVE_HOST_KIT_SCHEMA_VERSION,
            "object_type": "native_host_kit_release",
            "status": "complete",
            "kit_id": config.kit_id,
            "title": config.title,
            "target": dict(kit_plan["target"]),
            "endpoint": dict(kit_plan["endpoint"]),
            "dataset": dict(kit_plan["dataset"]),
            "scenario_plan": {
                **dict(kit_plan["scenario_plan"]),
                "source_verification": {
                    "status": plan_verification.status,
                    "artifact_count": plan_verification.artifact_count,
                    "deep_verification": dict(plan_verification.deep_verification),
                    "unregistered_file_count": plan_verification.unregistered_file_count,
                },
            },
            "selection": dict(kit_plan["selection"]),
            "requirements": dict(kit_plan["requirements"]),
            "safety": dict(kit_plan["safety"]),
            "payload_file_count": len(inventory),
            "payload_total_bytes": sum(int(row["size_bytes"]) for row in inventory),
            "payload_source_file_count": sum(
                str(row["path"]).startswith("source/") for row in inventory
            ),
            "configuration": _relative_reference(tracker.run_dir, config_output),
            "payload_archive": _relative_reference(tracker.run_dir, archive_path),
            "payload_inventory_json": _relative_reference(
                tracker.run_dir,
                inventory_json_path,
            ),
            "payload_inventory_csv": _relative_reference(
                tracker.run_dir,
                inventory_csv_path,
            ),
            "readme": _relative_reference(tracker.run_dir, readme_path),
            "generation": {
                "archive_format": "zip-stored-fixed-metadata-v1",
                "archive_timestamp": "1980-01-01T00:00:00Z",
                "payload_symlinks_allowed": False,
                "simulator_contacted": False,
                "simulator_mutated": False,
                "secrets_captured": False,
            },
        }
        _write_json(descriptor_path, descriptor)

        payload_paths = [
            config_output,
            descriptor_path,
            archive_path,
            inventory_json_path,
            inventory_csv_path,
            readme_path,
        ]
        _write_checksum_index(tracker.run_dir, payload_paths, checksum_path)
        roles = {
            config_output: "native_host_kit_configuration",
            descriptor_path: "native_host_kit_release_manifest",
            archive_path: "native_host_kit_payload_archive",
            inventory_json_path: "native_host_kit_payload_inventory_json",
            inventory_csv_path: "native_host_kit_payload_inventory_csv",
            readme_path: "native_host_kit_readme",
            checksum_path: "native_host_kit_checksum_index",
        }
        for path, role in roles.items():
            tracker.register_artifact(
                path,
                role=role,
                metadata={
                    "kit_id": config.kit_id,
                    "dataset_id": config.dataset_id,
                    "planned_capture_count": planned_capture_count,
                },
            )

    return {
        "kit_id": config.kit_id,
        "kit_dir": str(tracker.run_dir),
        "manifest": str(tracker.manifest_path),
        "payload_archive": str(archive_path),
        "descriptor": descriptor,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic, checksum-verified native CARLA collection "
            "kit for CPython 3.12 on Windows/Linux x86-64"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--kits-root", default="native_kits")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_native_host_kit(
        config_path=args.config,
        kits_root=args.kits_root,
        cli_args=vars(args),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


__all__ = [
    "NATIVE_HOST_KIT_SCHEMA_VERSION",
    "NativeHostKitConfig",
    "build_native_host_kit",
    "load_native_host_kit_config",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    raise SystemExit(main())
