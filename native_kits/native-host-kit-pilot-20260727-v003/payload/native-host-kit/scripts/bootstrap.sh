#!/usr/bin/env bash
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
