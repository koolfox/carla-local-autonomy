#!/bin/sh
set -eu

BUNDLE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
WORK_DIR=${1:-"$PWD/reproduced-workspace"}

if [ -e "$WORK_DIR" ]; then
  echo "Refusing to overwrite existing reproduction directory: $WORK_DIR" >&2
  exit 2
fi

mkdir -p "$WORK_DIR"
unzip -q "$BUNDLE_DIR/source/source.zip" -d "$WORK_DIR"
cd "$WORK_DIR"

uv sync --frozen --all-groups
uv run pytest -q
uv run ruff format --check .
uv run ruff check .
uv run python -m compileall -q carla_vision carla_yolo
uv run carla-native-preflight --help
uv run carla-operator-ui --help
