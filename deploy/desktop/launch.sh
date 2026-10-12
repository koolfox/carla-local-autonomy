#!/bin/sh
set -eu
DesktopScriptDir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ApplicationRoot=$(CDPATH= cd -- "$DesktopScriptDir/../.." && pwd)
if [ ! -x "$ApplicationRoot/.venv/bin/python" ]; then
    sh "$DesktopScriptDir/install.sh" --in-place --no-launch
fi
cd "$ApplicationRoot"
export PYTHONUTF8=1
exec "$ApplicationRoot/.venv/bin/python" -m carla_vision.launcher --workspace "$ApplicationRoot" tui
