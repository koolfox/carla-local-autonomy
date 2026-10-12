#!/bin/sh
set -eu
DesktopScriptDir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ApplicationRoot=$(CDPATH= cd -- "$DesktopScriptDir/../.." && pwd)
if [ ! -x "$ApplicationRoot/.venv/bin/python" ] || ! "$ApplicationRoot/.venv/bin/python" -c 'import textual' >/dev/null 2>&1; then
    sh "$DesktopScriptDir/install.sh" --in-place --no-launch
fi
cd "$ApplicationRoot"
export PYTHONUTF8=1
exec "$ApplicationRoot/.venv/bin/python" -m carla_vision.launcher --workspace "$ApplicationRoot" tui
