#!/bin/sh
set -eu
case "$(uname -s):$(uname -m)" in
    Linux:x86_64) ;;
    *) printf '%s\n' 'Same-computer setup supports Linux x86-64 (Windows has Install.cmd).' >&2; exit 1 ;;
esac
DesktopScriptDir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SourceRoot=$(CDPATH= cd -- "$DesktopScriptDir/../.." && pwd)
NoLaunch=false
InPlace=false
Destination=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --in-place) InPlace=true ;;
        --no-launch) NoLaunch=true ;;
        --destination)
            [ "$#" -ge 2 ] || { printf '%s\n' '--destination needs a directory.' >&2; exit 1; }
            Destination=$2; shift ;;
        *) printf '%s\n' 'Usage: install.sh [--in-place] [--no-launch] [--destination DIRECTORY]' >&2; exit 1 ;;
    esac
    shift
done
ApplicationRoot=$SourceRoot
if [ "$InPlace" = false ] && [ -f "$SourceRoot/installer-files.json" ]; then
    if [ -z "$Destination" ]; then
        XdgData=$(printenv XDG_DATA_HOME || true)
        if [ -z "$XdgData" ]; then XdgData="$HOME/.local/share"; fi
        Destination="$XdgData/carla-vision"
    fi
    mkdir -p "$Destination/app"
    ApplicationRoot=$(CDPATH= cd -- "$Destination/app" && pwd)
fi
command -v curl >/dev/null 2>&1 || { printf '%s\n' 'Install curl with your OS package manager first.' >&2; exit 1; }
command -v tar >/dev/null 2>&1 || { printf '%s\n' 'Install tar with your OS package manager first.' >&2; exit 1; }
command -v sha256sum >/dev/null 2>&1 || { printf '%s\n' 'Install coreutils with your OS package manager first.' >&2; exit 1; }
if [ -L "$ApplicationRoot/.launcher" ]; then
    printf '%s\n' 'The launcher state directory must not be a symbolic link.' >&2
    exit 1
fi
mkdir -p "$ApplicationRoot/.launcher/uv"
Uv="$ApplicationRoot/.launcher/uv/uv"
if [ ! -x "$Uv" ]; then
    UvStage=$(mktemp -d "$ApplicationRoot/.launcher/uv-install.XXXXXXXX")
    trap 'rm -rf -- "$UvStage"' 0
    trap 'exit 130' HUP INT TERM
    curl --fail --location --proto '=https' --tlsv1.2 --output "$UvStage/uv.tar.gz" \
        'https://github.com/astral-sh/uv/releases/download/0.11.2/uv-x86_64-unknown-linux-gnu.tar.gz'
    ActualHash=$(sha256sum "$UvStage/uv.tar.gz" | cut -d ' ' -f 1)
    if [ "$ActualHash" != "7ac2ca0449c8d68dae9b99e635cd3bc9b22a4cb1de64b7c43716398447d42981" ]; then
        printf '%s\n' 'Setup runtime checksum mismatch.' >&2
        exit 1
    fi
    tar -xzf "$UvStage/uv.tar.gz" -C "$UvStage"
    cp "$UvStage/uv-x86_64-unknown-linux-gnu/uv" "$Uv"
    chmod 755 "$Uv"
    rm -rf -- "$UvStage"
    trap - 0 HUP INT TERM
fi
export UV_PYTHON_INSTALL_DIR="$ApplicationRoot/.launcher/python"
export UV_CACHE_DIR="$ApplicationRoot/.launcher/cache"
export UV_PROJECT_ENVIRONMENT="$ApplicationRoot/.venv"
export PYTHONUTF8=1
"$Uv" run --no-project --python 3.12 --managed-python python \
    "$SourceRoot/deploy/desktop/copy_install.py" --source "$SourceRoot" \
    --target "$ApplicationRoot" --uv "$Uv" --shortcut
cd "$ApplicationRoot"
printf '%s\n' "Launcher installed: $ApplicationRoot" 'Use the CARLA Vision applications-menu entry for daily startup.'
if [ "$NoLaunch" = false ]; then
    exec "$ApplicationRoot/.venv/bin/python" -m carla_vision.launcher --workspace "$ApplicationRoot" tui
fi
