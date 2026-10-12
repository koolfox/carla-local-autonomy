#!/bin/sh
# Restore from a checkout, or save this file alone and run it in a new directory.
set -eu
umask 077
RestoreScriptDirectory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
case "$(uname -s):$(uname -m)" in
    Darwin:arm64)
        RestoreArchive=uv-aarch64-apple-darwin.tar.gz
        RestoreArchiveHash=4beaa9550f93ef7f0fc02f7c28c9c48cd61fe30db00f5ac8947e0a425c3fb282 ;;
    Darwin:x86_64)
        RestoreArchive=uv-x86_64-apple-darwin.tar.gz
        RestoreArchiveHash=a9c3653245031304c50dd60ac0301bf6c112e12c38c32302a71d4fa6a63ba2cb ;;
    Linux:x86_64)
        RestoreArchive=uv-x86_64-unknown-linux-gnu.tar.gz
        RestoreArchiveHash=7ac2ca0449c8d68dae9b99e635cd3bc9b22a4cb1de64b7c43716398447d42981 ;;
    *) printf '%s\n' 'Supported bootstrap platforms: macOS arm64/x86-64 and Linux x86-64.' >&2; exit 1 ;;
esac
for RestoreTool in curl tar mktemp; do
    command -v "$RestoreTool" >/dev/null 2>&1 || { printf 'Required tool missing: %s\n' "$RestoreTool" >&2; exit 1; }
done
restore_hash() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d ' ' -f 1
    else
        shasum -a 256 "$1" | cut -d ' ' -f 1
    fi
}
RestoreCacheDirectory="$HOME/.local/share/carla-restore-bootstrap/uv-0.11.2"
if [ -L "$RestoreCacheDirectory" ]; then
    printf '%s\n' 'The bootstrap cache must not be a symbolic link.' >&2; exit 1
fi
mkdir -p "$RestoreCacheDirectory"
RestoreStage=$(mktemp -d "$RestoreCacheDirectory/stage.XXXXXXXX")
trap 'rm -rf -- "$RestoreStage"' 0
trap 'exit 130' HUP INT TERM
RestoreUv="$RestoreStage/uv"
printf '%s\n' 'Preparing the verified setup runtime (no system Python changes)...'
curl --fail --location --proto '=https' --proto-redir '=https' --retry 3 \
    "https://github.com/astral-sh/uv/releases/download/0.11.2/$RestoreArchive" \
    --output "$RestoreStage/uv.tar.gz"
if [ "$(restore_hash "$RestoreStage/uv.tar.gz")" != "$RestoreArchiveHash" ]; then
    printf '%s\n' 'Setup runtime checksum mismatch.' >&2; exit 1
fi
mkdir "$RestoreStage/unpacked"
tar -xzf "$RestoreStage/uv.tar.gz" -C "$RestoreStage/unpacked"
for RestoreCandidate in "$RestoreStage"/unpacked/*/uv; do
    if [ -f "$RestoreCandidate" ]; then cp "$RestoreCandidate" "$RestoreUv"; fi
done
[ -f "$RestoreUv" ] || { printf '%s\n' 'Unexpected setup runtime archive layout.' >&2; exit 1; }
chmod 700 "$RestoreUv"
RestoreEngine="$RestoreScriptDirectory/restore_project.py"
if [ ! -f "$RestoreEngine" ]; then
    RestoreEngine="$RestoreStage/restore_project.py"
    curl --fail --location --proto '=https' --proto-redir '=https' --retry 3 \
        'https://raw.githubusercontent.com/koolfox/carla-local-autonomy/refs/heads/main/restore_project.py' \
        --output "$RestoreEngine"
fi
export UV_PYTHON_INSTALL_DIR="$RestoreCacheDirectory/python"
export UV_CACHE_DIR="$RestoreCacheDirectory/cache"
export UV_NO_CONFIG=1
export PYTHONUTF8=1
"$RestoreUv" run --no-project --python 3.12 --managed-python \
    python -B "$RestoreEngine" --uv "$RestoreUv" "$@"
