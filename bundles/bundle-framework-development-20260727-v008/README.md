# CARLA monocular-vision framework development reproduction bundle v0.9

Bundle ID: `bundle-framework-development-20260727-v008`  
Purpose: `development`  
Sources: 5 verified research objects  
Source snapshot: 227 files  
Dependency locks: 1

This bundle archives the exact source tree selected by the frozen configuration,
the dependency lockfiles, verified source manifests, hardware/software metadata,
and tokenized reproduction commands. It does not silently fetch mutable model or
dataset aliases.

## Quick reproduction

Run `./reproduce.sh /path/to/new-empty-workspace`. The script refuses to
overwrite an existing directory, extracts the immutable source snapshot, and
executes these pre-registered commands:

1. `uv sync --frozen --all-groups`
2. `uv run pytest -q`
3. `uv run ruff format --check .`
4. `uv run ruff check .`
5. `uv run python -m compileall -q carla_vision carla_yolo`
6. `uv run carla-verify runs/scenario-plan-native-integration-pilot-v1 --reject-unregistered`
7. `uv run carla-build-native-kit --config configs/native_host/native_pilot_kit_v3.json --kits-root native_kits`
8. `uv run carla-verify-native-kit native_kits/native-host-kit-pilot-20260727-v003`
9. `uv run carla-build-evidence-index --workspace . --registry-id evidence-index-reproduction-20260727-v001`
10. `uv run carla-verify runs/evidence-index-reproduction-20260727-v001 --reject-unregistered`
11. `uv run carla-operator-ui --help`

## Integrity

Before use, run:

```text
uv run carla-verify-reproduction bundle-framework-development-20260727-v008
uv run carla-verify bundle-framework-development-20260727-v008 --reject-unregistered
```

## Declared limitations

- This is a development bundle from an uncommitted dirty worktree and is not confirmatory thesis evidence.
- CARLA server binaries, Unreal assets, pretrained model weights, collected datasets, and live videos are not embedded in the source archive.
- The current Apple arm64 host can run the live MessagePack viewer but cannot import the official CARLA PythonAPI required for native synchronous collection.
- The portable native kit targets CPython 3.12 on Windows/Linux x86-64 and has not been bootstrapped on the eventual execution host.
- The retained native preflight reports not ready; the 50-frame pilot has not been executed and no simulator mutation is claimed.
- A reproduced evidence registry records the canonical path of its new clean workspace, so its exact bytes are not expected to match a registry built in another directory.
- The evidence index is a point-in-time manifest and invalid-source inventory, not a portable copy of all scientific payloads or a signed release.
- The local operator panel is intentionally loopback-only and opens the native OpenCV viewer in a separate desktop window.
- Real native collection, human label QA, CARLA-specific multi-seed training, locked-test evaluation, and vision-only closed-loop control remain manual future stages.
