# CARLA monocular-vision framework development reproduction bundle v0.8

Bundle ID: `bundle-framework-development-20260727-v006`  
Purpose: `development`  
Sources: 5 verified research objects  
Source snapshot: 214 files  
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
7. `uv run carla-build-native-kit --config configs/native_host/native_pilot_kit_v2.json --kits-root native_kits`
8. `uv run carla-verify-native-kit native_kits/native-host-kit-pilot-20260727-v002`
9. `uv run carla-verify native_kits/native-host-kit-pilot-20260727-v002 --reject-unregistered`
10. `uv run carla-operator-ui --help`

## Integrity

Before use, run:

```text
uv run carla-verify-reproduction bundle-framework-development-20260727-v006
uv run carla-verify bundle-framework-development-20260727-v006 --reject-unregistered
```

## Declared limitations

- This is a development bundle from an uncommitted dirty worktree and is not confirmatory thesis evidence.
- CARLA server binaries, Unreal assets, pretrained model weights, collected datasets, and live videos are not embedded in the source archive.
- The current Apple arm64 host can run the live MessagePack viewer but cannot import the official CARLA PythonAPI required for native synchronous collection.
- The portable native kit targets CPython 3.12 on Windows/Linux x86-64 and has not been bootstrapped on the eventual execution host.
- The retained native preflight reports not-ready; the 50-frame pilot has not been executed and no simulator mutation is claimed.
- The native-to-QA-to-training integration test uses a fake actor/sensor boundary and does not validate real spawning, sensor phase, cleanup, or world restoration.
- The local operator panel is intentionally loopback-only and opens the native OpenCV viewer in a separate desktop window.
- Live simulator state, actor IDs, route position, traffic, and package-index availability cannot be reconstructed from this bundle alone.
- Real native collection, human label QA, CARLA-specific multi-seed training, locked-test evaluation, and vision-only closed-loop control remain manual future stages.
