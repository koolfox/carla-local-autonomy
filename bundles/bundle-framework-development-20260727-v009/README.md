# CARLA monocular-vision framework development reproduction bundle v0.9 corrected

Bundle ID: `bundle-framework-development-20260727-v009`  
Purpose: `development`  
Sources: 5 verified research objects  
Source snapshot: 228 files  
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
7. `uv run carla-build-evidence-index --workspace . --registry-id evidence-index-reproduction-20260727-v001`
8. `uv run carla-verify runs/evidence-index-reproduction-20260727-v001 --reject-unregistered`
9. `uv run carla-operator-ui --help`

## Integrity

Before use, run:

```text
uv run carla-verify-reproduction bundle-framework-development-20260727-v009
uv run carla-verify bundle-framework-development-20260727-v009 --reject-unregistered
```

## Declared limitations

- This is a development bundle from an uncommitted dirty worktree and is not confirmatory thesis evidence.
- CARLA server binaries, Unreal assets, pretrained model weights, collected datasets, and live videos are not embedded in the source archive.
- The current Apple arm64 host can run the live MessagePack viewer but cannot import the official CARLA PythonAPI required for native synchronous collection.
- The retained v003 native-host kit is a verified historical source; exact reconstruction belongs to the v0.8 source snapshot that originally produced it.
- The superseded v008 bundle is retained because clean reproduction exposed that rebuilding fixed kit ID v003 from changed source produces different bytes under the same ID.
- This corrected bundle does not regenerate a historical immutable kit ID; any future native-host package must use a new configuration and kit ID.
- A reproduced evidence registry records the canonical path of its new clean workspace, so its exact bytes are not expected to match a registry built in another directory.
- The evidence index is a point-in-time manifest and invalid-source inventory, not a portable copy of all scientific payloads or a signed release.
- The local operator panel is intentionally loopback-only and opens the native OpenCV viewer in a separate desktop window.
- Real native collection, human label QA, CARLA-specific multi-seed training, locked-test evaluation, and vision-only closed-loop control remain manual future stages.
