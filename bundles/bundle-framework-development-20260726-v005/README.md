# CARLA monocular-vision framework development reproduction bundle v0.7

Bundle ID: `bundle-framework-development-20260726-v005`  
Purpose: `development`  
Sources: 6 verified research objects  
Source snapshot: 196 files  
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
6. `uv run carla-native-preflight --help`
7. `uv run carla-operator-ui --help`

## Integrity

Before use, run:

```text
uv run carla-verify-reproduction bundle-framework-development-20260726-v005
uv run carla-verify bundle-framework-development-20260726-v005 --reject-unregistered
```

## Declared limitations

- This is a development bundle from an uncommitted dirty worktree and is not confirmatory thesis evidence.
- CARLA server binaries, Unreal assets, pretrained model weights, datasets, and live videos are fingerprinted by source manifests but are not embedded in the source archive.
- The current Apple arm64 host can reach and query CARLA 0.9.16 read-only but cannot import an official compatible PythonAPI wheel.
- The native preflights explicitly report not-ready and do not authorize world reload or exclusive world.tick ownership.
- The 50-frame native integration pilot is planned and preflighted but has not been collected against the real simulator.
- The multi-episode native-to-QA-to-training handoff uses a fake session at the actor and sensor boundary and does not validate real spawning or sensor delivery.
- The local operator panel launches host subprocesses and is intentionally loopback-only; it is not a remotely deployable multi-user application.
- The live simulator endpoint, actor state, route position, and transient traffic cannot be reconstructed from this bundle alone.
- Real native synchronous collection, CARLA-specific multi-seed training, locked-test evaluation, and vision-only closed-loop control remain outside this development bundle.
