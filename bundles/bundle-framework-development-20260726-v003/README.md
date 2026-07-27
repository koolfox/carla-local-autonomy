# CARLA monocular-vision framework development reproduction bundle v0.5

Bundle ID: `bundle-framework-development-20260726-v003`  
Purpose: `development`  
Sources: 7 verified research objects  
Source snapshot: 168 files  
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

## Integrity

Before use, run:

```text
uv run carla-verify-reproduction bundle-framework-development-20260726-v003
uv run carla-verify bundle-framework-development-20260726-v003 --reject-unregistered
```

## Declared limitations

- This is a development bundle from an uncommitted dirty worktree and is not confirmatory thesis evidence.
- CARLA server binaries, Unreal assets, pretrained model weights, datasets, and live videos are fingerprinted by source manifests but are not embedded in the source archive.
- The live simulator endpoint, actor state, route position, and transient traffic cannot be reconstructed from this bundle alone.
- The included live matrix used sequential, non-identical scenes and cannot support detector-accuracy ranking.
- The hazard-stop policy is a non-learned, non-actuating systems baseline and does not establish safe autonomous driving.
- Real native synchronous collection, CARLA-specific multi-seed training, and locked-test evaluation remain outside this development bundle.
