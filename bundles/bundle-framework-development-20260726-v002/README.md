# CARLA monocular-vision framework development reproduction bundle v0.4

Bundle ID: `bundle-framework-development-20260726-v002`  
Purpose: `development`  
Sources: 11 verified research objects  
Source snapshot: 147 files  
Dependency locks: 1

This bundle archives the exact source tree selected by the frozen configuration,
the dependency lockfiles, verified source manifests, hardware/software metadata,
and tokenized reproduction commands. It does not silently fetch mutable model or
dataset aliases.

## Quick reproduction

Run `./reproduce.sh /path/to/new-empty-workspace`. The script refuses to
overwrite an existing directory, extracts the immutable source snapshot, and
executes these pre-registered commands:

1. `uv sync --frozen`
2. `uv run python -m unittest discover -s tests -p 'test_*.py'`
3. `uv run ruff format --check carla_vision carla_yolo tests`
4. `uv run ruff check .`
5. `uv run python -m compileall -q carla_vision carla_yolo`

## Integrity

Before use, run:

```text
uv run carla-verify-reproduction bundle-framework-development-20260726-v002
uv run carla-verify bundle-framework-development-20260726-v002 --reject-unregistered
```

## Declared limitations

- This is a development bundle from an uncommitted dirty worktree and is not confirmatory thesis evidence.
- CARLA server binaries, Unreal assets, pretrained model weights, datasets, and videos are fingerprinted by their source manifests but are not embedded in the source archive.
- The live simulator endpoint and its transient actor state cannot be reconstructed from this bundle alone.
- The retained pilot dataset has only three unassigned frames, so no validation-only threshold selection or generalization claim is possible.
- The included v0.4 report and prior bundle are development evidence and inherit their recorded limitations.
