# Repository and local-artifact audit — 2026-10-12

Scope: reconcile this local checkout with `koolfox/carla-local-autonomy`, publish
the maintainer-requested historical samples and videos, and document remaining
work. After the initial inventory, the maintainer deferred further tests and new
scripts until 2026-10-13 because the CARLA server is unavailable. No runtime
source change, simulator mutation or new test script belongs to this delivery.

## Repository state

- Fetched GitHub `main`: `ad8e00b7b57dc264f582e475d236d7320c6ff617`.
- Local `main` and GitHub `main` have identical source content and ancestry;
  the initial tracked worktree was clean.
- GitHub had only the `main` branch, no open PRs and no published releases.
- GitHub reports public visibility. The older release assessment's private
  visibility and open-PR #120 observations are historical; PRs through #124
  have merged.
- Twenty issues remain open. Their implementation, live acceptance and research
  gates were inspected; a historical sample upload does not close those gates.
- The latest existing main CI run succeeded:
  [run 34778637835](https://github.com/koolfox/carla-local-autonomy/actions/runs/34778637835).
  That run covers the existing main commit, not this archive branch or a live simulator.
- Delivery branch: `codex/repository-audit-samples`; PR creation and merging
  remain maintainer-controlled.

## Local differences and publication decision

| Local material | Finding and decision |
| --- | --- |
| `output/` | Retain all five short MP4s, seven PNGs and five JSON summaries under `samples/recordings/`. |
| `datasets/` | Retain the three-frame stationary labelled pilot under `samples/data/`, including original manifests, labels and hashes. |
| `reports/` | Retain all eight historical report trees under `samples/historical-reports/`; distinguish file integrity from unavailable upstream provenance and portability. |
| DeiT class CSVs in `models/` | Retain both 64- and 68-class files under `samples/model-labels/`, alongside the sample guide. |
| Model weights and Hugging Face cache | Keep local; not part of this small historical archive. |
| `native_tasks/teacher_capture/source/` | Generated source export: 176 Python files match the current checkout; all 21 differing files match older blobs already in Git history. No new module or unrecorded Python file was found. Preserve the local export rather than overwriting current source with older code. |
| `operator_sessions/` | Machine-local execution state, requests and logs; preserve locally. |
| `tools/`, `runs/`, `dist/` | Only bytecode/cache state in `tools/` and `runs/`, plus a rebuildable wheel in `dist/`; no missing source files to upload. |
| `.env.local`, virtual environment, browser dependencies, caches and session marker | Preserve locally and exclude from the publication. Cache/session ignore rules are explicit. |

The original payloads are copied without modification. The archive inventory
contains 123 payload files and 10,563,700 bytes. Each entry records the source
workspace-relative path, size and SHA-256. Original manifests retain the original
Git and dirty-state declarations; today's archive baseline is not relabelled as
their capture commit. Local originals remain intact.

## Publication limitations

The five legacy movies are MPEG-4 Part 2, not H.264, so browser playback may
require conversion; their original bytes remain downloadable. No new conversion
script is introduced. Encoded duration differs from wall-clock capture duration,
and the old summaries lack complete model/checkpoint and control-owner identities.

The dataset's three frames are stationary and unassigned, with 27 annotations.
Its retained training YAML points to partitions that are absent from this pilot.
It is a sample for inspection, not a training release or independent benchmark.

The reports retain absolute paths and refer to historical objects that are no
longer all available. Hashes establish retained-byte identity; full portable
reproduction and current semantic compatibility are not claimed. No sample or
project license is newly assigned by this upload.

## Verification already observed and work deferred

Before the maintainer narrowed the scope, local Python lint, Svelte type checking
and all 38 frontend tests passed. An initial full Python run passed 1,082 cases
plus 228 subtests; 49 failures and 15 setup errors all reached the environment's
blocked local-socket boundary. It was not a successful full-suite run. The
unrestricted rerun was cancelled, and no further test execution was performed.
No live-CARLA acceptance result was produced.

On 2026-10-13, after the simulator is available, resume the separate live and
code-verification work against the acceptance criteria of the existing issues:

- [#27](https://github.com/koolfox/carla-local-autonomy/issues/27) and
  [#56](https://github.com/koolfox/carla-local-autonomy/issues/56): repeated
  Garage lifecycle, dense settings, recovery, prepared-scene handoff and owned-actor cleanup.
- [#55](https://github.com/koolfox/carla-local-autonomy/issues/55) and
  [#45](https://github.com/koolfox/carla-local-autonomy/issues/45): preset
  readiness, persistence and supported UI acceptance.
- [#58](https://github.com/koolfox/carla-local-autonomy/issues/58) and
  [#33](https://github.com/koolfox/carla-local-autonomy/issues/33): release
  hygiene, clean installation and a new reproducible demo with complete identities.
- Existing research/model issues remain separately gated by real datasets,
  training, exact-frame evidence and closed-loop evaluation; sample filenames or
  detector overlays do not satisfy those requirements.

No new automation or scheduled action is created by this handoff.
