# Project cleanup and recovery — 2026-10-12

Cleanup began from the clean, published `a59fdbf5` checkout. Work is on
`codex/project-restore-cleanup`; `main` and existing branches were preserved.

Before deletion, ignored state was inventoried and an independently verified
private archive was written outside the project. It contains 417 declared local
payloads, including local settings, sessions, native task packages, retained
build artifacts and orphaned bytecode. An environment inventory was retained
privately. These files are not published to GitHub.

Removed logical file size: **4,018,685,714 bytes (3.743 GiB)**. This is not a
claim of equal APFS physical-space gain. Removed items were:

- an abandoned Git temporary pack and empty temporary object with no open
  handles; valid objects, two actual pack files, refs and reflogs were retained;
- rebuildable mypy, pytest, Ruff and project Python caches;
- Finder metadata, including `.git/refs/.DS_Store`, which caused bad-ref errors;
- seven duplicate Hugging Face blob-cache files after their regular published
  snapshot counterparts were checked against the exact model inventory hashes.

All model snapshot payloads and the Git LFS object cache were retained. The
installed `.venv`, frontend `node_modules`, active Svelte cache, original
recordings/reports/datasets, source packages, private settings and sessions were
preserved. Services were not stopped or started by cleanup.

After cleanup, `git fsck --connectivity-only --no-dangling` passed and Git reported
zero garbage. No pruning, Git garbage collection, branch deletion, package
removal or simulator manipulation was performed.

The [recovery guide](../restore_project.md) documents the independent stdlib
engine, bootstrap entry points, commit-pinned retrieval, model/sample verification,
backup import and host roles. The new tools are included in future installer
build inputs; the previously published preview ZIP/TAR was not rebuilt here.

Delivery checks cover Python syntax/lint, shell syntax, PowerShell parsing,
argument forwarding and CLI help/plan, plus byte integrity. These checks do not
establish fresh-machine or live-CARLA runtime acceptance. Runtime restoration,
installer/TUI acceptance and CARLA testing remain deferred at the maintainer's
request.
