# Default-branch integration — 2026-10-12

The maintainer explicitly authorized merging all current project branches after
checking for disruption. Remote state contained only `main` and these three
feature branches:

| Branch | Published commit | Included in recovery branch |
| --- | --- | --- |
| `codex/repository-audit-samples` | `2c600765` | Yes |
| `codex/single-host-launcher` | `a59fdbf5` | Yes |
| `codex/project-restore-cleanup` | `3faa6b97` | Yes |

They form one ancestry chain above `main` (`ad8e00b7`), so one integration PR
includes every branch's change. The integration uses a separate feature branch
and a merge commit to retain original source identities referenced by model and
installer inventories. No direct feature push to `main` or forced update is needed.

Recovery defaults and clone documentation now select `main`. Integrity/preservation
tests cover the standalone recovery engine, including original private data,
rejected corrupt archives/downloads, existing-project protection and process-lock
exclusion. Windows lock initialization uses file size instead of reading a byte
that another process may have locked. Recovery selects the exact lowercase
`samples/archive.json` filename, which is required on case-sensitive Linux
filesystems. CI adds Linux, Windows and macOS recovery
checks alongside the existing Operator, frontend and research regression jobs.
The native-research HTTP fixture injects an explicitly offline CARLA loader so
installing the optional SDK cannot make it contact a real localhost simulator.
It still verifies crash receipts, world-owner release and the next capture.

These checks do not require CARLA and do not establish complete native installation,
TUI appearance, live inference or simulator acceptance. Those gates remain deferred
as requested. Existing user data, installed environments, private backup archives
and historical payload bytes are preserved by the integration.
