# Branch audit after integration PR #125

Snapshot: 2026-10-12, after [PR #125](https://github.com/koolfox/carla-local-autonomy/pull/125)
was merged with **Squash**. This is a dated audit, not a live branch inventory.

`main` is `96836cb66923d6cddaa7e289e7a7277000f0b382`. Its complete Git tree is
identical to the integration branch at `8e91c1bd06cb96e16b81d9d125f91cba257c74f4`:
both resolve to tree `24fbf6d8699cbfaf2ec08e4a6ba6baa417cf9395`.

| Remaining branch | Audited tip | Result |
| --- | --- | --- |
| `codex/repository-audit-samples` | `2c600765e34862becc4f236d99291b295604f973` | Ancestor of the integration tip; final integrated changes are on main |
| `codex/single-host-launcher` | `a59fdbf58b5affcf70cd751a0c5bcbfb22757066` | Ancestor of the integration tip; final integrated changes are on main |
| `codex/project-restore-cleanup` | `3faa6b97f4fbdd1aeb2ec619f18f1d0c9e0c35f6` | Ancestor of the integration tip; final integrated changes are on main |
| `codex/main-integration-verification` | `8e91c1bd06cb96e16b81d9d125f91cba257c74f4` | Entire tree identical to main |

Squash creates a new commit rather than preserving the feature-branch ancestry.
Git can therefore list these branches as unmerged despite their integration
being complete. No additional merge is needed for these audited tips. The older
branch trees can differ from main because the integration contains later fixes.

No branches were deleted during this audit. Fetch and inspect their current tips,
worktrees, stashes and any uncommitted work before a later cleanup; this snapshot
does not cover commits added afterwards.

## Reproduce the content check

```bash
git fetch origin --prune
git rev-parse origin/main^{tree} origin/codex/main-integration-verification^{tree}
git diff --exit-code origin/codex/main-integration-verification origin/main
git merge-base --is-ancestor origin/codex/repository-audit-samples origin/codex/main-integration-verification
git merge-base --is-ancestor origin/codex/single-host-launcher origin/codex/main-integration-verification
git merge-base --is-ancestor origin/codex/project-restore-cleanup origin/codex/main-integration-verification
```

The merged commit passed [main CI](https://github.com/koolfox/carla-local-autonomy/actions/runs/38260870609).
Live CARLA and fresh-machine installer acceptance remain separate pending checks.
