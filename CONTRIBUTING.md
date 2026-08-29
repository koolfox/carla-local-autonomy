# Contributing

This project uses small, issue-scoped changes so the maintainer and Codex can
work in parallel without losing features or creating long-lived integration
branches.

## Start one task

1. Pick one open issue with acceptance criteria.
2. Comment on the issue with the branch name and the files you expect to own.
3. Start from the current remote `main`:

   ```bash
   git switch main
   git pull --ff-only origin main
   git switch -c feature/54-shared-session-config
   ```

   Codex-created branches use the `codex/` prefix. Human-created branches may
   use `feature/`, `fix/`, or `docs/`.
4. Do not put unrelated cleanup in the same branch. If the scope changes, note
   it on the issue before editing overlapping files.

Never push feature work directly to `main`. Never rewrite or force-push a
branch another contributor is using.

## Parallel work

The active convergence issues deliberately define file ownership:

- #53 owns frontend build and Python packaging.
- #54 owns shared scene configuration and Research-to-Garage mapping.
- #55 owns experiment preset state and serialization.
- #56 owns dense scene preparation progress and Worker-facing runtime state.
- #57 owns Garage preview presentation, camera fit, and responsive controls.
- #48 owns external model runtime hardening.
- #58 owns final structure and release cleanup, after feature parity.

If two tasks need the same file, sequence them instead of resolving a large
conflict after both are complete. GitHub issues are the source of truth for
current ownership and dependencies.

## Verify before a pull request

Run the smallest relevant checks first, then the affected suite.

Python changes:

```bash
uv run ruff check carla_vision tests
uv run pytest tests/test_relevant_module.py -q
```

Svelte changes:

```bash
cd web
npm install --no-audit --no-fund
npm run check
npm run build
```

Simulator-dependent behavior also needs a recorded live-CARLA acceptance run.
Unit tests and fake actors do not prove map changes, vehicle control, stream
quality, population capacity, or actor cleanup.

## Pull requests

Keep one product outcome per pull request. The description must state:

- the issue it closes or advances;
- the user-visible outcome;
- files or contracts intentionally left unchanged;
- automated checks run;
- live-CARLA evidence, or an explicit statement that it is still pending;
- migration and rollback behavior when state or artifacts change.

Do not merge while required checks are failing or while a review has an
unresolved safety or data-integrity blocker. Delete the remote branch after the
pull request is merged; Git history retains the commit.

## Local and generated data

Do not commit `.env.local`, checkpoints, datasets, runs, reports, browser build
caches, Python caches, or operator session output. Keep shareable examples
small and add them deliberately through a release or sample-data decision.

The final removal of legacy code belongs to #58. Until feature parity is
verified, cleanup means removing proven duplication, not deleting the only
working implementation of a capability.
