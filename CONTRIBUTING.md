# Contributing

This project uses small, issue-scoped changes so the maintainer and Codex can
work in parallel without losing features or creating long-lived integration
branches.

Read [`docs/development.md`](docs/development.md) before changing product
behavior. It maps the runtime roles, authoritative sources, request/data paths,
vertical-slice workflow, and live-CARLA evidence boundary.

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

Do not copy a fixed ownership list into long-lived documentation. The active
GitHub issue must name its branch, expected files, dependencies, and acceptance
gate before implementation begins. The current roadmap is grouped into:

- **M1 — Developer-ready product convergence:** one SessionConfig, stable
  Operator boundary, no-CARLA developer mode, and Svelte/Garage parity;
- **M2 — Vision and voxel autonomy MVP:** NavigationIntent, canonical scene
  perception, DriverSceneFrame, and supervised hybrid control; and
- **M3 — Live evidence and OSS release:** real datasets/checkpoints,
  closed-loop acceptance, final cleanup, and release audit.

If two tasks need the same file, sequence them instead of resolving a large
conflict after both are complete. GitHub issues and milestones are the source
of truth for current ownership and dependencies.

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
npm ci --no-audit --no-fund
npm run check
npm run build
```

The production bundle in `carla_vision/operator/console_static/` is a reviewed
release asset. Commit it with the Svelte source change; never edit it by hand.
CI rebuilds it from `web/package-lock.json` and rejects stale output.

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
working implementation of a capability. Durable contract, safety, topology,
or artifact decisions use the ADR process in [`docs/adr/`](docs/adr/).
