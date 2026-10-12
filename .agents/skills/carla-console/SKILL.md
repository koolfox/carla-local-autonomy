---
name: carla-console
description: Workflow for changing the CARLA Vision Research Console repository (Svelte console in web/, Operator and World Worker in carla_vision/, launcher TUI, research jobs). Use before fixing a bug, adding a feature, updating docs, or opening a pull request in this repository. Covers finding a feature's owner and test, scoping a branch, verification commands, the live-CARLA evidence boundary, commit attribution for AI agents, and the pull request checklist.
---

# Working on the CARLA Vision Research Console

The repository is one product with four runtime roles. Most mistakes come from
editing the wrong layer, so locate the owner first and change only that layer.

```text
web/ (Svelte)            browser intent, presentation, input
  -> carla_vision/operator/   Operator on loopback :8765: sessions, models, files
       -> carla_vision/native/  World Worker on the CARLA host :8766: PythonAPI, actors
       -> research packages     dataset, training, evaluation, reports
carla_vision/launcher/   full-screen TUI that installs and starts the local stack
```

## 1. Scope the task

1. Confirm there is an issue with acceptance criteria. If the user gave the
   task directly, write the expected outcome in one sentence before editing.
2. Find the feature in the table in `docs/README.md` ("Find the feature, then
   its test"). Note the owning file, the first regression test, and whether a
   Windows/World Worker update is needed.
3. For request paths and ownership details, read the matching section of
   `docs/development.md`. For product constraints, read `PRODUCT.md`.
4. Branch from current `origin/main` with your agent prefix
   (`claude/`, `codex/`):

   ```bash
   git fetch origin
   git switch -c claude/<issue>-<short-name> origin/main
   ```

## 2. Change the owning layer

- Write or extend the focused test first, then fix the owner.
- Python contract and TypeScript mirror change together; a new API field needs
  both `carla_vision/operator/` and `web/src/lib/api/` or `web/src/lib/domain/`.
- Do not add a second configuration store, a retry loop around an existing
  owner, a new framework, or a new service to make a file shorter.
- Never hand-edit `carla_vision/operator/console_static/`; rebuild it.
- Never extend `carla_vision/operator/static/` (legacy rollback only).
- Keep `carla_vision/native/observable_world_worker.py` free of project and ML
  imports.
- Keep generated and private data out of Git: `.env.local`, `.launcher/`,
  `runs/`, `datasets/`, `reports/`, `operator_sessions/`, new model weights.
- Durable contract, safety, topology or artifact decisions need an ADR in
  `docs/adr/`.

## 3. Verify

Run the smallest relevant check first, then the affected suite.

```bash
uv sync --group dev --frozen
uv run ruff check carla_vision tests restore_project.py
uv run pytest tests/<focused_test>.py -q
uv run pytest -q                       # full suite, about 1.5 minutes
```

Launcher/TUI changes also need
`uv sync --extra launcher --group dev --frozen` and
`uv run pytest -q tests/test_launcher.py tests/test_operator_local_config.py`.

Svelte changes, from `web/`:

```bash
npm ci --no-audit --no-fund
npm run check
npm test
npm run build      # commit the regenerated console_static/ with the source
```

CI (`.github/workflows/ci.yml`) runs a named list of Python tests. If you add a
test file that guards Operator behavior, add it to that list.

### Live-CARLA boundary

Actors, map changes, vehicle control, camera timing, stream quality,
population capacity and cleanup cannot be proven by unit tests or fake actors.
Do not run untested control changes in a live driving session. Unless a real
CARLA acceptance run was recorded, state in the PR that live validation is
pending. Never claim it.

## 4. Commit with correct attribution

Follow `AGENTS.md` exactly. For original work written by the agent, the agent
is the commit author and the maintainer stays committer and co-author:

```bash
git commit --author="Claude <noreply@anthropic.com>" -m "<summary>" \
  -m "<why>" \
  -m "Co-authored-by: <maintainer name> <maintainer verified email>"
```

Codex uses `--author="Codex <codex@openai.com>"`. Use the maintainer's identity
from `git config user.name` / `git config user.email`. Do not change Git
config. If you only reviewed someone else's work, keep their authorship.

## 5. Open the pull request

Push the branch and open a PR against `main`. The description must state:

- the issue it closes or advances;
- the user-visible outcome;
- files or contracts intentionally left unchanged;
- automated checks run, with results;
- live-CARLA evidence, or an explicit statement that it is still pending;
- migration and rollback behavior when state or artifacts change.

Do not merge yourself. The maintainer merges, preferably with
**Create a merge commit** so the authored commit keeps its author.

## Debugging routine

1. Write the exact action and expected outcome.
2. Capture request/response, operation ID and Operator/Worker logs. Redact the
   Operator and Worker tokens before sharing.
3. Use the feature map to decide whether the first failure is browser state,
   HTTP translation, Operator lifecycle, native CARLA work or model output.
4. Reproduce at that boundary with a focused test. A timeout alone does not
   show whether the remote mutation happened; inspect operation state.
5. Fix the owning layer and re-run its tests plus one adjacent boundary test.
