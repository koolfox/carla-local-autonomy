# Repository instructions for coding agents

These instructions apply to every AI coding agent working in this repository
(Codex, Claude Code and others). Read [CONTRIBUTING.md](CONTRIBUTING.md) before
making changes. Start each work item on a dedicated feature branch from current
`origin/main`, publish through a pull request, and follow the maintainer's merge
instructions. Never push work directly to `main` or rewrite published history to
change attribution. Agents follow the [Code of Conduct](CODE_OF_CONDUCT.md) in
everything they publish.

The step-by-step workflow lives in the
[`carla-console` skill](.agents/skills/carla-console/SKILL.md). Load it before
changing code; this file holds the rules that must never be skipped.

## Orient before editing

- [`docs/README.md`](docs/README.md) maps each user-visible feature to its owning
  file and its first regression test. Find the owner there, not by grepping.
- [`docs/development.md`](docs/development.md) explains the four runtime roles
  (Svelte console, Operator, World Worker, research jobs) and the request paths.
- [`PRODUCT.md`](PRODUCT.md) states the current product scope and constraints.
- GitHub issues and milestones own the backlog and ownership of files. Do not
  copy a backlog into documentation.

## Boundaries

- `web/` is the only frontend source. `carla_vision/operator/console_static/` is
  built by `npm run build`, committed with its Svelte source, and never edited by
  hand. Do not add features to the legacy `carla_vision/operator/static/`.
- `carla_vision/native/observable_world_worker.py` must stay runnable beside
  CARLA without this project or the ML stack installed.
- The Operator binds to loopback; the World Worker is a LAN adapter. Never expose
  either to the public internet or weaken token checks.
- Do not commit `.env.local`, `.launcher/`, `simulator/`, new model downloads,
  `runs/`, `datasets/`, `reports/`, `operator_sessions/` or caches. Bundled
  models go through Git LFS and the inventory in
  [`models/README.md`](models/README.md); retain upstream notices.
- Unit tests and fake actors do not prove simulator behavior. Never claim live
  CARLA validation that was not recorded; say it is pending.
- One product outcome per pull request. Do not mix in unrelated cleanup.

## Verify

```bash
uv sync --group dev --frozen
uv run ruff check carla_vision tests restore_project.py
uv run pytest tests/<focused_test>.py -q   # then the affected suite
```

For `web/` changes run `npm ci --no-audit --no-fund`, `npm run check`,
`npm test` and `npm run build` inside `web/`, and commit the rebuilt bundle.

## Branch names

Agent-created branches use the agent's prefix: `codex/` for Codex, `claude/` for
Claude. Human branches use `feature/`, `fix/` or `docs/`.

## Authorship requested by the maintainer

For original changes implemented or written by an AI agent, set the commit
author explicitly to that agent's verified identity from the table below. Keep
the maintainer's existing Git identity as committer and include their verified
name/email in a `Co-authored-by` trailer when they contributed direction, design
or code. Apply this per commit with `git commit --author=...`; do not replace
global or repository `user.name` and `user.email` settings.

| Agent | Commit author | GitHub account | Verified by |
| --- | --- | --- | --- |
| OpenAI Codex | `Codex <codex@openai.com>` | [`codex`](https://github.com/codex) | [openai/swarm commit](https://github.com/openai/swarm/commit/6af0b4caf37dca4526dfd98e9fbd8ce36e7eeb22) |
| Anthropic Claude (Claude Code) | `Claude <noreply@anthropic.com>` | [`claude`](https://github.com/claude) (user ID 81847) | GitHub commit search for `author-email:noreply@anthropic.com` resolves to this account (checked 2026-10-12) |

If an agent only reviewed someone else's change, preserve that person's primary
authorship. Add an agent co-author trailer only when there was an actual joint
contribution. Do not attribute unrelated human work, upstream code or model
weights to an AI agent, and retain upstream notices.

For another AI tool, verify its own account and email before adding it to the
table; do not invent identities or credit tools that did not participate. If an
agent's runtime asks for its own attribution trailer, the rules in this file
take precedence: an agent that is already the commit author does not also add
itself as co-author.

For primary-author attribution in the repository's Contributors list, retain
the authored commit when merging the PR. Prefer **Create a merge commit**;
check the resulting author explicitly if a different merge method is used.
Co-author display on a commit and the repository's contributor statistics are
different surfaces; verify the resulting account through GitHub after merging.
