# Repository instructions for coding agents

Read [CONTRIBUTING.md](CONTRIBUTING.md) before making changes. Start each work
item on a dedicated feature branch from current `origin/main`, publish through
a pull request, and follow the maintainer's merge instructions. Never push work
directly to `main` or rewrite published history to change attribution.

## Authorship requested by the maintainer

For original changes implemented or written by OpenAI Codex, set the commit
author explicitly to `Codex <codex@openai.com>`. Keep the maintainer's existing
Git identity as committer and include their verified name/email in a
`Co-authored-by` trailer when they contributed direction, design or code.
Apply this per commit; do not replace global or repository `user.name` and
`user.email` settings.

If Codex only reviewed someone else's change, preserve that person's primary
authorship. Add a Codex co-author trailer only when there was an actual joint
contribution. Do not attribute unrelated human work, upstream code or model
weights to an AI agent, and retain upstream notices.

Use the [verified Codex account](https://github.com/codex) and the email mapping
shown in this [OpenAI repository commit](https://github.com/openai/swarm/commit/6af0b4caf37dca4526dfd98e9fbd8ce36e7eeb22).
For another AI tool, verify its own account and email before using them; do not
invent identities or credit tools that did not participate.

For primary-author attribution in the repository's Contributors list, retain
the authored commit when merging the PR. Prefer **Create a merge commit**;
check the resulting author explicitly if a different merge method is used.
Co-author display on a commit and the repository's contributor statistics are
different surfaces; verify the resulting account through GitHub after merging.
