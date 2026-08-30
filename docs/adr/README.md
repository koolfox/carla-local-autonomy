# Architecture decision records

ADRs preserve the reason behind durable project boundaries without turning
every pull request into permanent documentation.

Create an ADR when a change affects one or more of:

- a public/versioned API or model contract;
- control ownership or a fail-safe boundary;
- allowed versus privileged research inputs;
- artifact identity, lineage, or verification;
- the Windows Worker/research-computer topology; or
- the primary frontend/backend framework.

Do not create an ADR for routine refactors, styling, local algorithms, or a
choice that is easy to reverse inside one module.

Copy `0000-template.md`, allocate the next four-digit number, and use a short
lowercase filename such as `0001-session-config-ownership.md`. Submit the ADR
in the same pull request as the decision. An accepted ADR is immutable except
for typo/link corrections; a later decision supersedes it with a new ADR.

