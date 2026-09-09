# Release-readiness assessment

Reviewed 2026-09-10 at `b1460157` (recordings branch including the latest fetched
`main`, `73b5707d`). PR #120 was open at review time; its newer player is not
assumed released. This is a targeted architecture/product review, not a
line-by-line security audit or a fresh live-CARLA acceptance run.

## Verdict

Continue the project, but narrow the first public promise: **an independent
vision-research console for CARLA, for bringing a model, running a scene,
recording results, and inspecting evidence.** There is useful implementation
here. Market demand, maintainability by outsiders, and simulator reliability
still need evidence. A UI alone is not a novel contribution; the usable
workflow and trustworthy model/frame/artifact boundaries are the value.

Release an experimental research tool after the gates below, not a proven
self-driving stack. Existing custom checkpoints are real work, but their
presence does not validate a closed-loop autonomy claim. Voxel perception,
waypoint teacher labels, and future voxel-assisted control remain in scope;
they need separate labels for model estimates, privileged truth, and actuation.

## What this review actually found

| Evidence | Interpretation |
| --- | --- |
| 187 tracked Python files, 70,359 lines; 6,991 lines under `web/src/` | Large for a single-maintainer MVP. Counts include blank/comment lines, not runtime data; they are not a quality score. |
| `native/world_worker.py` 4,876 lines; `operator/drive.py` 2,396; `garage_preview.py` 1,675 | Lifecycle ownership is concentrated. In the last 100 commits, these paths appeared in 20, 21, and 12 commits respectively. High size plus change frequency makes them priority review boundaries, not automatic deletion targets. |
| Legacy static UI is 8,923 lines; 36 documentation files total 7,461 lines | A second UI and accumulated delivery notes impose maintenance cost. More prose alone is not the solution. |
| Detector contract, factory, newest-frame worker, independent road segmenters, voxel packages, artifact producers/verifiers exist | The architecture has usable seams; preserve them instead of starting again. |
| CI has lean Operator, Svelte build/test, full research-suite and wheel-content checks | A useful engineering foundation. CI still does not demonstrate real network/renderer/actor performance. |
| Local lint passed; focused configuration, console, perception, labels, segmentation, recordings/API tests: 63 passed + 23 subtests | Current focused regression evidence only; no fresh full-suite or trained-model accuracy claim. |
| 28 frontend tests, Svelte type-checking and 100 documentation links/file references passed | Useful maintenance checks; these do not replace a newcomer exercise or browser/live-CARLA acceptance. |
| GitHub reports private repository, no release, no license metadata, no security policy; no tracked LICENSE/CITATION.cff/SECURITY.md found | Public/OSS release work is not finished. CONTRIBUTING and developer docs already exist. |
| `.env.local` and runtime roots are not tracked in the current tree | Good, but not a secrets-history audit. Previously shared credentials must be reviewed/rotated before publication. |

Methods: `git ls-files` plus `wc -l` for source counts; `git log -100
--format= --name-only` for change frequency; package scripts, CI and focused
source/tests inspected directly; GitHub metadata/issues read without changing
them. No simulator mutation, checkpoint download, or production session was
used for this assessment.

## Concrete weaknesses to fix

1. **Contributor independence:** a maintainer still needs private context to
   distinguish entry points, compatibility paths and experiments. #68's
   deterministic no-CARLA fixture is still open. The normal offline launch
   is not equivalent to a working fake world.
2. **Change risk:** HTTP handler inheritance, large lifecycle classes, and
   mirrored Python/TypeScript config need small, tested boundaries (#67).
   Do not solve this by introducing another backend framework or retry layer.
3. **Documentation drift:** the developer guide simultaneously described
   incremental reconfiguration and claimed all non-weather edits rebuild the
   scene. M9 notes still showed an arrow between independent head labels.
   Contribution notes carried obsolete milestone titles. Those concrete
   contradictions are corrected in this documentation pass.
4. **Compatibility clarity:** this project's supported baseline is CARLA
   0.9.16, not every `ue5-dev` snapshot. List tested CARLA/PythonAPI, OS,
   Worker capability, and model-runtime combinations separately from project
   version `0.9.0`. Do not reset version history just to change positioning.
5. **Release rights/security:** select a license only after checking project
   ownership, reused code, dependencies, models and sample-data rights.
   Ultralytics describes AGPL-3.0 and enterprise licensing; do not assume that
   adding an MIT file resolves the integration's obligations. This is a
   release review requirement, not a legal conclusion. See its
   [licensing guidance](https://www.ultralytics.com/license).

## Where we fit in CARLA

[CARLA Studio](https://github.com/carla-simulator/carla-studio) already documents
GUI setup, simulation control, vehicle import and sensor calibration. The
official [carla-plugins](https://github.com/carla-simulator/carla-plugins)
repository lists CarlaViz, TestPilot and TELECARLA. Browser visualization and
teleoperation are therefore not unique selling points.

Our proposed niche is a lightweight, local, model-oriented research workflow:
connect an existing CARLA host, load a supported/custom detector, inspect road
and voxel estimates, save exact-frame evidence, and revisit a run. Whether
outsiders prefer it must be tested; this review does not establish superiority
over those projects or prove that nobody else offers similar workflows.

There are three different destinations, not one automatic "main registry":

- **Independent OSS tool:** maintain this repository and its supported public
  scope. This does not require inclusion in CARLA core.
- **Community listing/integration:** `carla-plugins` and the
  [CARLA ecosystem](https://ecosystem.carla.org/) are relevant places to ask
  maintainers about inclusion. Listing policies/current intake were not
  established by this review; acceptance is not guaranteed.
- **Upstream contribution:** submit a small independently reproducible CARLA
  bugfix, test, example, or documentation improvement after discussing it.
  Follow the [contribution guidance](https://carla.readthedocs.io/en/latest/cont_contribution_guidelines/)
  and confirm the target branch with maintainers; some documented branch names
  are historical. Do not propose merging this entire application into CARLA.

## A lean release plan with observable exits

| Step | Existing work | Exit condition |
| --- | --- | --- |
| Developer entry point | #58; this documentation pass | A newcomer can locate a feature and its test; stale instructions are corrected. |
| Offline development fixture | #68 | Fresh checkout opens useful stable Garage/Drive/Recordings states without CARLA, credentials or downloads; failures/slow responses can be reproduced. Fixture is visibly synthetic and not scientific evidence. |
| One lifecycle boundary at a time | #67 | Extract one tested use case from transport with unchanged API, actor ownership, timeouts and results. No global rename/restructure. |
| Core live acceptance | #27, #56, #45, #55 | Retained runs cover repeated auto-apply, spawn handoff, population shortfall, disconnect/reconnect, stop/save and playback; requested/applied state and cleanup are explicit. Record latency/FPS, hardware and versions instead of claiming universal 60 FPS. |
| Release hygiene and proof | #58, #33 | Approved license/attribution, security reporting path, history scan, fresh-install trial, redistributable sample, short real demo, limitations and release notes. |

License review and contributor work can proceed alongside reliability fixes.
M2 covers perception/scene integration; M3 covers trained driving validation;
M4 covers OSS release. Do not make full M3 completion a prerequisite for an
honestly scoped research-console release. Current scope lives in
[the milestones](https://github.com/koolfox/carla-local-autonomy/milestones),
not in another duplicated backlog here.

Before announcing broadly, ask three independent users to connect, run a
supported model, record, and find/play the result using only the docs. Record
where they stop and whether they return to use it. Ask one developer to make
and test a small change without chat history. These are proposed acceptance
trials, not results we already have. Use their blockers to choose the next PR.

## Cleanup policy

Keep one repository, the Svelte frontend, thin Worker deployment, model
contracts and research packages. Move nothing merely to shorten the root.
For each deletion candidate, prove its callers, installed commands, UI entry,
tests, replacement parity and rollback. The legacy UI can go only after its
remaining required capabilities have a tested home. Experimental imitation
and voxel modules should be clearly opt-in, not silently removed.

Reduce repeated ownership and supported paths first. A smaller maintainer
mental model is more valuable than an arbitrary line-count target.
