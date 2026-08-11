# CARLA Vision Research

A CARLA 0.9.16 research codebase for monocular RGB perception, data collection,
training, evaluation, local driving, temporal voxel occupancy/flow, and a local
Garage/Cockpit operator UI.

## Claim boundary

This README describes capabilities that are visible in the current source tree
and installed entry points. A code path, passing unit test, or available UI
option is **not** evidence that a driving policy performs well in live CARLA.
Closed-loop behavior must be measured against a running simulator.

## Install

```bash
uv sync --all-groups
```

## Current installed commands

The commands below are the exact `[project.scripts]` entries currently declared
in `pyproject.toml`:

```text
carla-vision
carla-analyze-run
carla-build-evidence-index
carla-collect-dataset
carla-audit-dataset
carla-plan-scenarios
carla-native-preflight
carla-native-collect
carla-record-behavior-teacher
carla-verify-teacher-episodes
carla-compare-teacher-episodes
carla-build-native-kit
carla-verify-native-kit
carla-train-detector
carla-evaluate-detector
carla-replay-models
carla-select-threshold
carla-mine-failures
carla-finalize-failure-review
carla-verify
carla-package-model
carla-build-report
carla-build-reproduction
carla-verify-reproduction
carla-verify-shadow
carla-shadow-matrix
carla-verify-shadow-matrix
carla-operator-ui
carla-local-drive
carla-voxel-test
carla-voxel-flow-capture
carla-train-voxel
carla-train-voxel-flow
carla-voxel-shadow
carla-benchmark-voxel
carla-build-voxel-readiness-evidence
carla-verify-voxel-actuation-readiness
carla-train-imitation
carla-evaluate-drive
carla-summarize-drive-evaluations
```

## Operator UI

Start the current Garage server with:

```bash
uv run carla-operator-ui --open-browser
```

`carla-operator-ui` currently resolves to
`carla_vision.operator.garage_server:main`. The Garage server is additive: it
reuses the existing operator server, Drive session engine, research jobs, and
artifact APIs, then adds Garage drive modes and an allow-listed research-job
endpoint.

### Garage driving modes

The source currently defines four control modes:

| Mode | Code-visible availability condition | Control source |
|---|---|---|
| Manual | always | browser keyboard/touch input |
| BehaviorAgent | CARLA PythonAPI + BehaviorAgent importable | BehaviorAgent |
| Imitation | CARLA PythonAPI importable | RGB + speed imitation policy |
| Voxel Planner | CARLA PythonAPI + BehaviorAgent importable | BehaviorAgent longitudinal control + supervised voxel steering |

Autonomous modes require explicit operator acknowledgement. While an
autonomous mode is active, browser driving input is locked out. The existing
Emergency Brake remains exposed by the Cockpit UI.

Imitation and Voxel modes require a workspace-contained checkpoint ending in
`.pt`, `.pth`, or `.ckpt`. The Garage discovers compatible checkpoint files
inside the workspace. Voxel mode may also receive a workspace-contained JSON
readiness report.

Selecting an autonomous mode disables the normal detector-overlay option for
that drive session; the policy path uses its own CARLA RGB sensor rather than
the browser JPEG stream.

### Traffic and pedestrians

Garage traffic and walker population controls are enabled only when the CARLA
PythonAPI is importable in the operator environment. Spawned traffic vehicles,
walkers, and walker controllers are owned by the drive session and are cleaned
up when that session closes.

The Garage does **not** turn a random spawn point into a planned road route.
Map reload and route planning are separate concerns and are not claimed here as
part of the Garage drive flow.

### Saved-run actions

After a successful saved drive, the Garage integration can:

- open the run in the research/evidence UI;
- launch the existing run analysis workflow;
- launch the existing recursive verification workflow.

These actions reuse existing project backends rather than implementing separate
analysis or verification logic in the browser.

## Garage research launcher

The Garage research endpoint accepts only the following fixed research kinds:

```text
teacher_capture
imitation_train
voxel_capture
voxel_flow_capture
voxel_train
voxel_flow_train
voxel_shadow
voxel_benchmark
closed_loop_evaluate
```

Requests cannot supply an arbitrary shell command or Python module. Each kind
maps to one fixed project module with a validated parameter contract.

Runtime boundaries currently enforced by the server include:

- Behavior teacher capture cannot start while a Garage drive is active.
- Live voxel capture, voxel-flow capture, voxel shadow, and closed-loop
  evaluation require a running Garage ego unless the request is a dry run.
- Voxel capture/shadow attach observation sensors and do not own vehicle
  control.
- The closed-loop evaluator is an observer for collision/lane/route/brake
  metrics and does not issue vehicle control.
- Voxel benchmark is offline over retained artifacts.

## Local drive CLI

`carla-local-drive` currently resolves to
`carla_vision.voxel.local_drive_actuation:main`.

The voxel actuation path is opt-in and retains the existing local-drive path
when voxel actuation is not enabled. Live actuation requires explicit
acknowledgement and a predictor factory/checkpoint configuration appropriate to
the selected predictor.

See [`docs/voxel_actuation_fa.md`](docs/voxel_actuation_fa.md) for the current
voxel-actuation CLI contract.

## Research components present in the source tree

The current package includes code for:

- RT-DETR/YOLO/custom detector runtime and recording;
- synchronized RGB plus privileged teacher capture;
- dataset QA, scenario planning, replay, threshold selection, failure mining,
  evidence indexing, model packaging, reporting, and reproduction bundles;
- official-PythonAPI native collection and BehaviorAgent teacher episodes;
- imitation-policy training and driving evaluation;
- temporal RGB-only voxel occupancy/flow training;
- privileged voxel/flow teacher capture;
- voxel shadow planning, benchmarking, readiness checks, and opt-in actuation;
- closed-loop driving evaluation and evaluation summaries;
- local Operator/Garage orchestration over those existing backends.

The presence of these modules is a software-capability statement only. It does
not imply that a trained checkpoint is present, that a particular model has
been trained successfully, or that closed-loop CARLA performance has been
validated.

## Safety and privileged-data boundary

Privileged CARLA signals may be used by teacher-data generation and evaluation
components. Deployable imitation and temporal voxel model paths are intended to
consume RGB-derived inputs rather than privileged simulator state.

Shadow/observer components are separate from actuation components. Do not infer
that a shadow result or offline benchmark authorizes vehicle actuation.

## Tests versus simulator validation

Use repository tests and CI for code/contract regressions. Use live CARLA runs
for simulator integration, and closed-loop metrics for driving behavior. Keep
those three claims separate:

```text
CODE / CONTRACT     -> static checks, unit/integration tests, CI
LIVE CARLA          -> spawn, sensors, actuation, cleanup, simulator response
DRIVING BEHAVIOR    -> collisions, lane events, route progress, braking, success
```

A green first line does not automatically make the other two green.
