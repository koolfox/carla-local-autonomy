# CARLA Vision Framework Validation Report v0.6

Date: 2026-07-26  
Status: development validation  
Primary sensor boundary: one forward-facing monocular RGB camera  
Primary detector family: RT-DETR  
Operator surface: loopback-only proof of concept  
CARLA target: 0.9.16 at `172.20.10.7:2000`

## 1. Executive conclusion

The framework now has a small local operator console over the existing
research CLIs. It controls live detection and recording, situation recipes,
offline scenario planning, native collection, shadow matrices, paired replay,
training, analysis, and verification without accepting arbitrary shell
commands.

Every launch becomes a checksum-tracked operator session containing the
normalized request, exact tokenized command, status, stdout, stderr, hardware
and environment envelope, and successful child-manifest fingerprint. Existing
scientific producers still own their outputs; the UI is only an orchestration
and provenance layer.

An API-driven smoke test used the UI backend to save a crowded Town10HD
situation and launch its real scenario planner. The job completed with code
zero and produced one deterministic training episode with 75 planned camera
captures. Planning was offline and neither movement nor simulator mutation was
authorized.

This validates the operator workflow and situation-to-plan path. It does not
validate native spawning, dataset collection, detector training, locked-test
accuracy, or vision-only closed-loop driving.

## 2. Operator POC

Start the console with:

```bash
uv run carla-operator-ui --open-browser
```

The server binds to:

```text
http://127.0.0.1:8765/
```

Its four surfaces are:

1. **Live & Record** — choose CARLA actors, RT-DETR/YOLO weights, device,
   threshold, annotated view, recording, privileged teacher motion, and a
   non-actuating policy shadow.
2. **Situation Builder** — choose map, weather, traffic, walkers, crossing
   probability, traffic-manager behavior, props, camera, duration,
   repetitions, and seed.
3. **Research Workflows** — plan/execute native capture and shadow matrices,
   replay models, train, analyze, and verify.
4. **Sessions** — inspect jobs, stop active processes, and open retained logs,
   manifests, images, and videos.

The UI deliberately opens the existing OpenCV live viewer instead of adding a
second browser-streaming video pipeline. It exposes the controls needed for a
thesis MVP without adding a database, scheduler, user system, or remote
deployment.

## 3. Safety and integrity boundaries

- The server refuses non-loopback bind addresses.
- State-changing HTTP requests require a per-process token.
- Requests select explicit workflows and typed parameters; shell strings are
  not accepted.
- Input paths must remain inside the workspace and existing outputs are never
  overwritten.
- Teacher motion, native world mutation, locked-test access, and real training
  require separate acknowledgements.
- `--control vision` remains disabled. Vision-policy proposals are display and
  logging artifacts only.
- Native dry-run and scenario planning do not import CARLA or contact the
  simulator.
- Each UI launch is retained under `operator_sessions/<job-id>/` and can be
  recursively verified.

The token is a local cross-origin guard, not remote-user authentication.
Because the panel can launch local processes and explicitly authorize
simulator mutation, it is intentionally not hosted as a public web service.

## 4. Situation smoke

The saved strict suite is:

```text
operator_configs/situations/town10-ui-poc-v1.json
```

Its main values are:

| Field | Value |
|---|---:|
| Map | `Town10HD_Opt` |
| Weather | soft rain at sunset |
| Traffic vehicles | 50 |
| Pedestrians | 40 |
| Crossing probability | 0.25 |
| Traffic speed difference | 25% |
| Following distance | 3 m |
| Props | construction sign, barrier, cone |
| Camera | 1280x720, 90 degree FOV |
| Capture rate | 5 FPS |
| Duration | 15 s |
| Repetitions | 1 |
| Master seed | 20260726 |

The real planner output is:

```text
runs/scenario-plan-operator-poc-v1
```

It contains one episode assigned to the development training partition and 75
planned exact-frame captures. The plan records fixed-step synchronous
execution, single tick ownership, world reload per repetition, exact sensor
frame matching, hierarchical seed allocation, the front-monocular-RGB runtime
contract, and the privileged-teacher disclosure.

The launching operator session is:

```text
operator_sessions/op-20260726T195126Z-scenario_plan-e715b442
```

It records:

```text
status                  success
returncode              0
motion_authorized       false
destructive             false
expected_output_exists  true
```

Its child scenario-plan manifest is fingerprinted as an external reference.

## 5. Retained defect evidence

The first verification smoke,
`op-20260726T194828Z-verify-d785b0f1`, exposed that invoking an internal module
with `python -m` could exit successfully when the module did not contain a
module-level entry point. That session is preserved as development evidence
and is not cited as successful verification.

The command builder now invokes each allow-listed internal `main()` explicitly.
The replacement real verification session,
`op-20260726T194948Z-verify-e6436e1d`, produced the expected recursive
verification output and passed generic artifact verification itself.

No historical artifact was rewritten.

## 6. Manifest-generated report

`reports/rpt-framework-validation-20260726-v004` aggregates five verified
sources:

1. the v0.5 framework report;
2. the v0.5 standalone reproduction bundle;
3. the successful operator verification session;
4. the crowded-situation scenario plan;
5. the operator scenario-planning session.

The release contains:

| Field | Value |
|---|---:|
| Verified source objects | 5 |
| Canonical metric rows | 23 |
| Source-artifact inventory rows | 44 |
| Registered report artifacts | 9 |
| Payload checksum entries | 8 |
| External source references | 5 |
| Unregistered files | 0 |

Operator metrics are deliberately operational: job success, return code,
motion authorization, destructive-operation flag, stop request, and expected
output existence. They are not detector-performance metrics.

The report manifest SHA-256 is:

```text
ce149284941dd190175001f0f7d48679edde384821f140bd869b144867dd373b
```

## 7. Regression evidence

The final pre-bundle regression completed:

```text
157 passed, 44 subtests passed
```

Additional gates passed:

- Ruff lint;
- Ruff format check over 162 files;
- Python byte-code compilation;
- JavaScript syntax check;
- HTML parser smoke;
- package lock consistency;
- wheel build with all three static UI assets;
- real HTTP bootstrap and job API smoke;
- deep verification of both successful operator sessions, the scenario plan,
  and the v0.6 report.

## 8. Defensible claim boundary

The new evidence supports this claim:

> The development framework can translate a small local operator form into
> strict immutable research commands, retain operational provenance, and
> create a deterministic crowded-situation plan without authorizing vehicle
> motion or simulator mutation.

It does not support these claims:

- every requested actor or prop can spawn in the live CARLA world;
- the planned dataset has been collected or audited;
- RT-DETR has been trained on the planned scenarios;
- any detector meets a thesis accuracy target;
- any vision policy safely controls the vehicle;
- a local development session is confirmatory thesis evidence.

## 9. Next acceptance sequence

1. Create a clean Git baseline for confirmatory provenance.
2. Install the matching official CARLA 0.9.16 PythonAPI on the execution host.
3. Run one short native preflight and manually inspect actor, prop, weather,
   RGB, and instance-label alignment.
4. Execute and audit a small multi-situation pilot before scaling density.
5. Freeze an episode-grouped train/validation/locked-test dataset.
6. Pass tiny-overfit, resume, and one-epoch training gates.
7. Train at least three RT-DETR seeds and fair comparison baselines.
8. Select thresholds on validation only, complete blinded failure review, and
   evaluate the locked test once.
9. Replay released model packages on the frozen scenario matrix.
10. Design temporal monocular state, route intent, safety supervision, and
    intervention accounting before enabling any vision actuation.

Until those gates pass, all current artifacts remain development evidence and
vision control remains disabled.
