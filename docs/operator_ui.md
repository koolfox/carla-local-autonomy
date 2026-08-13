# CARLA Vision Operator UI

This document describes the current browser Operator/Garage code in the source
tree. It separates software capability from live-simulator evidence: a UI option,
code path, or passing test does not establish closed-loop driving quality.

## Start

```bash
uv run carla-operator-ui --open-browser
```

For a network where the CARLA address is unknown:

```bash
uv run carla-discover
uv run carla-operator-ui --carla-host auto --open-browser
```

`carla-discover` scans only bounded, directly connected private IPv4 LANs and
validates each TCP candidate with read-only CARLA RPC. The Garage exposes the
same operation as **Find CARLA on this network** under Advanced settings through
the token-gated `POST /api/discovery/carla` endpoint. The endpoint accepts no
subnet or arbitrary-port parameter, does not change world state, and does not
retarget a running Operator. If exactly one server is found, `--carla-host auto`
selects it during startup; zero or multiple servers fail with an explicit error.
The World Worker URL/token are intentionally configured separately.

The installed command points to:

```text
carla_vision.operator.garage_server:main
```

The Garage server wraps the existing operator server rather than replacing its
core Drive, job, artifact, and research APIs. It serves the same canonical
`static/index.html`, `app.css`, and `app.js`, uses `GarageDriveSessionManager`
for Drive sessions, and adds one Garage-specific allow-listed research endpoint:

```text
POST /api/garage/jobs
```

The base Operator API remains present underneath it.

## Local HTTP boundary

The base server accepts only these bind names/addresses:

```text
127.0.0.1
::1
localhost
```

State-changing POST requests pass through the existing per-server UI-token check.
The Garage research endpoint uses the same authorization check.

Artifact serving is restricted to registered manifest artifacts under canonical
research roots. The server rejects traversal, absolute paths, unsafe path
components, symlink traversal, unregistered artifacts, and unsupported artifact
types. Untrusted artifact responses use a restrictive content-security policy.

## Two product surfaces

The default surface is a fullscreen CARLA Garage/Drive shell. Research tools are
opened explicitly and replace the game shell until the operator selects Back to
Garage:

```text
Fullscreen Garage / Drive
Research workspace
```

The running game shell removes the former dashboard header, static warning strip,
and top-level tab bar. Its only persistent controls are the status HUD, setup,
camera, experiments, research lab, vehicle carousel, and Start Drive. The base
Research tools surface and its five tabs remain intact.

## Human experiment layer

The Experiments drawer is part of the fullscreen Garage/Drive surface. It does
not start a separate job or introduce another AI controller. A preset applies a
small, visible configuration to the existing controls:

```text
Free Drive
Manual Handling
Autopilot Takeover
Perception Review
Traffic Stress
Adverse Weather
```

Unavailable capability-gated presets remain unavailable rather than silently
falling back. For example, Autopilot Takeover requires World Worker autopilot.
The selected preset is written to the Drive config and summary.

While a Drive is running, the operator can retain one of five human observations:

```text
interesting
false_detection
missed_object
autopilot_issue
scene_issue
```

`POST /api/drive/mark` accepts the active session ID, one strict label, and an
optional single-line note. The server—not the browser—adds elapsed time, current
camera and detector sequence, command source, control mode, and telemetry before
writing the event to `events.jsonl`. The summary retains label counts and the
total number of human markers. These are observational annotations;
`model_output_actuated` remains false for the marker itself.

The Research tools surface currently has five tabs:

```text
Live capture
Scene builder
Workflows
Saved results
Activity
```

## Drive architecture

The existing `carla_vision.operator.drive` engine remains responsible for the
browser camera stream, recording, base actor ownership, browser deadman behavior,
Emergency Brake, weather/prop cleanup, optional spectator restore, and Drive-run
artifact finalization.

`carla_vision.operator.garage_drive` layers additional command sources and
PythonAPI-owned traffic/pedestrian actors on top of that engine.

For model-controlled modes, the Garage code attaches a separate CARLA RGB sensor
for policy input. That sensor is distinct from the browser JPEG stream and from
the optional browser detector overlay.

## Garage runtime control modes

The Garage backend defines four internal Drive control modes:

```text
manual
behavior
imitation
voxel
```

The canonical fullscreen browser currently exposes only Manual and, when the
World Worker advertises it, Traffic Manager Autopilot. Behavior, Imitation, and
Voxel are retained runtime paths, not selectable game-shell modes. Their former
injected selector was deliberately removed with the duplicate Garage assets.

### Manual

Availability: always.

The browser keyboard/touch state is the normal command source. The base Drive
engine applies service-brake fail-safes when the Emergency Brake is latched,
the camera becomes stale, the browser control lease expires, or unsafe reverse
is requested while the vehicle is still moving too quickly.

### BehaviorAgent

Availability requires both:

- the CARLA PythonAPI to be importable by the operator process; and
- `agents.navigation.behavior_agent` to be importable.

The current Garage constructs CARLA `BehaviorAgent` with the selected behavior
style and target speed. It owns throttle, steering, and brake for this mode.
Browser manual-control requests are rejected while autonomy owns the Drive.

The behavior styles exposed by the current Garage are:

```text
cautious
normal
aggressive
```

The policy chooses a destination from CARLA map spawn points, preferring one at
least 80 m from the current ego location when possible, and calls
`BehaviorAgent.set_destination()`. When the agent reports completion it chooses a
new destination.

This is internal agent routing. The Garage UI does not expose an operator-selected
route control.

### Imitation

Availability requires the CARLA PythonAPI to be importable.

The current Garage constructs the imitation driver through:

```text
carla_vision.imitation.predictor:create_driver
```

The mode requires a workspace-contained `.pt`, `.pth`, or `.ckpt` checkpoint.
The policy receives the policy camera frame plus ego speed through the imitation
runtime observation contract.

Current fail-closed behavior visible in `garage_drive.py` includes:

- missing/stale policy camera -> service brake;
- model exception -> service brake;
- repeated model exceptions can latch an error state;
- speed above the configured model speed ceiling suppresses throttle and adds
  braking; and
- browser manual input is disabled while autonomy owns the vehicle.

The presence of this Drive mode does not prove that a trained/compatible
imitation checkpoint is present in a particular workspace.

### Voxel Planner

Availability requires both:

- the CARLA PythonAPI; and
- BehaviorAgent.

The current Garage predictor factory is:

```text
carla_vision.voxel.model_examples.temporal_flow:create_predictor
```

A workspace-contained `.pt`, `.pth`, or `.ckpt` checkpoint is required. A
workspace-contained JSON voxel-readiness report is optional.

The current control split is:

```text
BehaviorAgent -> longitudinal command source
Voxel runtime -> supervised steering decision
```

The Voxel policy obtains the BehaviorAgent command first, runs the camera-voxel
actuation runtime on RGB history, then retains the BehaviorAgent longitudinal
fields while replacing steering only when the supervisor authorizes the voxel
decision.

Current fail-closed behavior visible in the Garage path includes:

- BehaviorAgent baseline failure -> service brake;
- missing/stale voxel policy camera -> service brake;
- no completed voxel decision/warm-up -> service brake;
- decision older than 0.20 s -> service brake;
- supervisor emergency/rejection -> service brake; and
- voxel runtime exception -> service brake, with repeated failures able to latch
  an error state.

This wiring is not a claim of closed-loop voxel-driving quality. That requires a
live simulator run and behavioral metrics.

## Explicit autonomy acknowledgement

`behavior`, `imitation`, and `voxel` are autonomous Garage modes. Starting any
of them requires the explicit `acknowledge_autonomy` flag.

The start contract rejects an autonomous request without that acknowledgement.
The browser also locks manual driving controls while the autonomous session is
running. Emergency Brake remains part of the base Drive engine and is checked
before autonomous policy execution.

The normal detector-overlay option is disabled for autonomous Garage sessions.
This prevents the advisory browser detector path from being confused with the
autonomous policy input path.

## Checkpoint discovery versus compatibility

The Garage server recursively discovers workspace files with these suffixes:

```text
*.pt
*.pth
*.ckpt
```

It skips paths containing `.git`, `.venv`, `__pycache__`, or `node_modules`, and
skips symlink checkpoint files.

The browser applies filename preferences for Imitation and Voxel selections and
falls back to the discovered candidate list when no preferred filename matches.
This is discovery only. The catalog does **not** inspect model architecture or
checkpoint compatibility.

The Drive start contract resolves the selected checkpoint inside the workspace
and rejects a path that escapes it. Actual model compatibility is established
only when the selected runtime loads/uses the checkpoint.

## Traffic and pedestrians

The Garage Drive catalog adds these current capability keys:

```text
pythonapi
behavior_agent
traffic_population
walker_population
autopilot
imitation_drive
voxel_drive
```

Traffic and walker controls are enabled only when the CARLA PythonAPI is
importable.

When requested, the Garage population path can create:

- Traffic Manager-controlled vehicle actors;
- pedestrian actors; and
- AI walker controllers.

Traffic actors are configured for CARLA Traffic Manager autopilot. Walker
controllers are started and assigned navigation destinations when available.
The Garage population object tracks all of those actors/controllers and destroys
them during cleanup.

Their presence in the code does not establish that a particular CARLA server or
PythonAPI installation can spawn the requested population successfully.

## Route behavior: exact current boundary

Without the World Worker, Manual Garage chooses a seeded official CARLA spawn
point and does not create a route. With a capable Worker, the setup drawer
enables Free Drive or Random Destination; the latter requires
GlobalRoutePlanner plus Traffic Manager `set_path` support.

BehaviorAgent mode internally sets CARLA destinations as described above. Voxel
mode embeds the same BehaviorAgent baseline for longitudinal control and therefore
also uses that internal BehaviorAgent destination handling.

So the exact current claim is:

- route selection is capability-gated rather than always available;
- Free Drive has no planned destination;
- Random Destination is planned and enforced only when the Worker advertises it;
- internal BehaviorAgent routing exists for Behavior and Voxel modes.

## Run completion and research handoff

Drive output is retained under `runs/<run-id>/` through the existing Drive
artifact tracker. Depending on enabled options and whether frames/results were
produced, the run can include the manifest/config/summary, control/detection/event
logs, latest JPEG frames, and raw/overlay MP4 recordings.

For a human experiment, `config.json` records `experiment_preset`,
`events.jsonl` stores every marked moment with exact run context, and
`summary.json` stores `human_marker_counts` plus `human_markers_written`.

Manual control logs are marked as browser/manual or fail-safe sources and record
`model_output_actuated: false`. Autonomous Garage control logs include
`control_mode`, fail-safe state, applied command, telemetry, policy detail, and
explicit autonomy/model-actuation metadata.

After a Drive session is successfully saved, its path is shown in the game
shell. The base Research workspace retains Saved results, `analyze`, and
`verify`; refresh its catalog and select the saved run there. The cleaned shell
does not claim a direct handoff that it does not currently implement, and the
Garage does not implement separate analysis or verification engines.

# Research tools surface

## Base Operator job contract

The existing base Operator contract accepts exactly these ten job kinds:

```text
analyze
dataset_qa
live
native_capture
native_preflight
replay
scenario_plan
shadow_matrix
train
verify
```

Each request must contain only the strict schema fields expected by
`OperatorJobRequest`. The command builder maps each kind to validated parameters
and a fixed project module invocation. It does not accept a browser-provided shell
command.

## Live capture

The `Live capture` tab targets an existing CARLA vehicle and camera actor and
launches the existing `carla_vision.runtime` path.

The current form/contract exposes:

- CARLA host/port;
- existing vehicle and camera actor IDs;
- expected map;
- resolution/FPS/FOV;
- RT-DETR or YOLO/model-package selection;
- device/image size/confidence;
- no motion or privileged low-speed simulator-teacher motion;
- no policy or the built-in hazard-stop vision shadow;
- split/overlay/headless viewer selection;
- optional MP4 retention; and
- optional CARLA server spectator follow.

Teacher motion requires explicit acknowledgement. The vision-policy proposal on
this base Live surface is non-actuating; it is a shadow/advisory path. Optional
spectator follow is visualization, not vehicle-control authority.

## Scene builder

The `Scene builder` tab creates a strict situation recipe with current fields for
map, weather, static prop preset, ego spawn index, repetitions, seed, episode
duration, traffic count, pedestrian count, crossing probability, Traffic Manager
speed difference, following distance, camera dimensions/FOV/capture FPS, and ego
blueprint.

Saving a situation produces an Operator situation config. The scenario-plan form
then resolves a selected situation/suite plus split plan into the existing
scenario-planner workflow.

Situation saving and scenario-plan generation are offline operations. They do
not themselves contact CARLA or populate the currently running Drive session.

## Workflows tab

The current base UI exposes these workflow cards:

### Automated dataset QA

Uses the existing dataset-QA backend on a selected manifest-backed dataset and a
new QA run ID. It is a read-only QA/evidence workflow, not human signoff.

### Native readiness and dataset capture

The UI exposes read-only native preflight separately from native capture.
Native capture supports dry-run and requires explicit authorization for the
world-mutating/exclusive-tick path when executed for real.

### Live shadow matrix

Uses a selected shadow-matrix template and a new matrix ID. Planning can remain
o-motion; live execution is separately acknowledged when configured teacher
motion is requested.

### Paired replay

Uses a selected replay template, dataset, and evaluation config with a new replay
ID. The locked-test acknowledgement remains an explicit control in the UI.

### Detector training

Uses an existing training config, dataset, initial weights, run ID, and device.
Dry-run is exposed separately from real compute authorization.

### Analyze recorded run

Launches the existing post-hoc run-analysis backend against a selected retained
runtime/run and writes a separate analysis object.

### Verify research object

Launches the existing recursive verifier with the selected research object and
verification flags. It is distinct from merely opening an object in Saved
results.

## Saved results

The live catalog discovers immediate manifest-backed research objects under the
canonical roots defined by the evidence contract:

```text
datasets
runs
models
reports
bundles
native_kits
operator_sessions
```

The catalog records manifest-declared object status/type/artifact roles/counts
and reports browser inspection status as:

```text
verification_status: not_checked
```

Opening a Saved result reads manifest declarations and checks current file
availability. It does **not** recompute hashes or run semantic verification.

Artifact serving is manifest-only: the requested artifact must be registered by
the selected object's manifest, remain inside the selected object directory, not
traverse symlinks, and use an allow-listed file suffix. Some safe media/text
formats are served inline; model/archive/checksum/YAML-style artifacts use a
download response.

## Activity and operator sessions

Every base or Garage research job submitted through the shared `JobManager`
creates an `operator_sessions/<job-id>/` tracked session.

The job manager retains, when produced:

```text
manifest.json
request.json
command.json
status.json
logs/stdout.log
logs/stderr.log
```

The session records the validated request and resolved command plan. When the
expected child output contains a manifest, the operator session adds a manifest
fingerprint as an input reference.

The Activity surface can list current and retained historical sessions, inspect
status, and tail stdout/stderr. Stopping a running job requests process-group
termination through the job manager.

# Garage-specific research job API

`garage_research.py` accepts exactly these nine additional Garage research kinds:

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

The request schema rejects unknown kinds/parameters. Each kind maps to a fixed
internal project module; a caller cannot provide an arbitrary shell command or
Python module name. These allow-listed plans remain available through the
Garage backend contract, but the cleaned canonical browser does not render the
former injected launcher. Maintained browser workflows live in the base
Research workspace.

## Behavior teacher capture

Maps to:

```text
carla_vision.native.behavior_teacher
```

It requires an existing scenario-plan JSON, dataset ID, BehaviorAgent style,
maximum episode count, and explicit exclusive-tick/map-reload acknowledgement.
The command plan is marked as motion-authorized and destructive.

The Garage server rejects Behavior teacher capture while an interactive Garage
Drive is active.

## Imitation training

Maps to:

```text
carla_vision.imitation.runner
```

It accepts an existing dataset, output/run ID, device, epoch count, and dry-run
flag. Its Garage command plan does not authorize vehicle motion.

## Voxel capture

Maps to:

```text
carla_vision.voxel.capture
```

The Garage launcher supports `teacher` and `rgb-only` modes. RGB-only capture
requires a checkpoint and device and uses the fixed current predictor factory:

```text
carla_vision.voxel.model_examples.temporal_flow:create_predictor
```

A non-dry-run live capture requires a currently running Garage ego and targets
role name `research_drive_ego`. The command plan is non-actuating.

## Voxel flow capture

Maps to:

```text
carla_vision.voxel.flow_capture
```

A non-dry-run live flow capture requires a running Garage ego and targets the
same `research_drive_ego` role. The Garage plan does not authorize vehicle
control.

## Voxel training

Maps to:

```text
carla_vision.voxel.training.runner
```

It uses retained dataset input and writes under the configured voxel model
output root. The Garage plan does not authorize CARLA vehicle control.

## Voxel-flow training

Maps to:

```text
carla_vision.voxel.training.flow_runner
```

It uses retained dataset input and writes under the configured voxel-flow model
output root. The Garage plan does not authorize CARLA vehicle control.

## Voxel shadow

Maps to:

```text
carla_vision.voxel.shadow
```

It requires a checkpoint/device/frame count and uses the fixed temporal-flow
predictor factory. A non-dry-run shadow job requires a running Garage ego. The
command plan observes/scores trajectories and does not authorize actuation.

## Voxel benchmark

Maps to:

```text
carla_vision.voxel.benchmark
```

The Garage benchmark plan is offline over an existing retained run and writes a
new benchmark output. It does not authorize CARLA vehicle control.

## Closed-loop observer

Maps to:

```text
carla_vision.closed_loop_cli
```

The Garage plan targets `research_drive_ego`, records the supplied driver label,
duration, and output run, and is marked non-motion-authorized/non-destructive.
A non-dry-run observer requires a running Garage ego.

This observer plan must not be confused with the active Drive controller. Its
purpose is to observe/retain driving metrics while another Drive control mode
owns the ego.

# Relationship to `carla-local-drive`

The installed local-drive CLI is separate from the browser Garage entrypoint:

```text
carla-local-drive = carla_vision.voxel.local_drive_actuation:main
```

The local-drive voxel actuation path is opt-in and has its own acknowledgement,
predictor, supervisor, and fail-closed contract. Running a voxel shadow job or
voxel benchmark does not enable actuation.

See [`voxel_actuation_fa.md`](voxel_actuation_fa.md) for that CLI contract.

# Canonical fullscreen shell

There is no longer a second Garage JavaScript/CSS bundle injected into the base
page. On load, the canonical application moves the real Garage stream and Drive
camera into one full-viewport stage, moves technical controls into one setup
drawer, and keeps Research as a separate workspace. Preview and Drive retain two
image elements because their transports and lifecycles differ, but only one is
visible in each phase.

The Garage camera menu provides Orbit, Front, Rear, Top, and Cockpit transforms.
The carousel updates the actual Worker-owned CARLA ego rather than swapping a
synthetic browser image.

# Base catalog flag versus Garage autonomy

The base Operator catalog still reports:

```text
vision_control_enabled: false
```

That flag belongs to the original advisory vision-policy path. Internal Garage
runtime capabilities and World Worker Traffic Manager Autopilot are separate
actuation paths. Do not interpret the base flag as proof that every Garage
runtime path is non-actuating.

# Validation model

Three different statements must remain separate.

## 1. Code/contract validation

Examples:

- parsing and validation succeed;
- imports resolve;
- unit/integration tests pass; and
- CI is green.

This validates software contracts, not simulator integration or driving quality.

## 2. Live CARLA integration

A real simulator run must establish facts such as:

- Drive ego creation/attachment succeeds;
- policy camera frames arrive;
- requested traffic/walkers spawn;
- the selected command source actually affects the ego;
- Emergency Brake and failure braking behave as intended;
- owned actors are cleaned up; and
- optional spectator/weather restoration behaves as expected when the CARLA
  episode has not changed.

## 3. Closed-loop driving behavior

Behavioral claims require retained metrics from CARLA, for example:

- collision events;
- lane events;
- route/destination progress where applicable;
- braking/fail-safe activity; and
- completion/success criteria selected for the experiment.

A passing code/CI layer is not a substitute for the other two layers.

# Current factual limitations

- A matching CARLA PythonAPI must actually be installed/importable for
  PythonAPI-dependent Garage features to be offered.
- BehaviorAgent must actually be importable for Behavior and Voxel Garage modes.
- Imitation/Voxel Drive start requires a checkpoint candidate, but suffix/name
  discovery does not prove model compatibility.
- The repository does not guarantee that a trained Imitation or Voxel checkpoint
  exists in a particular clone/workspace.
- The main shell exposes free drive and a random destination only when the World
  Worker confirms GlobalRoutePlanner and Traffic Manager path support.
- Situation Builder plans do not automatically populate a live Garage Drive.
- Saved-results inspection is not cryptographic/semantic verification.
- Behavior/Imitation/Voxel runtime paths exist, but their former injected browser
  selector is intentionally absent from the canonical game shell.
- The supplied CARLA `ue5-dev` PythonAPI source currently targets 0.10.0 and is
  used only as a design reference. The bridge, PythonAPI client, and simulator
  must match at CARLA 0.9.16.
- Cockpit is a generic best-effort interior transform; exact framing can vary
  across vehicle blueprints.
- The browser server is intentionally local-only and is not a multi-user remote
  service.
- Model/policy quality is not established by the existence of the UI/runtime
  path or by passing repository tests.
- Live CARLA results should be retained as experiment evidence rather than
  inferred from API compatibility alone.
