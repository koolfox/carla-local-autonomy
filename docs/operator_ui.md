# CARLA Vision Operator UI

This document describes the current browser Operator/Garage code. It does not
claim that an autonomous policy has good closed-loop driving performance merely
because its UI and runtime path exist.

## Start

```bash
uv run carla-operator-ui --open-browser
```

The installed command points to:

```text
carla_vision.operator.garage_server:main
```

The Garage server wraps the existing operator server rather than replacing its
core Drive, job, artifact, and research APIs. It injects Garage JavaScript/CSS,
uses `GarageDriveSessionManager` for Drive sessions, and adds one allow-listed
Garage research endpoint:

```text
POST /api/garage/jobs
```

State-changing Garage research requests pass through the existing UI-token
authorization check.

## Drive architecture

The existing `carla_vision.operator.drive` engine remains responsible for the
browser camera stream, recording, actor ownership, deadman behavior, Emergency
Brake, and cleanup. `garage_drive.py` layers additional command sources on top
of that engine.

For model-controlled modes, the Garage code attaches a separate CARLA RGB
sensor for policy input. That policy sensor is distinct from the browser JPEG
stream.

## Control modes

The Garage defines exactly four modes:

```text
manual
behavior
imitation
voxel
```

### Manual

Availability: always.

The browser keyboard/touch state is the normal command source. This is the
existing manual Drive behavior.

### BehaviorAgent

Availability requires:

- the CARLA PythonAPI to be importable by the operator process; and
- the CARLA BehaviorAgent package to be importable.

BehaviorAgent owns steering, throttle, and brake. Browser driving input is
locked while the autonomous session is running.

The Garage exposes the current behavior styles:

```text
cautious
normal
aggressive
```

and a target-speed selector.

### Imitation

Availability requires the CARLA PythonAPI to be importable.

The current Garage code constructs the imitation driver through:

```text
carla_vision.imitation.predictor:create_driver
```

The mode requires a workspace-contained `.pt`, `.pth`, or `.ckpt` checkpoint.
The policy receives RGB plus the speed observation expected by the imitation
runtime. Browser driving input is locked. Runtime/policy failure is handled as
a braking failure-safe in the Garage drive path.

The existence of this mode does not prove that a trained imitation checkpoint
is present in the workspace.

### Voxel Planner

Availability requires:

- the CARLA PythonAPI; and
- BehaviorAgent.

The current predictor factory used by the Garage is:

```text
carla_vision.voxel.model_examples.temporal_flow:create_predictor
```

A workspace-contained `.pt`, `.pth`, or `.ckpt` checkpoint is required. A
workspace-contained JSON voxel-readiness report is optional.

The current Garage control split is:

```text
BehaviorAgent -> longitudinal control
Voxel planner  -> supervised steering
```

The UI states that rejected voxel predictions apply full brake. Browser driving
input is locked while the autonomous mode is active.

This wiring is not a claim of closed-loop voxel-driving quality. That requires a
live simulator run and behavioral metrics.

## Explicit autonomy acknowledgement

`behavior`, `imitation`, and `voxel` are autonomous modes. Starting any of them
requires the operator to set the explicit autonomy acknowledgement flag.

The start contract rejects an autonomous request without that acknowledgement.
The Cockpit keeps Emergency Brake available while browser driving input is
blocked.

The normal detector-overlay option is disabled for autonomous Garage sessions.
That avoids treating the advisory browser detector path as the autonomous
policy input path.

## Traffic and pedestrians

The Garage drive catalog exposes these capability flags from the current
operator environment:

```text
pythonapi
behavior_agent
traffic_population
walker_population
autopilot
imitation_drive
voxel_drive
```

Traffic and walker population controls are enabled only when the CARLA
PythonAPI is available.

When requested, the Garage session can create:

- Traffic Manager controlled vehicle actors;
- pedestrian actors; and
- AI walker controllers.

Those actors/controllers are tracked by the Garage session and cleaned up when
the population/session closes.

Traffic and pedestrian availability in the code is not evidence that a specific
local CARLA installation is correctly configured; the operator environment must
actually provide the matching PythonAPI at runtime.

## What the Garage still does not claim

The current Garage does not turn its seeded random spawn into a validated
road-following route. Route planning is not part of the current Garage drive
contract.

The additive Garage work also does not make the Situation Builder automatically
populate a live Drive session. Situation/scenario planning and interactive
Garage driving remain distinct paths unless explicitly connected by a specific
workflow.

## Checkpoint discovery

The Garage server recursively discovers these checkpoint suffixes under the
workspace:

```text
*.pt
*.pth
*.ckpt
```

It skips common implementation directories such as `.git`, `.venv`,
`__pycache__`, and `node_modules`, and does not accept symlink checkpoint files.

The start contract resolves the selected policy checkpoint inside the workspace
and rejects paths that escape it.

## Run completion and research handoff

After a Drive session is successfully saved, the Garage integration provides
three direct actions:

- **Inspect saved run** — refreshes the catalog and opens the exact saved run in
  the evidence/results surface.
- **Analyze saved run** — launches the existing run-analysis workflow for that
  run.
- **Verify saved run** — launches the existing recursive verifier for that run.

The Garage does not implement a second analysis or verification engine; it
reuses the existing project workflows.

## Garage research jobs

`garage_research.py` accepts exactly these research kinds:

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

The request schema is fixed and rejects unknown kinds/parameters. The browser
cannot provide an arbitrary shell command or Python module name. Each research
kind maps to a fixed internal project module.

### Behavior teacher capture

Maps to:

```text
carla_vision.native.behavior_teacher
```

It requires explicit acknowledgement because it can load scenario maps, own
world ticks, and record BehaviorAgent controls. The Garage server rejects this
job while an interactive Garage drive is active.

### Imitation training

Maps to the existing imitation trainer. Training accepts an existing dataset,
output/run ID, device, epoch count, and dry-run flag. It does not issue CARLA
vehicle control.

### Voxel capture

Maps to:

```text
carla_vision.voxel.capture
```

The current Garage launcher supports teacher and RGB-only capture modes. The
RGB-only mode requires a compatible checkpoint and predictor device.

A non-dry-run live capture requires a currently running Garage ego. The capture
path attaches temporary observation sensors and does not own vehicle control.

### Voxel flow capture

Maps to:

```text
carla_vision.voxel.flow_capture
```

A non-dry-run live flow capture requires a running Garage ego. It is an
observation/data path, not a vehicle-control owner.

### Voxel training and voxel-flow training

These map to the existing voxel training modules and operate on retained
training data. They do not issue CARLA vehicle commands.

### Voxel shadow

Maps to:

```text
carla_vision.voxel.shadow
```

A non-dry-run shadow job requires a running Garage ego. The shadow path observes
and scores trajectories; it is not the actuation path.

### Voxel benchmark

Maps to:

```text
carla_vision.voxel.benchmark
```

The Garage benchmark plan is offline over retained run artifacts and does not
connect to CARLA vehicle control.

### Closed-loop observer

Maps to:

```text
carla_vision.closed_loop_cli
```

The Garage labels this as an observer for collision, lane, route, and braking
metrics. The plan does not authorize vehicle control. A non-dry-run observer
requires a running Garage ego.

## Relationship to `carla-local-drive`

The installed local-drive CLI is separate from the browser Garage entrypoint:

```text
carla-local-drive = carla_vision.voxel.local_drive_actuation:main
```

The local-drive voxel actuation path is opt-in. Do not infer that starting a
voxel shadow job or running a voxel benchmark enables actuation.

## Validation model

Three different statements must remain separate:

### 1. Code/contract validation

Examples:

- parsing and validation succeed;
- imports resolve;
- unit/integration tests pass;
- CI is green.

This validates software contracts, not driving quality.

### 2. Live CARLA integration

A real simulator run must establish facts such as:

- ego actor attachment/spawn succeeds;
- policy camera frames arrive;
- requested traffic/walkers spawn;
- the selected command source actually affects the ego;
- Emergency Brake/failure braking behaves as intended;
- owned actors are cleaned up.

### 3. Closed-loop driving behavior

Behavioral claims require metrics from CARLA, for example collision events,
lane events, route progress, braking/failsafe activity, and completion/success
criteria chosen for the experiment.

A passing code/CI layer is not a substitute for the other two.

## Current factual limitations

- A compatible CARLA PythonAPI must actually be installed/importable for the
  PythonAPI-dependent Garage features to become available.
- BehaviorAgent must actually be importable for Behavior and Voxel Garage modes.
- Imitation/Voxel modes require a compatible checkpoint file; the repository
  does not guarantee that a trained checkpoint exists in a particular clone.
- The Garage uses a seeded spawn point, not a route planner.
- Model/policy quality is not established by the existence of the UI or runtime
  path.
- Live CARLA results should be recorded as experiment evidence rather than
  inferred from API compatibility alone.
