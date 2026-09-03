# Developer guide

This is the shortest supported path from a clean checkout to a safe,
reviewable feature. It describes the product as it exists now; future work is
linked as an issue instead of being presented as current behavior.

## Start here

Prerequisites:

- Python 3.11-3.13;
- `uv`;
- Node.js 24 only when changing `web/`; and
- in the normal production topology, the matching official CARLA 0.9.16
  PythonAPI only on the Windows CARLA host.

The opt-in BehaviorAgent, Imitation, and Voxel Garage modes currently also
import the PythonAPI in the Operator process. That is an experimental legacy
constraint, not the target thin-Worker topology.

Install the lean developer profile:

```bash
uv sync --group dev
```

For a running CARLA/Worker pair, copy the environment template and put the same
fresh Worker token used on Windows in `.env.local`:

```bash
cp .env.example .env.local
```

Then start the packaged product:

```bash
uv run carla-operator-ui \
  --carla-host auto \
  --world-worker-url auto \
  --open-browser
```

When `.env.local` contains a Worker token and no Worker URL was supplied, the
packaged entrypoint selects `auto` implicitly and follows the resolved CARLA
host on port 8766. An explicit CLI or environment URL still wins.

Automatic discovery intentionally fails when no CARLA server is present. For
frontend/API work with the simulator offline, use an explicit unreachable local
endpoint and omit the Worker:

```bash
uv run carla-operator-ui --carla-host 127.0.0.1 --open-browser
```

The Operator will expose unavailable capabilities, but live preview, world
mutation, and driving will not work. A deterministic no-CARLA product fixture
with useful runtime states is tracked in Issue #68.

For Svelte development, keep the Operator on port 8765 and run Vite in another
terminal:

```bash
cd web
npm ci --no-audit --no-fund
npm run dev
```

Open `http://127.0.0.1:5173/`. Vite proxies `/api` to
`http://127.0.0.1:8765` by default. Set `CARLA_OPERATOR_API` only when the
Operator uses a different local address.

## Four runtime roles

Do not read the repository as one large process. It has four roles with
deliberately different dependencies and ownership.

| Role | Runs where | Owns | Does not own |
| --- | --- | --- | --- |
| Svelte console | Browser/research computer | Operator intent, forms, live presentation, touch/keyboard input | CARLA actors, model execution, artifact truth |
| Operator application | Research computer | Session lifecycle, validation, inference, recording, research jobs, retained artifacts | Native CARLA world mutation |
| World Worker | Windows CARLA host | Official PythonAPI calls, map/weather/actors, Traffic Manager, camera relay, cleanup | Torch, Ultralytics, research workflows, browser state |
| Research/model jobs | Research computer or an explicit native capture environment | Dataset, training, evaluation, reports, model packages, verification | Interactive UI state or hidden simulator control |

The normal deployment is:

```text
browser/Svelte
      |
      | local JSON + MJPEG
      v
Operator :8765 -------- model inference / recording / artifacts
      |
      | authenticated LAN Worker protocol
      v
World Worker :8766 ---- official PythonAPI ---- CARLA :2000
```

The Operator HTTP server binds to loopback. The World Worker is a narrow LAN
adapter and must never be exposed to the public internet.

## Authoritative source map

| If you are changing... | Start here | Then inspect |
| --- | --- | --- |
| Svelte page composition | `web/src/routes/+page.svelte` | `web/src/lib/components/` |
| Garage menus and modal ownership | `web/src/lib/components/GarageMenu.svelte` | `Modal.svelte`, task-specific components |
| Shared Scene/Research fields | `web/src/lib/components/SceneWorldFields.svelte` | `SceneSettings.svelte`, `SceneBuilder.svelte` |
| Shared Garage/Drive vehicle selector | `web/src/lib/components/VehiclePicker.svelte` | `GaragePreview.svelte`, `DriveSettings.svelte` |
| Editable session state | `web/src/lib/domain/config.ts` | `web/src/lib/stores/configuration.ts` |
| Browser API/runtime state | `web/src/lib/api/operator.ts` | `web/src/lib/stores/runtime.ts` |
| Canonical Operator configuration | `carla_vision/operator/configuration.py` | `carla_vision/operator/drive_contracts.py` |
| Preview lifecycle | `carla_vision/operator/garage_preview.py` | `carla_vision/operator/world_worker_client.py` |
| Drive lifecycle/control | `carla_vision/operator/garage_drive.py` | `carla_vision/operator/drive.py` |
| HTTP transport and research jobs | `carla_vision/operator/server.py` | `commands.py`, `jobs.py` |
| Windows CARLA behavior | `carla_vision/native/observable_world_worker.py` | `carla_vision/native/world_worker.py` |
| External runtime models | `carla_vision/model_package_contracts.py` | `model_registry.py`, `torchscript_driver.py` |
| Detector adapter | `carla_vision/detectors/` | `carla_vision/perception.py` |
| Imitation research | `carla_vision/imitation/` | teacher code under `carla_vision/native/` |
| Voxel research/control | `carla_vision/voxel/` | voxel configs and focused tests |
| Dataset/evaluation/report artifacts | the producer package under `carla_vision/` | its `contracts.py` and verifier |

The installed backend entrypoint currently resolves through this chain:

```text
pyproject.toml: carla-operator-ui
  -> operator/product_console.py       packaged Svelte transport
  -> operator/local_entrypoint.py      .env.local and launch resolution
  -> operator/garage_server.py         Garage/preview/session routes
  -> operator/server.py                base application, jobs, artifacts, Drive API
```

This inheritance chain is current implementation, not the desired location for
new business logic. Issue #67 moves use cases behind a stable application seam
one slice at a time.

`pyproject.toml` under `[project.scripts]` is the authoritative installed-command
registry. The primary operational commands are:

| Command | Role |
| --- | --- |
| `carla-operator-ui` | packaged Svelte product plus local Operator API |
| `carla-world-worker` | thin official-PythonAPI Worker on the CARLA host |
| `carla-discover` | bounded read-only LAN discovery |
| `carla-vision` | lower-level vision runtime |

Dataset, training, evaluation, replay, failure, report, and verification
commands in the same registry are research producers. Their `--help`, focused
package documentation, and retained artifact contract are authoritative; do
not add a second wrapper command merely to expose an existing producer.

### Source, generated output, and local state

- `web/` is the only product frontend source.
- `carla_vision/operator/console_static/` is generated by `npm run build`, is
  reviewed and committed with the Svelte source, and must never be hand-edited.
- `carla_vision/operator/static/` is a temporary legacy rollback surface. Do
  not add product features there.
- `models/`, `datasets/`, `runs/`, `reports/`, `operator_sessions/`, and
  `.env.local` are local/runtime state and are not source code.
- `carla_vision/native/observable_world_worker.py` must remain runnable beside
  CARLA without installing this project or the ML stack.

## Current request and data paths

These traces are the quickest way to find the owner of a behavior.

### Garage preview

```text
GaragePreview.svelte
  -> OperatorApi.configureGaragePreview
  -> POST /api/garage/preview/configure with one SessionConfig
  -> operator/configuration.py canonical preview mapping
  -> GaragePreviewManager.configure
  -> WorldWorkerClient.prepare_scene (first load) / configure_scene (deltas)
  -> World Worker actor ownership
  -> preview state + persistent MJPEG frame stream
```

While Garage is visible, valid scene/camera edits apply automatically after
300 ms without further edits. One browser request runs at a time; the latest
pending selection replaces earlier selections. There is no ordinary Apply
button. Retry appears only after a failure. The shared projection and queue
live in `web/src/lib/domain/garagePreview.ts` and are tested with `npm test`.
Starting Drive suspends this queue. Model, recording and other Drive-only
fields never trigger a Garage rebuild.

### What a Garage edit changes

| Edit | Native work |
| --- | --- |
| Weather, speed difference, following distance | Direct CARLA/TM setters |
| Pedestrian crossing factor | Set factor, then refresh existing walker destinations |
| Traffic / walker count | Spawn/remove only the difference; preserve other owned actors |
| Fixed-item preset | Replace only scene-owned props |
| Vehicle / color | Replace only ego at the same transform; retain unparented camera and population |
| Camera profile / FOV | Replace only RGB camera; keep the browser stream and last good frame |
| Route mode | Update prepared route; no population rebuild |
| Map / seed | New prepared scene; map loading occurs only when the map differs |

These deltas require `prepared_scene_reconfigure` on the Worker. An older
Worker safely falls back to scene preparation; pull and restart the Windows
Worker as well as the Mac Operator to use the fast path. No ML dependencies
were added to the Worker. The scene remains parked until Start.

`POST /v1/scenes/{id}/configure` uses the existing SceneConfig plus lease token.
The native response includes confirmed scene/camera configuration. After an
ambiguous or partially failed update, the Operator reads it back before
accepting another update or transferring a scene to Drive. Unknown state is
an error, never proof that requested settings were applied. A dense update
must not trigger camera-stale cleanup of its own scene.

Camera transport close interrupts active HTTP reads rather than waiting for
the normal frame timeout. The public frame sequence stays increasing when
the replacement sensor starts again at zero. Apply acknowledges camera
configuration without a second blocking first-frame wait; the image updates
when the new sensor produces a frame. Camera restart/map loading and real
population capacity still depend on CARLA, not the browser.

#### Native population and camera lifecycle

`native/world_worker.py` uses the official `carla.command.SpawnActor` /
`client.apply_batch_sync` pattern from
[generate_traffic.py](https://github.com/carla-simulator/carla/blob/0a5ce0d5b4952bd8294a163c12d49f197bdb2aba/PythonAPI/examples/generate_traffic.py).
Vehicles are prepared in chunks of at most 32; walkers and their controllers
use separate batches of at most 24. Every live batch passes `do_tick=False`.
The Worker observes natural ticks before resolving newly created actors; it
never becomes a synchronous tick owner. Clients without the command API retain
the serial compatibility path.

Spawn responses establish ownership before snapshot lookup. Known command
failures may retry only the deficit. A timed-out batch, unobserved successful
ID, or unconfirmed rollback aborts preparation instead of blindly spawning
replacements. Timeout recovery can identify current scene-tagged actors and
controllers attached to owned walkers; it cannot prove cleanup if CARLA stays
unreachable. Requested population counts are never silently reduced.

Camera close gates new callbacks, stops the listener, checks encoder shutdown,
and allows the 0.4-second native drain interval used by CARLA's
[rapid camera-switch fix](https://github.com/carla-simulator/carla/commit/ffd9d275cb07d0f9cdc49c45c1e3e31c110d1d67).
Failed detach/drain retains ownership for retry instead of destroying the
sensor underneath a callback. A confirmed-dead sensor still drains queued work.

Regression tests cover command counts, delayed snapshots, failed commands,
ownership rollback, camera switching and standalone execution. These fakes do
**not** establish Windows simulator latency. After pulling on Windows, restart
the existing Worker command and compare the same map/seed/population on a real
run. No new dependency or control mode is introduced.

For **Free Drive**, Start transfers a matching ready Garage lease to the Drive
session. It does not stop/prepare the population again. Preview transports and
heartbeat close before transfer; Drive renews ownership and replaces only the
unparented Garage camera with an ego-attached front camera. The ego remains
braked until Drive has a frame and activates the selected control mode. Random
Destination is planned during Garage preparation rather than at Start.

This requires the Worker's `prepared_scene_handoff` capability. Older Workers,
changed scene settings, and controlled experiment presets retain the fresh
prepare path. Validation runs before consuming the preview. Runtime status
reports `reused_garage_scene`; the recorded config states
`scene_origin: garage_preview` or `fresh`. Continuing a preview is not a fresh
seeded experiment reset.

Remaining stabilization work: Stop & Save still releases the Drive scene and
reopening Garage creates a fresh scene. Non-weather scene changes still rebuild
the population; incremental vehicle/population editing is a separate change.
Do not mistake batching or start handoff for completion of those acceptance gates.

### Drive

```text
SessionLaunchBar.svelte
  -> POST /api/session/start with one SessionConfig
  -> canonical SessionConfig-to-drive mapping
  -> GarageDriveSessionManager
  -> manual / CARLA baseline / registered model session
  -> controls, frames, detections, events, video
  -> run manifest and final cleanup
```

Emergency braking and cleanup are application/runtime responsibilities, not UI
conventions.

### Research job

```text
validated OperatorJobRequest
  -> POST /api/jobs
  -> allow-listed CommandPlan (token array, never a shell string)
  -> JobManager subprocess
  -> operator session manifest + request + command + logs + status
  -> the selected research producer's own verified output
```

The Operator launches and observes research producers; it does not duplicate
their scientific artifact formats.

### Registered model

```text
models/<model-id>/model.json
  -> discovery and manifest validation (no deserialization)
  -> explicit trust acknowledgement for executable content
  -> runtime-specific adapter
  -> canonical perception or ModelControl contract
  -> safety/freshness gate
  -> exact package, artifact, adapter hashes in run lineage
```

A loose `.pt` file is not a runnable model identity. Downstream code must not
import framework-specific result objects.

### Retained artifact

```text
producer
  -> unique output directory
  -> manifest + registered files + hashes + source references
  -> semantic verifier
  -> Evidence/Saved Results presentation
```

File existence is not verification. The producer and verifier own artifact
truth; the UI displays their result.

## Add one feature as a vertical slice

Use this order. It prevents UI-only controls and backend-only options from
becoming disconnected islands.

1. **Write the outcome and acceptance criteria.** Use one GitHub issue and
   identify live-CARLA evidence if simulator behavior changes.
2. **Change the canonical meaning once.** Add the field or operation with one
   definition of units, range, default, and capability rules. Today the Python
   contract in `operator/configuration.py` and its TypeScript mirror in
   `web/src/lib/domain/config.ts` must both change; #54/#67 remove silent
   parallel assumptions rather than pretending that duplication is solved.
3. **Implement an application use case.** Validation and ownership decisions
   belong here, not inside an HTTP route or Svelte component.
4. **Use a narrow adapter.** CARLA, the World Worker, model runtimes, and the
   filesystem stay behind explicit boundaries.
5. **Expose a boring transport.** HTTP parses, calls the use case, and
   serializes a stable success/error shape.
6. **Connect the Svelte store and component.** A component edits the canonical
   store; it must not create a hidden second copy.
7. **Return evidence.** Simulator settings should distinguish requested,
   resolved/clamped, and actually applied values.
8. **Test in layers.** Contract, use case, adapter, UI/build, then live CARLA
   when the behavior depends on the simulator.

### Example: pedestrian crossing factor

The intended unified trace is:

```text
SceneSettings.svelte
  -> SessionConfig.scene.pedestrianCrossingFactor
  -> configuration store
  -> POST /api/session/start or explicit Preview
  -> Operator canonical mapping
  -> WorldWorkerClient request
  -> world.set_pedestrians_cross_factor(...)
  -> requested/resolved/applied scene evidence
```

Do not add a separate Research-only crossing value. Research presets and
situation recipes must patch or reference the same `SessionConfig` field.
Preview, Drive, and Situation recipe creation now map this value through
`operator/configuration.py`. `SceneWorldFields.svelte` is reused by the Scene
and Research modals and edits the same store. New workflows must consume that
field or a retained SessionConfig reference rather than introducing another
crossing control.

## Where should a new file go?

- Browser rendering or interaction: `web/src/lib/`.
- Shared editable session meaning: the canonical SessionConfig contracts.
- A user operation spanning components: Operator application/session layer.
- CARLA PythonAPI call or actor lifecycle: `carla_vision/native/`.
- Network translation to the Worker: `operator/world_worker_client.py` or its
  future adapter package.
- Model-specific preprocessing/output: an adapter beside the relevant model
  contract, never the UI or planner.
- Research algorithm: its focused package, with a contract and verifier when
  it produces a retained object.
- Generated data: an ignored runtime root, not a new source directory.

If a change seems to belong in three unrelated places, define the contract and
application boundary before adding files.

## Verification commands

Run the smallest focused check first, then the affected suite.

Python:

```bash
uv run ruff check carla_vision tests
uv run pytest tests/test_relevant_module.py -q
```

Svelte:

```bash
cd web
npm ci --no-audit --no-fund
npm run check
npm run build
```

The build updates `carla_vision/operator/console_static/`. Commit that generated
bundle with its Svelte source. CI rebuilds it and rejects drift.

Complete local suite:

```bash
uv sync --all-extras --all-groups
uv run pytest -q
```

Useful entrypoint checks:

```bash
uv run carla-world-worker --help
uv run carla-operator-ui --help
uv run carla-discover --help
uv run carla-vision --help
```

Fake/unit tests prove contracts and failure handling. They do not prove map
reloads, actor counts, pedestrian crossing behavior, remote FPS, vehicle
control, or cleanup in real CARLA. Retain a live acceptance result for those
claims.

## One issue, branch, and pull request

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/<issue>-<short-name>
```

Before editing, comment on the issue with the branch and expected file
ownership. Keep unrelated cleanup out of the branch. A pull request must state:

- the issue it closes or advances;
- the user-visible outcome;
- contracts intentionally unchanged;
- automated checks run;
- live-CARLA evidence or an explicit pending statement; and
- migration/rollback behavior where state or artifacts change.

After merge, delete the branch. Do not keep an integration branch as a second
source of truth.

## Definition of done

A feature is done only when:

- its contract and ownership are explicit;
- unsupported states fail factually instead of silently resetting;
- focused tests pass;
- Svelte source and generated bundle agree when the UI changed;
- artifacts record the inputs/model/configuration that produced them;
- simulator-dependent behavior has live evidence or remains visibly pending;
- docs describe the supported path; and
- the issue is closed by a merged, green pull request.

## Architecture decisions

Use an ADR only when a decision changes a durable boundary: public contract,
control/safety ownership, privileged-data policy, artifact identity, runtime
topology, or primary framework. Small implementation choices stay in code and
the pull request.

ADR instructions and the template live in `docs/adr/`.
