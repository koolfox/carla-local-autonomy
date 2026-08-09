# CARLA Vision Operator UI

Status: local proof-of-concept operator console  
Binding policy: loopback only  
Control policy: vision proposals never actuate the vehicle

## Purpose

The operator UI gives a thesis researcher one simple place to:

- start a live detector and annotated recording;
- select RT-DETR or YOLO weights and runtime settings;
- enable a non-actuating vision-policy shadow;
- compose traffic, pedestrians, crossing probability, weather, props, camera,
  duration, and seed without editing scenario JSON;
- resolve a situation into deterministic episode plans;
- dry-run or explicitly authorize native dataset collection;
- plan or execute a live detector/policy matrix;
- replay models over an identical dataset order;
- dry-run or explicitly authorize training;
- create post-hoc metrics and plots;
- verify retained datasets, runs, models, reports, bundles, and native-host
  kits;
- browse manifest-backed evidence across all canonical roots;
- inspect job status, logs, registered overlays, and videos.

It is intentionally a small orchestration layer over the existing strict
Python CLIs. It does not duplicate simulator, dataset, training, replay, or
verification logic.

## Start

```bash
uv run carla-operator-ui --open-browser
```

The default address is:

```text
http://127.0.0.1:8765/
```

The two header indicators describe different paths. `CARLA · reachable` is
the requirement for Live & Record. `PythonAPI · missing` blocks native
synchronous collection on that machine, but does not block the lightweight
MessagePack live viewer, RT-DETR/YOLO overlay, MP4 recording, or optional
teacher motion.

The server refuses non-loopback bind addresses. It launches local processes
and can authorize simulator mutation, so it must not be exposed as an
unauthenticated network service.

## Five operator surfaces

### Live & Record

The form selects:

- CARLA host/port and existing vehicle/camera actor IDs;
- expected map, resolution, camera FPS, and FOV;
- RT-DETR or YOLO weight file, device, image size, and confidence floor;
- no motion or privileged low-speed teacher motion;
- no policy or the built-in hazard-stop shadow;
- split/overlay/headless view, duration, stale limit, and MP4 retention;
- optional fixed chase view on the monitor attached to the CARLA server.

Teacher motion requires a per-launch acknowledgement. The runtime opens its
existing exact-frame OpenCV viewer. Policy proposals are shown in the HUD and
written to `policy_shadow.jsonl`; they are never applied. Server-monitor follow
is unchecked by default and is best-effort operator visualization. It neither
changes the recorded front-RGB stream nor grants motion authority. On streamed
maps CARLA may use spectator position as a streaming reference, so leave it off
for benchmark runs unless the protocol explicitly includes it. The runtime
restores the previous spectator pose only if the CARLA episode and spectator
actor have not changed.

### Situation Builder

The situation form maps a small operator-facing schema onto the full existing
`ScenarioSuite` contract. It exposes the controls most useful for a thesis
MVP:

- map and ego spawn index;
- named coherent weather preset;
- number of traffic vehicles and pedestrians;
- pedestrian crossing probability;
- Traffic Manager speed difference and following distance;
- none, cones, construction, or accident static-prop preset;
- camera resolution/FOV and capture FPS;
- episode duration, repetitions, and master seed.

Saving produces:

```text
operator_configs/situations/<situation-id>.json
```

The output is validated by the same strict scenario classes used by the native
worker. It can then be resolved through the operator development split plan
into a normal checksum-tracked scenario-plan run.

Saving and planning are offline. They do not connect to CARLA.

### Research Workflows

The UI exposes six existing workflows:

1. **Native preflight and capture** — the preflight verifies the plan,
   endpoint, versions, PythonAPI, output ID, and manual gates through
   read-only operations and creates a standalone readiness artifact. Capture
   remains dry-run by default. Real execution requires the official matching
   PythonAPI plus acknowledgement of map reload, actor destruction, and
   exclusive `world.tick()` ownership.
2. **Live shadow matrix** — derives a new immutable matrix ID from a selected
   template. Plan mode is offline; teacher-driven execution requires
   acknowledgement.
3. **Paired replay** — derives a new replay ID while retaining the selected
   frozen template values and resolved model paths.
4. **Training** — dry-run by default. Real training requires a separate
   compute acknowledgement.
5. **Analysis** — creates source-linked frame CSV, metrics, and plots from a
   recorded runtime.
6. **Verification** — performs recursive read-only hash and semantic checks.

The backend accepts no shell string. Every workflow has an explicit parameter
allow-list and is launched as a token array through the current Python
environment. Workspace inputs cannot escape the project path. Existing output
IDs are never overwritten.

### Evidence

The Evidence tab is a searchable, read-only view of manifest-backed research
objects under:

```text
datasets/  runs/  models/  reports/  bundles/  native_kits/  operator_sessions/
```

It reports total and per-root counts and supports free-text, root, and
manifest-declared status filters. Selecting an object shows only the artifacts
registered by that object's `manifest.json`, including declared role, path,
byte size, SHA-256, metadata, current file availability, and a declared or
safely inferred media type.

The distinction between declaration and verification is deliberate:

- object status, artifact size, and artifact hash shown in this tab are
  manifest declarations;
- file availability means only that a contained, non-symlink regular file
  currently exists;
- opening or previewing an artifact does not recompute its digest or run its
  semantic verifier;
- the tab therefore labels every inspected object `not_checked`.

The **Select in verification workflow** action copies the exact
workspace-relative object path into the existing Verification form. It does
not claim success or launch verification automatically; the operator reviews
and starts that separately.

Artifact access is manifest-only. The server accepts one immediate child of an
allow-listed root, rejects traversal, absolute or drive-style paths and
symlinks, and serves only an exact registered artifact path with an allow-listed
suffix. Registered PNG, JPEG, and SVG images and MP4 videos can be previewed.
Text, JSON, JSONL, CSV, Markdown, and logs can be opened, while registered ZIP,
model, engine, YAML, and checksum files use safe download responses. Missing,
unsafe, unsupported, and unregistered files are not exposed. Artifact
responses use a separate restrictive browser policy that disables scripts,
including for registered SVG files.

This live catalogue is not the sealed point-in-time evidence registry. The
registry builder performs strict verification and retains canonical tables,
plots, and a report; see [`evidence_index.md`](evidence_index.md).

### Sessions

Each launch creates:

```text
operator_sessions/<operator-job-id>/
├── manifest.json
├── request.json
├── command.json
├── status.json
└── logs/
    ├── stdout.log
    └── stderr.log
```

The session uses `RunArtifactTracker`, so all five payloads receive SHA-256,
byte-size, role, environment, hardware, Git, and configuration provenance.
When a child run/dataset exists, its manifest fingerprint is added as an input
reference.

Verify an operator session with:

```bash
uv run carla-verify \
  operator_sessions/<operator-job-id> \
  --reject-unregistered
```

A stopped or failed session remains evidence and can be inspected with
`--allow-non-success`.

The Sessions inspector obtains links and previews from the operator-session
manifest and, when present, the child object's manifest. It does not guess
conventional paths such as a “latest overlay” or “overlay video.” While a
running job has not finalized its manifest, the UI reports that evidence
finalization is pending rather than probing undeclared files.

## Safety model

- The UI binds only to `127.0.0.1`, `::1`, or `localhost`.
- State-changing requests require a per-server same-origin token.
- Arbitrary commands and shell evaluation are not supported.
- Workspace file selections are path-contained and must already exist.
- Output directories must not exist.
- `vision` control is absent from the UI and still rejected by the runtime.
- Teacher motion, native mutation, locked-test access, and real training have
  separate visible acknowledgements.
- Native dry-run does not import CARLA or contact the simulator.
- Closing the UI asks active child processes to terminate; runtime watchdog
  and final-stop behavior remain authoritative.

The token is a local cross-origin request guard, not user authentication. This
POC is deliberately not a remotely hosted multi-user service.

## Current limitations

- The UI launches the existing native OpenCV viewer rather than re-encoding a
  low-latency browser video stream.
- It offers named weather/prop presets instead of exposing every CARLA
  attribute.
- It creates one situation recipe at a time; multi-recipe thesis suites remain
  JSON-configured.
- It does not edit training hyperparameters, ontologies, or model-package
  contracts in the browser.
- Replay and matrix runs derive new IDs from templates but do not automatically
  seal a new evidence-registry snapshot.
- Restarting the UI preserves completed session files, but it does not resume
  interrupted subprocesses.
- There is no authentication, remote access, scheduling queue, database, or
  role system by design.

These constraints keep the interface at the requested thesis POC/MVP scale.

## Validated development smoke

The UI backend was exercised through its real HTTP API to save
`town10-ui-poc-v1` and launch `scenario-plan-operator-poc-v1`. The retained
operator session completed with code zero, no motion or destructive
authorization, and a verified child manifest containing one episode and 75
planned captures.

The first verification smoke exposed an internal module-entry-point defect.
That session was retained, the allow-listed invocation was corrected, and a
new real verification session produced the expected recursive output. See
[Validation report v0.6](validation_report_v0.6.md) for IDs, hashes, test
counts, and the claim boundary.

The native-preflight button was also exercised through the real HTTP API. It
launched a non-destructive operator job, created
`native-preflight-ui-20260726-v1`, fingerprinted the child manifest, and
correctly reported that the CARLA 0.9.16 endpoint is reachable while the local
Apple arm64 host lacks the official PythonAPI required for collection.
