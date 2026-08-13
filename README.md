# CARLA Vision Research

A model-neutral, artifact-first framework for monocular-vision research in
CARLA 0.9.16. The primary detector is RT-DETR, but the runtime accepts
Ultralytics YOLO or any custom PyTorch-compatible adapter that implements the
small `Detector` protocol.

The current deployment defaults to CARLA at `172.20.10.7:2000`, vehicle actor
`24`, and front RGB camera actor `25`. The Apple-silicon client uses a
version-coupled MessagePack/TCP bridge, so it does not require a native CARLA
wheel for the lightweight bridge path.

## Claim boundary

A code path, passing unit test, green CI check, or available UI option is not
evidence that a driving policy performs well in live CARLA. Keep software
contract validation, live-simulator integration, and closed-loop driving
behavior as separate claims.

## What works now

- front monocular RGB inference with RT-DETR, YOLO, or a custom detector;
- exact-frame live overlays and raw/annotated split view;
- optional fixed chase view on the CARLA server monitor, with guarded pose restore;
- latest-frame-only inference, so a slow model drops stale queued frames;
- optional low-speed simulator-teacher driving with an independent watchdog;
- a strict front-RGB-only policy observation contract with non-actuating
  shadow proposals, on-screen status, input auditing, and semantic verification;
- sequential live multi-model shadow matrices with opt-in teacher motion,
  isolated child artifacts, videos, logs, and parent/child integrity checks;
- MP4, PNG, JSONL, summary, environment/model provenance, and SHA-256 manifest;
- read-only post-hoc run analysis with CSV and four plots;
- synchronized RGB plus privileged instance-segmentation dataset capture;
- CARLA semantic ontology and visible-mask boxes;
- canonical COCO plus derived YOLO labels, per-frame metadata, retained masks,
  dataset checksum index, and split-leakage guards;
- read-only dataset QA that reproduces labels/exports and emits plots/montage;
- deterministic scenario recipes, hierarchical seeds, map/weather OOD
  partitions, and integrity-verifiable episode plans;
- a native official-PythonAPI collection worker with synchronous fixed-delta
  stepping, Traffic Manager seeding, traffic/walkers/props, and exact-frame
  RGB/instance capture;
- BehaviorAgent teacher-episode recording, verification, and comparison;
- a deterministic Windows/Linux native-host ZIP with a verified 50-frame
  plan, hash-pinned dependencies, read-only preflight, and guarded collection;
- immutable-dataset verification before training or evaluation;
- RT-DETR, YOLO, and custom training adapters with resolved configs, seed
  schedules, logs, plots, checkpoint hashes, and dry-run validation;
- canonical COCO/operating-point evaluation with episode bootstrap,
  stratification, calibration, latency, plots, and failure montages;
- paired multi-model replay over one checksum-locked RGB order, retaining each
  child evaluation plus per-image disagreements, episode-cluster bootstrap,
  latency/accuracy tables, PNG/SVG plots, and a comparison montage;
- validation-only operating-threshold selection with deterministic grids,
  episode-bootstrap stability, plots, and a sealed threshold artifact;
- validation-only failure categorization, diversity-capped review queues, and
  immutable human-reviewed failure catalogs that cannot migrate validation
  RGB into training;
- a read-only recursive verifier for runs, datasets, plans, models, reports,
  paired replays, checksum indexes, external references, and clean-Git
  publication gates;
- a sealed point-in-time evidence index spanning all seven canonical research
  roots, retaining verification failures alongside verified objects and
  generating canonical tables, plots, checksums, and a human-readable report;
- immutable model promotion with packaged weights, input/inference contracts,
  ontology, model card, licenses, training lineage, and checksum index;
- manifest-driven report releases with canonical CSV tables, Markdown,
  raster/vector plots, source-graph verification, and no manual metric
  transcription;
- canonical experiment/configuration identities plus privacy-safe CPU, memory,
  accelerator, deterministic-backend, tool, and dependency-lock provenance;
- standalone checksum-indexed reproduction bundles containing a deterministic
  source archive, file inventory, source manifests, lockfiles, environment
  envelope, and tokenized rerun commands;
- imitation-policy training and an internal Garage Imitation runtime path;
- temporal RGB-only voxel occupancy/flow training plus privileged voxel/flow
  teacher capture;
- voxel shadow planning, benchmarking, readiness checks, and opt-in actuation;
- closed-loop driving evaluation and evaluation summaries;
- a local Garage/Cockpit flow with browser Manual and World Worker Traffic
  Manager Autopilot, plus internal BehaviorAgent/Imitation/Voxel runtime paths;
  and
- retained Drive outputs that can be selected by the existing inspection,
  analysis, and verification workflows.

The base `carla-vision --control vision` mode remains intentionally unavailable.
Its detector/shadow policy path is advisory. Garage autonomous modes and the
`carla-local-drive` voxel-actuation path are separate control paths with their
own acknowledgement and fail-closed contracts.

Privileged CARLA pose, segmentation, actor IDs, optical-flow/teacher signals,
and similar simulator state remain teacher/evaluation inputs rather than
runtime inputs to the deployable imitation/temporal RGB model paths.

## Install

```bash
uv sync --all-groups
```

The installed commands below are the exact current `[project.scripts]` entries:

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
carla-discover
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

Start the current local Garage/Operator server:

```bash
uv run carla-operator-ui --open-browser
```

On a new LAN, discover the simulator before starting the Operator:

```bash
uv run carla-discover
uv run carla-operator-ui --carla-host auto --open-browser
```

Discovery inspects only active, directly connected RFC1918 IPv4 interfaces,
bounds large networks to the local `/24`, ignores tunnels/link-local adapters,
probes the configured CARLA port, and then validates candidates with read-only
CARLA `version` and `get_map_info` RPC calls. An open non-CARLA port is not
reported. The same action is available under Advanced settings in the Garage.
It reports the address but does not mutate the simulator or silently retarget an
already running Operator. World Worker discovery/configuration remains separate.

`carla-operator-ui` resolves to `carla_vision.operator.garage_server:main`.
The server is local-only and accepts only loopback bind names/addresses. The
Garage reuses the existing Operator server, Drive engine, job manager, research
catalog, and artifact APIs behind one canonical browser shell.

The browser has two deliberately separate surfaces:

```text
Fullscreen Garage / Drive
Research workspace
```

Garage is the default. The live CARLA RGB stream fills the viewport, with a
small status HUD, a bottom vehicle carousel, camera shortcuts, setup and
experiment drawers, and a single Start Drive action. The old dashboard header,
tab bar, safety strip, and duplicate injected Garage assets are not part of the
running game shell.

The Research tools surface still contains:

```text
Live capture
Scene builder
Workflows
Saved results
Activity
```

The base Operator job contract still accepts exactly:

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

The Garage adds a separate allow-listed research endpoint for:

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

Neither command layer accepts an arbitrary browser-provided shell command.

### Fullscreen Garage controls

The main game shell currently exposes browser Manual control and, when the World
Worker advertises it, CARLA Traffic Manager Autopilot with manual takeover. Map,
weather, traffic, pedestrians, props, and free/random-destination route options
are capability-gated in the setup drawer. Owned actors are cleaned up with the
existing scene lease and episode guards.

The live Garage ports the CARLA `vehicle_gallery.py` orbit pattern through the
0.9.16-compatible bridge and
offers Orbit, Front, Rear, Top, and Cockpit views. Its 720p stream, actual CARLA
vehicle replacement, and bottom carousel are separate from the front camera used
once Drive starts.

The Experiments drawer turns the same Garage/Drive session into a small human
evaluation protocol. It offers Free Drive, Manual Handling, Autopilot Takeover,
Perception Review, Traffic Stress, and Adverse Weather presets. Presets only
configure existing CARLA and recording controls; they do not add another model
or silently actuate the vehicle. During a run the operator can mark an
interesting moment, false detection, missed object, autopilot issue, or scene
issue. Each marker is retained in `events.jsonl` with the experiment name,
elapsed time, exact camera/detector sequence, control source, and telemetry.
Marker totals and the experiment preset are also retained in the run summary.

Traffic Manager Autopilot in these protocols remains CARLA's own baseline. The
research contribution is the reproducible scene/perception/recording/evidence
layer and the human evaluation record around that baseline, not a claim that
CARLA's autopilot was implemented by this repository.

The supplied `ue5-dev` PythonAPI tree currently targets CARLA 0.10.0. It is a
design reference only for this project: the Windows bridge and the simulator must
continue to use matching CARLA 0.9.16 PythonAPI/client/server binaries.

BehaviorAgent, Imitation, and Voxel implementations remain in the research/runtime
code, but the cleaned main shell does not silently inject their former secondary
selector. They must not be presented as game modes until their acknowledgement,
checkpoint, and control-owner UI is integrated into the canonical shell.

The nine Garage-specific research job plans remain strict backend API/runtime
contracts. They are not exposed as buttons in the cleaned game shell; the base
Research workspace continues to expose its maintained capture, workflow,
inspection, analysis, and verification jobs.

### Research tools retained from the base Operator

Live capture still targets an existing vehicle/camera pair and can run detector
inference/recording, optional privileged low-speed teacher motion, a non-actuating
vision-policy shadow, and optional CARLA spectator follow. Teacher motion remains
explicitly acknowledged.

Scene builder still saves a strict situation recipe and resolves it through the
scenario planner. Saving/planning is offline and does not itself mutate CARLA or
populate the active Drive session.

The Workflows tab still exposes dataset QA, native preflight/capture, live shadow
matrix, paired replay, detector training, run analysis, and recursive
verification through the existing strict command layer.

Saved results still enumerates immediate manifest-backed objects under the seven
canonical evidence roots (`datasets`, `runs`, `models`, `reports`, `bundles`,
`native_kits`, `operator_sessions`). Browser inspection reports manifest
information plus file availability and explicitly leaves verification as
`not_checked` until the verifier is run.

Activity remains backed by tracked `operator_sessions/` objects containing the
validated request, resolved command plan, status, stdout/stderr logs, and
manifest/provenance when produced.

See [Operator UI](docs/operator_ui.md) for current control ownership,
research-job contracts, and the validation boundary.

## Live RT-DETR

Perception-only mode is the default and does not move the vehicle:

```bash
uv run carla-vision \
  --detector rtdetr \
  --weights rtdetr-l.pt \
  --device mps \
  --view split \
  --duration 30 \
  --run-id exp-rtdetr-shadow-001
```

Window keys:

- `Q` or `Esc`: stop;
- `S`: split view;
- `O`: overlay only;
- `M`: toggle the view.

The privileged development teacher must be requested explicitly:

```bash
uv run carla-vision \
  --detector rtdetr \
  --weights rtdetr-l.pt \
  --device mps \
  --control teacher \
  --cruise-speed 2.0 \
  --view split \
  --duration 20 \
  --run-id exp-rtdetr-teacher-001
```

Add `--spectator-follow` when an operator at the CARLA machine should see a
fixed chase view on the server monitor. This bridge option is off by default,
does not change the front-RGB recording or policy inputs, and restores the
previous spectator pose when the CARLA episode and spectator actor remain the
same. It is best-effort operator visualization; do not enable it for a benchmark
unless the run protocol calls for it.

Every run is written below `runs/<run-id>/`:

```text
manifest.json
summary.json
images/latest_overlay.png
logs/detections.jsonl
video/overlay.mp4
```

Analyze a completed run without changing it:

```bash
uv run carla-analyze-run \
  --source-run runs/exp-rtdetr-teacher-001 \
  --runs-root runs \
  --run-id exp-rtdetr-teacher-001-analysis
```

## Live vision-policy shadow

A shadow policy receives only a copied front-RGB frame, RGB-derived
detections, sequence/timestamp, and camera FOV. It cannot receive CARLA pose,
speed, route, map, actor IDs, segmentation, or other privileged simulator
state through the observation contract. Its throttle/steer/brake proposal is
logged and displayed but never applied:

```bash
uv run carla-vision \
  --detector rtdetr \
  --weights rtdetr-l.pt \
  --device mps \
  --shadow-policy hazard-stop \
  --policy-options '{"confidence":0.35,"close_bottom":0.72}' \
  --control teacher \
  --expected-map Town10HD_Opt \
  --view split \
  --duration 12 \
  --run-id exp-rtdetr-policy-shadow-001
```

Create a preregistered sequential plan without contacting CARLA:

```bash
uv run carla-shadow-matrix \
  --config configs/shadow/rtdetr_yolo26_live_shadow_v1.json \
  --runs-root runs
```

Real teacher-driven execution is intentionally a separate opt-in action:

```bash
uv run carla-shadow-matrix \
  --config configs/shadow/rtdetr_yolo26_live_shadow_v1.json \
  --runs-root runs \
  --execute \
  --acknowledge-teacher-motion
```

Verify the parent and each child independently:

```bash
uv run carla-verify-shadow-matrix runs/shadow-rtdetr-yolo26-live-v1
uv run carla-verify-shadow \
  runs/shadow-rtdetr-yolo26-live-v1--rtdetr-l-hazard-stop-r00
uv run carla-verify \
  runs/shadow-rtdetr-yolo26-live-v1 \
  runs/shadow-rtdetr-yolo26-live-v1--rtdetr-l-hazard-stop-r00 \
  runs/shadow-rtdetr-yolo26-live-v1--yolo26n-hazard-stop-r00 \
  --reject-unregistered
```

The built-in hazard-stop policy is a contract/systems baseline, not a safe
driver. Custom policies use `--shadow-policy custom --policy-factory
package.module:create`; structural isolation prevents accidental privileged
inputs but is not a security sandbox for malicious custom code.

## Dataset Factory v0

This collector attaches a privileged instance-segmentation camera at exactly
the RGB camera pose. It retains a sample only when both sensor messages have
the same CARLA frame ID, timestamp, dimensions, FOV, and transform.

Start with a no-motion pilot:

```bash
uv run carla-collect-dataset \
  --dataset-id ds-carla0916-town10-pilot-v001 \
  --resolution 1280x720 \
  --samples 100 \
  --sample-every 3 \
  --control none \
  --split unassigned \
  --scenario-id scn-town10-static-clear \
  --episode-id ep-static-r00
```

The collector can use the disclosed privileged teacher route:

```bash
uv run carla-collect-dataset \
  --dataset-id ds-carla0916-town10-teacher-v001 \
  --resolution 1280x720 \
  --samples 250 \
  --sample-every 3 \
  --control teacher \
  --split train \
  --scenario-id scn-town10-clear-teacher \
  --episode-id ep-clear-r00
```

A dataset release contains lossless RGB PNGs, retained BGRA teacher masks,
per-frame JSON, COCO annotations, YOLO labels, `data.yaml`, `dataset.json`,
`checksums.sha256`, and a provenance `manifest.json`.

Audit it before training:

```bash
uv run carla-audit-dataset \
  --dataset datasets/ds-carla0916-town10-pilot-v001 \
  --runs-root runs \
  --run-id ds-carla0916-town10-pilot-v001-qa
```

The raw-bridge collector is useful for a non-destructive pilot against an
already-running world. Before any world reload, create a read-only native
readiness artifact for the deliberately small 50-frame integration plan:

```bash
uv run carla-plan-scenarios \
  --suite configs/scenarios/native_integration_pilot_v1.json \
  --split-plan configs/scenarios/split_plan_native_integration_pilot_v1.json \
  --run-id scenario-plan-native-integration-pilot-v1

uv run carla-native-preflight \
  --scenario-plan runs/scenario-plan-native-integration-pilot-v1 \
  --dataset-id ds-carla0916-native-pilot-v001 \
  --run-id native-preflight-pilot-001 \
  --host 172.20.10.7 \
  --partition train \
  --max-episodes 1
```

The preflight verifies the plan, selected episodes, unused Dataset ID, TCP
endpoint, read-only server version/map RPC, official PythonAPI import and
client/server version agreement, plus two manual gates. It produces a
manifest, JSON/CSV checks, selected-episode record, and Markdown report while
recording `simulator_mutated: false`.

A ready-to-transfer host package is retained at:

```text
native_kits/native-host-kit-pilot-20260727-v003/payload/native-host-kit.zip
```

It targets CPython 3.12 on Windows/Linux x86-64 and contains the exact
50-frame plan, collection source, verified hashed requirements, and PowerShell
and shell scripts. Collection is refused unless a matching verified preflight
is ready and the literal confirmation token is supplied. See
[Portable native-host kit](docs/native_host_kit.md) and
[First native pilot checklist](docs/native_pilot_checklist.md).

## Train and evaluate

Training first verifies the complete dataset checksum index and split
invariants. A dry run resolves all inputs without starting the backend:

```bash
uv run carla-train-detector \
  --config configs/training/rtdetr_pilot_v1.json \
  --dataset datasets/ds-carla0916-thesis-pilot-v001 \
  --weights rtdetr-l.pt \
  --dry-run
```

The current three-frame dataset has only an `unassigned` development
partition, so it is correctly rejected as a training input. Once a trainable
release exists, remove `--dry-run` to launch the configured backend.

Run model-neutral offline evaluation on an allowed development partition:

```bash
uv run carla-evaluate-detector \
  --config configs/evaluation/rtdetr_development_v1.json \
  --dataset datasets/ds-carla0916-town10-pilot-v001 \
  --detector rtdetr \
  --weights rtdetr-l.pt \
  --device mps \
  --run-id eval-rtdetr-pretrained-pilot-v1
```

Selecting a locked test partition requires the explicit
`--acknowledge-locked-test` gate. That flag records authorization; it does not
relax checksum, leakage, or configuration validation.

Run two or more detector adapters on the exact same immutable RGB order:

```bash
uv run carla-replay-models \
  --config configs/replay/rtdetr_yolo26_development_v1.json \
  --dataset datasets/ds-carla0916-town10-pilot-v001 \
  --evaluation-config configs/evaluation/model_comparison_development_v1.json \
  --runs-root runs
```

The retained development pilot
`runs/replay-rtdetr-yolo26-pilot-v1` compares RT-DETR-L and YOLO26n on all
three pilot frames. It recursively verifies both child evaluations and records
the sample-order digest, canonical predictions, paired TP/FP/FN outcomes,
latency, bootstrap deltas, tables, plots, and a visual disagreement panel.
The sample is one unassigned episode, and the shared `0.05` prediction cutoff
removes all YOLO26n predictions because its maximum score on these views is
lower. This is a useful calibration diagnostic, not a model-ranking claim.
Final comparisons must select model-specific operating thresholds on
validation data and freeze them before locked-test replay.

## Verify, index, package, and report

Verify one or more objects recursively without modifying them:

```bash
uv run carla-verify \
  datasets/ds-carla0916-town10-pilot-v001 \
  runs/scenario-plan-thesis-pilot-v1 \
  runs/eval-rtdetr-pretrained-pilot-v1 \
  runs/replay-rtdetr-yolo26-pilot-v1 \
  --reject-unregistered
```

`--require-clean-git` adds the confirmatory-publication gate. It correctly
fails for the referenced development artifacts because they were created before
the repository had a clean commit.

Seal a point-in-time workspace evidence index without contacting CARLA:

```bash
uv run carla-build-evidence-index \
  --workspace . \
  --registry-id evidence-index-workspace-20260727-v001

uv run carla-verify \
  runs/evidence-index-workspace-20260727-v001 \
  --reject-unregistered
```

The index scans only immediate children of the seven canonical roots, attempts
deep verification for every discovered object, retains failures instead of
omitting them, and emits JSON/CSV/Markdown plus PNG/SVG inventory plots. New
objects created later do not invalidate the snapshot; drift in any recorded
source manifest or in the non-following tree snapshot of an already-invalid
source does. See [Evidence index](docs/evidence_index.md).

After a successful real training run, promote its selected checkpoint using a
strict model-release configuration:

```bash
uv run carla-package-model \
  --config configs/model_release/<release>.json \
  --training-run runs/<successful-training-run> \
  --models-root models
```

The runtime and evaluator can then consume the entire verified contract:

```bash
uv run carla-vision --model-package models/<model-id> --device mps
uv run carla-evaluate-detector \
  --config configs/evaluation/<evaluation>.json \
  --dataset datasets/<dataset-id> \
  --model-package models/<model-id> \
  --device mps
```

Loose `--detector`, `--weights`, `--detector-factory`, and `--image-size`
options are mutually exclusive with `--model-package`, preventing accidental
drift from the released preprocessing/adapter contract.

Build the current development report entirely from verified manifests:

```bash
uv run carla-build-report \
  --config configs/reports/framework_validation_v8.json \
  --reports-root reports

uv run carla-verify \
  reports/rpt-framework-validation-20260727-v008 \
  --reject-unregistered
```

The generated release contains `report.md`, `report.json`, canonical inventory
and metric CSVs, PNG/SVG plots, a checksum index, and a tracker manifest.

Build a standalone development reproduction bundle:

```bash
uv run carla-build-reproduction \
  --config configs/reproduction/framework_development_bundle_v9.json \
  --bundles-root bundles

uv run carla-verify-reproduction \
  bundles/bundle-framework-development-20260727-v009
```

This bundle archives deterministic source bytes and `uv.lock`, not mutable
paths to the original research objects. The referenced bundle is development
point-in-time evidence, not proof of current Garage/autonomy behavior. See
[Reproducibility and provenance](docs/reproducibility.md) for the identity,
privacy, archive, semantic-verification, and confirmatory-gate contracts.

## Custom detector

Use `--detector custom --detector-factory package.module:create`. The callable
receives `DetectorConfig` and returns an object with:

```python
class Detector(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def metadata(self) -> DetectorMetadata: ...

    def infer(self, image_bgr: np.ndarray) -> tuple[Detection, ...]: ...

    def close(self) -> None: ...
```

Detections must use original-image pixel coordinates. This contract keeps
runtime, visualization, recording, risk logic, and analysis independent of the
model family.

## Verification

For the current Python test suite and maintained package/browser sources, use:

```bash
uv run ruff check carla_vision carla_yolo tests
uv run python -m compileall -q carla_vision carla_yolo tests
find carla_vision/operator -type f -name '*.js' -print0 | xargs -0 -r -n1 node --check
uv run pytest -q
```

Versioned validation reports and retained development artifacts in this
repository are point-in-time evidence for the experiments that produced them.
They should not be treated as proof that the current Garage/Imitation/Voxel
control paths have been revalidated in live CARLA after later code changes.
Current simulator behavior must be measured again when that claim matters.

## Research documentation

- [Architecture](docs/architecture.md)
- [Experiment protocol](docs/experiment_protocol.md)
- [Artifact policy](docs/artifact_policy.md)
- [Paired replay](docs/paired_replay.md)
- [Threshold and failure workflow](docs/threshold_and_failure_workflow.md)
- [Reproducibility and provenance](docs/reproducibility.md)
- [Evidence index](docs/evidence_index.md)
- [Operator UI](docs/operator_ui.md)
- [Voxel actuation](docs/voxel_actuation_fa.md)
- [Portable native-host kit](docs/native_host_kit.md)
- [First native pilot checklist](docs/native_pilot_checklist.md)
- [Validation report v0.9](docs/validation_report_v0.9.md)
- [Validation report v0.8](docs/validation_report_v0.8.md)
- [Validation report v0.7](docs/validation_report_v0.7.md)
- [Validation report v0.6](docs/validation_report_v0.6.md)
- [Validation report v0.5](docs/validation_report_v0.5.md)
- [Validation report v0.4](docs/validation_report_v0.4.md)
- [Validation report v0.3](docs/validation_report_v0.3.md)
- [Native host runbook](docs/native_host_runbook.md)

The implementation follows CARLA 0.9.16 behavior and uses the official
[Python examples](https://github.com/carla-simulator/carla/tree/0.9.16/PythonAPI/examples),
[sensor reference](https://carla.readthedocs.io/en/0.9.16/ref_sensors/), and
[synchronous-mode guidance](https://carla.readthedocs.io/en/0.9.16/adv_synchrony_timestep/)
as versioned references.

## Safety boundary

This is simulator research software. A 2D RGB box is not metric distance,
teacher control is not vision-only autonomy, and CARLA results do not establish
real-world vehicle safety. Autonomous Garage/local-drive code paths also do not
establish closed-loop quality merely because they exist or pass tests. Motion
remains opt-in/acknowledged in the paths that require it, and live behavior must
be validated against CARLA with retained metrics and cleanup evidence.
