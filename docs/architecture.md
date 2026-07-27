# CARLA Vision Research Framework Architecture

Status: implemented research-framework core with gated thesis roadmap  
Reference simulator: CARLA 0.9.16  
Primary runtime sensor contract: one forward-facing monocular RGB camera

## 1. Purpose and status vocabulary

This repository is intended to become a reproducible research framework for
training, comparing, and evaluating vision models in CARLA, and later for
closed-loop vision-only driving experiments. It must support thesis-quality
code, datasets, models, videos, plots, tables, and reports without coupling the
research pipeline to one detector family.

The following words are normative throughout the documentation:

- **Current**: implemented in this workspace and covered by source inspection,
  tests, or a recorded run.
- **Planned**: an architectural requirement that is not yet implemented.
- **Acceptance gate**: evidence that must exist before a planned component can
  be treated as research-ready.
- **Privileged data**: simulator state that would not be available to a real
  monocular camera system, including actor poses, depth, semantic or instance
  images, map coordinates, and perfect traffic-light state.

Passing a smoke test proves only that one configuration executed. It does not
prove generalization, safety, or thesis-level validity.

## 2. Non-negotiable research boundary

### 2.1 Runtime perception and policy inputs

The deployable perception and future driving policy may consume only:

- ordered BGR/RGB images from one rigid forward-facing monocular camera;
- timestamps and frame identifiers needed to maintain temporal order;
- the model's own recurrent or temporal state.

No CARLA pose, velocity, depth, segmentation, actor list, bounding box,
waypoint, map coordinate, collision signal, or traffic-light API value may
enter the deployable model or vision-control decision path.

The current detector contract receives only a `uint8 HxWx3` BGR array. The
current runtime records the contract string
`front_monocular_rgb_only` in its run manifest and summary.

The current shadow-policy contract receives a `VisionObservation` containing
only sequence, source timestamp, a read-only copied RGB array, RGB-derived
detections, and camera FOV. It produces a bounded
`VisionControlProposal`. The runtime logs and displays that proposal but never
sends it to CARLA. Policy evaluation occurs before privileged pose/speed is
computed for teacher control or evaluation.

### 2.2 Permitted privileged lanes

Privileged data is permitted only in the following explicitly marked lanes:

| Lane | Permitted use | Forbidden use |
|---|---|---|
| Dataset teacher | Create and quality-check labels; determine visibility, state, and attributes | Model input or inference-time feature |
| Simulator teacher | Generate expert trajectories and actions for imitation-learning data | Evidence that the final policy is vision-only |
| Evaluation | Compute ground-truth distance, speed, collision, route, and compliance metrics | Feed measurements back into a vision-only controller |
| Safety during development | Emergency stop and independent simulator supervision | Claim that the evaluated policy achieved the result unaided |

Every log field or dataset column derived from privileged state must carry
either a `teacher_` or `ground_truth_` prefix in future schemas. Every
experiment manifest must state which privileged lanes were active.

**Current limitation:** even in perception-only mode, the runtime derives
vehicle pose and speed from the CARLA camera transform for the HUD, log, and
evaluation summary. That value is not passed into the detector. In
`--control teacher`, pose is also consumed by the route controller.
`--control vision` is deliberately rejected by the CLI and is not implemented.

### 2.3 Route intent

A future policy may require a high-level navigation command such as
`follow`, `left`, `right`, or `straight`. Whether that command is allowed is a
research-design choice, not a camera sensor choice. It must be pre-registered
per experiment and must not contain metric waypoints or simulator pose.

## 3. System context

The current deployment has two logical sides:

```text
CARLA host (172.20.10.7:2000)             Research client (this repository)
--------------------------------          -----------------------------------
CARLA 0.9.16 simulation             RPC   actor/camera lifecycle
vehicle + rigid RGB camera           -->  latest-frame camera stream
camera TCP stream                    -->  detector adapter
actor control endpoint               <--  teacher control + fail-safe actuator
                                           frame-correct live viewer
                                           video, JSONL, summary, manifest
```

The current client intentionally does not require the native CARLA Python
wheel. `carla_vision.bridge.CarlaRpc` implements a small, version-coupled
MessagePack RPC client and `CarlaCameraStream` decodes the camera TCP stream.
This is useful for the Apple-silicon runtime, but it is not a general
replacement for the official Python API.

Dataset Factory v0 can use the same bridge to prove co-located RGB and
instance-segmentation capture with exact CARLA-frame matching. A separate
native worker is now implemented for the production path and must run where
the official CARLA 0.9.16 Python API is available, preferably on the simulator
host or another compatible machine. It owns synchronous stepping, actor
creation, Traffic Manager, weather, deterministic seeds, props, exact sensor
phase discovery, and per-episode provenance. A separate read-only preflight
now verifies plan selection, endpoint/version/map, PythonAPI compatibility,
output identity, and manual gates while retaining JSON/CSV/Markdown evidence.
A destructive integration run against the live simulator has not yet been
authorized or executed.

## 4. Current component architecture

| Component | Current responsibility | Evidence location |
|---|---|---|
| Contracts | Model-neutral `Detection`, `DetectorConfig`, `DetectorMetadata`, `Detector`, and frame-bound `PerceptionResult` | `carla_vision/contracts.py` |
| Detector factory | Select Ultralytics YOLO, Ultralytics RT-DETR, or a custom `module:callable` detector | `carla_vision/detectors/factory.py` |
| Ultralytics adapter | Normalize YOLO and RT-DETR boxes into the common detection contract | `carla_vision/detectors/ultralytics.py` |
| Camera bridge | CARLA 0.9.16 RPC, RGB actor creation, and newest-frame TCP reader | `carla_vision/bridge.py` |
| Perception worker | Run inference off the main thread and replace superseded pending frames instead of building latency | `carla_vision/perception.py` |
| Risk policy | Apply a separate image-space hazard heuristic to normalized detections | `carla_vision/risk.py` |
| Vision-policy contract | Expose only copied front RGB and RGB-derived detections; validate bounded non-actuating proposals and audit declared inputs | `carla_vision/policy/` |
| Live shadow matrix | Preregister or sequentially execute interchangeable detector/policy cells with explicit teacher-motion acknowledgement and parent/child verification | `carla_vision/shadow/` |
| Operator UI | Offer a loopback-only POC panel for live recording, situation/crowdedness recipes, native dry-run/execute, replay, training, analysis, verification, manifest-driven evidence browsing, job stop, and verifiable session logs without accepting shell strings | `carla_vision/operator/` |
| Live display | Render boxes on the exact inferred frame; offer overlay and raw-live/annotated split modes | `carla_vision/display.py` |
| Recorder | Write annotated video asynchronously with a bounded queue | `carla_vision/recording.py` |
| Teacher controller | Follow one hard-coded Town10HD loop using privileged pose | `carla_vision/controller.py` |
| Fail-safe actuator | Apply controls in a separate process and brake after heartbeat loss | `carla_vision/watchdog.py` |
| Run tracking | Create a unique run directory, provenance manifest, SHA-256 artifact entries, and failure record | `carla_vision/artifacts.py` |
| Reproducibility identity | Canonicalize resolved configurations, allocate thesis experiment IDs, and record privacy-safe hardware/backend and dependency-lock fingerprints | `carla_vision/reproducibility.py` |
| Runtime composition | Connect all current components and expose the CLI | `carla_vision/runtime.py` |
| Run analysis | Validate an immutable source run and produce metrics, CSV, and plots with lineage | `carla_vision/analysis.py` |
| Dataset ontology/labels | Decode CARLA BGRA instance IDs and extract visible boxes for official thing classes | `carla_vision/dataset/ontology.py`, `instance_labels.py` |
| Dataset synchronization | Retain only exact RGB/teacher CARLA-frame matches and reject geometry disagreement | `carla_vision/dataset/sync.py` |
| Dataset writer/collector | Write RGB, teacher masks, COCO, YOLO, metadata, and checksum-indexed releases | `carla_vision/dataset/writer.py`, `collector.py` |
| Dataset QA | Reproduce teacher labels/YOLO exports, verify all hashes, and create QA plots/montage | `carla_vision/dataset/qa.py` |
| Dataset release verifier | Verify manifests, checksum indexes, references, counts, episode leakage, and exact-image leakage before downstream use | `carla_vision/dataset/verified.py` |
| Scenario contracts/planner | Validate recipes, derive hierarchical seeds, assign group-safe seen/map-OOD/weather-OOD partitions, and write a recomputable episode plan | `carla_vision/scenarios/` |
| Native readiness preflight | Contact only read-only CARLA endpoints; verify plan selection, output ID, server/client versions, PythonAPI availability, and manual execution gates; retain a semantically verified readiness artifact | `carla_vision/native/preflight.py` |
| Native-host kit | Freeze one verified train episode, exact project source, hash-pinned CPython 3.12 Windows/Linux dependencies, guarded scripts, and deterministic inventories into a self-contained ZIP; bind execution to a matching ready preflight and literal confirmation | `carla_vision/native/host_kit.py`, `verified_host_kit.py`, `host_gate.py` |
| Native collection worker | Use the official PythonAPI as sole synchronous tick owner; reload worlds, spawn traffic/walkers/props/sensors, capture exact RGB/instance pairs, and clean up | `carla_vision/native/` |
| Training service | Verify inputs and launch Ultralytics RT-DETR, YOLO, or a custom trainer with frozen partitions, seeds, logs, plots, and checkpoint hashes | `carla_vision/training/` |
| Offline evaluation service | Normalize any detector through the common adapter and write COCO, operating-point, calibration, bootstrap, latency, strata, plot, and failure-panel artifacts | `carla_vision/evaluation/` |
| Paired replay service | Execute two or more loose development adapters or verified model packages on one checksum-locked RGB order; retain independently verifiable child evaluations and paired accuracy/latency/disagreement artifacts | `carla_vision/replay/` |
| Threshold selection | Consume only a verified validation evaluation, sweep a preregistered canonical grid, rerun selection under episode bootstrap, and seal the selected operating threshold | `carla_vision/thresholds/` |
| Failure mining and review | Categorize mutually exclusive validation failures, retain all candidates, build a deterministic diversity-capped queue, and finalize complete human decisions as a separate immutable catalog | `carla_vision/failure_mining/` |
| Research-object verifier | Recursively verify tracker envelopes, timestamps, safe paths, hashes, sizes, checksum indexes, external references, semantic object contracts, unregistered files, and clean-Git gates | `carla_vision/verification.py` |
| Evidence registry | Scan immediate children of all seven canonical roots, retain both passing and failing verification outcomes, and seal deterministic JSON/CSV/Markdown/PNG/SVG inventory evidence with source-manifest drift checks | `carla_vision/evidence/` |
| Model release service | Promote a selected verified checkpoint into a checksum-indexed package containing weights, ontology, input/inference contracts, licenses, model card, and complete lineage | `carla_vision/model_release/` |
| Report service | Verify source graphs and generate canonical CSV tables, Markdown, PNG/SVG plots, a report descriptor, and checksum index without manual metric transcription | `carla_vision/reporting/` |
| Reproduction bundle service | Verify source objects and archive deterministic source bytes, inventories, manifests, lockfiles, environment metadata, and tokenized commands into a standalone semantically verified release | `carla_vision/reproduction/` |

The automated suite checks contracts, adapter normalization, latest-frame
behavior, frame-correct rendering, GUI shutdown, risk separation, artifact
integrity, run analysis, exact dataset pairing, instance-label decoding,
COCO/YOLO writing, dataset QA and release verification, scenario planning,
read-only native preflight, scenario-to-fake-native-dataset-to-QA-to-training
handoff, deterministic native-host packaging and tamper rejection, training
with a fake backend, canonical evaluation with a fake
detector, recursive verification,
paired replay, threshold selection, failure mining/review, vision-only policy
observations, live shadow matrices, canonical configuration identity,
standalone reproduction bundles and tamper detection, model promotion, report
generation/source invalidation, sealed evidence registries, and secure
manifest-only evidence browsing. The latest exact test count and commands
live in the validation reports; native
interaction against the real 0.9.16 PythonAPI and vision-only driving remain
untested.

The operator UI is intentionally not a new application backend. It validates a
small form schema, emits existing scenario contracts, builds allow-listed
tokenized CLI invocations, and wraps each subprocess in a tracked operator
session. Scientific outputs remain owned and verified by their original
producer.

Recorded development runs currently demonstrate both YOLO and RT-DETR
perception. A recorded `framework-v0-rtdetr-live-drive` run demonstrates
RT-DETR visualization while the privileged Town10HD teacher moves the vehicle.
These are development artifacts, not benchmark results.

## 5. Detector abstraction

### 5.1 Current contract

Every current detector exposes:

```python
class Detector(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def metadata(self) -> DetectorMetadata: ...

    def infer(self, image_bgr: np.ndarray) -> tuple[Detection, ...]: ...

    def close(self) -> None: ...
```

Each normalized `Detection` contains:

- canonical-facing `class_id` and `label`;
- `confidence` in `[0, 1]`;
- `xyxy` in original-image pixel coordinates;
- optional `source_class_id`;
- optional attributes.

PyTorch is an implementation detail, not the interoperability contract.
Models can disagree on preprocessing, tensor layout, class indexing,
postprocessing, NMS, query selection, and output structure even when all use
PyTorch. The adapter owns those differences. Downstream display, recording,
risk, evaluation, and future tracking code must consume only normalized
contracts.

Current backend selection:

```text
--detector yolo    -> Ultralytics YOLO adapter
--detector rtdetr  -> Ultralytics RT-DETR adapter
--detector custom  -> user factory in module:callable form
```

### 5.2 Planned contract extensions

The following are required before broad model comparisons:

- a versioned canonical class ontology and explicit source-to-canonical class
  mapping;
- model-weight SHA-256 and immutable model identity in metadata;
- preprocessing metadata, including color order, resizing, normalization, and
  letterboxing;
- optional batch inference without changing single-frame semantics;
- capability flags for boxes, masks, traffic-light state, embeddings, and
  temporal state;
- ONNX Runtime and, where supported, TensorRT adapters;
- adapter conformance tests using fixed images and golden normalized outputs;
- validation-only threshold selection plus deterministic frozen thresholds per
  model and per class;
- model-specific lineage filters over the implemented workspace Evidence
  Explorer and sealed evidence-index tables.

An adapter may add output types, but it must never make downstream components
import a model-specific result object.

## 6. Live RGB perception pipeline

### 6.1 Current data flow

```text
CARLA RGB TCP
    |
    v
CarlaCameraStream -- retains newest frame only
    |
    v
PerceptionWorker -- one in-flight inference + one replaceable pending frame
    |
    v
PerceptionResult -- detections and exact source_bgr share sequence/frame/time
    |                 |                  |                    |
    v                 v                  v                    v
HazardPolicy    VisionObservation  OverlayRenderer      detections.jsonl
                      |                  |
                      v                  +--> LiveViewer + AsyncVideoRecorder
              VisionPolicy proposal
                      |
                      +--> policy_shadow.jsonl + HUD (NOT APPLIED)
```

The important invariant is:

> A detection overlay is drawn only on `PerceptionResult.source_bgr`, the exact
> image used for inference.

When inference is slower than the camera:

- the camera reader keeps the newest image;
- a superseded pending inference frame is dropped;
- overlay mode shows the exact inferred frame;
- split mode shows the newest raw frame on the left and the exact annotated
  inference frame on the right;
- an old detection is never painted over an unrelated newer frame.

The current HUD includes CARLA frame/sequence, detection count, inference
latency, source age, model, mode, simulator-derived speed, route progress,
processed/dropped counts, hazard state, and controls when applicable. `Q`,
Escape, or closing the window enters the same runtime shutdown path. `S`, `O`,
and `M` switch display modes.

### 6.2 Current control modes

| Mode | Current behavior | Sensor claim |
|---|---|---|
| `none` | Perception, viewer, logs, and optional video; no driving command | Detector input is RGB-only; simulator state is still logged for evaluation/HUD |
| `teacher` | Hard-coded Town10HD pure-pursuit route using privileged pose, gated by the image-space hazard heuristic | Not vision-only driving |
| `vision` | CLI rejects the request | Planned |

The current visual hazard rule uses class, confidence, box size, and an
image-space driving corridor. A 2D box is not metric distance. This heuristic
is a conservative development gate and must not be described as collision
avoidance or autonomous-driving validation.

A policy shadow can accompany either `none` or `teacher` mode. In teacher
mode, only the privileged teacher reaches the fail-safe actuator. The shadow
proposal is evaluated before privileged state is read, written to a separate
JSONL stream, marked `actuation_applied: false`, and shown on the HUD as
`NOT APPLIED`. `--expected-map` can fail the run before motion if the live map
does not match the preregistration.

### 6.3 Planned live extensions

- publish camera FPS, inference FPS, end-to-end latency percentiles, queue
  drops, frame age, and recorder drops as structured metrics;
- add a temporal tracker with IDs while preserving detector-frame provenance;
- add drivable-area/lane perception as a separate RGB model or multi-task
  adapter;
- support headless streaming for remote monitoring in addition to native
  OpenCV display;
- record both raw RGB and annotated video when the protocol requires it;
- replay a frozen RGB stream through multiple adapters for paired comparison
  is implemented offline; extend the same contract to long live recordings;
- add temporal policy state and high-level route intent only after their
  contracts, leakage audit, and closed-loop gates are preregistered.

## 7. Dataset Factory v0 and native worker

Dataset Factory v0 is implemented as a separate collector. A live three-frame
pilot on Town10HD_Opt retained only exact RGB/instance CARLA-frame matches,
created 27 visible-instance annotations, verified 15 checksum-indexed files,
reproduced every mask-derived label and YOLO export, and destroyed the
temporary teacher sensor. This proves the data contract, not the A2 gate.

The native official-PythonAPI worker is implemented behind an explicit
exclusive-tick-owner acknowledgement. The minimal bridge remains appropriate
for non-destructive pilots, while the native path verifies the plan, requires
exact client/server version agreement, reloads the world per repetition,
enables synchronous fixed-delta mode, seeds the Traffic Manager and
pedestrians, creates traffic/walkers/props, and captures co-located RGB and
instance frames. It records the actual actor inventory and cleanup outcome.

The worker has unit coverage, a no-import dry run, and an end-to-end
fake-session handoff through a multi-episode Dataset release, QA, and training
dry-run. The Dataset verifier semantically checks repeated per-episode
provenance, split/sample/annotation counts, cleanup, and world restoration.

The read-only preflight has also run against the real endpoint: server
`0.9.16` and `Town10HD_Opt` were observed without mutation, while the Apple
arm64 host was correctly classified as not ready because the official
PythonAPI is absent. The first real target is a one-episode, 50-frame
Town10HD integration plan. Its portable host kit is independently verified,
but a real run remains an acceptance task because map reload destroys existing
actors and because the native CARLA wheel must match the server.

### 7.1 Native data flow

```text
Scenario recipe + master seed
              |
              v
CARLA world setup (map, weather, actors, props, Traffic Manager)
              |
              v
synchronous world.tick() with fixed_delta_seconds
              |
              +--> deployable RGB camera --------------------+
              +--> co-located instance/semantic camera       |
              +--> optional depth for label QA               |
              +--> actor boxes, landmarks, states, pose      |
                                                             v
                                 frame synchronizer -> label builder
                                                             |
                                  +--------------------------+
                                  v
                        immutable dataset episode
               RGB + COCO labels + frame/episode manifest + QA
```

Teacher sensors must be co-located with the RGB camera and matched by CARLA
frame ID. Only RGB is written to the model-input image set. Teacher images may
be retained in a restricted debug tier or reduced to derived label metadata;
the dataset manifest must say which choice was made.

### 7.2 Label strategy

The current canonical target is the visible box derived from instance pixels.
The following enrichments remain planned:

- project actor and environment-object 3D boxes into the RGB image;
- fuse existing instance-mask visibility with projected actor/world-object
  geometry and same-tick snapshots;
- retain amodal projected boxes as teacher metadata, not the default detector
  target;
- compute truncation and occlusion flags;
- attach actor blueprint, canonical class, traffic-light state, sign subtype,
  and relevant attributes;
- reject boxes behind the camera, outside the image, below a declared visible
  pixel threshold, or inconsistent across teacher sources;
- store canonical COCO detection annotations and deterministic converters for
  model-specific formats.

Depth may be used to check occlusion and distance-stratified evaluation, but it
must never become a runtime model input under the monocular contract.

### 7.3 Scenario space

Versioned scenario contracts now explicitly control map, weather, traffic,
camera, capture cadence, and props. The pilot suite spans six map families,
six weather recipes, seven recipes, 23 episodes, and 3,450 planned captures.
Further recipe coverage must explicitly control:

- map family and road/route segment;
- plausible weather vector and time of day;
- traffic density and vehicle composition;
- pedestrian density and crossing behavior;
- parked and moving vehicles;
- static props and hazards such as cones, barriers, debris, bins, boxes, and
  roadworks;
- camera intrinsics and bounded mounting perturbation;
- rare events such as an occluded pedestrian, a cyclist entering the lane,
  construction, stalled vehicles, emergency vehicles, and traffic-light
  transitions.

Sampling every simulation tick would create highly correlated data. The
capture rate must be declared per dataset recipe, typically 2–5 Hz for ordinary
driving plus targeted event-triggered frames.

## 8. Training and evaluation services

Service status:

1. **Dataset validator — implemented.** It checks image/annotation
   correspondence, ontology, split membership, exact duplicate content,
   checksums, and sample references before training/evaluation.
2. **Trainer adapter — implemented and backend-tested.** It launches RT-DETR,
   YOLO, or a custom model under one experiment contract while preserving
   backend-specific configuration. A real training run awaits a trainable
   immutable dataset.
3. **Offline evaluator — implemented and smoke-tested with real RT-DETR.** It
   writes canonical predictions, COCO metrics, operating-point metrics,
   stratification, calibration, episode-bootstrap uncertainty, latency, plots,
   and qualitative failures. Locked-test use is separately gated.
4. **Replay evaluator — implemented.** It sends one immutable RGB sequence
   through multiple detector adapters and retains independently verifiable
   child evaluations plus paired latency, outcome, bootstrap, and
   disagreement artifacts.
5. **Live shadow evaluator — implemented for development.** The runtime
   invokes a structurally RGB-only policy without actuation, and the matrix
   planner/executor runs model/policy cells sequentially with input audits,
   videos, logs, post-hoc metrics, and semantic parent/child verification.
   A frozen multi-map/weather benchmark and preregistered pass targets remain.
6. **Closed-loop evaluator — planned.** It runs a frozen policy on a fixed
   suite of scenario recipes and seeds.
7. **Report builder — implemented for development summaries.** It recursively
   verifies source objects and generates canonical inventory/metric CSVs,
   Markdown, PNG/SVG plots, a source graph, and a checksum-indexed report
   release. PDF/LaTeX thesis rendering and cross-seed inferential comparison
   remain planned.

Selected training checkpoints can also be promoted into immutable model
packages. Runtime and evaluation accept `--model-package`, verify the whole
package and its source graph before loading, and derive backend, weight,
factory, image size, and options from the frozen inference contract.

Training code may use PyTorch, Ultralytics, or an upstream implementation, but
the experiment identity, dataset references, metrics, and artifacts must
follow `experiment_protocol.md` and `artifact_policy.md`.

## 9. CARLA 0.9.16 and `ue5-dev` compatibility policy

The active simulator and bridge are CARLA 0.9.16. Therefore:

1. The [official 0.9.16 examples][carla-examples-0916] and
   [0.9.16 API reference][carla-api-0916] are normative.
2. The [`ue5-dev` examples][carla-examples-ue5] are design references only.
3. No `ue5-dev` script may be copied or imported into the 0.9.16 path until its
   imports, API calls, blueprints, attributes, and behavior pass a compatibility
   test against the pinned server.
4. Adapted code must record the source branch/tag and upstream commit in its
   provenance.
5. A server/client version mismatch must fail fast for Dataset Factory and
   benchmark execution. Exploratory development may override the check only
   with an explicit manifest flag that makes the run non-publishable.

The CARLA documentation warns that development documentation can expose
features unavailable in packaged releases. Synchronous stepping, fixed delta,
bounding-box projection, sensor synchronization, traffic generation, and
manual-control patterns should be adapted from the matching 0.9.16 sources.

Compatibility must be isolated behind two interfaces:

- **runtime transport**: the current minimal RPC/camera bridge;
- **dataset/simulation backend**: the implemented, gated official PythonAPI
  0.9.16 worker.

Upgrading to UE5 is a separate migration experiment. It requires a new
simulator-build ID, map/asset inventory, sensor validation, dataset version,
and cross-version benchmark; it is not an in-place dependency update.

[carla-examples-0916]: https://github.com/carla-simulator/carla/tree/0.9.16/PythonAPI/examples
[carla-api-0916]: https://carla.readthedocs.io/en/0.9.16/python_api/
[carla-examples-ue5]: https://github.com/carla-simulator/carla/tree/ue5-dev/PythonAPI/examples

## 10. Safety and claim boundaries

- CARLA results do not establish real-world safety.
- Teacher control is not evidence of vision-only autonomy.
- Detector AP is not evidence of closed-loop driving ability.
- A live demo is not a statistically valid experiment.
- Ground-truth values may be used for scoring but may not silently enter the
  tested policy.
- Every control experiment must default to no motion, use bounded speed, stop
  on stale perception or process failure, and verify the final stopped state.
- Vision-only control remains disabled until its own interface, tests,
  scenario suite, and acceptance gates exist.

## 11. Acceptance gates

| Gate | Required evidence | Current status |
|---|---|---|
| A0 — framework unit integrity | Contract, adapter, display, perception, risk, artifact, analysis, and dataset tests pass | **Current:** full offline suite passes; see the latest validation report rather than a hard-coded count |
| A0b — evidence discoverability | Every canonical research root is inventoried; invalid objects remain visible; registered files, verification outcomes, plots, checksums, and source-manifest fingerprints are retained | **Current:** the sealed evidence registry and manifest-driven UI explorer are implemented; snapshots remain workspace-bound development artifacts until created from a clean tagged release |
| A1 — live perception smoke | Successful manifest, exact CARLA version/map, synchronized overlay, detections log, summary, video, and hashes for YOLO and RT-DETR | **Current:** development smoke artifacts exist; not a benchmark |
| A2 — Dataset Factory v0 | Deterministic 1,000-frame pilot; zero frame-ID mismatches; zero split conflicts; 100-frame visual audit with at least 99% correct critical-object class/box labels; complete manifest and checksums | **In progress:** exact-sync three-frame pilot, COCO/YOLO/checksums, and automated QA pass; scale, deterministic world ownership, and human audit remain |
| A3 — training pipeline | Re-run from clean commit and immutable dataset; all required artifacts; no NaN; best checkpoint recoverable; metrics reproducible within declared tolerance | **In progress:** launcher, adapters, dataset gate, seed schedule, logs, plots, explicit best/last checkpoint registration, and immutable model promotion pass end-to-end fake-backend tests; real tiny-overfit/resume/training runs remain |
| A4 — detector benchmark | Locked test split; at least three training seeds for primary comparisons; episode-bootstrap confidence intervals; latency on target hardware; qualitative failure report | **In progress:** canonical evaluator and a real pretrained RT-DETR development smoke exist; locked multi-episode, multi-seed evaluation remains |
| A5 — live shadow benchmark | No vision-policy actuation; any teacher motion separately disclosed; frame alignment proven; p50/p95/p99 latency and stale/drop rates recorded across the frozen scenario matrix | **In progress:** real two-model Town10HD development matrix, input audits, videos, p50/p95 metrics, and semantic verification pass; multi-map/weather repetitions, p99 targets, raw replay evidence, and model packages remain |
| A6 — teacher-data generation | Deterministic expert trajectories, privileged-field audit, safe shutdown, and replayable scenario recipes across supported maps | **Planned**; one Town10HD development drive exists |
| A7 — vision-only closed loop | Policy-input audit proves RGB-only; fixed unseen scenario suite; safety monitor disclosure; driving metrics and failure videos; no unresolved critical gate violation | **Planned and intentionally disabled** |

Gate status must be inferred from registered evidence, not manually asserted.

## 12. Near-term implementation sequence

1. Commit and tag the cleanly tested framework baseline.
2. Rerun the read-only preflight on an isolated compatible CARLA 0.9.16
   PythonAPI host, then execute and audit the 50-frame native pilot.
3. Scale Dataset Factory v0 to the A2 sample/QA gate and freeze a
   leakage-safe episode split.
4. Exercise tiny-overfit, resume, and one-epoch training gates, then train an
   RT-DETR baseline and one adapter-based comparison under identical data and
   compute rules.
5. Scale the implemented live-shadow matrix over frozen scenario seeds,
   released model packages, maps, weather, and explicit latency/drop targets.
6. Exercise the implemented validation-only threshold and failure-review
   workflow on a released multi-episode validation partition.
7. Add cross-seed statistics and thesis PDF/LaTeX rendering.
8. Add temporal perception and only then specify and gate vision actuation.

The detailed experimental procedure is defined in
[`experiment_protocol.md`](experiment_protocol.md). Artifact identity,
retention, and integrity rules are defined in
[`artifact_policy.md`](artifact_policy.md). Canonical configuration identity,
hardware/lockfile provenance, and standalone source bundles are specified in
[`reproducibility.md`](reproducibility.md).
