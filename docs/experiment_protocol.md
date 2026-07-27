# Experiment Protocol

Status: active thesis protocol with implemented development gates explicitly marked  
Applies to: dataset generation, detector training, offline evaluation, replay,
live shadow evaluation, teacher driving, and future vision-only driving

## 1. Objective

This protocol makes each result traceable from a thesis table or figure back
to an immutable dataset, model, configuration, code revision, CARLA build, and
set of random seeds. It also prevents frame leakage, accidental use of
privileged simulator state, metric cherry-picking, and test-set contamination.

The experiment is the unit of evidence. A notebook screenshot, terminal
number, or unregistered video is exploratory material, not a thesis result.

## 2. Current versus planned capability

### Current

- A model-neutral live runtime accepts YOLO, RT-DETR, or a custom detector.
- Normalized detections retain the exact RGB frame used for inference.
- Overlay and split live views, annotated video, detections JSONL, run summary,
  and a SHA-256 artifact manifest are implemented.
- The CLI supports `none` and privileged `teacher` control. It refuses
  `vision` control.
- Development smoke runs exist for YOLO and RT-DETR on CARLA 0.9.16,
  Town10HD_Opt.
- Post-hoc run analysis produces checksum-linked metrics, CSV, and plots.
- Dataset Factory v0 performs exact RGB/instance frame pairing, visible-mask
  label extraction, COCO/YOLO export, per-sample provenance, and checksum
  indexing.
- Dataset QA independently verifies source hashes, regenerates labels and YOLO
  exports, checks episode split leakage, and creates plots and a montage.
- A live three-frame pilot produced 27 annotations and passed the automated QA
  checks. It is development evidence, not the 1,000-frame acceptance dataset.
- Strict scenario contracts implement the documented hierarchical seed
  derivation, `_Opt` map-family grouping, episode-level group assignment, and
  seen/map-OOD/weather-OOD partitions.
- The resolved pilot plan contains seven recipes, 23 episodes across six map
  families and six weather recipes, and 3,450 planned captures. Plan loading
  verifies hashes and recomputes every episode from the frozen source inputs.
- A gated native official-PythonAPI worker implements synchronous fixed-delta
  collection, Traffic Manager and pedestrian seeding, map reload per
  repetition, traffic/walkers/props, exact sensor phase matching, provenance,
  and cleanup. Its dry-run and fake-API tests pass; a real native-server run is
  pending.
- Training launchers for RT-DETR, YOLO, and custom adapters verify the dataset,
  exclude locked tests, freeze partitions/seeds/configuration, and register
  logs, histories, plots, and best/last checkpoint hashes. The end-to-end path
  passes with a fake backend; real training awaits a trainable release.
- The common offline evaluator writes canonical COCO predictions and metrics,
  frozen operating-point metrics, calibration, episode-bootstrap uncertainty,
  strata, latency, raster/vector plots, and a rule-selected failure montage.
  A real pretrained RT-DETR development smoke has completed.
- The paired replay service runs arbitrary RT-DETR, YOLO, custom, or packaged
  detectors on one verified sample order. It retains independently verifiable
  child evaluations, per-image outcomes, paired episode bootstrap, latency and
  disagreement tables, PNG/SVG plots, and a comparison montage. A real
  RT-DETR-L/YOLO26n development replay has completed.
- Validation-only threshold selection and failure mining are implemented as
  independent verified objects. Threshold selection reruns its rule under
  episode bootstrap. Mining retains all categorized failures and builds a
  deterministic, diversity-capped human-review queue. Review finalization
  archives the completed CSV and creates confirmed/label-issue catalogs
  without moving validation RGB into training.
- A versioned `VisionObservation`/`VisionControlProposal` contract exposes
  only copied front RGB and RGB-derived detections to a modular policy. Shadow
  proposals are logged and displayed but cannot reach the actuator.
- The live shadow matrix preregisters and sequentially executes detector/policy
  cells. Teacher motion requires explicit acknowledgement, policy-input audits
  are retained, and dedicated verifiers reconstruct parent commands, child
  lineage, proposal logs, summaries, and the no-actuation invariant. A real
  RT-DETR-L/YOLO26n Town10HD development matrix completed 2/2 child runs.
- A local operator POC exposes live recording, situation/crowdedness recipes,
  native collection, replay, training, analysis, and verification through
  allow-listed forms. It generates no shell strings, refuses output overwrite,
  retains every request/command/status/log as a verified operator session, and
  keeps destructive or expensive actions behind separate acknowledgements.
- A read-only native preflight verifies scenario selection, output identity,
  endpoint reachability, server version/map, official PythonAPI import,
  client/server agreement, and manual world/tick-owner gates. It creates
  JSON/CSV/Markdown evidence and never mutates CARLA. A real endpoint
  assessment correctly records the current Apple arm64 host as not ready.
- A portable native-host kit freezes the bounded 50-frame train episode,
  source code, hash-pinned CPython 3.12 Windows/Linux dependencies, and guarded
  scripts. Its collection entry point requires an exact, independently
  verified ready preflight and the literal reload/tick-owner confirmation.
- A generic verifier checks tracker status/timestamps, safe paths, SHA-256,
  sizes, sorted checksum indexes, external references, semantic dataset/plan/
  model/replay/report contracts, unregistered files, and an optional clean-Git
  gate.
- A point-in-time evidence registry scans all seven canonical roots, retains
  verification failures instead of hiding them, and produces deterministic
  object/artifact tables, plots, checksums, and a human-readable report. Its
  semantic verifier detects drift in recorded source manifests without
  invalidating older snapshots when later objects are added.
- Selected checkpoints can be promoted into immutable model packages carrying
  a frozen input/inference adapter contract, ontology, licenses, model card,
  source training/dataset lineage, and a checksum index. Runtime and evaluation
  can consume these packages directly.
- The report builder verifies every source first, extracts metrics without
  manual transcription, and writes canonical CSVs, Markdown, PNG/SVG plots,
  a source graph, and a checksum-indexed report release. The seven-source v0.5
  report includes live policy-shadow runs and systems plots; the five-source
  v0.6 report adds verified operator sessions and a deterministic
  crowded-situation plan.

### Planned

- real execution of the 50-frame native integration pilot on an isolated
  compatible 0.9.16 PythonAPI host;
- an A2-scale, human-audited, frozen leakage-safe dataset release;
- tiny-overfit/resume and real multi-seed model training;
- locked multi-episode detector benchmarks;
- systematic live benchmarks and long-stream replay;
- real human review and remediation cycles on the A2 validation release;
- cross-seed statistical comparison and PDF/LaTeX thesis rendering;
- signed evidence releases, supersession metadata, and repository-wide
  cross-object cycle analysis;
- temporal RGB perception and vision-only closed-loop control.

Current smoke artifacts must not be included in a final thesis comparison
unless they are regenerated after the applicable protocol and acceptance gates
are implemented.

## 3. Experiment lifecycle

Every confirmatory experiment follows these states:

```text
proposal
  -> pre-registered configuration
  -> input integrity check
  -> execution
  -> artifact finalization
  -> automated validation
  -> blinded/locked-test evaluation when applicable
  -> analysis
  -> report inclusion
```

An experiment that stops early still gets a manifest with `failed` status and
the redacted error. It is not silently deleted or reused under the same ID.

### 3.1 Pre-registration record

Before execution, save a resolved, immutable configuration containing:

- research question and hypothesis;
- stage: `data`, `train`, `eval`, `replay`, `live`, `teacher`, or
  `vision-control`;
- primary and secondary metrics;
- model adapter, architecture, weights initialization, and thresholds;
- dataset and ontology IDs;
- exact split names;
- master seed and repetition index;
- training budget and stopping rule;
- image resolution, camera parameters, and preprocessing;
- CARLA build, maps, scenario suite, and control mode;
- permitted privileged lanes;
- planned exclusions;
- target hardware;
- acceptance gate and failure conditions.

Changing any field after execution begins creates a new experiment ID. A
post-hoc analysis may reuse frozen predictions but must receive a new analysis
ID and must be labeled post-hoc.

## 4. Identifiers

The canonical experiment ID is:

```text
exp-<UTC>-<stage>-<model>-<cfg8>-s<seed>
```

Example:

```text
exp-20260726T171722Z-live-rtdetr-l-4fa19c2e-s42
```

Where:

- `UTC` is the start time in `YYYYMMDDTHHMMSSZ`;
- `stage` is from the controlled stage vocabulary;
- `model` is a short normalized adapter/model name;
- `cfg8` is the first eight hexadecimal characters of SHA-256 over the
  canonical resolved configuration, excluding start time and output paths;
- `seed` is the training or scenario master seed.

`carla_vision.reproducibility.canonical_experiment_id` implements this
convention. `RunArtifactTracker` allocates and verifies the canonical ID when
stage, model, and master seed are provided together; otherwise it retains the
legacy user-supplied or timestamp-plus-random development ID behavior.
Confirmatory experiments must use the canonical path.

Dataset, model, scenario, and report identifiers are defined in
[`artifact_policy.md`](artifact_policy.md).

## 5. Reproducibility and seeds

### 5.1 Hierarchical seed derivation

One integer `master_seed` is the root of all randomness. Independent
sub-seeds are derived with:

```text
payload = "carla-vision-v1\0" + str(master_seed) + "\0" + namespace
derived_seed = uint64_be(SHA256(payload)[0:8]) mod (2^31 - 1)
```

Required namespaces include:

```text
python
numpy
torch
model_init
data_order
augmentation
world
traffic_manager
vehicles
walkers
walker_crossing
props
weather
route
camera_jitter
failure_mining
bootstrap
```

The manifest records the master seed, every namespace, the derivation version,
and the resulting integer. Code must never use an unseeded default RNG in a
confirmatory experiment.

### 5.2 CARLA generation settings

The native Dataset Factory worker is designed to:

- use CARLA synchronous mode;
- use one declared fixed `fixed_delta_seconds`, initially `0.05` seconds unless
  the dataset recipe justifies another value;
- make sensor ticks integer multiples of the fixed delta;
- designate exactly one client as the world-tick owner;
- run Traffic Manager in synchronous mode;
- set Traffic Manager, pedestrian, Python, NumPy, world, actor, prop, weather,
  route, and camera-jitter seeds explicitly;
- record actor blueprint IDs, spawn transforms, attributes, and destruction
  status;
- record the exact CARLA server build and map package;
- fail on missing frames rather than pairing data by arrival order.

Official CARLA guidance on [synchronous mode and fixed time-step][carla-sync]
is normative for the 0.9.16 implementation.

[carla-sync]: https://carla.readthedocs.io/en/0.9.16/adv_synchrony_timestep/

### 5.3 Training determinism

Each training run must record:

- Python, NumPy, and PyTorch seeds;
- data-loader worker seeds and sample order;
- deterministic-algorithm flags;
- CUDA, cuDNN, MPS, compiler, mixed-precision, and TF32 settings;
- package versions, GPU model, driver, and operating system;
- batch size, gradient accumulation, optimizer state, and scheduler state.

Exact bitwise replay is required when the backend and hardware support it.
When it does not, the manifest must say `determinism: statistical`, identify
the nondeterministic operations, and report independent repetitions. Primary
model comparisons require at least three training seeds unless a power or
compute analysis registered in advance justifies more or fewer.

### 5.4 Reproduction levels

| Level | Requirement |
|---|---|
| R0 — configuration | Resolved configuration and source revision are recoverable |
| R1 — artifact | Existing outputs pass checksum verification |
| R2 — execution | Same inputs and environment rerun successfully |
| R3 — deterministic | Canonical predictions or dataset files are byte-identical |
| R4 — statistical | Repeated metrics remain within a pre-registered tolerance or confidence interval |

Every published result declares the highest demonstrated level. Do not call a
run reproducible merely because a seed was saved.

## 6. Dataset generation protocol

### 6.1 Dataset Factory v0 scope

The first acceptance milestone is deliberately small enough to audit:

- CARLA 0.9.16;
- Town10HD_Opt;
- one fixed forward monocular RGB camera;
- approximately 1,000 retained RGB frames;
- 8–12 driving-critical detection classes;
- multiple lighting and weather recipes;
- RGB/teacher frame synchronization;
- canonical COCO annotations;
- episode, frame, split, and artifact manifests;
- 100-frame human QA sample.

This validates the pipeline, not the final model.

### 6.2 Camera and frame contract

Each frame record must contain:

- `dataset_id`, `scenario_id`, `episode_id`, and split;
- CARLA frame ID and simulation timestamp;
- image relative path, width, height, SHA-256, and byte size;
- camera blueprint, transform relative to vehicle, world transform,
  intrinsics, FOV, exposure-related attributes, and sensor tick;
- map family, map package, OpenDRIVE hash, road/lane identifiers when
  available;
- weather recipe ID and resolved weather vector;
- references to annotations and teacher provenance.

Initial dataset capture should use `1280x720` unless a pilot measurement shows
that the simulator or storage cannot sustain it. Runtime inference may benchmark
different resize values, but the source capture should preserve small and
distant signs.

RGB, instance segmentation, semantic segmentation, and optional depth teacher
data must share the same CARLA frame ID. The synchronizer must reject:

- missing teacher frames;
- duplicate frame IDs;
- mismatched image dimensions or intrinsics;
- out-of-order frames that cannot be resolved without guessing;
- actor snapshots from a different simulation tick.

### 6.3 Canonical annotation

COCO JSON is the canonical detection format. Model-specific YOLO text files or
other layouts are derived exports and must be reproducible from the canonical
release.

Each object annotation must include:

- canonical category ID and label;
- source semantic tag, actor or environment-object ID, and blueprint/type ID;
- visible `bbox_xywh` derived with instance visibility;
- amodal projected box as teacher metadata when available;
- visible mask pixel count or visible fraction;
- truncation and occlusion categories;
- traffic-light state or traffic-sign subtype/value where valid;
- teacher distance and bearing for stratified evaluation only;
- `iscrowd` and ignore reason where applicable;
- provenance method and label-schema version.

The label builder must use the official CARLA 0.9.16 actor bounding-box and
sensor APIs. The [official bounding-box tutorial][carla-boxes] is a reference,
not a drop-in label-quality guarantee.

[carla-boxes]: https://carla.readthedocs.io/en/0.9.16/tuto_G_bounding_boxes/

### 6.4 Ontology policy

Version 1 begins with driving-critical boxes:

- pedestrian;
- rider;
- bicycle;
- motorcycle;
- car;
- van;
- truck;
- bus;
- emergency vehicle where reliably distinguishable;
- red, yellow, and green traffic-light states;
- selected stop, yield, and speed-limit signs;
- cone, barrier, roadwork warning, and road obstacle.

Lane markings, road edge, curb, crosswalk, and drivable area are not forced
into object boxes. They belong to a planned segmentation or geometry task.
Context such as weather and map is metadata, not a detection class.

An ontology version is immutable. Renaming, merging, or splitting a class
creates a new ontology and a new derived dataset release.

### 6.5 Scenario matrix

Each dataset release contains an explicit matrix, not unconstrained random
sampling:

| Factor | Minimum planned levels |
|---|---|
| Map | Multiple map families; `_Opt` and non-`_Opt` forms grouped together |
| Light | dawn, daytime, sunset/backlight, night |
| Weather | clear, cloudy, wet, light rain, heavy rain, fog/haze using plausible vectors |
| Traffic | empty, light, medium, heavy |
| Pedestrians | none/low, normal, dense/crossing |
| Vehicle state | moving, parked, stopped/stalled |
| Road context | ordinary, junction, crosswalk, construction, obstruction |
| Camera | nominal plus bounded, realistic mounting/exposure variations |

Weather components must be sampled as named plausible recipes. Independently
uniform weather parameters can create physically incoherent scenes and are not
allowed in confirmatory data.

Ordinary capture is sampled at a declared 2–5 Hz to limit adjacent-frame
correlation. Rare events may use denser event windows, but those frames share
one episode/group ID and cannot cross splits.

## 7. Leakage-safe split protocol

### 7.1 Forbidden split method

Randomly splitting individual video frames is forbidden. Neighboring frames
share scene geometry, actors, textures, lighting, and pose and would inflate
validation and test performance.

### 7.2 Group identity

The atomic grouping key is:

```text
simulator_build
+ map_family
+ route_or_road_region_id
+ scenario_recipe_id
+ static_layout_seed
+ episode_id
```

`Town01` and `Town01_Opt`, for example, have the same `map_family` for split
purposes. Images, clips, crops, augmentations, annotations, and mined examples
derived from one group inherit its split.

### 7.3 Required benchmark partitions

Before data collection, `split_plan.yaml` must freeze:

1. at least one entire map family for `val_map_ood`;
2. at least two entire map families for the locked `test_map_ood`;
3. named weather/light combinations for `val_weather_ood` and locked
   `test_weather_ood`;
4. for remaining map/weather families, deterministic group-hash assignment:
   0–84 to `train`, 85–92 to `val_seen`, and 93–99 to `test_seen`.

The group bucket is the first eight bytes of SHA-256 over the canonical group
key, interpreted unsigned and reduced modulo 100. The dataset release stores
the computed assignment rather than recomputing it on demand.

Map-family choices must be made using only asset/class-coverage metadata, not
model performance. If a held-out family lacks a required class, report the
missing stratum and add a separate targeted test set; do not move high-error
episodes into training.

### 7.4 Leakage audit

Before release, automated validation must prove:

- one group appears in exactly one split;
- one source frame and all of its derivatives share a split;
- no matching image SHA-256 appears across splits;
- no near-duplicate perceptual hash above the declared similarity threshold
  crosses splits without a reviewed exception;
- no base/`_Opt` map pair crosses map-family boundaries;
- train-derived normalization statistics exclude validation and test;
- class and scenario counts are reported per split;
- locked-test labels and predictions are access-controlled in the workflow.

The final test sets are never used for threshold selection, failure mining,
early stopping, qualitative example selection, or ontology design.

## 8. Model training protocol

### 8.1 Baselines

The first primary baseline is an Ultralytics RT-DETR checkpoint initialized
from declared pretrained weights, initially `rtdetr-l.pt`. A lighter official
RT-DETR/RT-DETRv2 implementation or another detector may be compared through
an adapter, but all comparisons must use:

- the same immutable training and evaluation split;
- the same canonical ontology;
- equivalent augmentation disclosure;
- the same input source resolution;
- a declared compute budget;
- independent, pre-registered training seeds;
- canonical predictions for shared evaluation.

PyTorch compatibility does not make two trainers methodologically equivalent.
Backend-specific preprocessing, optimizer defaults, image resize, and
postprocessing must be resolved and recorded.

### 8.2 Pilot sequence

1. Overfit a tiny audited subset to detect label or implementation errors.
2. Run a one-epoch smoke experiment and verify all artifacts.
3. Train the pilot baseline on the frozen pilot split.
4. Evaluate only on pilot validation partitions.
5. Inspect failure strata and decide whether the dataset pipeline or ontology
   requires a new version.
6. Freeze a full dataset release.
7. Run confirmatory multi-seed experiments.

### 8.3 Fair comparison rules

- Select hyperparameters on validation only.
- Select confidence and any NMS/query thresholds on validation only, retain
  the selection artifact, and freeze them before locked-test inference.
- Report both equal-training-budget and best-practical-configuration results
  when those answer different questions.
- Count failed and diverged seeds; do not replace them silently.
- Keep data augmentation stochasticity independent across repetitions while
  using the registered seed schedule.
- Change one causal factor per ablation unless the design explicitly studies an
  interaction.
- Store every resolved trainer configuration and the exact pretrained-weight
  digest.

## 9. Evaluation metrics

### 9.1 Offline detection

Required metrics:

- COCO `AP@[0.50:0.95]`;
- AP50 and AP75;
- AP for small, medium, and large objects;
- per-class AP, precision, recall, and false-negative rate;
- critical-class recall at the frozen operating threshold;
- localization error and class confusion;
- traffic-light state and traffic-sign subtype confusion;
- calibration metrics and reliability diagram when confidence is used for
  control decisions.

All metrics must also be stratified, where sample count permits, by:

- map family and road context;
- weather and light;
- traffic density;
- teacher distance range;
- visible pixel area;
- occlusion and truncation;
- ordinary versus rare-event scenario;
- nominal versus perturbed camera.

Report support counts beside every stratified metric. A metric with too few
instances is marked insufficient rather than pooled invisibly.

### 9.2 Temporal and live runtime

Required metrics:

- camera acquisition FPS;
- detector throughput FPS;
- inference latency p50, p95, and p99;
- end-to-end source-frame age p50, p95, and p99;
- submitted, processed, and superseded-before-inference frame counts;
- recorder submitted, written, and dropped counts;
- stale-frame duration and events per minute;
- temporal detection flicker and persistence on annotated sequences;
- peak memory and target-device description.

Latency must be measured after warm-up and separated into preprocessing,
model, postprocessing, and end-to-end values where the backend permits it.
Never report `1 / mean_latency` as measured pipeline FPS.

### 9.3 Future closed-loop driving

When vision-only control is implemented, required metrics include:

- route completion;
- collisions per kilometer, separated by pedestrian, vehicle, and static
  object;
- red-light, stop-sign, and speed violations;
- lane departure and off-road duration;
- intervention or independent-safety-stop count;
- minimum ground-truth time-to-collision for evaluation only;
- mean speed and stopped-time ratio;
- longitudinal jerk and lateral acceleration;
- scenario success rate and time to completion.

The policy-input audit and safety-supervisor intervention log must accompany
every closed-loop metric. A route completed by teacher control is not a
vision-policy result.

### 9.4 Statistical analysis

- Treat episode or scenario recipe/seed as the independent sampling unit, not
  individual frames.
- Use paired scenario seeds when comparing models in replay or closed-loop
  experiments.
- Report mean and standard deviation across training seeds.
- Report 95% episode-cluster bootstrap confidence intervals, using a registered
  bootstrap seed and at least 10,000 resamples for final tables.
- Pre-register one primary metric per hypothesis.
- Label exploratory subgroup analyses and correct or disclose multiple testing
  when making confirmatory claims.
- Include effect size and confidence interval, not only a p-value.

## 10. Failure mining

Failure mining is an iterative data-development process, not permission to
train on the test set.

### 10.1 Sources

Failures may be mined from:

- training diagnostics;
- validation partitions;
- dedicated exploratory scenarios;
- live shadow runs;
- user-reported development runs.

Locked test partitions are excluded.

### 10.2 Failure record

Each candidate receives:

- failure ID and source experiment ID;
- frame/clip and episode identifiers;
- model and dataset IDs;
- expected and predicted objects;
- category: false negative, false positive, misclassification, localization,
  state error, temporal flicker, latency/staleness, or control consequence;
- class and severity;
- map/weather/light/distance/occlusion strata;
- human review decision and reviewer;
- proposed scenario or labeling remedy.

### 10.3 Mining cycle

1. Rank candidates by safety severity, rarity, uncertainty, and diversity.
2. De-duplicate by source group, embeddings or perceptual hash, and scenario
   metadata.
3. Human-review labels and root causes.
4. Generate or select new examples with new scenario IDs and seeds.
5. Add them only to a new dataset version.
6. Preserve the old dataset and model results.
7. Retrain under a new experiment ID.
8. Measure the originally failing validation slice and general validation
   metrics to detect regression.

No mined frame may migrate from validation or test into training. If a similar
scenario is generated for training, it must use a new map/route grouping or
seed as required by the frozen split plan.

## 11. Acceptance gates

### G0 — code and manifest gate

- all tests for the touched subsystem pass;
- source revision is recorded;
- confirmatory runs use a clean committed worktree;
- configuration is resolved and hashed;
- manifest starts before execution and finalizes on success or failure;
- required files pass checksum verification.

**Current evidence:** the full offline suite passes across runtime, artifacts,
dataset, scenario, native-worker orchestration, training, evaluation, policy
input isolation, and live shadow orchestration; use
the latest validation report for the exact count and command. Runtime manifests
also retain development runs. The repository state was dirty/untracked during
the audit, so current runs are development-only.

### G1 — Dataset Factory v0

- approximately 1,000 retained RGB frames with complete frame records;
- zero RGB/teacher frame-ID mismatch;
- zero annotation references to missing images;
- zero split/group conflicts;
- zero checksum failures;
- deterministic replay of at least three scenario recipes produces the same
  actor/spawn manifest and frame labels, or documented rendering-level
  exceptions;
- 100 frames sampled by a fixed QA seed are visually reviewed;
- at least 99% of critical visible objects in that sample have an acceptable
  class and box under the written QA rubric;
- every discrepancy is retained in the QA report.

### G2 — dataset release

- ontology and split plan are frozen;
- class, distance, weather, map, and occlusion coverage reports exist;
- cross-split exact and near-duplicate checks pass;
- canonical COCO validator passes;
- dataset tree checksum and release manifest are signed off;
- a fresh environment can load and visualize a deterministic sample.

### G3 — training pipeline

- tiny-subset overfit and one-epoch smoke pass;
- no NaN/Inf in loss, gradients, weights, or metrics;
- interrupted training resumes from a checkpoint with documented tolerance;
- best and last checkpoints are registered and loadable;
- validation predictions reproduce from the saved best checkpoint;
- training plots and machine-readable history agree.

**Current evidence:** configuration and split gates, deterministic seed
application, input/checkpoint hashing, logging, and artifact registration pass
an end-to-end fake-backend run. Explicit best/last roles and immutable model
promotion also pass consumer-side verification and tamper tests. A real
tiny-subset overfit, interruption/resume, and validation-prediction
reproduction have not yet passed.

### G4 — detector benchmark

- primary configurations complete all registered seeds, or failures are
  reported;
- frozen thresholds and checkpoints are evaluated once on locked test;
- canonical metrics, stratification, latency, uncertainty, plots, and
  qualitative failure panels are registered;
- every thesis table cell links to experiment IDs;
- no primary conclusion depends only on a smoke run or one cherry-picked seed.

**Current evidence:** a real pretrained RT-DETR development evaluation
produced canonical predictions, COCO and operating-point metrics, calibration,
latency, episode bootstrap, plots, CSV/JSON sources, and a failure montage.
Only three images from one unassigned episode were used, so it is neither a
locked benchmark nor statistical evidence. A derived report release proves
machine extraction and source-graph verification, not the validity of those
development metrics. A paired replay additionally evaluated RT-DETR-L and
YOLO26n in the exact same RGB order and retained both child runs plus paired
tables and plots. The shared prediction cutoff exposed a large score-
calibration difference, so the result is a pipeline/calibration diagnostic and
not a fair final ranking.

### G5 — live shadow

- vision-policy actuation is disabled; any teacher motion is separately
  disclosed and cannot be presented as a policy result;
- every overlay is frame-correct;
- camera, inference, and frame-age distributions are recorded;
- no unexplained stream reset or frame-ID regression;
- stale and drop rates satisfy a pre-registered model/hardware target;
- raw or replayable input evidence exists for critical failures.

**Current evidence:** a two-cell RT-DETR-L/YOLO26n matrix completed against the
live CARLA 0.9.16 Town10HD_Opt endpoint with 2/2 successful children, separate
proposal logs, terminal overlays, MP4 videos, input audits, p50/p95 systems
metrics, verified final stops, and `actuation_applied: false` for every policy
proposal. It is a short sequential development run, not the frozen
multi-scenario A5 benchmark; it lacks p99 targets and raw replayable RGB.

### G6 — teacher driving

- teacher use of pose is declared;
- camera mount validation passes;
- actuator watchdog and final stop verification pass;
- fixed scenario suite and seeds are replayable;
- control, detection, and privileged evaluation logs are aligned by frame/time;
- the report labels results as teacher-driven.

### G7 — vision-only closed loop

- code-level input audit proves the policy sees only allowed RGB temporal data
  and declared high-level route intent;
- no teacher pose or ground truth reaches the policy process;
- independent safety supervision is logged separately;
- frozen unseen scenario suite is run with paired seeds;
- all required driving, latency, and failure artifacts exist;
- critical collision or compliance regressions are resolved or explicitly
  reported.

Until G7 is implemented and passed, `--control vision` must remain unavailable.

## 12. Required reporting

Each final experiment report contains:

- research question and pre-registered hypothesis;
- experiment, model, dataset, code, CARLA, and hardware IDs;
- exact train/validation/test split;
- privileged-data declaration;
- methods generated from the resolved configuration;
- aggregate and stratified metrics with supports and uncertainty;
- runtime performance;
- plots with machine-readable source data;
- representative successes selected by a fixed rule;
- failures selected by severity/diversity, not appearance;
- limitations, deviations, and failed repetitions;
- links to all registered artifacts.

Artifact structure and retention rules are normative in
[`artifact_policy.md`](artifact_policy.md).
