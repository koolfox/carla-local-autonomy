# CARLA Vision Research — Validation Report v0.3

Date: 2026-07-26  
Framework scope: CARLA 0.9.16, front monocular RGB, model-neutral detection  
Evidence class: development validation, not a publishable thesis benchmark

## 1. Executive result

The framework core is operational from live RGB acquisition through
frame-correct inference, artifact capture, exact-frame dataset generation,
dataset verification, deterministic scenario planning, training orchestration,
and canonical offline evaluation.

The full offline regression suite passes 123 tests. Ruff format/lint and Python
bytecode compilation pass. The latest retained live run connected to CARLA
0.9.16 at `172.20.10.7:2000`, reused vehicle actor 24 and RGB actor 25, ran
RT-DETR on Apple MPS, rendered and recorded exact inferred frames, and exited
without moving or destroying either actor. All 55 artifacts across the seven
principal evidence objects listed below independently match their registered
SHA-256 digests; the recursive verifier also reports zero unregistered files.

This is not yet an end-to-end thesis result. The native collector has not been
run against a compatible official PythonAPI host because doing so reloads the
map and destroys existing actors. The current dataset has three images in one
`unassigned` episode, so it is intentionally unusable for training and
statistical claims. No vision-only driving policy exists; `--control vision`
remains rejected.

## 2. Scientific boundary

- The deployable detector receives only the ordered front RGB image.
- Frame IDs and timestamps may preserve temporal order but are not perception
  features.
- Instance segmentation, actor IDs, CARLA pose, weather, map, and other
  simulator state are privileged teacher/evaluation data.
- The optional teacher controller uses privileged pose and cannot support a
  vision-only autonomy claim.
- A 2D detection box is not metric distance, and detector AP is not a
  closed-loop driving metric.

The runtime records `front_monocular_rgb_only`. The native dataset worker uses
co-located instance segmentation only to construct labels and provenance.

## 3. Validation environment

| Item | Observed value |
|---|---|
| Research client | Apple arm64, Darwin 25.5.0 |
| Project interpreter | CPython 3.12.11 through `uv` |
| `uv` | 0.11.2 |
| PyTorch | 2.13.0 |
| Ultralytics | 8.4.106 |
| NumPy | 2.5.1 |
| OpenCV package | 5.0.0.93 |
| Accelerator used for live/eval RT-DETR | Apple MPS |
| CARLA endpoint | `172.20.10.7:2000` |
| CARLA server | 0.9.16 |
| Live map | `Carla/Maps/Town10HD_Opt` |
| Live vehicle/RGB actors | 24 / 25 |
| Git state | repository has no commit; files are untracked/dirty |

The system shell currently resolves a newer Python, but every reported project
check and run uses the pinned `uv` environment. The native official CARLA
PythonAPI is deliberately imported lazily and is not installed on the Apple
client.

## 4. Verification matrix

| Subsystem | Evidence | Result |
|---|---|---|
| Contracts/adapters | Unit tests for normalized boxes, metadata, custom factories, and invalid configs | Pass |
| Latest-frame inference | Concurrency tests for one in-flight and one replaceable pending frame | Pass |
| Overlay correctness | Detections rendered only on the exact inferred source frame | Pass |
| Live lifecycle | GUI close/key paths, recorder drain, no-control mode, watchdog/controller boundaries | Pass |
| Artifact tracking | Atomic manifests, containment, redaction, hashing, success/failure finalization | Pass |
| Run analysis | Immutable source verification, lineage, CSV, summary, four plots | Pass |
| Dataset synchronization | Exact CARLA frame/timestamp/geometry/FOV/transform agreement | Pass |
| Label generation | CARLA instance-ID decode, visible boxes, ontology mapping, COCO/YOLO | Pass |
| Dataset release | Checksums, sample references, counts, ontology, episode leakage, exact RGB leakage | Pass |
| Dataset QA | Recreated masks/boxes/YOLO, plots, montage, checksum verification | Pass |
| Scenario plan | Strict recipes, seed derivation, group-safe split assignment, plan recomputation | Pass |
| Native worker | Fake-API orchestration tests plus no-import/no-mutation dry run | Pass locally |
| Native real integration | Requires compatible 0.9.16 PythonAPI and exclusive tick ownership | Not run |
| Training | Config/split gates and fake-backend end-to-end artifacts/checkpoints | Pass locally |
| Real training | Requires a verified dataset with train/validation partitions | Not run |
| Offline evaluation | Fake perfect-detector tests and real pretrained RT-DETR development run | Pass |
| Generic verification | Safe paths, timestamps, SHA-256/size, sorted checksum indexes, nested references, clean-Git and unregistered-file gates | Pass |
| Model promotion | Fake trained checkpoint to sealed weights/config/ontology/model-card package; tamper and contract tests | Pass locally |
| Real model package | Requires a real successful training checkpoint and confirmed licenses | Not produced |
| Report release | Recursive five-source verification, canonical CSVs, Markdown, PNG/SVG plots, source invalidation, checksum index | Pass |
| Vision-only policy/control | Deliberately absent and rejected by the CLI | Not implemented |

## 5. Live RT-DETR evidence

Run: `runs/framework-v0.2-rtdetr-smoke`

- Status: success.
- Mode: perception only; requested duration 3 seconds.
- Camera: existing actor 25, 640×384, 90° FOV, 10 Hz.
- Model: `rtdetr-l.pt`, 66,511,432 bytes, SHA-256
  `6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f`.
- Frames: 20 submitted, 17 processed, one superseded before inference.
- Recording: 17 submitted, 17 written, zero recorder drops.
- Vehicle: zero travelled distance, zero maximum simulator speed.
- Retained evidence: manifest, summary, detections JSONL, final overlay PNG,
  annotated MP4.

The derived analysis run
`runs/framework-v0.2-rtdetr-smoke-analysis` reports:

| Measure | Value |
|---|---:|
| Analyzed frames | 17 |
| Effective source FPS | 3.2616 |
| Detections | 78 |
| Pipeline latency median | 265.81 ms |
| Model inference median | 175.80 ms |
| First/cold model inference | 2608.11 ms |
| Estimated travelled distance | 0.0 m |

This short run validates transport, adapter, frame association, display,
recording, provenance, and shutdown. It is too short to characterize runtime
tails or detection accuracy.

## 6. Dataset evidence

Dataset: `datasets/ds-carla0916-town10-pilot-v001`  
QA run: `runs/ds-carla0916-town10-pilot-v001-qa`

- Three exact RGB/instance matched images from one Town10HD_Opt episode.
- 27 visible annotations: 3 cars, 6 traffic lights, 18 traffic signs.
- 15 checksum-index entries verified.
- Three teacher masks and three YOLO exports independently reproduced.
- No episode split leakage.
- No exact RGB duplicate group across partitions.
- Three images included in the deterministic QA montage.
- Teacher BGRA instance masks are retained and explicitly marked privileged.

All samples are in `unassigned`. The training resolver correctly refuses this
dataset because it has no frozen train/validation partition. The dataset is a
contract pilot, not the planned 1,000-frame A2 release.

## 7. Deterministic scenarios and native collection

Plan: `runs/scenario-plan-thesis-pilot-v1`

| Plan property | Value |
|---|---:|
| Recipes | 7 |
| Episodes | 23 |
| Planned captures | 3,450 |
| Map families | 6 |
| Weather recipes | 6 |
| Train episodes | 11 |
| Validation episodes, all types | 5 |
| Test episodes, all types | 7 |

Partitions are `train`, `val_seen`, `test_seen`, `val_map_ood`,
`test_map_ood`, `val_weather_ood`, and `test_weather_ood`. `_Opt` and base map
names share a map-family identity. Every episode carries a hierarchical seed
bundle derived from master seed 20260726. Loading the plan verifies the
artifact hashes and recreates the episode records from the resolved suite and
split plan.

A native dry run selecting `test_seen` verified one Town10HD episode and 150
planned captures without importing CARLA, connecting to the server, creating a
dataset, or mutating the simulator. A real run requires:

1. an official CARLA 0.9.16 PythonAPI compatible with the execution host;
2. no other client calling `world.tick()`;
3. permission to load/reload every selected map and destroy old world actors;
4. `--acknowledge-exclusive-tick-owner`.

The worker restores asynchronous settings and destroys actors it created in
its cleanup path, but map reload itself is intentionally treated as
destructive.

## 8. Training and evaluation evidence

### Training

The training service verifies the immutable dataset before it creates a run,
never exposes a locked test partition to the trainer, freezes the resolved
data YAML and seed schedule, fingerprints pretrained weights, and supports:

- Ultralytics RT-DETR;
- Ultralytics YOLO;
- a custom `TrainerBackend`.

An end-to-end fake backend proves registration of the training log,
machine-readable history, plot, best checkpoint, last checkpoint, resolved
configuration, seeds, and summary. Configuration validation rejects batch
zero, empty device/optimizer/cache strings, zero initial learning rate,
overrides of controlled options, overlap between train/validation, and test
partition use.

This validates orchestration but not convergence, resume fidelity, model
quality, or hardware capacity.

### Model package and common consumption

The model promotion service requires one verified successful training run, one
selected best/last checkpoint, an explicit input contract, an inference
adapter contract, validation-only selection metadata, non-placeholder license
statements, intended use, and limitations. It creates packaged weights,
`model.json`, `model-card.md`, the resolved release configuration, and a sorted
checksum index. Consumer verification recursively checks the source training
run and dataset before returning a `DetectorConfig`.

Runtime and offline evaluation accept `--model-package`. They reject loose
detector, weight, factory, or image-size options when a package is selected,
preventing silent drift from the released adapter/preprocessing identity. This
path passes with a deterministic fake training backend; no real CARLA-trained
model has yet been promoted.

### Real pretrained RT-DETR development evaluation

Run: `runs/eval-rtdetr-pretrained-pilot-v1`

| Measure | Value |
|---|---:|
| Images / episodes | 3 / 1 |
| Ground-truth annotations | 27 |
| Source / operating confidence | 0.05 / 0.25 |
| TP / FP / FN at operating point | 3 / 0 / 24 |
| Precision | 1.0000 |
| Recall | 0.1111 |
| False-negative rate | 0.8889 |
| F1 | 0.2000 |
| COCO AP@[.50:.95] | 0.2723 |
| COCO AP50 / AP75 | 0.3333 / 0.3333 |
| First-image latency | 2504.85 ms |
| Warm two-image mean/median latency | 317.74 / 317.74 ms |
| Registered artifacts | 20 |

The evaluator mapped 467 low-confidence predictions to the canonical ontology
and disclosed 388 unmapped predictions. The bootstrap correctly marks its
interval as degenerate because only one episode exists. The numerical results
must not be used to compare models or claim accuracy: the sample is tiny,
unassigned, dominated by very small objects, and the weights were pretrained
rather than trained on this ontology.

The useful result is structural: the run retained canonical COCO predictions,
per-frame logs, COCO output, operating-point metrics, per-class and stratum
tables, calibration bins, episode bootstrap, latency, five plots in both PNG
and SVG, and a rule-selected qualitative failure montage.

## 9. Manifest-driven development report

Report: `reports/rpt-framework-validation-20260726-v001`

- Five recursively verified sources: dataset, QA, scenario plan, live runtime
  analysis, and pretrained RT-DETR evaluation.
- 43 canonical metric rows.
- 40 source-artifact inventory rows.
- Four generated plots: source inventory and evaluation summary in PNG/SVG.
- 11 registered report artifacts and ten checksum-index entries.
- Zero unregistered files.
- Source manifests are external fingerprinted dependencies; changing a source
  artifact invalidates recursive report verification.
- Confirmatory report mode requires every source to record a clean Git commit.

The report is explicitly marked `development`; its generator proves traceable
extraction, not scientific validity of the tiny pilot metrics.

## 10. Artifact integrity

Independent SHA-256 verification was rerun against every artifact entry in:

| Run | Registered files | Digest failures |
|---|---:|---:|
| `ds-carla0916-town10-pilot-v001` | 5 | 0 |
| `framework-v0.2-rtdetr-smoke` | 4 | 0 |
| `framework-v0.2-rtdetr-smoke-analysis` | 6 | 0 |
| `scenario-plan-thesis-pilot-v1` | 4 | 0 |
| `eval-rtdetr-pretrained-pilot-v1` | 20 | 0 |
| `ds-carla0916-town10-pilot-v001-qa` | 5 | 0 |
| `rpt-framework-validation-20260726-v001` | 11 | 0 |
| **Total** | **55** | **0** |

Manifests record a dirty worktree and no commit because this repository has no
initial commit. These objects are valid development evidence but fail the
confirmatory publication gate.

## 11. Commands rerun for this report

```bash
uv run ruff format --check .
uv run ruff check .
uv run python -m compileall -q .
uv run python -m unittest discover -s tests -v

uv run carla-native-collect \
  --scenario-plan runs/scenario-plan-thesis-pilot-v1 \
  --dataset-id ds-validation-dry-run \
  --partition test_seen \
  --dry-run

uv run carla-vision --help
uv run carla-analyze-run --help
uv run carla-collect-dataset --help
uv run carla-audit-dataset --help
uv run carla-plan-scenarios --help
uv run carla-native-collect --help
uv run carla-train-detector --help
uv run carla-evaluate-detector --help
uv run carla-verify --help
uv run carla-package-model --help
uv run carla-build-report --help

uv run carla-verify \
  datasets/ds-carla0916-town10-pilot-v001 \
  runs/framework-v0.2-rtdetr-smoke \
  runs/framework-v0.2-rtdetr-smoke-analysis \
  runs/scenario-plan-thesis-pilot-v1 \
  runs/eval-rtdetr-pretrained-pilot-v1 \
  runs/ds-carla0916-town10-pilot-v001-qa \
  reports/rpt-framework-validation-20260726-v001 \
  --reject-unregistered
```

## 12. Remaining acceptance work

Priority order:

1. Create the first clean Git commit and freeze the environment/lockfile
   identity.
2. Prepare an isolated CARLA 0.9.16 native-PythonAPI execution host.
3. Run one short native episode, inspect actor placement, frame timing, labels,
   cleanup, and provenance, then scale to the 1,000-frame A2 pilot.
4. Complete the fixed-seed 100-frame human annotation audit and freeze a
   train/validation/test release.
5. Run tiny-subset overfit, interruption/resume, and one-epoch RT-DETR gates.
6. Train at least three registered RT-DETR seeds and a fair comparison model.
7. Evaluate once on locked, multi-episode partitions with non-degenerate
   episode bootstrap and target-hardware latency.
8. Add replay/live-shadow matrix automation, cross-seed statistical reporting,
   PDF/LaTeX rendering, registry discovery, and signed reproduction bundles.
9. Exercise the implemented model package path on real trained checkpoints and
   reproduce their canonical validation predictions.
10. Design temporal monocular perception and a separately audited vision-only
   policy before enabling any `vision` control mode.

Until those items pass, the defensible claim is that the framework plumbing
and development evidence are validated—not that a trained vision-only driver
or thesis benchmark has been completed.
