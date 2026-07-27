# CARLA Vision Framework Validation Report v0.4

Date: 2026-07-26  
Status: development validation  
Primary sensor boundary: front monocular RGB only  
Primary detector family: RT-DETR  
Comparison adapter exercised: YOLO26n  
CARLA target: 0.9.16 at `172.20.10.7:2000`

## 1. Executive conclusion

The framework now has an end-to-end, artifact-first research path for live
monocular perception, exact RGB/teacher dataset capture, deterministic scenario
planning, model-neutral training/evaluation, paired detector replay,
validation-only threshold selection, failure mining and human review, model
promotion, report generation, canonical provenance, and standalone source
reproduction bundles.

The complete offline suite contains 135 passing tests. A real development
reproduction bundle and a seven-source v0.4 report release both pass recursive
hash, source-reference, semantic, checksum-index, and unregistered-file
verification.

This does **not** establish a thesis benchmark or an autonomous driver. The
current empirical dataset contains three unassigned frames from one episode.
The detector weights are pretrained COCO weights, the workspace is dirty and
has no commit, the native official-PythonAPI collector has not run on a
compatible host, and vision-only control remains intentionally disabled.

## 2. Scope added after v0.3

### 2.1 Paired model replay

`carla_vision/replay/` now executes two or more model adapters over one
checksum-locked RGB order. Development configs may use loose RT-DETR, YOLO, or
custom PyTorch adapters; confirmatory replay requires verified model packages.

Each model receives an independently retained child evaluation. The parent
release stores:

- canonical sample order and SHA-256;
- child manifests and exact model lineage;
- aggregate and per-image metrics;
- paired TP/FP/FN and latency differences;
- episode-cluster bootstrap samples and intervals;
- disagreement tables;
- PNG and SVG accuracy, latency, outcome, and error-heatmap plots;
- a qualitative paired montage;
- a human-readable report and sorted checksum index.

The verifier reconstructs the sample order, model/sample table, paired rows,
bootstrap identities, source graph, and every child evaluation. Rehashed
semantic tampering is rejected.

### 2.2 Evaluation semantic verification

The common evaluator is no longer trusted only because its outer files hash.
`carla_vision/evaluation/verified.py` now:

- reopens the immutable dataset;
- reconstructs the exact selected image order;
- verifies the retained prediction floor and ontology;
- proves that flattened per-image predictions equal canonical COCO
  predictions;
- recomputes the selected operating-point TP, FP, FN, precision, recall, F1,
  and false-negative rate;
- checks summary and source lineage consistency.

Generic deep verification invokes this contract only when the complete
evaluation role set is present. A full-suite regression exposed that a minimal
unit-test report fixture also used the role name `evaluation_summary`; dispatch
was corrected to require all four semantic evaluation roles rather than one
ambiguous role.

### 2.3 Validation-only threshold selection

`carla_vision/thresholds/` consumes a verified evaluation whose partitions all
begin with `val`. Train, unassigned, and locked-test sources are rejected.
Supported preregistered objectives are maximum F1, minimum recall, minimum
precision, and weighted error.

The release retains:

- complete linear or logarithmic threshold sweep;
- deterministic 15-significant-digit thresholds;
- feasibility, tie-break, FP-cost, and FN-cost rules;
- sealed operating threshold and operating metrics;
- episode-bootstrap reruns of the complete selection procedure;
- infeasible replicate fraction;
- PNG/SVG metric, error, and threshold-stability plots;
- report, descriptor, source graph, and checksum index.

The grid cannot begin below the evaluator's retained-prediction floor. An
end-to-end test caught the floating-point boundary
`0.30000000000000004`; canonical threshold generation now prevents that
selection error.

No threshold artifact was produced from the retained real pilot. Its partition
is `unassigned`, so doing so would violate the validation-only contract.

### 2.4 Failure mining and human review

`carla_vision/failure_mining/` categorizes validation errors in a fixed,
mutually exclusive order:

1. correct-class IoU matches are removed;
2. wrong-class matches become misclassification failures;
3. same-class lower-IoU matches become localization failures;
4. unmatched ground truth becomes false negatives;
5. unmatched predictions become false positives.

All candidates are retained with stable failure IDs, full GT/prediction
context, lineage, and a deterministic priority. The review queue applies
image/episode diversity caps without discarding the complete candidate table.
It includes CSV/JSONL queues, an immutable blank review template, category and
priority plots, a montage, report, descriptor, and checksum index.

Review finalization requires every queued ID exactly once, preserves immutable
fields, validates reviewer identity and decision vocabulary, and requires a
remedy for confirmed failures or label issues. It archives the exact completed
CSV and emits normalized decisions, confirmed/label-issue catalogs, plots,
report, and checksum index. Reviewed validation images remain references; they
are never copied into training.

The synthetic test fixture exercises misclassification, localization, false
negative, and false positive paths plus semantic tampering. The real pilot has
no valid validation partition, so no real failure review was fabricated.

### 2.5 Canonical provenance

`carla_vision/reproducibility.py` implements:

- deterministic sorted-key UTF-8 JSON;
- exact resolved-configuration and output-location-independent identity
  digests;
- canonical `exp-<UTC>-<stage>-<model>-<cfg8>-s<seed>` IDs;
- dependency-lock SHA-256 records;
- privacy-safe CPU, memory, PyTorch, CUDA, cuDNN, MPS, ROCm, determinism,
  compiler, Git, uv, and container fields.

New tracker manifests embed this envelope. The generic verifier recomputes
configuration hashes and canonical experiment IDs and enforces the allow-listed
hardware schema. Legacy schema-1.0 objects remain valid with explicitly absent
new metadata; old provenance is not rewritten.

### 2.6 Standalone reproduction bundles

`carla_vision/reproduction/` builds a portable release after recursively
verifying its source objects. Its source ZIP uses fixed metadata, sorted paths,
normalized modes, no compression, and no symlinks. JSON and CSV inventories
carry a digest, byte size, and mode for every source file. The bundle also
contains source manifests, exact dependency locks, tracker environment and
hardware metadata, tokenized commands, an executable non-overwriting script,
README, descriptor, and checksum index.

The semantic verifier independently reconstructs the ZIP inventory, source
graph, lock copies, environment, command arrays, script bytes, descriptor
counts, and confirmatory gates. It does not need the original source
directories.

## 3. Retained empirical evidence

### 3.1 Live RT-DETR drive

The retained `framework-v0-rtdetr-live-drive` run used the privileged
simulator teacher only for motion. RT-DETR consumed front RGB frames. The
analysis contains:

- 74 inferred frames over 11.7999 simulator seconds;
- 6.1865 effective source FPS;
- 552 retained detections across nine COCO labels;
- median pipeline latency 215.902 ms;
- p95 pipeline latency 270.825 ms;
- estimated travel distance 16.387 m;
- one recorded stale-perception brake mode.

These measurements are a development demonstration on the recorded client,
not a target-hardware benchmark and not evidence of vision-only control.

### 3.2 Exact-frame dataset pilot

`ds-carla0916-town10-pilot-v001` retains three exact RGB/instance-frame pairs
and 27 visible-box annotations. Its only partition is `unassigned`. Dataset QA
reproduced all teacher-mask boxes and derived YOLO exports. The release is
suitable for pipeline verification only.

### 3.3 Deterministic scenario plan

`scenario-plan-thesis-pilot-v1` contains 23 deterministic episodes and 3,450
planned captures across the frozen scenario and group-safe split contracts.
This proves planning and recomputation, not native execution.

### 3.4 Pretrained RT-DETR evaluation

On the three pilot frames and 27 annotations:

| Metric | Value |
|---|---:|
| COCO AP@[.50:.95] | 0.272277 |
| AP@.50 | 0.333333 |
| AP@.75 | 0.333333 |
| TP / FP / FN at operating point | 3 / 0 / 24 |
| Precision | 1.000000 |
| Recall | 0.111111 |
| F1 | 0.200000 |
| Warm median inference latency | 317.744 ms |
| Cold first inference | 2504.852 ms |

The single episode makes the episode bootstrap degenerate. The small,
unassigned sample and pretrained weights prohibit accuracy or generalization
claims.

### 3.5 Paired RT-DETR and YOLO26 replay

Both models consumed three exact images in the order identified by:

```text
92ad61e34ee6199ab17435b4fe7ac4f10f7dcc0d812c7dd2acc5a66a0ddb80f6
```

| Model | AP@[.50:.95] | TP | FP | FN | Recall | F1 | Warm median |
|---|---:|---:|---:|---:|---:|---:|---:|
| RT-DETR-L COCO | 0.272277 | 3 | 0 | 24 | 0.111111 | 0.200000 | 555.664 ms |
| YOLO26n COCO | 0.000000 | 0 | 0 | 27 | 0.000000 | 0.000000 | 30.655 ms |

The replay records three outcome disagreements. YOLO26n's maximum confidence
on the first pilot view was approximately 0.0403, below the shared 0.05
retention floor. At lower diagnostic floors the adapter returns detections.
The zero row is therefore a threshold-calibration finding, not proof that the
adapter failed and not a fair model ranking.

## 4. Reproduction artifact

`bundle-framework-development-20260726-v001` passed both the dedicated and
generic verifier:

| Field | Retained value |
|---|---:|
| Verified source objects | 10 |
| Source files | 143 |
| Source bytes | 1,477,172 |
| Archived dependency locks | 1 |
| Tokenized commands | 4 |
| Registered artifacts | 22 |
| Checksum-index entries | 21 |
| External references needed to verify bundle | 0 |
| Unregistered files | 0 |

The bundle was also tested in a temporary standalone location after its
original source object was removed. Verification still passed.

The bundle snapshot was generated before the final v0.4 documentation and
report edits. It remains immutable development evidence. A later clean bundle
must receive a new ID rather than overwriting it.

## 5. Manifest-generated report release

`rpt-framework-validation-20260726-v002` aggregates seven verified sources:
dataset, QA, scenario plan, live analysis, RT-DETR evaluation, paired replay,
and reproduction bundle. It contains:

- 65 canonical metric rows;
- 82 source-artifact inventory rows;
- four PNG/SVG plots;
- Markdown and JSON reports;
- three canonical CSV tables;
- a ten-entry payload checksum index;
- 11 registered artifacts;
- zero unregistered files.

Report values are extracted from machine-readable sources. The report builder
now understands paired replay, threshold selection, failure mining/review,
reproduction bundle, and prior report source types.

## 6. Verification results

The exact offline command:

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
```

completed:

```text
Ran 135 tests in 12.040s
OK
```

The suite covers runtime contracts, adapters, display/frame correctness,
latest-frame dropping, safety separation, artifact manifests, analysis,
dataset synchronization/labels/QA, scenario planning, fake-CARLA native
orchestration, training, evaluation, replay, validation-only threshold
selection, failure mining/review, model packaging, reporting, canonical
provenance, reproduction bundles, shallow and deep tamper cases.

Unique retained research objects reverified in this phase:

| Object | Registered artifacts | Integrity failures |
|---|---:|---:|
| Dataset pilot | 5 | 0 |
| Dataset QA | 5 | 0 |
| Scenario plan | 4 | 0 |
| Live RT-DETR drive | 4 | 0 |
| Live drive analysis | 6 | 0 |
| RT-DETR pilot evaluation | 20 | 0 |
| RT-DETR replay child | 20 | 0 |
| YOLO26 replay child | 20 | 0 |
| Paired replay parent | 20 | 0 |
| Reproduction bundle v001 | 22 | 0 |
| Validation report v002 | 11 | 0 |
| **Total** | **137** | **0** |

The environment lock was checked with `uv lock --check`; Python compilation
and Ruff checks passed for the new code before artifact generation. A final
repository-wide format/lint/compile pass is rerun after documentation changes.

## 7. Claim boundary

The defensible claim is:

> The development framework can collect, track, compare, analyze, report, and
> independently verify monocular-RGB CARLA perception research artifacts with
> explicit privileged-data boundaries and portable source provenance.

The evidence does not support these claims:

- a trained CARLA-specific RT-DETR model exists;
- model performance generalizes;
- YOLO26 is inferior to RT-DETR;
- threshold selection is statistically stable;
- the native dataset factory has executed at thesis scale;
- a vision-only controller drives safely;
- CARLA results transfer to public-road safety.

## 8. Remaining acceptance sequence

1. Create and tag the first clean Git baseline.
2. Build a new confirmatory reproduction bundle from that clean commit.
3. Install the exact official CARLA 0.9.16 PythonAPI on an isolated native
   execution host.
4. Run one short synchronous episode and audit timing, placement, labels,
   cleanup, privileged lanes, and source provenance.
5. Scale to the preregistered dataset/QA gate and freeze episode-grouped
   train/validation/locked-test releases.
6. Run tiny-overfit, interruption/resume, and one-epoch training gates.
7. Train at least three RT-DETR seeds and a fair model comparison under equal
   data and compute rules.
8. Select per-model thresholds on validation only, complete blinded failure
   review, then freeze thresholds.
9. Evaluate once on the locked multi-episode test set with episode bootstrap
   and target-hardware latency.
10. Add live-shadow matrix automation and policy-input auditing.
11. Design temporal monocular perception and a separately specified
   vision-only policy before enabling any closed-loop vision control.
12. Add repository registry/signoff and PDF/LaTeX thesis rendering.

Until these gates pass, development artifacts must retain their limitations
and dirty-Git evidence exactly as recorded.
