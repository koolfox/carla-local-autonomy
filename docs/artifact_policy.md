# Artifact and Provenance Policy

Status: active thesis policy; implemented development evidence and remaining
release requirements are distinguished explicitly

## 1. Purpose

Every scientific claim must be traceable to stable, integrity-checked evidence.
For this project, evidence includes code, configurations, manifests, datasets,
annotations, model weights, predictions, logs, plots, tables, images, videos,
statistical outputs, and reports.

This policy has five invariants:

1. **Identity:** every research object has a globally unambiguous ID.
2. **Provenance:** every derived object points to its inputs and producing
   experiment.
3. **Integrity:** every retained file has a SHA-256 digest and byte size.
4. **Immutability:** finalized objects are never edited in place.
5. **Reconstructability:** plots, tables, videos, and claims can be regenerated
   from machine-readable inputs and a recorded environment.

## 2. Current implementation

`carla_vision.artifacts.RunArtifactTracker` currently provides:

- one unique `runs/<run_id>/` directory;
- manifest schema version `1.0`;
- `created`, `running`, `success`, and `failed` statuses;
- UTC created, started, updated, and finished timestamps;
- Git availability, commit, and dirty flag;
- sanitized CLI arguments and configuration;
- an allow-listed Python/platform/package environment snapshot;
- CARLA endpoint, version, and map;
- model and dataset references;
- SHA-256, byte size, MIME type, role, timestamp, and metadata for each
  registered file;
- atomic manifest replacement;
- path containment within the run directory;
- secret redaction for common credential fields and free text;
- retained failure type, module, and redacted message.

The current live runtime normally produces:

```text
runs/<run_id>/
├── manifest.json
├── summary.json
├── images/
│   └── latest_overlay.png
├── logs/
│   └── detections.jsonl
└── video/
    └── overlay.mp4
```

These files are registered with hashes after writing. The manifest does not
hash itself, avoiding a self-referential digest.

Additional implemented producers and consumers:

- scenario planning registers the frozen suite, split plan, episode JSONL, and
  summary, and later verifies their hashes plus deterministic recomputation;
- dataset releases carry complete checksum indexes, auxiliary provenance
  artifacts, and a strict consumer-side verifier for counts, references,
  ontology, episode splits, and exact cross-partition RGB duplicates;
- training registers the source configuration, resolved configuration, seed
  schedule, derived data file, complete backend output tree, log, summary, and
  best/last checkpoint digests;
- evaluation registers configuration, canonical predictions, metric
  JSON/CSV, bootstrap data, latency, calibration/confusion/strata sources,
  PNG/SVG plots, montage, and summary;
- paired replay registers the exact RGB order and digest, every child
  evaluation manifest, aggregate/per-image/disagreement tables, paired
  episode-bootstrap deltas, PNG/SVG plots, a qualitative panel, a
  human-readable report, and a sorted checksum index;
- threshold selection registers the complete validation sweep, selected
  operating point, episode-bootstrap stability, plots, report, descriptor, and
  checksum index;
- failure mining retains all candidates plus the selected review queue,
  original review template, plots/montage, and source evaluation/threshold
  lineage; review finalization archives the completed human CSV and writes
  normalized, confirmed, rejected/issue summary artifacts as a new object;
- reproduction bundles retain a deterministic source ZIP, JSON/CSV file
  inventory, verified source manifests, exact lockfiles, environment/hardware
  envelope, tokenized commands, executable script, descriptor, and checksum
  index without depending on the original source paths;
- run analysis and dataset QA consume source objects read-only and record
  lineage.
- native preflight consumes a verified scenario plan, performs only
  read-only endpoint/version/map/world checks, records automated and manual
  readiness gates, and retains JSON/CSV/Markdown evidence with
  `simulator_mutated: false`;
- native-host kits retain their normalized configuration, deterministic ZIP,
  JSON/CSV payload inventories, exact Windows/Linux hashed requirements,
  operator README, outer checksum index, and a descriptor proving the build
  neither contacted nor mutated CARLA;
- operator UI launches retain the normalized request, exact tokenized command,
  terminal status, stdout/stderr, and tracker manifest under
  `operator_sessions/`; successful child manifests are fingerprinted as input
  references.
- evidence registries scan immediate children of all seven canonical roots,
  attempt strict deep verification for every discovered object, retain failed
  objects and normalized errors, and seal machine-readable tables, plots,
  Markdown, checksums, and source-manifest fingerprints.

Current limitations:

- a local resolved model weight is now SHA-256 fingerprinted, but remote model
  registry identities and signed release manifests are not implemented;
- pilot dataset releases now include per-sample hashes and a complete
  `checksums.sha256` index, but canonical tree-derived dataset IDs and signed
  immutable publication releases are not implemented;
- new manifests include CPU, memory, PyTorch/CUDA/cuDNN/MPS/ROCm,
  deterministic-backend, compiler/tool, container, canonical configuration,
  and dependency-lock metadata; exact accelerator driver and container image
  digests remain unavailable unless supplied by the execution host;
- manifests do not yet have a structured seed/determinism block;
- standalone reproduction bundles now retain deterministic source archives,
  per-file inventories, source manifests, lockfiles, environment envelopes,
  tokenized commands, and checksum indexes; signed archives are not
  implemented;
- run-analysis, dataset-QA, training-run, offline-evaluation, paired replay,
  immutable model package, and manifest-driven report artifacts are
  implemented; no real
  trained checkpoint has yet passed promotion, and cross-seed inferential
  comparisons plus PDF/LaTeX thesis rendering remain;
- finalized files are protected by convention, not operating-system or object
  store immutability;
- a repository-wide, point-in-time evidence registry and a manifest-driven
  browser are implemented; there is no mutable database, signed registry,
  global supersession policy, or cross-registry graph/cycle service;
- `output/` contains legacy development files without the current run-manifest
  guarantees.

The currently recorded framework smoke runs have `dirty: true` and no Git
commit in their manifests. They are useful development evidence but are not
publishable thesis experiments.

## 3. Canonical storage layout

The canonical roots are:

```text
runs/<experiment_id>/       one execution or analysis
datasets/<dataset_id>/      immutable released datasets
models/<model_id>/          immutable model/checkpoint packages
reports/<report_id>/        thesis reports
bundles/<bundle_id>/        standalone reproduction bundles
native_kits/<kit_id>/        portable native-host packages
operator_sessions/<job_id>/ operator request, command, status, and logs
```

All seven roots now have writer or builder support. Operator sessions are
operational provenance wrappers rather than replacements for the child
research object. The current retained report
and reproduction bundle are development-only, and model packaging is covered
by end-to-end tests but awaits a real training checkpoint. Source code and
small schemas belong in Git. Large immutable artifacts may be
stored outside Git, but their canonical URI, digest, byte size, and retrieval
instructions must be in the manifest. A local path alone is not sufficient
provenance.

Temporary files belong under a run-local `tmp/` or an operating-system
temporary directory and are excluded from the final manifest. Any temporary
file needed to support a result must be promoted to a named artifact before
finalization.

The legacy `output/` directory is for development only. No thesis figure,
metric, or model may cite it.

## 4. Identifier policy

IDs contain lowercase ASCII letters, digits, and hyphens. They are never
reused, including after a failed run.

### 4.1 Experiment

```text
exp-<YYYYMMDDTHHMMSSZ>-<stage>-<model>-<cfg8>-s<seed>
```

Example:

```text
exp-20260726T171722Z-live-rtdetr-l-4fa19c2e-s42
```

The canonical configuration hash and seed rules are defined in
[`experiment_protocol.md`](experiment_protocol.md).

### 4.2 Scenario and episode

```text
scn-<map-family>-<recipe>-s<scenario-seed>
ep-<scenario-id>-r<replicate>
```

Examples:

```text
scn-town10hd-night-rain-crossing-s81721
ep-scn-town10hd-night-rain-crossing-s81721-r03
```

### 4.3 Dataset

```text
ds-carla0916-<ontology>-v<NNN>-<tree8>
```

Example:

```text
ds-carla0916-road-v001-91c8ab27
```

`tree8` is derived from the finalized release checksum index. A change to any
image, annotation, split, ontology, or metadata file creates a new dataset ID.

### 4.4 Model

```text
mdl-<architecture>-<dataset8>-<cfg8>-s<seed>-<checkpoint>
```

Example:

```text
mdl-rtdetr-l-91c8ab27-4fa19c2e-s42-best
```

`best` is meaningful only when the selection metric and epoch are written in
the model manifest. Exported ONNX or TensorRT engines receive distinct model
IDs and point to the source checkpoint.

### 4.5 Report, figure, and table

```text
rpt-<topic>-<YYYYMMDD>-v<NNN>
fig-<chapter-or-study>-<slug>
tbl-<chapter-or-study>-<slug>
```

Figure and table names are stable logical names inside a versioned report.
Rendered formats and machine-readable source data share the same stem.

## 5. File naming

- Use lowercase `kebab-case` for stable artifact file names.
- Use UTC timestamps only when time is part of the scientific identity.
- Include units in column names, for example `latency_ms` and `speed_mps`.
- Include schema versions inside data, not only in file names.
- Do not encode mutable words such as `new`, `final-final`, or `best2`.
- `latest` may be a disposable user-interface alias, never a cited canonical
  identity.
- Use zero-padded frame or sequence numbers when a directory requires lexical
  ordering.
- Store paths in manifests as forward-slash relative paths or canonical URIs.

The current run-local `images/latest_overlay.png` is an end-of-run snapshot
whose identity is disambiguated by the enclosing run ID and manifest. It must
not become a global archival alias.

## 6. Manifest requirements

### 6.1 Common envelope

Every dataset, model, experiment, and report manifest must contain:

```text
schema_name
schema_version
object_type
object_id
status
created_at_utc
started_at_utc
finished_at_utc
producers
description
license_and_provenance
inputs
outputs
artifacts
```

Every artifact entry contains:

```text
relative_path_or_uri
role
media_type
sha256
size_bytes
created_at_utc
schema_name
schema_version
metadata
```

All timestamps use ISO 8601 UTC with a trailing `Z`.

### 6.2 Experiment manifest

The thesis experiment manifest extends the current run manifest with:

- canonical experiment ID and configuration digest;
- research question, hypothesis, stage, and repetition;
- source repository URI, commit, branch/tag, dirty flag, and source archive or
  patch when necessary;
- exact command and fully resolved configuration;
- lockfile digest and installed package versions;
- host OS, CPU, memory, GPU/accelerator, driver, CUDA/cuDNN/MPS, compiler, and
  container image digest;
- CARLA client and server versions, build IDs, map package, OpenDRIVE digest,
  endpoint role, fixed delta, synchronous-mode state, and rendering settings;
- runtime camera/sensor contract;
- permitted privileged lanes and a policy-input audit result;
- master seed, derivation version, all derived seeds, and determinism level;
- immutable dataset, ontology, model, and scenario-suite references with
  digests;
- preprocessing and postprocessing identity;
- resource budget and measured resource use;
- structured metric summaries and confidence intervals;
- deviations from the pre-registration record;
- status, failure, and finalization errors.

The current manifest already covers a subset of these fields. Missing fields
must not be inferred from filenames or memory.

### 6.3 Dataset manifest

The release-grade dataset manifest contains:

- dataset ID, parent dataset ID, and release state;
- CARLA build and asset/map inventory;
- ontology ID and class definitions;
- generation-code experiment/commit;
- split-plan ID and exact group assignments;
- scenario, episode, frame, and annotation counts;
- camera and teacher sensor definitions;
- privileged-data retention policy;
- capture frequency and filtering rules;
- class/size/distance/occlusion/map/weather counts by split;
- exact-duplicate and near-duplicate audit results;
- human QA sample seed, rubric, reviewers, and discrepancies;
- licenses and upstream asset/source provenance;
- release checksum-index digest.

### 6.4 Model manifest

The release-grade model manifest contains:

- model ID, architecture, backend adapter, and upstream source;
- source and canonical class maps;
- initialization-weight ID and digest;
- producing experiment and dataset IDs;
- input color order, size, resize/letterbox, and normalization;
- output and postprocessing contract;
- checkpoint epoch/step, selection metric, and selection split;
- raw and exponential-moving-average weights when applicable;
- parameter count, serialized format, ops estimate if available;
- validation metrics, limitations, and intended use;
- framework and export versions;
- weight and export-file digests;
- license and security warning for executable/pickle-based formats.

### 6.5 Report manifest

The release-grade report manifest contains:

- report ID, title, authors, version, and status;
- included hypotheses and experiment IDs;
- exact figure/table IDs;
- source-data and generation-code references;
- document source, rendered PDF, supplementary material, and checksums;
- thesis chapter/section mapping;
- deviations and known limitations.

## 7. Integrity and checksums

SHA-256 is the mandatory integrity algorithm.

### 7.1 Individual files

Hash a file only after its writer is closed and flushed. Record:

- digest over exact bytes;
- byte size from the same stable version;
- media type;
- relative path or canonical URI.

The current tracker checks size and modification time before and after hashing
and fails if the file changes during the operation.

### 7.2 Directory releases

A finalized dataset, model, or report release contains
`checksums.sha256`:

```text
<64 lowercase hex characters><two spaces><POSIX relative path><LF>
```

Entries are sorted by UTF-8 path bytes. The checksum index excludes itself but
includes the release manifest and every retained payload file. The release
digest is SHA-256 over the exact `checksums.sha256` bytes and is recorded by
every consuming object's reference or the external artifact registry.

Files with unstable metadata containers must still hash their exact archived
bytes. Re-encoding a PNG, video, Parquet file, or checkpoint creates a new
digest even if its semantic content is intended to be equivalent.

### 7.3 Verification

`carla-verify` currently supports:

- one artifact;
- one run;
- one release tree and its checksum index;
- fingerprinted external references, including nested relative references;
- semantic dataset, scenario-plan, model, and report verification;
- recursive report source verification;
- semantic evidence-registry reconstruction against the recorded source
  manifest set, while ignoring objects created after the snapshot;
- rejection of unregistered files and a clean-Git confirmatory gate.

Verification fails on a missing file, unexpected file in a sealed release,
size mismatch, digest mismatch, schema mismatch, unresolved reference, or
unsafe/symlinked path. Evidence-registry objects are excluded from scans to
avoid registry-of-registry cycles. Broader cross-object cycle analysis and
signed registry publication remain future work.

No training, evaluation, or report-generation job may consume an immutable
input until verification passes. Training and evaluation already invoke the
strict dataset verifier; generic run/model/report verification remains.

## 8. Required artifacts by stage

### 8.1 Dataset generation

```text
datasets/<dataset_id>/
├── dataset-manifest.json
├── checksums.sha256
├── ontology.yaml
├── split-plan.yaml
├── annotations/
│   ├── instances-train.json
│   ├── instances-val-seen.json
│   ├── instances-val-map-ood.json
│   ├── instances-test-seen.json
│   └── instances-test-map-ood.json
├── metadata/
│   ├── scenarios.jsonl
│   ├── episodes.jsonl
│   └── frames.parquet
├── images/
│   └── <split>/<episode-id>/<frame>.png
└── qa/
    ├── validation-report.json
    ├── coverage.csv
    ├── class-distribution.csv
    ├── split-audit.json
    ├── duplicate-audit.json
    ├── reviewed-sample.jsonl
    ├── contact-sheets/
    └── sample-video.mp4
```

Teacher masks/depth, if retained, are stored in a clearly named restricted
subtree with their own retention rule. They are never mixed into `images/`.

### 8.2 Training

Each training run must retain:

- pre-registration and resolved configuration;
- stdout/stderr or structured event log;
- environment/hardware record;
- initialization model reference and digest;
- data-order/seed record;
- epoch/step history in CSV or JSON;
- best and last checkpoints;
- optimizer/scheduler/scaler state required for resume;
- canonical validation predictions;
- loss and metric curves with source CSV;
- precision-recall curves;
- confusion matrix and normalized confusion matrix;
- representative prediction panels selected by a fixed seed/rule;
- summary and run manifest;
- failure/deviation record.

TensorBoard, Weights & Biases, or another tracker may be used as a convenience,
but it cannot be the sole copy of any required artifact.

### 8.3 Offline evaluation

Required artifacts:

- evaluation configuration and frozen thresholds;
- model and dataset references;
- canonical per-image predictions;
- aggregate metrics JSON;
- per-class and per-stratum CSV;
- bootstrap samples or sufficient statistics plus seed;
- calibration and confusion data;
- latency and resource measurements;
- plots in vector and raster forms;
- source CSV/JSON for every plot;
- qualitative success and failure panels with selection rule;
- evaluation summary and manifest.

### 8.4 Replay and live runs

Required artifacts:

- run manifest and summary;
- scenario/replay input reference;
- camera and model configuration;
- frame-level detections or policy outputs;
- frame/sequence/time alignment log;
- camera, inference, frame-age, and queue-drop metrics;
- annotated video;
- raw video or immutable replay reference when required;
- latest/end snapshot for convenience;
- control and safety logs when motion is enabled;
- privileged evaluation log in a separate named stream;
- final stopped-state evidence for controlled runs.

The current runtime provides a subset: manifest, summary, detections JSONL,
annotated video, and synchronized final overlay.

### 8.5 Failure-mining cycle

Retain:

- ranked candidate table;
- immutable source frame/clip references;
- reviewed failure records;
- reviewer decisions;
- diversity/de-duplication report;
- proposed and accepted scenario recipes;
- parent and new dataset IDs;
- regression comparison and plots.

### 8.6 Reports and thesis figures

For each figure:

```text
fig-<slug>.pdf or .svg    primary vector render where appropriate
fig-<slug>.png            review/presentation render
fig-<slug>.csv or .json   exact source data
fig-<slug>.yaml           generation options and source experiment IDs
```

For each table:

```text
tbl-<slug>.csv            canonical values
tbl-<slug>.tex            generated thesis form when applicable
tbl-<slug>.md             review form when useful
tbl-<slug>.yaml           source experiment IDs and formatting rules
```

For each video:

- MP4 with declared codec, dimensions, and intended playback FPS;
- frame-index mapping to source CARLA frame/sequence IDs;
- model, dataset, experiment, and scenario IDs;
- disclosure of dropped, duplicated, or time-scaled frames;
- no stale detections painted on newer images;
- optional subtitle or JSON sidecar for metrics and events.

Aesthetically selected qualitative examples must be labeled illustrative.
Evidence panels must use a registered sampling or ranking rule.

## 9. Provenance graph

Every artifact forms part of a directed acyclic graph:

```text
source commit + CARLA build + scenario recipes
                    |
                    v
                 dataset
                    |
        pretrained model + train config
                    |
                    v
               trained model
                    |
       test dataset + evaluation config
                    |
                    v
          predictions and metrics
                    |
                    v
          plots, tables, and report
```

References use object ID plus release digest. A path is a retrieval hint, not
identity. The report builder must traverse the graph and fail if any referenced
object is missing, failed, mutable, or checksum-invalid.

## 10. Immutability and versioning

- A finalized object is read-only.
- A correction creates a new ID and points to `supersedes`.
- Failed runs remain retained with their original IDs.
- Model weights are never overwritten under the same model ID.
- Dataset images or labels are never edited in a released directory.
- Derived exports point to their canonical source release.
- `best` and `last` are checkpoint roles, not mutable filenames shared across
  experiments.
- Schema changes follow semantic versioning. Readers reject an unsupported
  major version.
- A published report pins exact object IDs and digests, never a branch,
  `latest`, or mutable URL.

Exploratory storage may be cleaned after promotion, but no file referenced by
a manifest may be deleted without first creating a formally marked tombstone
and demonstrating that the retention policy permits it.

## 11. Retention and backup

### Permanent through thesis defense and archival submission

- released datasets and their manifests;
- checkpoints used in tables or figures;
- canonical predictions and metric source data;
- all thesis plots, tables, videos, and report sources;
- successful and failed confirmatory experiment manifests;
- source commits, lockfiles, and reproduction instructions.

### Retain at least until the corresponding study is frozen

- optimizer and scheduler states;
- detailed training event logs;
- non-selected checkpoints;
- teacher masks/depth retained for QA;
- failure-mining candidates.

### Disposable after verified promotion

- caches;
- package downloads;
- temporary extracted frames;
- duplicated local convenience copies;
- aborted exploratory outputs with no registered scientific reference, subject
  to the project's audit policy.

Use at least two independently administered copies for irreplaceable releases,
with one copy off the primary machine. Periodically verify stored digests and
record the verification date and result.

## 12. Security, privacy, and licensing

- Never serialize the complete process environment.
- Redact password, token, secret, authorization, API-key, credential, and
  credential-bearing URL values.
- Store external-service credentials only in the runtime secret mechanism, not
  configs, logs, reports, or manifests.
- Treat PyTorch pickle-based checkpoints as executable/untrusted input; retain
  source and hash and load only from trusted provenance.
- Record CARLA, model, code, asset, and external dataset licenses.
- Synthetic CARLA images normally contain no personal data, but any future
  real-world images require a separate privacy, consent, and retention review.

The current tracker already applies an allow-list to environment metadata and
redacts common secret forms. This is necessary but does not replace log review
before publication.

## 13. Acceptance gates

### Artifact gate for any successful run

- manifest status is `success`;
- every required artifact exists;
- every registered size and SHA-256 verifies;
- no file changed during hashing;
- all input references resolve and verify;
- configuration and command are present;
- failure and finalization-error fields are empty;
- artifact roles are unique or deliberately repeatable;
- secrets scan reports no finding.

### Additional gate for confirmatory thesis experiments

- Git commit exists and the recorded worktree is clean;
- lockfile and source revision are archived;
- CARLA server/client/build and map identities are exact;
- dataset and initial/final model digests exist;
- all seeds and determinism settings exist;
- hardware/accelerator metadata is complete;
- pre-registration exists and deviations are documented;
- metrics have machine-readable source data;
- required plots, tables, and qualitative evidence are generated by code;
- statistical outputs identify the independent sampling unit;
- a clean environment passes the reproduction check.

### Dataset release gate

- release manifest and checksum index verify;
- ontology and split plan are immutable;
- frame, annotation, class, and split counts reconcile;
- exact/near-duplicate and leakage audits pass;
- human QA record meets the threshold in `experiment_protocol.md`;
- teacher-data boundary is documented;
- a deterministic sample viewer reproduces the QA selection.

### Model release gate

- checkpoint and every export verify;
- training experiment and dataset references resolve;
- class map and preprocessing are explicit;
- saved weights reproduce canonical validation predictions within declared
  tolerance;
- model card states intended use and limitations;
- license and executable-format warning are present.

### Report release gate

- every result points to successful verified experiment IDs;
- every figure/table has source data and a generation recipe;
- numbers in prose match canonical tables;
- all cited datasets/models use immutable IDs and digests;
- PDF and source bundle verify;
- limitations and failed confirmatory runs are not omitted.

Current development runs do not pass the confirmatory gate because their
manifests record a dirty worktree and no commit. This is an expected early
framework state, not a reason to rewrite their provenance.

## 14. Implementation priorities

1. Create the first clean tagged baseline and issue a confirmatory
   reproduction bundle using the implemented canonical identity, hardware, and
   lockfile envelope.
2. Extend manifests with structured sensor/privileged lanes, exact accelerator
   driver/container digests, seed derivation blocks, model/dataset identities,
   and structured metric summaries.
3. Complete immutable dataset promotion and release-signoff semantics around
   the existing checksum-indexed writer/verifier.
4. Exercise the implemented model packager on a real trained checkpoint and
   reproduce canonical validation predictions.
5. Extend the implemented report release with cross-seed statistics and
   PDF/LaTeX thesis output.
6. Extend the implemented evidence registry with cross-object cycle audit,
   signatures, supersession metadata, and independent release signoff.
