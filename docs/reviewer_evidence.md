# Reviewer evidence kit: executable runbook

This is an offline **software and evidence-completeness** workflow. It is not a
certificate of model quality, novelty, real-CARLA behavior, rights, or Q1 acceptance.
The active delivery status is in [#142](https://github.com/koolfox/carla-local-autonomy/issues/142)
and its parent [#140](https://github.com/koolfox/carla-local-autonomy/issues/140).
See [ADR 0002](adr/0002-reviewer-evidence-and-shard-contracts.md), the
[paper evidence checklist](paper/reviewer_requirements.md), and [method text](paper/methods_update.md).

## 1. Reproduce software checks, with no running CARLA

Use a full checkout for the complete suite; a source-only archive deliberately
omits historical model/recovery assets. The CARLA Python SDK may be installed by
`--all-extras`; no simulator server is started or contacted by these tests.

```bash
uv sync --all-extras --all-groups --frozen
uv run ruff check carla_vision tests restore_project.py
uv run pytest -q tests/test_reviewer_evidence.py tests/test_reviewer_storage.py tests/test_reviewer_workflows.py tests/test_reviewer_operator.py tests/test_reviewer_voxel_training.py
uv run pytest -q --junitxml=reviewer-tests.xml
```

Keep test output outside tracked source. Frontend changes additionally use the
existing `npm ci`, `npm run check`, `npm test`, `npm run build`, and current-bundle
checks. The TypeScript mirror exposes the six research job kinds; no alternate
front end or new network endpoint is introduced.

The test fixtures include deliberately false metadata, exact small geometric
answers, known k-center selections, duplicate COCO identifiers, real tiny CPU
optimization in the previous integrity suite, a real JobManager subprocess,
source-archive races, and metric replay after deleting its original workspace.
These are artificial inputs for software assertions, not publishable ADAS scores.

## 2. Freeze a source snapshot and register reviewer evidence

```bash
uv run carla-source-snapshot --repository . --commit HEAD --output ../source-review.zip
uv run carla-reviewer-evidence init --output ../submission.json
uv run carla-reviewer-evidence check ../submission.json
```

The last command intentionally returns 1 for a fresh template. It cannot become
ready by passing an empty custom requirement list. Fill `source_commit` with the
actual selected full Git SHA; retain the generated archive fingerprint and Git
object proof. `SOURCE_COMMIT.txt` alone is not source authenticity. Templates under
`configs/reviewer/` contain null/false values until actual artifacts exist.

The fixed roles are source snapshot, dependency lock, JUnit, registered study
protocol, geometric audit, independent label audit, result matrix, actual costs,
claim registry, rights review, dataset card, model card, environment, baseline
registry, manuscript inventory and independent reproduction. `offline_fixture`
mode checks the software-package subset and **never** returns empirical readiness.
`empirical_simulation` and `empirical_real_world` require every role; the latter
also needs a real-world claim. This catalog is a project review protocol, not a
universal rule imposed by all journals. Human scientific review remains necessary.

Every file reference is `{path, sha256, size_bytes}` with actual values produced
by `carla_vision.artifacts.fingerprint_file`. Relative paths are interpreted at the
referencing artifact; traversal and symlinks are rejected. Unknown schema fields,
invalid numeric values and duplicate JSON keys are errors, not ignored metadata.
Do not use an in-place rehash utility to conceal modifications to evidence.

## 3. Verify observed route independence

New native collection records an OpenDRIVE description using the existing read-only
map API. It records the returned map name/hash in sample context; unsupported
legacy APIs report that map evidence is unavailable, not a made-up map hash.

```bash
uv run carla-audit-trajectories --dataset datasets/release-train --dataset datasets/release-test --map-document datasets/release-train/maps/episode-a.opendrive.json --clearance-m 2 --maximum-step-m 5 --runs-root runs --run-id observed-corridors
```

Paths and numeric settings above illustrate the command shape; use actual release
paths and **preregistered** clearance/gap values, not these example numbers as a
study requirement. The command verifies the datasets, reconstructs trajectories
across canonical shards, rejects duplicate release identities, and evaluates
segment-interior distances in 3D. A transitive connected component may not span
partitions. Overpasses can remain distinct according to measured vertical distance.
The reviewer checker recomputes this audit from original captures rather than
trusting supplied route arrays. Empty, stationary or undersampled evidence cannot
establish unseen-route claims. A Town-name change alone is not independence.

## Voxel compatibility and model selection

The experimental voxel trainers preserve source partitions, require nonempty
validation for normal training and do not automatically open a test loader. Their
post-training flow check reloads the selected best checkpoint and evaluates only
validation. The explicit `--allow-training-selection` option is development-only
and labels its score accordingly; `--allow-legacy-splits` does not authorize it
implicitly. Nested dataset paths now serialize in the recorded configuration and
nonfinite settings fail before model execution. These repairs do not promote the
voxel model or teacher to a validated empirical driving baseline.

## 4. Bounded dataset generation and consumption

Native worker and BehaviorAgent teacher commands accept `--shard-samples N`.
Single-release output remains the default. N bounds samples per active canonical
writer; each finalized shard has its normal manifests and semantic verification.

```bash
uv run carla-verify-shards datasets/my-sharded-release
uv run carla-verify-shards datasets/incomplete-release --recover-inventory
```

Recovery is a read-only inventory and returns nonzero: it does not repair, delete,
overwrite, or promote incomplete evidence. Explicitly start a fresh release after
investigating the crash. Across-shard identities and sample order are checked before
training roots are returned. Both detector training and imitation consume verified
shards; data YAML references the actual shard image directories without copying RGB.

For evaluations produced independently per shard with the same model/policy:

```bash
uv run carla-pool-evaluations --evaluation runs/eval-shard-0 --evaluation runs/eval-shard-1 --runs-root runs --run-id pooled-evaluation
```

This recomputes pooled AP from remapped source annotations/predictions and rejects
duplicate source samples or changed model/ontology/policy. It is not an average
of shard AP. Frame/group identities remain in the output. The evaluator is
memory-resident with an explicit image cap; sharding does not make COCO unlimited.

## 5. Supervision, diagnosis, baselines and budgets

`carla-score-diagnosis --observations ... --reference ...` first predicts actions
from allowed signals; only then does it read held-out fault/reference labels.
The oracle outcome is not an acquisition feature. The output retains confusion
matrices, clean-hard behavior, unknown decisions and all case/group counts.
`tests/test_reviewer_workflows.py` includes an executable fixture in which changing
only truth labels leaves predictions unchanged. Independent human review still has
to establish whether real captures/labels/actions are correct.

`carla-select-coreset --protocol ...` implements **Sener & Savarese, Algorithm 1,
greedy k-center** on frozen, equal-cost feature rows, with a disclosed initial
labeled set and deterministic tie order. It does not implement their robust
mixed-integer solver or noisy-oracle detection. The seven original internal
heuristics are not relabeled QActor, NAS or ADAA. For a novelty claim, implement and
validate the closest task-matched baseline, disclose adaptation and hyperparameter
search budgets, and charge embeddings, probes and review costs to every method.
The feature pool records labels/partitions and encoder hashes but does not by
itself prove the encoder was never fitted on held-out data.

`RunArtifactTracker.resource_usage` measures elapsed wall time, current-process CPU
and registered output bytes on success/failure. GPU time is null if not measured;
child processes and overlapping nested scopes are explicitly not additive. The
all-attempt `CostLedger` requires known units and retains rejected/failed/cancelled
work. Complete external-stage instrumentation and audit are still needed to claim
end-to-end monetary, energy or GPU-hour savings.

## 6. Bind real results to runs and compute manuscript tables

The results role contains `protocol` (a fingerprinted matrix specification),
`rows`, `synthetic`, and `producers`. The matrix specifies methods, paired seeds,
one holdout digest, the common budget and explicit resource conversion weights.
Each row has method, seed, unique run ID, outcome, actual cost, holdout digest and
metric. Each producer binds the real manifest, registered summary, metric field
and cost ledger. The producing manifest's `invocation.config.review_binding` must
match matrix SHA, method, seed, holdout and ledger SHA. Empirical mode additionally
requires non-synthetic status and the selected clean Git source. Register these
bindings in the experimental producer; do not edit a completed manifest afterwards.

The complete synthetic construction is `bound_result_matrix` and
`test_seed_analysis_emits_recomputed_table_from_bound_producers` in the workflow
tests. It is a schema example, not an adapter claiming that older runs have these
bindings. Failed/missing cells make a matrix incomplete, and their cost is retained.

```bash
uv run carla-analyze-study-seeds --results results/registered-matrix.json --baseline random --source-commit FULL_ACTUAL_GIT_SHA --runs-root runs --run-id paired-analysis
```

The generated CSV/JSON are recomputed from registered results, not hand-entered
summary numbers. This CLI conditions on one generation condition and fixed holdout.
Use `paired_crossed_analysis` for a complete generation-seed by training-seed
factorial design; it resamples factor levels separately instead of treating every
cell as independent. Group-bootstrap COCO intervals condition on fixed weights.
These uncertainties answer different questions and must not be conflated.

`plan_paired_sample_size` accepts independent pilot pairs only, not the final test,
and reports its normal-approximation/Bonferroni assumptions. `holm_adjust` controls
a declared multiple-testing family when valid p-values have been computed. A
bootstrap interval or small-N seed average is not an automatic superiority verdict.

## 7. Integrate through the existing Operator

The existing `/api/jobs`/`OperatorJobRequest` path recognizes `dataset_derive`,
`acquisition_plan`, `acquisition_pool`, `diagnosis_benchmark`, `coreset_select`, and
`reviewer_check`. Its existing workspace restrictions, token arrays and JobManager
remain the authority. `operator/research_cycle.py::execute_cycle_job` advances one
eligible stage with a durable receipt, real job completion and verified producer
output. It never retries an uncertain submission. Inspect the receipt and actual
job state to reconcile a crash. Human audit/approval stages are deliberately not
synthesized by the dispatcher. Native capture uses the existing job kind and
motion-authorization boundary; it was not executed in this offline work.

## 8. Move the package and reproduce metrics

```bash
uv run carla-reviewer-evidence pack submission.json --output ../review-bundle
uv run carla-reviewer-evidence verify ../review-bundle --sha256 ACTUAL_CLOSURE_INDEX_SHA256
uv run carla-replay-bundled-metrics ../review-bundle --evaluation-manifest-sha256 ACTUAL_EVALUATION_MANIFEST_SHA256 --closure-sha256 ACTUAL_CLOSURE_INDEX_SHA256 --runs-root runs --run-id reviewer-metric-replay
```

`pack` may retain an incomplete submission for inspection but still returns a
nonzero readiness status. `verify` checks portable byte/reference integrity, not
all paper semantics. Metric replay supports canonical and pooled COCO evaluations,
uses saved predictions and writes its own manifest/log/comparison. Original paths
may be absent. It explicitly records that training and inference were **not**
reexecuted. Reproduce actual training separately with legally available inputs and
an independently resolved runtime; this package does not execute untrusted code.

The bundle includes original path strings and author metadata. Review privacy,
licensing and anonymous-review requirements before sharing; there is no automatic
public upload, license grant, or anonymization. Archive the final code/data with
persistent identifiers only after owner approval and preserve the prior versions.

## 9. Bind all review roles to the same study

Individual role checks are necessary, not sufficient. Both empirical modes now
also require the `study_linkage` check described in [ADR 0003](adr/0003-cross-role-study-evidence.md).
Use the data-custodian workflow to freeze the **already assigned** source releases:

```bash
uv run carla-reviewer-evidence freeze-split \
  --dataset datasets/release-train \
  --dataset datasets/release-validation \
  --dataset datasets/release-test \
  --maximum-samples 100000 --output study/split-v1.1.json
```

These paths and capacity illustrate syntax, not bundled empirical data or a
recommended sample size. Include all releases used by the study, including
explicit training-side acquisition/derived releases as appropriate. The tool
supports canonical and sealed sharded sources; it does not create partitions.
An existing output is never overwritten. Choose the budget before running it;
exhausting the bound is a failure, not a certificate of completeness.

Record the emitted split fingerprint in the **new** study protocol. The frozen
inventory contains exact `(manifest_sha256, sample_id)` identities, original
partitions, scene roots and RGB digests. Its `holdout_sha256` is the canonical
JSON digest of the sorted sample rows whose original partition has test role.
Use that digest in the matrix and in producer bindings. Changing a model seed
must not alter the inventory. Group-only schema 1.0 remains valid for legacy
software planning but cannot satisfy empirical review.

The protocol's `primary_metric` must be the canonical dotted field actually used
by the result producer (for example a field in its registered summary), not a
human-readable alias. The result matrix's methods and training seeds must match
the protocol; the baseline registry must cover those methods. The geometry audit
must reference all frozen dataset manifests, and producer dataset references must
belong to that inventory. This is additional to, not a substitute for, each role's
existing semantic checks.

In the manuscript record, use claim ID strings matching the selected claim
registry, and `tables` entries with **exactly** `table_id` and `results`, where
`results` is the selected result file's actual fingerprint. In annotation audits,
add the optional `dataset_manifest_sha256` when a local sample ID/group appears in
more than one release. Ambiguous or unrelated sample references fail. The template
stays incomplete until actual evidence and author approval exist.

The global cost ledger must contain exactly the logical events allocated among
all per-result producer ledgers. Event IDs may not be reused in two producers;
kind, outcome, resources and evidence reference must be identical. Hash-chain
positions may differ because each ledger has its own ordering. An unallocated
failed probe or review is not omitted from method budgets: it makes empirical
linkage fail. Define and record shared-cost allocation explicitly; the tool does
not invent such an allocation or measure uninstrumented work.

Verification of links does not parse article prose, regenerate typeset tables or
prove that a producer evaluated every declared test sample. The matrix currently
represents one condition/learner/regime and paired training seeds, not automatic
coverage of the entire generation/learner/ablation Cartesian design. Check those
additional dimensions and replay saved predictions independently. The checker
may reject artifacts that are useful for development but inadequate for the
selected empirical claim. Keep them; do not repair history by rehashing it.

Focused verification, with no CARLA server:

```bash
uv run pytest -q tests/test_reviewer_study_links.py
uv run carla-reviewer-evidence check study/submission.json
```

The report distinguishes individual role failures from `study_linkage`. A passing
link gate is **consistency only**, never evidence of superiority or guaranteed Q1
acceptance. Completion is tracked in Issue #144 rather than a parallel doc backlog.

## 10. Verify observed processing, not only the declared split

[ADR 0004](adr/0004-producer-consumption-receipts.md) adds a second empirical gate,
`study_consumption`, on top of section 9. Supply the **same frozen inventory** to
an instrumented producer before it runs:

```bash
uv run carla-evaluate-detector --help
uv run python -m carla_vision.imitation.runner --help
uv run carla-evaluate-imitation --help
uv run carla-pool-evaluations --help
uv run carla-reviewer-evidence check-consumption --help
```

The producer flag is `--split-inventory study/split-v1.1.json`. The exact installed
detector evaluation command is in `pyproject.toml`; do not replace a registered
command with a shell wrapper. These are interface examples, not existing runs.
After an actual run, verify its registered trace and selected weights:

```bash
uv run carla-reviewer-evidence check-consumption runs/evaluation \
  --split-inventory study/split-v1.1.json --full-holdout
```

Omit `--full-holdout` for a canonical partial evaluation or training receipt.
For pooled detection outputs, use `--full-holdout`: this rechecks each child,
source identities and image-ID remapping. It does not rerun inference.
The full flag means all test-role samples in that inventory, not an arbitrary
subset. Freeze a new justified inventory for a different registered study;
never relabel historical partitions to make a failed gate pass.

The actual producer records a plan, a JSONL stream of acknowledged batches and a
summary under `consumption/`. Each sample key is `manifest_sha256:sample_id`;
outputs explicitly include an empty detection array for processed negative
images. Training is exactly once per scheduled epoch, not once across all
epochs. Configured training shuffling is retained as observed order. Invalid or
failed batches do not create success records. Failure diagnostics describe an
uncertified suffix; a failure after computation but before fsync cannot establish
whether that suffix was computed. No silently repaired partial success is allowed.

For the empirical model card add both arrays:

```json
{
  "training_consumption": [],
  "pretrained_baselines": []
}
```

These are additional fields alongside the existing required card fields, not a
complete card. In `training_consumption`, put fingerprints of actual completed
instrumented training `manifest.json` files. In `pretrained_baselines`, each
entry has exactly `weights` (a real fingerprint) and a nonempty `reason`.
An empty list is permitted only for a genuinely absent category. Every evaluated
checkpoint must match exactly one category; training-utility claims need evaluated
observed-training weights. Existing claim, method, cost and source checks remain.

The imitation loop and canonical detector evaluation are instrumented. Detector
training adapters receive `TrainingRequest.sample_consumption` only when they
advertise `consumption_contract_version = "1.0"`. Such adapters must bracket
**actual** completed batches with `recorder.batch(pass_id, keys)` and set one
`outputs` record per key. Merely iterating the declared filenames to synthesize
receipts is invalid. The built-in Ultralytics YOLO/RT-DETR adapters now implement
this boundary **only for the pinned restricted profile in section 11**. Other
configurations fail before training rather than claiming observed use.
Non-audited training is unchanged and cannot satisfy this empirical gate. Fresh fixed-epoch imitation is supported;
audited resume, early-stop/replacement sampling and mosaic/mixup are not supported
by this receipt version. The ordinary non-audited resume path remains available.
Opaque factory-only weights are not an attested evaluation model identity.

### Source-bound test results

From a clean committed checkout, run the fixed pytest wrapper, writing outside
the source tree:

```bash
uv run python -m carla_vision.reproduction.test_evidence \
  --repository . --output /tmp/reviewer-tests-new
```

The output directory must not exist. The wrapper retains `pytest.log`, `tests.xml`
and `test-evidence.json` after a completed subprocess, including a failed result.
Use the fingerprint of `test-evidence.json` for the empirical `tests` role; bare
JUnit cannot bind the selected source. Tests and subtests are counted once as
JUnit cases. A nonzero process, missing/empty JUnit, skipped/failed cases, changed
source or lock cannot yield a passed receipt. Source commit and verified Git tree
must match the submitted archive. Preserve selected test arguments; a narrow
passing suite is not proof of complete regression coverage. The standard research
CI uses the complete suite. CI environment fields are provenance, not independent
attestation or proof that the asserted journal requirements were met.

Expected failures include `study_consumption` for legacy/unattested outputs,
wrong weights or incomplete holdout; `test_source_binding` for unrelated source
or lock; and consumption coverage errors for duplicates, omissions or edits.
Do not rehash old artifacts as a repair. Re-execute the appropriate producer in
a new output directory, retaining original failures. Semantic revalidation still
requires resolvable source paths; these checks do not add portable inference,
independent retraining, universal cost instrumentation or empirical ADAS results.


## 11. Audited YOLO/RT-DETR training with the pinned framework

[ADR 0005](adr/0005-pinned-ultralytics-consumption.md) documents the actual
framework hooks, source-safe loader and deliberately restricted execution profile.
Keep the existing complete training config, but explicitly set:

```json
{
  "workers": 0, "amp": false, "cache": false,
  "patience": 0, "warmup_epochs": 0.0,
  "extra_options": {"mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0}
}
```

This is a **fragment**, not a complete config or an empirical experiment. Select
`backend=yolo` or `backend=rtdetr`, `custom_factory=null`, fixed batch/epochs and
an explicit single device. The pinned version is 8.4.137. No dependency upgrade
is needed. Do not enable richer sampling through `extra_options` to bypass the
fixed-pass contract. Training initialization loads a trusted local checkpoint;
no new pretrained assets are provided by this change.

```bash
uv run carla-train-detector --help
uv run pytest -q tests/test_ultralytics_consumption.py
```

Use the existing detector-training command with `--split-inventory` on actual
verified source releases. In each new run inspect `backend/consumption_view/`,
`backend/fit/args.yaml`, `consumption/plan.json`, `consumption/batches.jsonl` and
`consumption/summary.json`. The final declared pass is `validation/final`; its
weight digest matches the selected `best.pt`. It is validation, **not** exposure
to the test partition. Continue to use the independent instrumented detector
assessment for held-out AP and `check-consumption --full-holdout` on that result.

An image's trace includes source RGB/label hashes, normalized input tensor hash
and shape, transformed-target hash, and a completed operation boundary. Training
loss is explicitly a **batch loss repeated alongside each member**, not a
per-sample loss. Validator predictions use preprocessed-image coordinates, not
canonical image coordinates. Empty predictions still count as observed inputs.

The profile sets nominal batch size to actual configured batch size, disables
gradient accumulation and fixes several sampling/precision controls. These
constraints are reported, affect the experiment, and must be identical for all
comparators. Repeated hash checks, uncached decode, tensor hashing, finite-value
checks and fsync add overhead; do not cite normal training throughput for this
profile. GPU/MPS performance and reliability are not established by CPU tests.
Model and dataset semantics, source licensing, complete resource metering, and
empirical learning gains remain separate checks. No old release is reclassified.
