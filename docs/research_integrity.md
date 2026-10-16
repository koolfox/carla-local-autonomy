# Research integrity and offline hypothesis workflow

This guide specifies executable contracts, not measured ADAS performance.
Issue [#140](https://github.com/koolfox/carla-local-autonomy/issues/140) owns delivery
status. [ADR 0001](adr/0001-research-data-and-evidence-boundaries.md) records the
durable decisions. No CARLA instance, downloaded checkpoint or model-quality
benchmark is required for the regression tests below.

## Run the offline checks

```bash
uv sync --all-extras --all-groups --frozen
uv run ruff check carla_vision tests restore_project.py
uv run pytest -q tests/test_research_integrity.py tests/test_research_evaluation.py tests/test_research_methods.py
uv run pytest -q
```

The full suite needs the locked optional profiles and the repository's retained
recovery assets; a source-only review archive intentionally does not contain those
assets. For the source-only environment use the focused tests, then explicitly
report missing-profile/recovery exclusions. Do not describe that as a full clean
checkout test. JUnit output is retained by the research CI workflow even on failure.

The dedicated tests include hand-derived mask/box answers, semantically corrupted
but rehashed releases, independent source partitions, alternating temporal
contracts, per-model replay thresholds, duplicate-group COCO resampling, actual
tiny CPU epoch-resume equivalence, and invalid capture/review/cost evidence. The
`research_fixtures.rehash_dataset` helper is **test-only**: it is not a production
repair utility and must not be used to conceal edits to published artifacts.

## Data and model migration

The authoritative `sample.split` survives detector and imitation consumption.
`--val-fraction`, `--test-fraction` and the legacy imitation `split_seed` argument
remain accepted for old invocation compatibility but do not repartition source
samples. `unassigned` is not a trainable source. `acquisition_pool` is reserved for
training-side acquisition and must be explicitly frozen into a new train release.
Validation/test diagnostics remain in the existing validation-only mining path.

Train/validation source-group inventories and exact RGB hashes are registered in
the checkpoint split summary. A changed model seed does not change these source
partitions. Scene roots survive parent/child/grandchild derivation. A geometry
fingerprint identifies one polyline; it does not prove spatial disjointness or
replace a registered corridor-overlap policy. Legacy data without route evidence
remain episode-scoped, and must not support an unseen-route claim.

New BehaviorAgent supervision uses `ObservationActionTransition` at the captured
observation/decision frame. `applies_before_frame` is the following world frame;
`resulting_frame=null` means a resulting observation has not been certified.
Traffic Manager trace targets are not silently treated as decision-aligned
BehaviorAgent targets. Old BehaviorAgent releases without timing evidence are
rejected by default by the imitation loader. The development-only
`--allow-legacy-action-targets` option preserves inspection capability without
rewriting history. A future decision paired with a past image is rejected.

The supported imitation input is **RGB plus speed**, not strict RGB-only. The
existing strict-RGB shadow contract is unchanged. The compact voxel semantic map
now explicitly uses CARLA 0.9.16 tags (road=1, pedestrian=12, car=14, road-line=24).
Old compact labels or checkpoints are not automatically revalidated: regenerate
teacher targets from a version-identified source before retraining. In 0.9.16,
wall=4 and terrain=10 must not be classified as dynamic road users.

## Imitation dry run, train, resume and explicit test

`carla-train-imitation --dry-run` is a forward-only synthetic shape smoke when no
dataset is supplied, and a plan-only read when a dataset is supplied. Neither
mode constructs an optimizer or saves weights. The default real training path
requires non-empty train and validation partitions. `--purpose confirmatory`
also requires a clean recorded Git revision and rejects bypass/legacy/truncation
options. An explicit `--allow-training-selection` development run reports
`score_source=train_development_only` and `best_validation_loss=null`.

```bash
uv run carla-train-imitation --dataset datasets/train-release --dataset datasets/val-release --output models/imitation-r1 --device cpu --epochs 2
uv run carla-train-imitation --dataset datasets/train-release --dataset datasets/val-release --output models/imitation-r2 --device cpu --epochs 4 --resume models/imitation-r1/checkpoint_last.pt
```

These are example paths to **real verified releases**, not bundled training data.
Resume writes a new run directory; the old run is immutable. Epoch-boundary resume
includes optimizer, RNG and loader state, split/source identities, augmentation
settings and best-model state. It is not arbitrary mid-batch resumption or a
cross-device bitwise determinism guarantee. RMSE and brake precision/recall are
pooled over samples, not averaged over batch-level nonlinear metrics.

Training never evaluates the test. Run `carla-evaluate-imitation --help` for the
separate locked-test command. A test in the same source release requires explicit
`--acknowledge-locked-test`. For a physically separate test release also provide
`--evaluation-protocol` with this JSON schema:

```json
{
  "schema_version": "1.0",
  "checkpoint": {"path": "...", "sha256": "...", "size_bytes": 0},
  "training_manifest_sha256": ["..."],
  "test_manifests": [{"path": "...", "sha256": "...", "size_bytes": 0}],
  "group_policy": "disjoint_scene_roots_and_exact_rgb"
}
```

Replace every placeholder with actual `fingerprint_file` output before use. The
protocol binds the selected checkpoint and both source sets, and rejects overlap
with stored training/validation groups or RGB hashes. It does not establish when
an operator first inspected the test. Freeze the protocol before opening the
holdout and retain that event in the study evidence.

## Teacher and dataset verification

`carla-verify-teacher-episodes` now builds on the common verified dataset envelope,
then checks teacher context and timing. Empty datasets, missing fingerprints,
failed producer trackers and semantic COCO inconsistencies do not count as a
verified training release. Independent human audit is still necessary for shared
producer errors or rendered semantic mistakes.

`render_instance_id` explicitly names the sensor's encoded render identity. The
legacy `actor_id` serialization alias remains for archived schema compatibility;
it is not authorization to call world actor lookup for distance or subtype labels.
A shader/world mapping requires separate version-bound evidence.

Native queues have a bounded default. Missing required target frames fail rather
than being replaced. Partial spawn IDs are owned before errors propagate, and
cleanup failures are recorded without replacing the primary error. A same-host
lease prevents cooperative collectors sharing one endpoint; other hosts and
uncooperative clients remain outside that lock. Unexpected frame jumps fail.
The native isolated-collection reset policy is explicitly asynchronous baseline,
not restoration of the previous map, actors or Traffic Manager random state.

The writer has explicit sample/annotation capacity limits and fails before the
next over-limit sample is written. This bounds growth but is **not** a sharded
high-throughput writer or a measured capacity guarantee. Profile representative
runs before increasing limits; the previous in-memory release representation is
not claimed to scale indefinitely.

## Thresholds, latency and uncertainty

Each replay model entry may contain an optional `threshold_selection` path to a
verified validation-only threshold artifact. Offline evaluation also accepts
`--threshold-selection`. The artifact must match model weights/adapter, options,
input size, ontology and IoU/alias policy. Child and parent operating metrics use
that exact threshold. Omitting the field is an explicitly common-raw-threshold
study, not model-specific operating-point selection.

The new supported operating-point metrics use null with denominators when no
support exists; legacy zero-valued fields remain identified for compatibility.
Optional `ap_bootstrap_replicates` in evaluation config reruns COCO over group
resamples with new image/annotation IDs for repeated groups. The API also supports
paired prediction comparisons. Single-group and undefined draws are reported,
not given artificial certainty. Training-seed variation requires a separate
registered multi-seed analysis; these intervals condition on fixed weights.

`measured_inference` uses explicit CPU, CUDA or MPS completion boundaries. It times
the adapter call including work in that call, not file decode, network transport,
preview render or storage. CPU/GPU/latency/storage units must not be mixed. The
existing replay warm-up exclusion affects latency, not accuracy. ECE/Brier remain
conditional on the retained detection population; they do not measure missed
objects. The closed-loop observer remains labelled development-only: high-brake
transitions are not human interventions, and route/model ownership and stop-line
violations are not certified by this observer.

## Executable acquisition study

Use `carla-prepare-acquisition-pool --help` to convert an existing **verified
training-side** detector evaluation plus an independent blinded audit into action
candidates. Validation/test evaluations are rejected. The audit format is defined
by `failure_mining/acquisition_pool.py`; each review references a source sample,
scene group, reviewer and separately fingerprinted evidence. A conflicting or
ambiguous review remains unknown. Label and action review tasks are distinct.

The current adapter consumes successful verified captures. Rejected native capture
diagnostics require a separate caller supplying `AttributionEvidence`; this CLI
does not turn an invalid release into a verified one. A model error alone cannot
establish a bad label. Quality estimates are pooled group-level audit bounds,
not instance-level calibrated probabilities. The conservative lower bound remains
below one even for a small all-valid audit. Per-condition audit sampling must be
registered and externally carried out; no fictional reviewer is generated.

`carla-plan-acquisition-study` reads a strict protocol and evidence-backed candidate
pool. The executable schema is in `scenarios/study.py`; the focused test
`test_study_plan_is_executable_but_not_empirical_result` constructs a complete
**synthetic software fixture**, including all paths and fingerprints. Protocols
freeze hypotheses, effect size, primary metric, methods/ablations, generation and
training seeds, budgets, group partitions, clean/fault regimes, stopping rule and
power/compute justification. Candidate pools reject unknown fields such as an
injected-fault oracle.

The selector implements seven transparent heuristic strategies: random,
stratified, uncertainty, failure-only, noise-aware, capability-matched and
error-source-aware. Required coverage, exploration, group caps, deterministic
ordering and cost budget use the same machinery. Ablations remove quality,
attribution, coverage or exploration. A greedy failure to fill quotas is a failed
plan, not a proof of global infeasibility. These strategies are **not reproductions
of QActor, NAS or ADAA**; published algorithm baselines require a separately
matched implementation before a comparative novelty claim.

Only estimated costs guide this planner. `CostLedger` separately records actual
render/probe/audit/repair/train/evaluation/storage resources, including rejected,
failed and cancelled work. The caller must instrument every stage; the current
library does not claim automatic GPU accounting throughout the product. Journals
are hash-linked and fsynced, but are not signed or multi-writer databases.

`CycleJournal` validates stage order, frozen identities and referenced producer
artifacts, and can resume a nonterminal journal. It is an integration primitive
for the existing job owner, not a second scheduler or automatic CARLA dispatcher.
Capture, training and evaluation still use their existing producers. Stage files
must be real outputs: a plan cannot stand in for a trained model or a dataset.

`carla-derive-training-dataset` implements immutable FREEZE and independently
reviewed RELABEL using the canonical writer. Parent bytes stay unchanged;
COCO/YOLO are regenerated and lineage is retained. RECAPTURE and NEW_SCENE cannot
be faked by copying RGB; they must be produced by native collection. The pipeline
does not upgrade old labels, RGB views or recorded MP4s into new capture evidence.

## Claim and redistribution gates

`carla-check-research-claims registry.json` validates evidence scopes:
implementation, offline contract, training utility, simulator driving, real world.
A synthetic fixture cannot pass an empirical claim. The JSON shape is exercised
in `test_publication_gate_does_not_promote_fixture_to_empirical_claim`. Rights
clearance requires an actual license reference and owner-reviewed asset inventory;
the command does not pick a project license, grant model rights or guarantee Q1
acceptance. CITATION authors and existing model/sample notices are unchanged.

Keep algorithm tests, tiny synthetic learning checks, real training results,
live acceptance, blinded human audits and independent reproduction separate.
Unimplemented optional matched interventions and real-world transfer remain
outside the current empirical claims. The scientific hypothesis is still open:
no held-out ADAS performance improvement is asserted by this software change.

## Reviewer-evidence extension

See [the reviewer runbook](reviewer_evidence.md) for source object proofs, recorded
corridor audits, canonical shards, source-partitioned voxel data, bound result
tables and portable saved-prediction replay. It supersedes the earlier limitations
on those implemented software paths; empirical, privacy and rights gates remain.
