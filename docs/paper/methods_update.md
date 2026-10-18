# CarlaPyVision methods update: reviewer-evidence implementation

**Manuscript status:** source-grounded method/implementation text for revision;
not a completed empirical paper. Replace result placeholders only with approved
registered outputs. No author list, legal approval, DOI or experimental score is
invented. See the companion reviewer matrix for primary-source attributions.

## Proposed title

**CarlaPyVision: Error-Source-Aware Data Acquisition and Auditable Supervision for
ADAS Perception**

## Abstract working text

Simulation can supply large amounts of annotated driving data, but an observed
model-label disagreement does not uniquely identify a model failure. It can also
reflect invalid capture, ambiguous supervision or an uncertain diagnosis. We
present a CARLA-based research pipeline that keeps those error sources and their
associated actions explicit, while preserving the provenance required to evaluate
data utility. The software distinguishes recapture, independently reviewed relabeling,
new-condition acquisition and unresolved cases, and separates acquisition data from
model-selection validation and locked evaluation. An executable evidence layer binds
source revisions, sample lineage, actual cost records, selected models and resulting
tables. Bounded canonical shards support collection and training, and pooled metrics
are recomputed from predictions rather than averaged across shards. A portable
reference closure permits independent recalculation of retained detection metrics
without access to the original machine. The empirical study is designed to test
whether error-source-aware action selection improves held-out perception under
matched acquisition, review and training budgets. Software tests establish contract
behavior only; they do not establish the direction or magnitude of that empirical
effect. The pending evaluation compares task-matched baselines and ablations across
independently defined scenes, generation seeds and model initializations.

## Research question and contribution boundary

The proposed contribution is a measurable decision process around dataset creation,
not CARLA itself, a new object detector, a checksum algorithm or merely a unified
interface. The hypothesis is that selecting an appropriate action after distinguishing
capture, supervision and model errors can improve data utility at equal total cost,
without removing valid difficult conditions. This must be tested against a control
with the **same review and collection capabilities** but no source-aware diagnosis,
and against nearest noise-aware acquisition methods. Prior work already covers
synthetic data, active acquisition, privileged supervision and noisy-oracle learning.

## Formal objects and action selection

An observation is associated with image, target, sensor/frame context, source group,
partition and immutable file references. The diagnosis consumes allowed capture
checks, independently audited supervision information and model-error measurements;
it does not consume the injected-fault label. The current diagnosis and selection
rules are transparent heuristics with explicit unknown states, not calibrated
per-instance posterior probabilities. Group-pooled conservative quality bounds must
not be called exact label correctness probabilities.

Let q denote a candidate and a an allowed action. A registered selection policy
chooses a finite set S of candidate-action pairs subject to a common estimated
budget, condition coverage, exploration and per-group limits. Actual costs C(S)
include all probes, rejected attempts, audits, repairs and training, not just retained
images. Selection estimates and measured costs are stored separately. A greedy
failure to satisfy constraints is not a proof of global infeasibility.

## Temporal and geometric validity

Decision supervision distinguishes observation, decision and actuation frames.
Legacy trace controls are not silently shifted or treated as verified reactive
policy targets. The implemented imitation track uses RGB plus speed, whereas
strictly visual shadow inference has a separate input contract.

For route-independent claims, a capture records its actual OpenDRIVE bytes and
sample poses. Build a graph whose vertices are captured route segments grouped by
source scene roots. Add an edge for a common scene root or when the minimum 3D
distance between segment interiors on the same captured map is at most registered
clearance. Connected components may not cross partitions. This implementation
checks observed piecewise-linear trajectories, rejects long sampling gaps and
preserves transitivity. It does not establish independence of unobserved motion or
alternative roads. Source split labels remain authoritative for detector, imitation
and voxel consumption; absent legacy voxel splits require development-only opt-in.

## Collection and evaluation at bounded scale

Opt-in sharding seals each bounded group of samples using the canonical writer and
semantic verifier. The root inventory preserves order, counts, auxiliary map files
and cross-shard grouping. An interrupted release stays incomplete; read-only recovery
lists retained shards instead of overwriting them. Training references the sealed
shards without duplicating RGB. This bounds active writer buffers; it is not evidence
of measured long-run throughput or unlimited memory.

Evaluation retains canonical predictions and image identities. Pooled evaluation
requires equal model, ontology and policy, remaps image/annotation identities, and
recomputes COCO on the pooled population. Averaging shard AP is forbidden. Confidence
thresholds originate from validation and remain bound to both parent and child runs.
Group-bootstrap AP estimates uncertainty conditional on fixed weights; separate
paired training-seed or crossed generation/training-seed analyses address different
variation. Unsupported metrics remain null with denominators, not fabricated zeros.

## Result provenance and reproducibility

A paper result row binds the method, seed, holdout identity, selected source,
registered output metric and all-attempt cost ledger. An empirical claim must point
to the actual registered empirical result of a non-synthetic producer. A wrapper
cannot override the result's values or borrow an unrelated synthetic manifest.
Source archives carry raw Git commit/tree proofs and included blob bytes, so a
revision label alone cannot authenticate unrelated code.

The reference closure retains original bytes under content hashes and checks all
transitive dependencies. Relocated COCO replay recalculates metrics using those saved
predictions and writes a new comparison log. It explicitly does not retrain a model
or rerun inference. Independent model reproduction, actual simulator acceptance,
human QA and legal redistribution remain separate evidence categories.

## Tables to be populated from real runs

| Table | Input and computation | Current scientific status |
|---|---|---|
| Diagnosis by fault regime and clean-hard condition | `carla-score-diagnosis`, blinded references | empirical results not supplied |
| Learning utility under equal total cost | frozen method x seed runs and registered result matrix | empirical results not supplied |
| Component ablations and sensitivity | shared selection protocol and preserved condition support | empirical results not supplied |
| Held-out map/weather/route generalization | fixed grouped test and canonical/pooled predictions | empirical results not supplied |
| Paired effects and uncertainty | `carla-analyze-study-seeds`; group/crossed analysis as appropriate | empirical results not supplied |
| Production overhead and scale | measured stage ledgers, storage and representative captures | empirical results not supplied |
| Independent reproduction | original approved package and another executor's actual log | external execution not supplied |

## Limitations to retain in the paper

The implemented rules are not a calibrated causal diagnosis. The visible-instance
ontology does not automatically cover light-state recognition, speed-sign text or
all ADAS tasks. Spatial grouping is conditional on supplied capture fidelity and
sampling density. Classifiers/detectors and simulator outputs may share domain bias.
No sim-to-real or collision-reduction result follows from improved offline AP.
The behavioral teacher and closed-loop observer still require a separate, driver-
bound experimental protocol. Greedy k-center covers one published component only;
it does not remove the obligation to compare the nearest noisy-oracle competitor.
Fixtures and CI establish software consistency, not full scientific validity.


## Producer-consumed sample evidence (software implementation)

A frozen dataset inventory defines the expected population but is not evidence of
execution. The implementation now records sample identities at the successful
producer boundary: completed training work, inference, or validation metric
update. Preparation and prefetch are not counted. Each phase/epoch is checked
against immutable source membership; missing, duplicated or substituted samples
prevent completion. Evaluation of a trained arm binds the consumed training
producer and selected-best checkpoint. Sharded evaluation reconstructs the union
of actual child receipts and recomputes pooled metrics, without claiming another
inference. Recorded producer source bytes and completed pytest/JUnit receipts are
checked against the selected Git-proven code snapshot.

This instrumentation is tested with synthetic data and tiny real CPU model loops,
not a dataset-utility experiment. It supplies traceable software records, not
trusted-hardware attestation, independent human audits or a guarantee of first
holdout access. Initial detector hooks restrict distributed execution, multi-image
augmentation, accumulation and AMP to avoid unsupported consumption claims. These
restrictions and any alternative sampling protocol must be disclosed when the
empirical experiments are executed. No AP improvement or hypothesis result is
reported by this implementation alone.
