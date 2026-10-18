# ADR 0004: record sample use at the successful producer boundary

Status: accepted for the bounded offline contract; delivery is tracked in Issue
#144. This is not empirical validation or a guarantee of publication.

## Problem

ADR 0003 establishes the declared study inventory, not what a model consumed.
A prepared data YAML, prefetched item, completed wrapper process, unrelated JUnit,
or a checkpoint filename cannot prove a successful batch or full holdout coverage.

## Decision

Keep the existing producers and add an explicit `--sample-inventory` mode using
schema-1.1 frozen membership. `ConsumptionRecorder` stores a source-bound plan,
ordered hash-linked events and a terminal report inside the producer run. A
`prepare` token checks source identity, partition and input digest but is not a
successful event. Only the owning loop acknowledges it after successful model
work. Training includes backward and a completed optimizer call; inference includes
its output/metric update. Prefetch does not commit. An exception retains an aborted
journal and the failed producer, not a complete attestation. One recorder has one
process, one pending batch and one active pass.

Each pass enumerates exactly the declared phase membership. Repetition is allowed
across explicitly separate epochs, not within an exact-once pass. Evaluation can
also require the declared order. Missing, duplicate, substituted, extra, changed
or wrong-partition identities fail. Empty detections are an evaluated image, not
an omitted sample. Journals have explicit byte/event capacities. Input tensor
hashes include shape and dtype; RGB is hashed and decoded from the same buffer.

Instrumented producers bind initialization and actual selected/evaluated weights.
An evaluated trained arm names its immutable producing training manifest and best
checkpoint. Epoch-boundary imitation resume preserves the prior last checkpoint,
history and consumed passes recursively. No historical trace is manufactured for
an old checkpoint. A protocol declares each method `trained` or `external_frozen`;
the latter pins its external weight fingerprint. Merely presenting an external
model cannot establish training utility.

Canonical evaluation receipts are compared to saved frame rows and predictions.
Pooling verifies each child, remaps identities, reconstructs pooled truth and
predictions, recomputes AP and checks the exact frozen holdout, including whole
missing shards. Pooling never claims a second inference. Imitation evaluation and
validation control metrics are replayed from per-sample outputs/targets; training
augmented targets are distinguished from unaugmented validation targets.

`carla-test-execution` launches pytest in a clean source workspace into a new
external output directory. It retains the executed command, exit/timeout status,
log, JUnit and before/after Git tree/commit/lock/selected-source identities. A
passing receipt requires process completion, nonempty tests, zero errors/failures/
skips and unchanged committed source. A zero-failure XML from a hung process is
not success. The reviewer compares the receipt to a Git-object-proven source ZIP.
Raw legacy JUnit is only development evidence. Producer file-byte digests are
also compared to that source; an arbitrary clean cwd marker is insufficient.

## Supported boundaries

- Native detector evaluation, canonical and pooled saved-child evaluations.
- RGB-plus-speed imitation training, validation, epoch-boundary resume and test.
- Opt-in Ultralytics YOLO and RT-DETR training at the exact reviewed lock version.
  Upstream trainer/validator/dataset source hashes are checked. The adapter hooks
  preprocess, completed optimizer work, batch end and validator metric updates;
  it does not copy the upstream training loop. Verified selected inputs are
  projected into a bounded temporary directory, protecting canonical data from
  library caches. Training and final selected-best validation are separate passes.

The initial built-in detector profile requires one process, workers=0, no AMP,
no cache, no early stopping, no multi-image augmentation, full epochs and one
optimizer step per batch. Unsupported DDP, AMP skips, accumulation, mosaic/mixup/
cutmix/copy-paste, truncated sampling, custom transformations and unknown hook
versions fail before certified training rather than inventing lineage. Custom
backends must explicitly implement the receipt boundary; returning a checkpoint
is insufficient. Arbitrary plugins are not certified by declaring a capability.
Uninstrumented voxel/opaque/legacy paths cannot satisfy the empirical gate.

## Compatibility and evidence limits

Ordinary non-attested producer behavior remains supported. Existing artifacts are
immutable, readable as legacy evidence and not silently upgraded. Opt-in failed
runs must use new output directories. Revert code to roll back; never edit old
manifests, controls, predictions or weights to satisfy this contract.

These are trusted-process software records against accidental inconsistencies,
not hardware attestation against malicious Python, an operating-system sandbox,
proof of gradient magnitude per sample, or proof that labels are scientifically
correct. Digests of transformed inputs do not retain/reconstruct those tensors.
Recorded source files cover the specified producer boundaries, not every possible
external library, plugin, dependency or runtime monkeypatch. Independent execution
and environment/model provenance remain necessary. Self-authored timestamps do
not prove first holdout access. Protocol freezing is a separate external process.

Full crossed-factor studies, nearest-method reproduction, universal GPU/child
costs, independent audits, real training utility and Q1 acceptance are not supplied
by these contracts. Relocated byte closure and saved-metric replay remain separate
from semantic verification with resolved source/weight paths.

## Regression evidence

`tests/test_sample_consumption.py` exercises omission/repetition/substitution,
wrong source/partition/weights, prefetch/abort, pooling and altered saved outputs.
Tiny real CPU training/inference for both YOLO and RT-DETR and imitation exercises
actual loop hooks and selected weights. A narrow fixture binds source, real pytest
execution and consumption; it is not a full empirical reviewer submission.
`tests/test_execution_receipts.py` covers failed/skipped/hung subprocesses, source
changes, command tampering, stale output and relocated source/JUnit binding.
