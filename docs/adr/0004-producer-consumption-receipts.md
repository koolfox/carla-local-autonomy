# ADR 0004: producer-time consumed-sample receipts

Status: implemented software contract; external-backend and empirical limits apply.
Builds on [ADR 0003](0003-cross-role-study-evidence.md). Issue #144 tracks delivery.

## Problem

A source split and a result manifest can agree even when execution skipped an
image, substituted a batch, or returned cached predictions. A list of training
paths proves exposure, not actual model processing. A bare JUnit document also
has no necessary relationship to the submitted source revision.

## Decision

Add optional, fail-closed receipts at the existing producer boundary, not another
scheduler. A sample is identified by its canonical source manifest SHA-256 plus
sample ID. Shard-local names are not globally unique. The frozen schema-1.1 split
is reopened and its original memberships are verified before and after a run.

Each fixed pass declares its role, partitions, canonical sources, exact expected
keys and whether order is fixed. Training may shuffle but must acknowledge each
selected sample once per scheduled epoch. Evaluation has an exact order. A batch
is appended only after the instrumented model call (and, for imitation training,
optimizer step) returns successfully and its per-input outputs can be serialized.
Zero detections are an explicit empty output, not an omitted input. Prefetching,
reading a YAML file or constructing a DataLoader is not acknowledged processing.

The producer registers `consumption_plan`, `consumed_sample_batches` and
`sample_consumption`. The trace contains monotonically positioned batches,
actual sample keys, and per-sample output hashes. A complete summary binds the
chosen weights and exact pass coverage. Trace writes are flushed/fsynced. A failed
run retains the acknowledged prefix and uncertified omissions where shutdown is
possible; an abrupt kill or storage failure need not leave a final summary.
An omission means **not durably certified**, not proof that no computation occurred.

RGB is hashed and decoded from the same read bytes in the instrumented native
Python consumers. Identities refer to raw source images before configured
transforms; this is not independent verification of an arbitrary plugin's tensors.
All evidence is locally produced software observation, not cryptographic remote
attestation, protection against a malicious producer rewriting all records,
proof of first holdout access, or a scientific correctness certificate.

## Supported consumers and boundaries

- Detector evaluation: instrument the existing actual inference loop and compare
  trace outputs with both frame logs and canonical predictions.
- Pooled evaluations: verify canonical child receipts, weights and policy, then
  reconstruct exact image-ID remapping and prediction union. Pooling explicitly
  declares that inference was not rerun.
- Imitation: instrument real CPU/GPU forward/optimizer batches and validation;
  the separate locked-test command records its own selected checkpoint. Existing
  input remains RGB plus speed. Fresh fixed-epoch runs support receipts.
- External detector trainers: require explicit `consumption_contract_version=1.0`
  capability and use the supplied recorder at actual processing. A capability
  declaration without events cannot complete. The current built-in Ultralytics
  YOLO/RT-DETR trainers are **not instrumented** and audited requests are rejected
  before training. Unattested legacy training is still available.

Receipt version 1 does not support replacement sampling, dropped batches,
multisource mosaic/mixup, dynamic early stopping, distributed reductions, or
chained audited resume. Audited imitation resume is rejected until a verifiable
parent-epoch chain exists; ordinary epoch resume without receipts is unchanged.
A passing custom-backend fixture is not validation of a third-party training API.
The trace inspection limit is 256 MiB and the pass limit is 10,000; fail rather
than silently truncate. This is not a throughput/scale benchmark.

## Reviewer integration

Empirical submissions additionally require `study_consumption`. Each evaluated
model must cover exactly the frozen holdout, with no duplicates, missing keys or
foreign keys. Models are explicitly classified as observed-training checkpoints
or pretrained controls. A training-utility claim cannot be supported solely by
pretrained controls. Partial condition runs must be pooled for the complete
registered holdout; a count matching another set is not coverage.

The tests role in an empirical bundle now requires a `pytest_execution_receipt`:
clean source commit/tree before and after, fixed subprocess invocation and return
code, lock digest, JUnit bytes and log. The selected archive's actual Git objects
bind its tree; the returned tests receipt and lock must match. Bare JUnit remains
readable only in the explicitly offline-fixture mode. This is an execution
receipt, not signed CI authority or proof of sufficient test quality/coverage.

## Migration and rollback

Use `--split-inventory` explicitly on supported producers. Old artifacts stay
immutable/readable but cannot be upgraded to observed use by copying a manifest,
retrofitting trace files, or replaying metrics. Re-run the instrumented producer
into a new run. To roll back, revert code, not historical run bytes. The frontend,
Operator APIs, standalone Worker, dependencies, assets and licenses are unchanged.

## Verification

`tests/test_consumed_samples.py` exercises actual producer loops, tiny real CPU
training, no-detection coverage, failed optimization, changed raw bytes/weights,
wrong/duplicate/reordered/omitted samples, sharded union, mutable schedules and
uninstrumented or falsely declared backend support. `tests/test_test_evidence.py`
executes tiny isolated pytest subprocesses and verifies source/output binding.
Synthetic data and fake detector outputs establish software contracts only.
