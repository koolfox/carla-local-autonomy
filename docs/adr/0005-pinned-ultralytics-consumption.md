# ADR 0005: observed batches in the pinned Ultralytics trainer

Status: implemented restricted software contract. Extends ADR 0004; Issue #144
owns delivery. These tests do not establish empirical ADAS performance.

## Problem and scope

ADR 0004 instrumented the native inference and imitation loops but deliberately
rejected audited YOLO/RT-DETR training. A generic callback that simply lists
files, or runs on prefetch, does not close that gap. Ultralytics may discover
extra directory images, use label/image caches, combine images, accumulate
multiple batches, retry OOM, stop early, or validate the selected checkpoint
again after fitting. Each changes the meaning of a fixed-pass receipt.

## Decision

Add a lazy adapter for **Ultralytics 8.4.137**, the unchanged lock version. Use
subclasses of its actual DetectionTrainer/RTDETRTrainer and validators, without
copying the optimization loop or globally monkeypatching framework functions.
Read the version-matched sources rather than assuming latest callback behavior:

- `ultralytics/engine/trainer.py`: preprocess, forward/backward, optimizer step,
  `on_train_batch_end`, epoch validation, checkpoint stripping, final validation.
- `ultralytics/engine/validator.py`: inference/postprocess before `update_metrics`.
- `ultralytics/models/{yolo/detect,rtdetr}/{train,val}.py`: task-owned hooks.
- `ultralytics/data/{base,dataset}.py`: image/label loading and collating.

Upstream source: <https://github.com/ultralytics/ultralytics/tree/v8.4.137/ultralytics>.
Upstream notices and license apply to that dependency; this change does not grant
new redistribution rights or choose a license for this project.

## Restricted execution profile

Audited runs require explicit `workers=0`, `amp=false`, `cache=false`,
`patience=0`, `warmup_epochs=0`, fixed positive batch size and one explicitly
selected CPU/MPS/CUDA device. Each of `mosaic`, `mixup`, `cutmix` and `copy_paste`
must be explicitly zero in `extra_options`. Extra options use a small allow-list;
custom trainers, class filters, external configuration, automatic batches,
distributed execution, resume, time limits, compilation and richer sampling
are not supported by this profile. Version mismatch fails before model loading.
GPU/MPS branches require hardware-specific validation before claims about those
devices; the retained integration tests run actual CPU optimization.

Resolved arguments additionally fix `nbs=batch_size`, `close_mosaic=0`,
`single_cls=false`, `classes=null`, `fraction=1`, `multi_scale=0`,
`distill_model=null`, `compile=false`, `quantize=null`, `time=null` and normal
checkpoint/validation saving. These are **visible experiment constraints**, not
an invisible speed or accuracy optimization. In particular, nominal batch size
and disabled accumulation may change optimization relative to ordinary defaults.
Every compared method must use the same preregistered profile. Ordinary training
without `--split-inventory` keeps its previous backend behavior.

## Source and observation boundary

The source view uses verified canonical/sharded members, not a filesystem glob.
Its custom read-only dataset parses the exact hashed YOLO label bytes, hashes and
decodes the same RGB bytes, and bypasses `.cache`, `.npy`, image repair and label
rewrite logic. No RGB copies or caches are created in source releases. Only
run-local source lists, YAML and identity mapping are written. One-source image
transforms still run in the upstream task pipeline; target/image tensor digests
and shapes are recorded immediately before the model operation. This is not
independent mathematical verification of every transform. Shape/byte scans at
initialization do **not** create processed-sample events.

A training batch is opened at preprocessing and acknowledged only after one
completed optimizer step and the successful upstream batch-end callbacks.
Nonfinite gradients/weights, missing callbacks and automatic pipeline rebuilds
fail instead of silently issuing receipts. Upstream first-epoch OOM retries and
NaN-recovery restarts cannot change the audited schedule. A failed or interrupted
batch is uncertified, even if some computation happened before the failure.

Every epoch has separate train and validation passes. `validation/final` is a
separate declared pass using the actual stripped `best.pt` loaded by the upstream
validator. The file is fingerprinted before/after that validation and must match
the selected weights sealed by the producer. Validation outputs include per-input
postprocessed predictions, even when empty, in preprocessed-image coordinates.
They are not a substitute for the canonical held-out COCO evaluation producer.

The verifier checks framework/profile identity, the configured epoch schedule,
source RGB/target digests, batch-level optimizer-step order, per-image prediction
shape and the final checkpoint binding. This extends the common exact-membership
and source-reopening checks; it never promotes ordinary unobserved runs.

## Evidence, migration and rollback

`tests/test_ultralytics_consumption.py` runs tiny random-weight YOLO and RT-DETR
models through their actual trainer/optimizer/validator, including sharded data,
multiple epochs and a partial last batch. It also injects failures before/after
optimization and semantic trace corruption after superficial rehashing. Network
update/font lookups alone are stubbed in these tests; ML execution is not mocked.
These are software fixtures, not trained ADAS models or effect-size estimates.

Create a new run with a valid frozen split and the explicit profile. Do not attach
new receipts to old checkpoints or rewrite historic manifests. Revert source code
to roll back; prior artifacts remain immutable. Audited resume chains, composite
image provenance, replacement/distributed sampling, device-scale performance,
complete resource accounting and empirical model utility remain separate gates.
Receipts are locally recorded observations, not independent or signed proof
against malicious full-record rewriting, first holdout access, or journal acceptance.
