# ADR 0001: Preserve research data identity and separate evidence scopes

Status: proposed in the research-validity change associated with issue #140.

## Context

The detector and imitation paths did not enforce the same dataset envelope. The
imitation loader rehashed source partitions, and its training seed changed the
split. Teacher controls recorded before a tick were consumed with images from
after that tick. Several development metrics were easy to mistake for validation
or driving-safety results. Existing datasets and model payloads must not be
rewritten to make these gaps disappear.

## Decision

Preserve every source partition. A separately collected training-side
`acquisition_pool` may enter `train` only through a new explicit FREEZE release.
Validation and locked tests are not adaptive acquisition inputs. Scene-root IDs
survive all derived releases; route-endpoint grouping is conservative, not proof
of unseen-route geometry. Model, augmentation, and acquisition seeds do not
redefine a frozen partition. Keep physically separate test releases when their
bytes must be inaccessible to the training process; verification of a combined
release necessarily reads its artifacts, but test targets are not model inputs.

Use the same tracker/hash/semantic verifier for datasets and teacher consumers.
Compare mask-derived annotations, metadata, COCO and YOLO semantically, including
IDs, classes, boxes and areas. Golden fixtures and independent blinded review are
separate checks: regenerating with the producer function is not an independent
semantic oracle. Render instance IDs are not automatically CARLA world actor IDs.

New BehaviorAgent targets describe a decision at the captured observation frame,
with the next application frame and an explicitly unknown resulting observation
until it is actually captured. Old action targets are not shifted or upgraded
silently. The imitation model consumes front RGB plus speed; its checkpoint and
inference contract say so. Sensor truth, action supervision, and model inputs are
separate lanes.

Dry runs never optimize or save checkpoints. Confirmatory training rejects
verification bypasses and training-loss fallback; locked evaluation is a separate
command. Epoch-boundary resume binds source, split, model and training settings
and restores optimizer and RNG/loader state. Model-specific validation threshold
artifacts bind both replay children and parent operating metrics. Group-bootstrap
COCO AP reruns COCO with unique resampled image/annotation identities.

Research acquisition separates capture faults, independent supervision concerns,
model errors on supported data, and unknowns. Cost and cycle journals retain
failures. Plans are not captures, relabels are not new scenes, and passing software
tests is not empirical model superiority. Publication claims require appropriately
scoped evidence; hashes establish consistency, not truth or authenticity.

## Consequences and limits

Historical samples, weights, notices and manifests are retained byte-for-byte.
Some old imitation data now require an explicit development-only legacy-target
flag. Users who relied on implicit resplitting must create a reviewed source
release instead. New training runs need an actual validation partition, unless
explicitly labelled development-only training-loss selection is requested.

A cooperative same-host lease is not protection against arbitrary external CARLA
clients. Resetting an isolated native collection session to asynchronous baseline
is deliberate, not restoration of a destroyed scene. Simulator behavior, timing,
semantic render identity, throughput and cleanup still require live acceptance.
The observer's red-light and braking metrics remain development proxies and are
not silently promoted to a closed-loop benchmark.

The acquisition methods are explicit heuristic baselines, not reproductions of
published active-learning algorithms or a claim of novelty. Independent route
geometry validation, end-to-end job dispatch, full resource instrumentation,
large-scale storage profiling, empirical comparison, human audits, rights
clearance and independent reproduction remain separate acceptance gates.
