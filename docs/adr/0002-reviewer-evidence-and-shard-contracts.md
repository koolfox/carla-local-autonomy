# ADR 0002: source-bound, portable reviewer evidence

Status: implemented for offline software validation; empirical use needs the
separate evidence gates in the reviewer runbook. Depends on ADR 0001.

## Context

A directory of files or a successful unit-test run does not establish that a
paper's table came from the cited model, source, partition or cost budget. Paths
in retained manifests commonly refer to the original computer. Deriving a Git
revision from a text marker alone also allows unrelated source bytes to borrow
that identity. Episodic source identifiers do not establish corridor separation.

## Decisions

1. Preserve original artifacts. Store an acyclic, content-addressed reference
   closure with explicit roots, file sizes, hashes and edges. Resolve packed
   references by digest rather than old absolute paths. Verify an externally
   retained index digest; local rehashing alone is not authenticity.
2. Source snapshots contain selected tracked source/test/configuration files plus
   raw Git commit/tree object proofs. Recompute the commit and tree IDs and each
   included blob ID when inspecting the archive. Do not download excluded model
   payloads or claim that excluded assets were verified. Publish files without
   replacing another writer's output, including a race after an existence check.
3. Keep software-fixture, synthetic-training, simulated-ADAS and real-world
   evidence scopes distinct. A self-declared empirical wrapper is insufficient:
   the actual registered result and producing manifest must agree. Claim predicates
   are evaluated on that result, not on the wrapper.
4. Retain canonical source partitions in voxel as well as detector/imitation
   consumers. Legacy missing partitions need an explicit development-only opt-in.
   Define geometric groups from recorded metric 3D trajectories and captured
   OpenDRIVE identities. Audit transitive corridor overlap, including segment
   interiors; unobserved gaps and exhausted comparison budgets fail closed.
5. Bound collection memory with opt-in canonical shards. Every sealed shard uses
   the existing writer and verifier; the top-level index verifies across-shard
   frame order, group partitions and all auxiliary assets. Do not resume into or
   silently promote an incomplete release. Restart explicitly without changing
   old bytes. Training adapters consume verified shards; pooled detector metrics
   remap identities and rerun COCO rather than averaging shard-level AP.
6. Add actions through the existing Operator command planner and JobManager.
   A stage receipt precedes submission, captures job/output identities, and exposes
   ambiguous outcomes. No new service, shell executor or blind retry is added.
7. Distinguish costs: local tracker CPU/wall/output bytes are measured; unavailable
   GPU/child-process totals are not imputed. Selection estimates are not actual
   costs. Complete all-attempt ledgers, including failures, bind result rows.

## Consequences and limitations

The reviewer checker verifies a fixed, documented artifact contract. It cannot
certify author honesty, reviewer independence, semantics of arbitrary prose,
completeness of undisclosed attempts, legal rights, scientific novelty, or journal
acceptance. Packed files retain original paths and possibly author metadata; they
are not anonymized or cleared for public distribution. Metric replay uses retained
predictions, not newly trained models. Full training reproduction still requires
legal datasets/weights, a resolved environment and an independent executor.

Canonical shard writing bounds active sample buffers, not the number of scene
identities, total output size or RAM used by COCO evaluation. The geometry audit
covers recorded piecewise-linear trajectories at the registered sampling gap,
not all possible roads or unrecorded simulator motion. Operator integration is a
bounded, explicit stage API; human audits remain human actions. Native changes
have only fake-actor/contract tests until live acceptance is recorded.

## Migration and rollback

Default collection remains single-release; `--shard-samples` is opt-in. Preserve
old sample/weight inventories. Use `--allow-legacy-splits` only for development of
historical voxel material, never as silent confirmation of old splits. New source
snapshot verification requires Git object proof; the previous source-only tar
archives remain historical artifacts, not retroactively certified snapshots.
Reverting code does not rewrite new or old research releases.
