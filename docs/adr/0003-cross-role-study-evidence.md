# ADR 0003: bind reviewer roles to a frozen source-sample inventory

Status: accepted for the offline evidence contract. Delivery and empirical work
remain tracked in Issue #144; this ADR does not certify a paper's results.

## Problem

A collection of individually valid files may describe different studies. A source
snapshot and passing tests cannot bind a table's methods, seeds and test population
to a protocol. An unrelated route audit or annotation review may pass its own
checks. A global cost log can contain attempts excluded from all comparison rows.

## Decision

Keep the fixed reviewer role catalog and its individual semantic validators. Add
an independent `study_linkage` gate for both empirical modes. `offline_fixture`
continues to validate software contracts and never becomes empirical evidence.

`carla-reviewer-evidence freeze-split` publishes schema **1.1** from verified
canonical releases or their sealed shards. It reads every sample's original
partition, scene root and RGB digest. Sample identity is the pair of canonical
manifest SHA-256 and sample ID, not a local shard basename. The inventory records
original release descriptors, canonical manifests, samples, group assignments and
a digest of the sorted test-role sample rows. It rejects duplicate sources,
cross-partition scene/RGB overlap, unassigned samples and exhausted budgets.
`verify_split` reopens the sources and recomputes this inventory. Publication is a
new-file-only operation; concurrent writers cannot replace the winner's output.
No operation repartitions, relabels or edits a source release.

The empirical gate requires:

- the selected study protocol and its verified 1.1 split inventory;
- the same unique methods and training seeds in the selected results matrix,
  that inventory's test digest, and the exact primary metric field;
- geometry source coverage of every frozen canonical dataset manifest, and
  source membership for all result-producer dataset references;
- manuscript claim IDs matching the selected registry, and nonempty unique table
  IDs referencing the selected result bytes;
- annotation-audit sample/group identities belonging unambiguously to the frozen
  study (use `dataset_manifest_sha256` to disambiguate repeated local IDs);
- exact one-to-one allocation of global cost events to the per-producer ledgers,
  comparing logical events independently of each ledger's hash-chain position.

An extra global event is not free work: empirical linkage fails until a registered
cost policy accounts for it in a run ledger. Shared-cost allocation across methods
is not inferred, duplicated or silently discounted by this checker.

## Compatibility and limits

Legacy schema 1.0 group-only manifests remain usable by software development
planners. They are insufficient for empirical reviewer linkage. Create a new 1.1
inventory and a new protocol/matrix/producer generation rather than editing a
completed run. Existing historical artifacts stay unchanged.

This gate checks **recorded relationships**, not their scientific truth. It does
not prove that each producer actually evaluated all claimed samples, that a model
never saw held-out data, that reviewers were independent humans, that a paper's
rendered numbers equal its table inventory, or that prior test access never
occurred. Saved-prediction recomputation, producer-selected-sample attestations,
independent reproduction and human review remain distinct requirements. A valid
method can lose its comparison; this gate never chooses an effect direction.

The present matrix binds one registered generation condition/learner/regime at a
time. It is not a complete multidimensional run-lattice validator. The selected
protocol may require additional cells; verify that coverage separately before
submission. Source grouping and observed-corridor geometry do not prove separation
of all possible roads or unrecorded motion.

Freezing may include train, acquisition, validation and test *metadata*, but must
not be used to disclose test content to adaptive acquisition. Access control and
preregistration timestamps need an external process. CLI inventory inspection
necessarily reads verified source bytes and should be run by the data custodian.

Budgets cap inventory count and serialized JSON size; the existing verifier may
still load a whole canonical shard. Use bounded shards for large releases and
measure memory/time on representative data. This is not unlimited streaming.

Portable closure verification preserves the original inventory and source bytes.
It is byte/reference verification, not automatic path rebasing or a rerun of all
empirical semantics after moving the bundle. No publication, license grant,
anonymization or journal acceptance is performed.

## Verification

`tests/test_reviewer_study_links.py` includes immutable publication and race tests,
canonical/sharded identity checks, malformed/repartitioned inventories, a real
planner boundary, table/claim/audit/protocol mismatches, changed/unallocated cost
events and relocated byte closure. All data are explicitly software fixtures.
