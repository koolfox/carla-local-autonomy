# CARLA Vision Framework Development Validation v0.9

Report ID: `rpt-framework-validation-20260727-v008`  
Purpose: `development`  
Authors: Hesam, CARLA Vision Research Project

## Thesis context

Traceable v0.9 thesis-framework handoff after implementing repository-wide point-in-time evidence discovery, deterministic semantic reconstruction, secure manifest-only operator browsing, and report/reproduction integration while preserving the simulator and scientific claim boundaries.

## Registered claims

- The framework now creates a deterministic point-in-time evidence index across the seven canonical research roots without contacting CARLA, mutating the simulator, or modifying source research objects.
- Every discovered immediate-child source is retained even when strict deep verification fails; the release includes normalized failures, registered-artifact inventory, canonical JSON/CSV/Markdown, PNG/SVG plots, checksums, and source-manifest fingerprints.
- Dedicated semantic verification reconstructs the recorded source set, rejects registry tampering and recorded-source drift, excludes evidence registries to prevent recursive snapshots, and intentionally ignores objects created after the snapshot.
- The loopback operator UI now provides a searchable manifest-driven Evidence Explorer for every canonical root, exposes only registered contained artifacts, labels catalogue data as not checked, and hands selected objects to the existing verifier without guessing artifact paths.
- Evidence-registry sources are accepted by the report and standalone-reproduction contracts, so registry counts and safety booleans can flow into thesis evidence without manual metric transcription.
- The v0.9 source snapshot, tests, documentation, retained report, and reproduction commands preserve the distinction between development infrastructure and unperformed real CARLA collection or vision-only driving.

## Verified source inventory

| source_id | label | kind | status | artifact_count | artifact_bytes | git_commit | git_dirty |
|---|---|---|---|---|---|---|---|
| rpt-framework-validation-20260727-v007 | Final framework validation report v0.8 | report | success | 9 | 205720 |  | True |
| bundle-framework-development-20260727-v007 | Final framework reproduction bundle v0.8 | reproduction_bundle | success | 17 | 2561816 |  | True |
| evidence-index-workspace-20260727-v001 | Sealed workspace evidence index v0.9 | evidence_registry | success | 12 | 471757 |  | True |
| native-host-kit-pilot-20260727-v003 | Portable native 50-frame pilot host kit | native_host_kit | success | 7 | 1383892 |  | True |

## Machine-readable metrics

| label | kind | metric | value | unit | support | note |
|---|---|---|---|---|---|---|
| Final framework validation report v0.8 | report | sources | 5 | objects |  |  |
| Final framework validation report v0.8 | report | metric_rows | 43 | rows |  |  |
| Final framework validation report v0.8 | report | artifact_inventory_rows | 42 | rows |  |  |
| Final framework reproduction bundle v0.8 | reproduction_bundle | sources | 5 | objects |  |  |
| Final framework reproduction bundle v0.8 | reproduction_bundle | source_files | 217 | files |  |  |
| Final framework reproduction bundle v0.8 | reproduction_bundle | source_bytes | 2168594 | bytes |  |  |
| Final framework reproduction bundle v0.8 | reproduction_bundle | dependency_locks | 1 | lockfiles |  |  |
| Final framework reproduction bundle v0.8 | reproduction_bundle | commands | 10 | commands |  |  |
| Sealed workspace evidence index v0.9 | evidence_registry | objects | 47 | objects |  |  |
| Sealed workspace evidence index v0.9 | evidence_registry | source_manifests | 47 | objects |  |  |
| Sealed workspace evidence index v0.9 | evidence_registry | verified_objects | 46 | objects |  |  |
| Sealed workspace evidence index v0.9 | evidence_registry | verification_failures | 1 | objects |  |  |
| Sealed workspace evidence index v0.9 | evidence_registry | registered_artifacts | 437 | artifacts |  |  |
| Sealed workspace evidence index v0.9 | evidence_registry | failed_source_tree_entries | 10 | entries |  |  |
| Sealed workspace evidence index v0.9 | evidence_registry | failed_source_tree_files | 8 | files |  |  |
| Sealed workspace evidence index v0.9 | evidence_registry | failed_source_tree_bytes | 1388800 | bytes |  |  |
| Sealed workspace evidence index v0.9 | evidence_registry | excluded_registries | 0 | objects |  |  |
| Sealed workspace evidence index v0.9 | evidence_registry | carla_contacted | 0 | boolean |  |  |
| Sealed workspace evidence index v0.9 | evidence_registry | simulator_mutated | 0 | boolean |  |  |
| Sealed workspace evidence index v0.9 | evidence_registry | source_objects_modified | 0 | boolean |  |  |
| Portable native 50-frame pilot host kit | native_host_kit | payload_files | 114 | files |  |  |
| Portable native 50-frame pilot host kit | native_host_kit | payload_source_files | 95 | files |  |  |
| Portable native 50-frame pilot host kit | native_host_kit | payload_bytes | 1322086 | bytes |  |  |
| Portable native 50-frame pilot host kit | native_host_kit | selected_episodes | 1 | episodes |  |  |
| Portable native 50-frame pilot host kit | native_host_kit | planned_captures | 50 | images |  |  |
| Portable native 50-frame pilot host kit | native_host_kit | requirements_packages | 14 | packages |  |  |
| Portable native 50-frame pilot host kit | native_host_kit | carla_wheel_hashes | 6 | hashes |  |  |
| Portable native 50-frame pilot host kit | native_host_kit | requirements_hash_pinned | 1 | boolean |  |  |
| Portable native 50-frame pilot host kit | native_host_kit | preflight_read_only | 1 | boolean |  |  |
| Portable native 50-frame pilot host kit | native_host_kit | ready_preflight_required | 1 | boolean |  |  |
| Portable native 50-frame pilot host kit | native_host_kit | confirmation_token_required | 1 | boolean |  |  |
| Portable native 50-frame pilot host kit | native_host_kit | build_contacted_simulator | 0 | boolean |  |  |
| Portable native 50-frame pilot host kit | native_host_kit | build_mutated_simulator | 0 | boolean |  |  |

## Limitations

- All sources were created from a dirty, uncommitted workspace and are development evidence rather than confirmatory thesis experiments.
- The evidence index is workspace-bound: its canonical absolute workspace path and recorded source objects must remain available for semantic verification.
- The registry inventories source manifests and failed-source fingerprints rather than embedding every source payload; portable archival remains the responsibility of the standalone reproduction bundle.
- The live Evidence Explorer reports manifest declarations and current file availability only; operators must run the separate verifier before relying on hashes or semantics.
- Previously retained invalid historical objects remain visible as failures and are not rewritten or silently omitted.
- A mutable database, cross-object cycle graph, signed release, supersession policy, PDF/LaTeX thesis renderer, and independent release signoff are not introduced by this MVP milestone.
- No native CARLA episode, new dataset, real training run, locked-test benchmark, or vision-only closed-loop control result is introduced by this milestone.

## Reproducibility

Every source object was verified read-only before this report was created.
The CSV tables are canonical; this Markdown document and all plots are derived
from them. Verify the sealed report with:

```bash
uv run carla-verify reports/rpt-framework-validation-20260727-v008 --reject-unregistered
```
