# CARLA Vision Framework Development Validation v0.8 Final

Report ID: `rpt-framework-validation-20260727-v007`  
Purpose: `development`  
Authors: Hesam, CARLA Vision Research Project

## Thesis context

Final traceable implementation handoff after deterministic portable collection packaging, explicit preflight binding, immutable safety gates, operator-catalog discovery, and closure of the only warning exposed by independent clean-workspace reproduction.

## Registered claims

- The framework now builds a deterministic self-contained native-host kit for the bounded 50-frame pilot without contacting or mutating CARLA.
- The final kit carries the exact verified scenario plan, project collection sources, platform-specific CPython 3.12 lock files with required hashes, guarded Windows and Linux scripts, and complete payload inventories.
- Collection cannot start through the kit scripts unless a semantically verified ready preflight exactly matches the kit and the operator supplies the literal world-reload and exclusive-tick confirmation token.
- The kit archive is independently reconstructed and semantically checked for deterministic order, timestamps, permissions, content hashes, Python compilation, scenario-plan recomputation, dependency pins, and safety gates.
- The operator catalog discovers native-host kits as normal research objects while live MessagePack capture remains independent of native PythonAPI availability.
- The Python 3.13 training-log teardown warning found by independent reproduction is covered by a regression test and closed in the final source snapshot.

## Verified source inventory

| source_id | label | kind | status | artifact_count | artifact_bytes | git_commit | git_dirty |
|---|---|---|---|---|---|---|---|
| rpt-framework-validation-20260727-v006 | Pre-final framework validation report v0.8 | report | success | 9 | 192304 |  | True |
| bundle-framework-development-20260727-v006 | Pre-final framework reproduction bundle v0.8 | reproduction_bundle | success | 17 | 2550952 |  | True |
| native-host-kit-pilot-20260727-v003 | Final portable native 50-frame pilot host kit | native_host_kit | success | 7 | 1383892 |  | True |
| scenario-plan-native-integration-pilot-v1 | Minimal native integration scenario plan | scenario_plan | success | 4 | 7313 |  | True |
| native-preflight-pilot-20260726-v1 | Real endpoint read-only native preflight | native_preflight | success | 5 | 16250 |  | True |

## Machine-readable metrics

| label | kind | metric | value | unit | support | note |
|---|---|---|---|---|---|---|
| Pre-final framework validation report v0.8 | report | sources | 5 | objects |  |  |
| Pre-final framework validation report v0.8 | report | metric_rows | 43 | rows |  |  |
| Pre-final framework validation report v0.8 | report | artifact_inventory_rows | 43 | rows |  |  |
| Pre-final framework reproduction bundle v0.8 | reproduction_bundle | sources | 5 | objects |  |  |
| Pre-final framework reproduction bundle v0.8 | reproduction_bundle | source_files | 214 | files |  |  |
| Pre-final framework reproduction bundle v0.8 | reproduction_bundle | source_bytes | 2159219 | bytes |  |  |
| Pre-final framework reproduction bundle v0.8 | reproduction_bundle | dependency_locks | 1 | lockfiles |  |  |
| Pre-final framework reproduction bundle v0.8 | reproduction_bundle | commands | 10 | commands |  |  |
| Final portable native 50-frame pilot host kit | native_host_kit | payload_files | 114 | files |  |  |
| Final portable native 50-frame pilot host kit | native_host_kit | payload_source_files | 95 | files |  |  |
| Final portable native 50-frame pilot host kit | native_host_kit | payload_bytes | 1322086 | bytes |  |  |
| Final portable native 50-frame pilot host kit | native_host_kit | selected_episodes | 1 | episodes |  |  |
| Final portable native 50-frame pilot host kit | native_host_kit | planned_captures | 50 | images |  |  |
| Final portable native 50-frame pilot host kit | native_host_kit | requirements_packages | 14 | packages |  |  |
| Final portable native 50-frame pilot host kit | native_host_kit | carla_wheel_hashes | 6 | hashes |  |  |
| Final portable native 50-frame pilot host kit | native_host_kit | requirements_hash_pinned | 1 | boolean |  |  |
| Final portable native 50-frame pilot host kit | native_host_kit | preflight_read_only | 1 | boolean |  |  |
| Final portable native 50-frame pilot host kit | native_host_kit | ready_preflight_required | 1 | boolean |  |  |
| Final portable native 50-frame pilot host kit | native_host_kit | confirmation_token_required | 1 | boolean |  |  |
| Final portable native 50-frame pilot host kit | native_host_kit | build_contacted_simulator | 0 | boolean |  |  |
| Final portable native 50-frame pilot host kit | native_host_kit | build_mutated_simulator | 0 | boolean |  |  |
| Minimal native integration scenario plan | scenario_plan | recipes | 1 | recipes |  |  |
| Minimal native integration scenario plan | scenario_plan | episodes | 1 | episodes |  |  |
| Minimal native integration scenario plan | scenario_plan | planned_captures | 50 | images |  |  |
| Minimal native integration scenario plan | scenario_plan | partition_count.train | 1 | episodes |  |  |
| Real endpoint read-only native preflight | native_preflight | selected_episodes | 1 | episodes |  |  |
| Real endpoint read-only native preflight | native_preflight | planned_captures | 50 | images |  |  |
| Real endpoint read-only native preflight | native_preflight | checks | 14 | checks |  |  |
| Real endpoint read-only native preflight | native_preflight | checks_passed | 7 | checks |  |  |
| Real endpoint read-only native preflight | native_preflight | checks_failed | 1 | checks |  |  |
| Real endpoint read-only native preflight | native_preflight | checks_pending | 2 | checks |  |  |
| Real endpoint read-only native preflight | native_preflight | checks_skipped | 3 | checks |  |  |
| Real endpoint read-only native preflight | native_preflight | checks_warnings | 1 | checks |  |  |
| Real endpoint read-only native preflight | native_preflight | socket_latency | 7.508499998948537 | ms |  |  |
| Real endpoint read-only native preflight | native_preflight | read_only_rpc_latency | 22.15020899893716 | ms |  |  |
| Real endpoint read-only native preflight | native_preflight | read_only | 1 | boolean |  | 0=false, 1=true |
| Real endpoint read-only native preflight | native_preflight | simulator_contacted | 1 | boolean |  | 0=false, 1=true |
| Real endpoint read-only native preflight | native_preflight | simulator_mutated | 0 | boolean |  | 0=false, 1=true |
| Real endpoint read-only native preflight | native_preflight | automated_ready | 0 | boolean |  | 0=false, 1=true |
| Real endpoint read-only native preflight | native_preflight | manual_ready | 0 | boolean |  | 0=false, 1=true |
| Real endpoint read-only native preflight | native_preflight | ready_for_native_execution | 0 | boolean |  | 0=false, 1=true |
| Real endpoint read-only native preflight | native_preflight | rpc_reachable | 1 | boolean |  | 0=false, 1=true |
| Real endpoint read-only native preflight | native_preflight | native_pythonapi_importable | 0 | boolean |  | 0=false, 1=true |

## Limitations

- All sources were created from a dirty, uncommitted workspace and are development evidence rather than confirmatory thesis experiments.
- The portable kit targets CPython 3.12 on Windows or Linux x86-64 and has not been bootstrapped on the eventual execution host.
- No native episode was executed; world reload, actor spawning, synchronous sensor delivery, cleanup, and restoration still require manual acceptance against CARLA.
- The retained real preflight remains not ready on Apple arm64 because the official CARLA PythonAPI is unavailable there and the two manual mutation gates are pending.
- The dependency locks were resolved and hash checked, but external package availability at future bootstrap time remains outside this artifact.
- The kit deliberately contains no CARLA server binary, model weights, secrets, prior datasets, or full 23-episode thesis plan.
- No CARLA-specific RT-DETR training, locked-test evaluation, or vision-only closed-loop driving result is introduced by this milestone.

## Reproducibility

Every source object was verified read-only before this report was created.
The CSV tables are canonical; this Markdown document and all plots are derived
from them. Verify the sealed report with:

```bash
uv run carla-verify reports/rpt-framework-validation-20260727-v007 --reject-unregistered
```
