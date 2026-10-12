# CARLA Vision Framework Development Validation v0.7

Report ID: `rpt-framework-validation-20260726-v005`  
Purpose: `development`  
Authors: Hesam, CARLA Vision Research Project

## Thesis context

Traceable framework validation after adding a read-only native readiness artifact, operator preflight control, a deliberately small first native dataset plan, stronger multi-episode dataset provenance verification, and a non-destructive scenario-to-training integration test.

## Registered claims

- The framework now creates a dedicated read-only native-collection preflight artifact that separates successful assessment from readiness to execute and records automated, advisory, and manual gates.
- Two real preflight assessments contacted only the CARLA read-only version and map RPCs, observed server version 0.9.16 and Town10HD_Opt, selected checksum-verified train episodes, and recorded simulator_mutated as false.
- The current Apple arm64 host is correctly classified as not ready because the official version-matched CARLA PythonAPI is not importable and the destructive world-reload and exclusive-tick-owner confirmations remain pending.
- A deliberately small native integration scenario plan contains one Town10HD train episode, moderate traffic, two construction props, and 50 planned exact-frame RGB/instance captures instead of using the 3,450-capture thesis plan for the first destructive test.
- The operator HTTP workflow launched a real native preflight as a non-destructive, non-actuating job and retained both the operator session and verified child manifest.
- The multi-episode dataset verifier now permits repeated native episode provenance while semantically checking episode IDs, partitions, sample and annotation counts, cleanup success, and asynchronous-world restoration.

## Verified source inventory

| source_id | label | kind | status | artifact_count | artifact_bytes | git_commit | git_dirty |
|---|---|---|---|---|---|---|---|
| rpt-framework-validation-20260726-v004 | Framework validation report v0.6 | report | success | 9 | 189122 |  | True |
| bundle-framework-development-20260726-v004 | Framework reproduction bundle v0.6 | reproduction_bundle | success | 15 | 2208292 |  | True |
| scenario-plan-native-integration-pilot-v1 | Minimal native integration scenario plan | scenario_plan | success | 4 | 7313 |  | True |
| native-preflight-pilot-20260726-v1 | Minimal 50-frame native pilot preflight | native_preflight | success | 5 | 16250 |  | True |
| native-preflight-ui-20260726-v1 | Operator-launched thesis-plan native preflight | native_preflight | success | 5 | 16120 |  | True |
| op-20260726t203533z-native-preflight-34a7877d | Operator native-preflight session | operator_session | success | 5 | 5156 |  | True |

## Machine-readable metrics

| label | kind | metric | value | unit | support | note |
|---|---|---|---|---|---|---|
| Framework validation report v0.6 | report | sources | 5 | objects |  |  |
| Framework validation report v0.6 | report | metric_rows | 23 | rows |  |  |
| Framework validation report v0.6 | report | artifact_inventory_rows | 44 | rows |  |  |
| Framework reproduction bundle v0.6 | reproduction_bundle | sources | 3 | objects |  |  |
| Framework reproduction bundle v0.6 | reproduction_bundle | source_files | 186 | files |  |  |
| Framework reproduction bundle v0.6 | reproduction_bundle | source_bytes | 1850789 | bytes |  |  |
| Framework reproduction bundle v0.6 | reproduction_bundle | dependency_locks | 1 | lockfiles |  |  |
| Framework reproduction bundle v0.6 | reproduction_bundle | commands | 6 | commands |  |  |
| Minimal native integration scenario plan | scenario_plan | recipes | 1 | recipes |  |  |
| Minimal native integration scenario plan | scenario_plan | episodes | 1 | episodes |  |  |
| Minimal native integration scenario plan | scenario_plan | planned_captures | 50 | images |  |  |
| Minimal native integration scenario plan | scenario_plan | partition_count.train | 1 | episodes |  |  |
| Minimal 50-frame native pilot preflight | native_preflight | selected_episodes | 1 | episodes |  |  |
| Minimal 50-frame native pilot preflight | native_preflight | planned_captures | 50 | images |  |  |
| Minimal 50-frame native pilot preflight | native_preflight | checks | 14 | checks |  |  |
| Minimal 50-frame native pilot preflight | native_preflight | checks_passed | 7 | checks |  |  |
| Minimal 50-frame native pilot preflight | native_preflight | checks_failed | 1 | checks |  |  |
| Minimal 50-frame native pilot preflight | native_preflight | checks_pending | 2 | checks |  |  |
| Minimal 50-frame native pilot preflight | native_preflight | checks_skipped | 3 | checks |  |  |
| Minimal 50-frame native pilot preflight | native_preflight | checks_warnings | 1 | checks |  |  |
| Minimal 50-frame native pilot preflight | native_preflight | socket_latency | 7.508499998948537 | ms |  |  |
| Minimal 50-frame native pilot preflight | native_preflight | read_only_rpc_latency | 22.15020899893716 | ms |  |  |
| Minimal 50-frame native pilot preflight | native_preflight | read_only | 1 | boolean |  | 0=false, 1=true |
| Minimal 50-frame native pilot preflight | native_preflight | simulator_contacted | 1 | boolean |  | 0=false, 1=true |
| Minimal 50-frame native pilot preflight | native_preflight | simulator_mutated | 0 | boolean |  | 0=false, 1=true |
| Minimal 50-frame native pilot preflight | native_preflight | automated_ready | 0 | boolean |  | 0=false, 1=true |
| Minimal 50-frame native pilot preflight | native_preflight | manual_ready | 0 | boolean |  | 0=false, 1=true |
| Minimal 50-frame native pilot preflight | native_preflight | ready_for_native_execution | 0 | boolean |  | 0=false, 1=true |
| Minimal 50-frame native pilot preflight | native_preflight | rpc_reachable | 1 | boolean |  | 0=false, 1=true |
| Minimal 50-frame native pilot preflight | native_preflight | native_pythonapi_importable | 0 | boolean |  | 0=false, 1=true |
| Operator-launched thesis-plan native preflight | native_preflight | selected_episodes | 1 | episodes |  |  |
| Operator-launched thesis-plan native preflight | native_preflight | planned_captures | 150 | images |  |  |
| Operator-launched thesis-plan native preflight | native_preflight | checks | 14 | checks |  |  |
| Operator-launched thesis-plan native preflight | native_preflight | checks_passed | 7 | checks |  |  |
| Operator-launched thesis-plan native preflight | native_preflight | checks_failed | 1 | checks |  |  |
| Operator-launched thesis-plan native preflight | native_preflight | checks_pending | 2 | checks |  |  |
| Operator-launched thesis-plan native preflight | native_preflight | checks_skipped | 3 | checks |  |  |
| Operator-launched thesis-plan native preflight | native_preflight | checks_warnings | 1 | checks |  |  |
| Operator-launched thesis-plan native preflight | native_preflight | socket_latency | 7.454582999343984 | ms |  |  |
| Operator-launched thesis-plan native preflight | native_preflight | read_only_rpc_latency | 22.60549999482464 | ms |  |  |
| Operator-launched thesis-plan native preflight | native_preflight | read_only | 1 | boolean |  | 0=false, 1=true |
| Operator-launched thesis-plan native preflight | native_preflight | simulator_contacted | 1 | boolean |  | 0=false, 1=true |
| Operator-launched thesis-plan native preflight | native_preflight | simulator_mutated | 0 | boolean |  | 0=false, 1=true |
| Operator-launched thesis-plan native preflight | native_preflight | automated_ready | 0 | boolean |  | 0=false, 1=true |
| Operator-launched thesis-plan native preflight | native_preflight | manual_ready | 0 | boolean |  | 0=false, 1=true |
| Operator-launched thesis-plan native preflight | native_preflight | ready_for_native_execution | 0 | boolean |  | 0=false, 1=true |
| Operator-launched thesis-plan native preflight | native_preflight | rpc_reachable | 1 | boolean |  | 0=false, 1=true |
| Operator-launched thesis-plan native preflight | native_preflight | native_pythonapi_importable | 0 | boolean |  | 0=false, 1=true |
| Operator native-preflight session | operator_session | returncode | 0 |  |  |  |
| Operator native-preflight session | operator_session | job_succeeded | 1 | boolean |  | 0=false, 1=true |
| Operator native-preflight session | operator_session | motion_authorized | 0 | boolean |  | 0=false, 1=true |
| Operator native-preflight session | operator_session | destructive_operation | 0 | boolean |  | 0=false, 1=true |
| Operator native-preflight session | operator_session | stop_requested | 0 | boolean |  | 0=false, 1=true |
| Operator native-preflight session | operator_session | expected_output_exists | 1 | boolean |  | 0=false, 1=true |

## Limitations

- All sources were created from a dirty, uncommitted workspace and are development evidence rather than confirmatory thesis experiments.
- No native dataset episode was executed because this Apple arm64 host has no compatible official CARLA 0.9.16 PythonAPI and SSH access to the simulator host is unavailable.
- The read-only raw RPC confirms endpoint, version, and current map but cannot substitute for the official PythonAPI checks required for Traffic Manager, synchronous world ownership, actor spawning, sensors, and cleanup.
- The two manual readiness gates are deliberately pending; no authority to reload the current world or claim exclusive world.tick ownership is inferred from a read-only preflight.
- The 50-frame native pilot is a plan, not a collected dataset; actor and prop availability, sensor phase, labels, cleanup, and QA remain unverified against the real simulator.
- The scenario-to-dataset-to-QA-to-training handoff is covered by a fake-native integration test and does not prove behavior of CARLA actor spawning or sensor delivery.
- No trained CARLA-specific checkpoint, locked-test detector result, or vision-only closed-loop driving result is introduced by this milestone.

## Reproducibility

Every source object was verified read-only before this report was created.
The CSV tables are canonical; this Markdown document and all plots are derived
from them. Verify the sealed report with:

```bash
uv run carla-verify reports/rpt-framework-validation-20260726-v005 --reject-unregistered
```
