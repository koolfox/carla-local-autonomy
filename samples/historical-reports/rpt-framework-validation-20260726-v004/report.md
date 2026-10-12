# CARLA Vision Framework Development Validation v0.6

Report ID: `rpt-framework-validation-20260726-v004`  
Purpose: `development`  
Authors: Hesam, CARLA Vision Research Project

## Thesis context

Traceable framework validation after adding a loopback-only operator console, compact situation and crowdedness controls, safe allow-listed workflow launch, persistent operator sessions, and an API-driven deterministic scenario-planning smoke test.

## Registered claims

- The local operator workflow launched a real read-only verification job and a real offline scenario-planning job, and both operator sessions completed successfully with registered requests, tokenized command plans, status records, logs, and SHA-256 manifests.
- The situation builder produced a strict CARLA 0.9.16 Town10HD_Opt recipe with 50 vehicles, 40 walkers, soft-rain sunset weather, construction props, and one forward-facing monocular RGB camera.
- The generated scenario plan contains one deterministic training episode and 75 planned captures while preserving the front-monocular-RGB runtime input contract and the privileged-teacher disclosure.
- The scenario-planning operator session recorded motion_authorized and destructive as false and fingerprinted the successful child scenario-plan manifest as an external input.
- All newly reported operator and scenario objects pass registered SHA-256, external-reference, unregistered-file, and applicable object-specific semantic verification.

## Verified source inventory

| source_id | label | kind | status | artifact_count | artifact_bytes | git_commit | git_dirty |
|---|---|---|---|---|---|---|---|
| rpt-framework-validation-20260726-v003 | Framework validation report v0.5 | report | success | 11 | 400154 |  | True |
| bundle-framework-development-20260726-v003 | Framework reproduction bundle v0.5 | reproduction_bundle | success | 19 | 2064909 |  | True |
| op-20260726T194948Z-verify-e6436e1d | Operator real verification session | operator_session | success | 5 | 2497 |  | True |
| scenario-plan-operator-poc-v1 | Operator crowded-situation scenario plan | scenario_plan | success | 4 | 7612 |  | True |
| op-20260726T195126Z-scenario_plan-e715b442 | Operator scenario-planning session | operator_session | success | 5 | 3059 |  | True |

## Machine-readable metrics

| label | kind | metric | value | unit | support | note |
|---|---|---|---|---|---|---|
| Framework validation report v0.5 | report | sources | 7 | objects |  |  |
| Framework validation report v0.5 | report | metric_rows | 54 | rows |  |  |
| Framework validation report v0.5 | report | artifact_inventory_rows | 69 | rows |  |  |
| Framework reproduction bundle v0.5 | reproduction_bundle | sources | 7 | objects |  |  |
| Framework reproduction bundle v0.5 | reproduction_bundle | source_files | 168 | files |  |  |
| Framework reproduction bundle v0.5 | reproduction_bundle | source_bytes | 1662106 | bytes |  |  |
| Framework reproduction bundle v0.5 | reproduction_bundle | dependency_locks | 1 | lockfiles |  |  |
| Framework reproduction bundle v0.5 | reproduction_bundle | commands | 5 | commands |  |  |
| Operator real verification session | operator_session | returncode | 0 |  |  |  |
| Operator real verification session | operator_session | job_succeeded | 1 | boolean |  | 0=false, 1=true |
| Operator real verification session | operator_session | motion_authorized | 0 | boolean |  | 0=false, 1=true |
| Operator real verification session | operator_session | destructive_operation | 0 | boolean |  | 0=false, 1=true |
| Operator real verification session | operator_session | stop_requested | 0 | boolean |  | 0=false, 1=true |
| Operator crowded-situation scenario plan | scenario_plan | recipes | 1 | recipes |  |  |
| Operator crowded-situation scenario plan | scenario_plan | episodes | 1 | episodes |  |  |
| Operator crowded-situation scenario plan | scenario_plan | planned_captures | 75 | images |  |  |
| Operator crowded-situation scenario plan | scenario_plan | partition_count.train | 1 | episodes |  |  |
| Operator scenario-planning session | operator_session | returncode | 0 |  |  |  |
| Operator scenario-planning session | operator_session | job_succeeded | 1 | boolean |  | 0=false, 1=true |
| Operator scenario-planning session | operator_session | motion_authorized | 0 | boolean |  | 0=false, 1=true |
| Operator scenario-planning session | operator_session | destructive_operation | 0 | boolean |  | 0=false, 1=true |
| Operator scenario-planning session | operator_session | stop_requested | 0 | boolean |  | 0=false, 1=true |
| Operator scenario-planning session | operator_session | expected_output_exists | 1 | boolean |  | 0=false, 1=true |

## Limitations

- All sources were created from a dirty, uncommitted workspace and are development evidence rather than confirmatory thesis experiments.
- The operator panel is a loopback-only proof of concept; it launches the existing research CLIs and opens the existing OpenCV live viewer rather than streaming simulator video into the browser.
- The planned crowded scenario was not executed because the compatible native CARLA PythonAPI is not installed on this host and native collection can mutate the simulator world.
- Scenario planning validates configuration, seed allocation, split assignment, capture counts, and provenance but does not demonstrate that every requested CARLA blueprint will spawn successfully at runtime.
- An earlier retained UI smoke session exposed an internal module-invocation defect and exited without performing verification; it is superseded by the successful real verification session included here and is not used as positive evidence.
- No detector training, locked-test evaluation, or vision-only closed-loop driving result is introduced by this operator-console validation.
- The live shadow evidence inherited through the prior report remains sequential systems evidence and cannot be used to rank detector accuracy.

## Reproducibility

Every source object was verified read-only before this report was created.
The CSV tables are canonical; this Markdown document and all plots are derived
from them. Verify the sealed report with:

```bash
uv run carla-verify reports/rpt-framework-validation-20260726-v004 --reject-unregistered
```
