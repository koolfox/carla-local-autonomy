# CARLA Vision Framework Development Validation v0.5

Report ID: `rpt-framework-validation-20260726-v003`  
Purpose: `development`  
Authors: Hesam, CARLA Vision Research Project

## Thesis context

Traceable framework validation after adding a modular vision-policy contract, non-actuating live shadow execution, sequential multi-model orchestration, semantic safety verification, live videos, and source-linked latency and throughput analyses.

## Registered claims

- The live shadow matrix and both child runs pass registered SHA-256, external-reference, unregistered-file, and object-specific semantic verification.
- RT-DETR-L and YOLO26n executed sequentially against the configured CARLA 0.9.16 Town10HD_Opt vehicle and front RGB camera while the privileged teacher was the only actuation source.
- Every logged policy observation passed the front-monocular-RGB structural input audit, direct CARLA objects were not exposed, and vision-policy actuation authorization remained false.
- The two twelve-second child runs produced checksum-tracked overlay videos, terminal overlays, detections, policy proposals, summaries, and immutable source-linked post-hoc analyses.
- Observed throughput and latency are extracted from verified frame logs without manual metric transcription and are explicitly reported as systems measurements rather than detector-accuracy comparisons.

## Verified source inventory

| source_id | label | kind | status | artifact_count | artifact_bytes | git_commit | git_dirty |
|---|---|---|---|---|---|---|---|
| rpt-framework-validation-20260726-v002 | Framework validation report v0.4 | report | success | 11 | 313853 |  | True |
| bundle-framework-development-20260726-v002 | Framework reproduction bundle v0.4 | reproduction_bundle | success | 23 | 1936814 |  | True |
| shadow-rtdetr-yolo26-live-v1 | RT-DETR and YOLO26 live shadow matrix | shadow_matrix | success | 11 | 41992 |  | True |
| shadow-rtdetr-yolo26-live-v1--rtdetr-l-hazard-stop-r00 | RT-DETR-L RGB policy shadow run | vision_shadow | success | 6 | 2895291 |  | True |
| shadow-rtdetr-yolo26-live-v1--yolo26n-hazard-stop-r00 | YOLO26n RGB policy shadow run | vision_shadow | success | 6 | 3416913 |  | True |
| shadow-rtdetr-yolo26-live-v1--rtdetr-l-hazard-stop-r00-analysis | RT-DETR-L live systems analysis | runtime_analysis | success | 6 | 317250 |  | True |
| shadow-rtdetr-yolo26-live-v1--yolo26n-hazard-stop-r00-analysis | YOLO26n live systems analysis | runtime_analysis | success | 6 | 349371 |  | True |

## Machine-readable metrics

| label | kind | metric | value | unit | support | note |
|---|---|---|---|---|---|---|
| Framework validation report v0.4 | report | sources | 7 | objects |  |  |
| Framework validation report v0.4 | report | metric_rows | 65 | rows |  |  |
| Framework validation report v0.4 | report | artifact_inventory_rows | 82 | rows |  |  |
| Framework reproduction bundle v0.4 | reproduction_bundle | sources | 11 | objects |  |  |
| Framework reproduction bundle v0.4 | reproduction_bundle | source_files | 147 | files |  |  |
| Framework reproduction bundle v0.4 | reproduction_bundle | source_bytes | 1506421 | bytes |  |  |
| Framework reproduction bundle v0.4 | reproduction_bundle | dependency_locks | 1 | lockfiles |  |  |
| Framework reproduction bundle v0.4 | reproduction_bundle | commands | 5 | commands |  |  |
| RT-DETR and YOLO26 live shadow matrix | shadow_matrix | planned_runs | 2 | runs |  |  |
| RT-DETR and YOLO26 live shadow matrix | shadow_matrix | successful_runs | 2 | runs |  |  |
| RT-DETR and YOLO26 live shadow matrix | shadow_matrix | failed_runs | 0 | runs |  |  |
| RT-DETR and YOLO26 live shadow matrix | shadow_matrix | vision_policy_actuation_authorized | 0 | boolean |  | 0=false, 1=true |
| RT-DETR-L RGB policy shadow run | vision_shadow | elapsed | 12.116897125000833 | s |  |  |
| RT-DETR-L RGB policy shadow run | vision_shadow | distance_travelled | 15.965974588928374 | m |  |  |
| RT-DETR-L RGB policy shadow run | vision_shadow | max_simulator_speed | 2.4611802950136425 | m/s |  |  |
| RT-DETR-L RGB policy shadow run | vision_shadow | perception_submitted | 78 | frames |  |  |
| RT-DETR-L RGB policy shadow run | vision_shadow | perception_processed | 74 | frames |  |  |
| RT-DETR-L RGB policy shadow run | vision_shadow | perception_dropped_before_inference | 3 | frames |  |  |
| RT-DETR-L RGB policy shadow run | vision_shadow | recording_written | 73 | frames |  |  |
| RT-DETR-L RGB policy shadow run | vision_shadow | policy_proposals | 73 | proposals |  |  |
| RT-DETR-L RGB policy shadow run | vision_shadow | policy_throttle_proposals | 73 | proposals |  |  |
| RT-DETR-L RGB policy shadow run | vision_shadow | policy_braking_proposals | 0 | proposals |  |  |
| RT-DETR-L RGB policy shadow run | vision_shadow | policy_latency_median | 0.01504199462942779 | ms |  |  |
| RT-DETR-L RGB policy shadow run | vision_shadow | policy_latency_p95 | 0.029024996911175535 | ms |  |  |
| RT-DETR-L RGB policy shadow run | vision_shadow | vision_policy_actuation_authorized | 0 | boolean |  | 0=false, 1=true |
| YOLO26n RGB policy shadow run | vision_shadow | elapsed | 12.014783791993978 | s |  |  |
| YOLO26n RGB policy shadow run | vision_shadow | distance_travelled | 20.748599638921906 | m |  |  |
| YOLO26n RGB policy shadow run | vision_shadow | max_simulator_speed | 2.499440191556413 | m/s |  |  |
| YOLO26n RGB policy shadow run | vision_shadow | perception_submitted | 104 | frames |  |  |
| YOLO26n RGB policy shadow run | vision_shadow | perception_processed | 104 | frames |  |  |
| YOLO26n RGB policy shadow run | vision_shadow | perception_dropped_before_inference | 0 | frames |  |  |
| YOLO26n RGB policy shadow run | vision_shadow | recording_written | 103 | frames |  |  |
| YOLO26n RGB policy shadow run | vision_shadow | policy_proposals | 103 | proposals |  |  |
| YOLO26n RGB policy shadow run | vision_shadow | policy_throttle_proposals | 103 | proposals |  |  |
| YOLO26n RGB policy shadow run | vision_shadow | policy_braking_proposals | 0 | proposals |  |  |
| YOLO26n RGB policy shadow run | vision_shadow | policy_latency_median | 0.012832999345846474 | ms |  |  |
| YOLO26n RGB policy shadow run | vision_shadow | policy_latency_p95 | 0.014499993994832039 | ms |  |  |
| YOLO26n RGB policy shadow run | vision_shadow | vision_policy_actuation_authorized | 0 | boolean |  | 0=false, 1=true |
| RT-DETR-L live systems analysis | runtime_analysis | frames | 73 | frames |  |  |
| RT-DETR-L live systems analysis | runtime_analysis | effective_source_fps | 5.947617105228151 | fps |  |  |
| RT-DETR-L live systems analysis | runtime_analysis | detections | 520 | objects |  |  |
| RT-DETR-L live systems analysis | runtime_analysis | hazard_frame_rate | 0.0 | fraction |  |  |
| RT-DETR-L live systems analysis | runtime_analysis | pipeline_latency_median | 202.20250000420492 | ms |  |  |
| RT-DETR-L live systems analysis | runtime_analysis | pipeline_latency_p95 | 286.2106999993557 | ms |  |  |
| RT-DETR-L live systems analysis | runtime_analysis | model_latency_median | 152.86149999883492 | ms |  |  |
| RT-DETR-L live systems analysis | runtime_analysis | estimated_distance | 15.734784893266989 | m |  |  |
| YOLO26n live systems analysis | runtime_analysis | frames | 103 | frames |  |  |
| YOLO26n live systems analysis | runtime_analysis | effective_source_fps | 8.641475646684011 | fps |  |  |
| YOLO26n live systems analysis | runtime_analysis | detections | 129 | objects |  |  |
| YOLO26n live systems analysis | runtime_analysis | hazard_frame_rate | 0.0 | fraction |  |  |
| YOLO26n live systems analysis | runtime_analysis | pipeline_latency_median | 42.823000010685064 | ms |  |  |
| YOLO26n live systems analysis | runtime_analysis | pipeline_latency_p95 | 59.524500297266044 | ms |  |  |
| YOLO26n live systems analysis | runtime_analysis | model_latency_median | 25.460542005021125 | ms |  |  |
| YOLO26n live systems analysis | runtime_analysis | estimated_distance | 20.539963356527117 | m |  |  |

## Limitations

- All sources were created from a dirty, uncommitted workspace and are development evidence rather than confirmatory thesis experiments.
- The two live runs were sequential and observed different route positions and transient scenes, so detection counts and class frequencies must not be used to rank model accuracy.
- The detectors use pretrained COCO weights and different development confidence thresholds rather than checkpoints and thresholds selected on a released CARLA validation partition.
- The hazard-stop policy is a deliberately simple non-learned shadow baseline; it proposed no braking events in these short runs and has not demonstrated safe driving behavior.
- Teacher motion uses privileged simulator pose and speed, remains outside the runtime sensor contract, and cannot be presented as vision-only driving.
- Structural observation isolation prevents accidental privileged inputs but does not sandbox malicious custom policy code.
- Each child ran for only twelve seconds on one map, weather state, vehicle, route segment, and hardware configuration.
- The native synchronous collector and real multi-seed training protocol remain gated on a compatible official CARLA 0.9.16 PythonAPI host and a released dataset.

## Reproducibility

Every source object was verified read-only before this report was created.
The CSV tables are canonical; this Markdown document and all plots are derived
from them. Verify the sealed report with:

```bash
uv run carla-verify reports/rpt-framework-validation-20260726-v003 --reject-unregistered
```
