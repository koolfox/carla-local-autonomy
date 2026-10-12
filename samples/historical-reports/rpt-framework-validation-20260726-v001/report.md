# CARLA Vision Framework Development Validation

Report ID: `rpt-framework-validation-20260726-v001`  
Purpose: `development`  
Authors: CARLA Vision Research Project

## Thesis context

Traceable validation of the monocular-RGB research framework before native dataset scaling, detector training, and locked testing.

## Registered claims

- The retained development artifacts pass their registered SHA-256 and semantic verification checks.
- The live perception overlay and analysis preserve exact detector-frame provenance.
- The pilot dataset preserves exact RGB and privileged instance-frame pairing and reproducible labels.
- The scenario plan is reproducible from its frozen suite, split plan, and hierarchical seed protocol.
- The common evaluator produces canonical predictions, metrics, uncertainty, plots, and qualitative evidence for RT-DETR.

## Verified source inventory

| source_id | label | kind | status | artifact_count | artifact_bytes | git_commit | git_dirty |
|---|---|---|---|---|---|---|---|
| ds-carla0916-town10-pilot-v001 | Exact-frame dataset pilot | dataset | success | 5 | 24027 |  | True |
| ds-carla0916-town10-pilot-v001-qa | Dataset QA | dataset_qa | success | 5 | 2166498 |  | True |
| scenario-plan-thesis-pilot-v1 | Deterministic scenario plan | scenario_plan | success | 4 | 82279 |  | True |
| framework-v0.2-rtdetr-smoke-analysis | Live RT-DETR runtime analysis | runtime_analysis | success | 6 | 163590 |  | True |
| eval-rtdetr-pretrained-pilot-v1 | Pretrained RT-DETR development evaluation | evaluation | success | 20 | 1719231 |  | True |

## Machine-readable metrics

| label | kind | metric | value | unit | support | note |
|---|---|---|---|---|---|---|
| Exact-frame dataset pilot | dataset | samples | 3 | images |  |  |
| Exact-frame dataset pilot | dataset | annotations | 27 | objects |  |  |
| Exact-frame dataset pilot | dataset | split_count.unassigned | 3 | images |  |  |
| Dataset QA | dataset_qa | samples | 3 | images |  |  |
| Dataset QA | dataset_qa | annotations | 27 | objects |  |  |
| Dataset QA | dataset_qa | class_count.car | 3 | objects |  |  |
| Dataset QA | dataset_qa | class_count.traffic_light | 6 | objects |  |  |
| Dataset QA | dataset_qa | class_count.traffic_sign | 18 | objects |  |  |
| Dataset QA | dataset_qa | check.checksum_entries_verified | 15 |  |  |  |
| Dataset QA | dataset_qa | check.episode_split_leakage | 0 |  |  |  |
| Dataset QA | dataset_qa | check.montage_samples | 3 |  |  |  |
| Dataset QA | dataset_qa | check.teacher_masks_reproduced | 3 |  |  |  |
| Dataset QA | dataset_qa | check.yolo_exports_reproduced | 3 |  |  |  |
| Deterministic scenario plan | scenario_plan | recipes | 7 | recipes |  |  |
| Deterministic scenario plan | scenario_plan | episodes | 23 | episodes |  |  |
| Deterministic scenario plan | scenario_plan | planned_captures | 3450 | images |  |  |
| Deterministic scenario plan | scenario_plan | partition_count.test_map_ood | 4 | episodes |  |  |
| Deterministic scenario plan | scenario_plan | partition_count.test_seen | 1 | episodes |  |  |
| Deterministic scenario plan | scenario_plan | partition_count.test_weather_ood | 2 | episodes |  |  |
| Deterministic scenario plan | scenario_plan | partition_count.train | 11 | episodes |  |  |
| Deterministic scenario plan | scenario_plan | partition_count.val_map_ood | 2 | episodes |  |  |
| Deterministic scenario plan | scenario_plan | partition_count.val_seen | 1 | episodes |  |  |
| Deterministic scenario plan | scenario_plan | partition_count.val_weather_ood | 2 | episodes |  |  |
| Live RT-DETR runtime analysis | runtime_analysis | frames | 17 | frames |  |  |
| Live RT-DETR runtime analysis | runtime_analysis | effective_source_fps | 3.2615942278349053 | fps |  |  |
| Live RT-DETR runtime analysis | runtime_analysis | detections | 78 | objects |  |  |
| Live RT-DETR runtime analysis | runtime_analysis | hazard_frame_rate | 0.0 | fraction |  |  |
| Live RT-DETR runtime analysis | runtime_analysis | pipeline_latency_median | 265.8148750051623 | ms |  |  |
| Live RT-DETR runtime analysis | runtime_analysis | pipeline_latency_p95 | 795.7242914097135 | ms |  |  |
| Live RT-DETR runtime analysis | runtime_analysis | model_latency_median | 175.8041669963859 | ms |  |  |
| Live RT-DETR runtime analysis | runtime_analysis | estimated_distance | 0.0 | m |  |  |
| Pretrained RT-DETR development evaluation | evaluation | coco_ap_50_95 | 0.2722772277227723 |  | 3 |  |
| Pretrained RT-DETR development evaluation | evaluation | coco_ap_50 | 0.3333333333333333 |  | 3 |  |
| Pretrained RT-DETR development evaluation | evaluation | coco_ap_75 | 0.3333333333333333 |  | 3 |  |
| Pretrained RT-DETR development evaluation | evaluation | precision | 1.0 |  | 3 |  |
| Pretrained RT-DETR development evaluation | evaluation | recall | 0.1111111111111111 |  | 3 |  |
| Pretrained RT-DETR development evaluation | evaluation | false_negative_rate | 0.8888888888888888 |  | 3 |  |
| Pretrained RT-DETR development evaluation | evaluation | f1 | 0.19999999999999998 |  | 3 |  |
| Pretrained RT-DETR development evaluation | evaluation | images | 3 | images |  |  |
| Pretrained RT-DETR development evaluation | evaluation | ground_truth_annotations | 27 | objects |  |  |
| Pretrained RT-DETR development evaluation | evaluation | episodes | 1 | episodes |  |  |
| Pretrained RT-DETR development evaluation | evaluation | inference_median | 317.74393749947194 | ms |  |  |
| Pretrained RT-DETR development evaluation | evaluation | inference_p95 | 325.5225812565186 | ms |  |  |

## Limitations

- All sources were created from a dirty, uncommitted workspace and are not publishable thesis experiments.
- The dataset has only three images from one unassigned episode and cannot support training or statistical accuracy claims.
- The reported RT-DETR weights are pretrained rather than trained on this CARLA ontology.
- The episode bootstrap is degenerate because the evaluation contains one episode.
- The native synchronous collector has not yet been executed against a compatible official CARLA 0.9.16 PythonAPI host.
- No vision-only driving policy has been implemented or evaluated.

## Reproducibility

Every source object was verified read-only before this report was created.
The CSV tables are canonical; this Markdown document and all plots are derived
from them. Verify the sealed report with:

```bash
uv run carla-verify reports/rpt-framework-validation-20260726-v001 --reject-unregistered
```
