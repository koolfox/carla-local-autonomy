# Paired Detector Replay

Status: implemented development workflow  
Runtime input contract: front monocular RGB only  
Schema versions: replay configuration `1.0`, replay release `1.0`

## Purpose

Paired replay removes input-order ambiguity from detector comparisons. Every
model receives the same verified RGB files in the same order. A replay is an
orchestrator, not a replacement for the canonical evaluator: each model keeps
its own complete child evaluation, while the parent release adds paired
per-image and per-episode comparisons.

Supported model sources are:

- a verified immutable model package;
- a loose RT-DETR or YOLO weight for development;
- a custom `module:callable` detector implementing the common protocol.

Confirmatory replay rejects loose weights and custom development
specifications. Every model must be a verified package, the Git gate must be
clean, and locked test partitions still require explicit acknowledgement.

## Run

```bash
uv run carla-replay-models \
  --config configs/replay/rtdetr_yolo26_development_v1.json \
  --dataset datasets/ds-carla0916-town10-pilot-v001 \
  --evaluation-config configs/evaluation/model_comparison_development_v1.json \
  --runs-root runs
```

The configuration freezes:

- replay identity, purpose, title, and master seed;
- reference model and execution order;
- model package or adapter, weight digest, device, input size, and options;
- episode-bootstrap replicate count and confidence;
- number of first images excluded from stable latency statistics;
- deterministic qualitative-panel size.

The evaluation configuration separately freezes partitions, minimum retained
score, operating score, matching IoU, ontology aliases, calibration, and COCO
settings. Replay and evaluation purposes must match.

## Output contract

```text
runs/<replay-id>/
├── manifest.json
├── replay.json
├── replay.md
├── resolved_replay_config.json
├── child_runs.json
├── paired_bootstrap.json
├── checksums.sha256
├── tables/
│   ├── sample_order.csv
│   ├── aggregate_metrics.csv
│   ├── per_image_metrics.csv
│   ├── paired_differences.csv
│   └── disagreements.csv
├── plots/
│   ├── accuracy_comparison.{png,svg}
│   ├── latency_comparison.{png,svg}
│   ├── operating_outcomes.{png,svg}
│   └── error_heatmap.{png,svg}
└── qualitative/
    └── paired_disagreements.png
```

Every child evaluation is a sibling run with its own manifest, canonical COCO
predictions, frame log, tables, bootstrap, plots, and qualitative panel. The
parent references each child manifest by absolute path, SHA-256, and byte size.

`carla-verify --reject-unregistered` checks:

- parent and child tracker envelopes and all registered file fingerprints;
- external dataset, configuration, model-weight/package, and child references;
- the sorted replay checksum index;
- model identities and order;
- contiguous sample order and exact RGB hashes;
- equality of every child frame order with the preregistered order;
- aggregate, per-image, pair, disagreement, and bootstrap table structure;
- confirmatory clean-Git and model-package requirements.

## Statistical meaning

Accuracy and error deltas use candidate minus reference. The resampling unit is
the full episode, never an individual frame. Precision, recall, F1, FP, FN,
error burden, and mean latency receive paired episode-cluster bootstrap
intervals. Final work requires at least 10,000 preregistered replicates and
enough independent episodes; a one-episode interval is marked degenerate and
is descriptive only.

The first configured images remain in accuracy metrics but are excluded from
stable latency summaries. If those warm-up images consume an entire episode,
latency bootstrap samples only episodes with eligible measurements while
accuracy still uses every episode.

## Threshold policy

Confidence scores from different model families are not necessarily
calibrated. A common raw threshold can therefore compare score calibration
rather than detector capacity. The development pilot demonstrates this:
YOLO26n's maximum score on the first pilot image is about `0.0403`, below the
preregistered retained-prediction cutoff of `0.05`, while RT-DETR emits many
higher-scored candidates.

The final protocol is:

1. retain predictions at a sufficiently low preregistered score;
2. choose a model-specific operating threshold on validation data only, using
   a preregistered objective or safety constraint;
3. package the selected threshold with the model;
4. freeze it before any locked-test replay;
5. report threshold-free COCO AP as well as frozen operating-point metrics;
6. never retune from test failures or qualitative test inspection.

The existing three-frame replay is deliberately retained as development
evidence. It validates orchestration, pairing, provenance, latency collection,
and calibration diagnostics; it does not rank RT-DETR and YOLO26.
