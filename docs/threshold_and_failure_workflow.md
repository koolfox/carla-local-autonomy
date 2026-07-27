# Threshold Selection and Failure Review

Status: implemented validation-only workflow  
Runtime model input: front monocular RGB only  
Privileged lane: ground-truth scoring and post-hoc grouping only

## Why these are separate research objects

Detector families can assign very different confidence values to similar
boxes. A raw threshold shared across RT-DETR and YOLO can therefore measure
score calibration rather than useful operating performance. Threshold
selection is kept separate from evaluation so the selected value has explicit
validation lineage and cannot be silently retuned on test data.

Failure mining is also separate from evaluation. It is a development workflow
that ranks validation failures for human inspection. Human decisions form a
third immutable release. This preserves:

```text
verified validation evaluation
    -> threshold sweep and selected operating point
    -> deterministic failure queue
    -> completed human-review CSV
    -> reviewed failure catalog
```

Every arrow is a SHA-256 verified external reference. The source evaluation,
mining release, and completed review remain unchanged.

## 1. Retain low-score validation predictions

Run the canonical evaluator on validation partitions only. Its
`minimum_prediction_confidence` must be at or below the threshold-grid minimum.
Locked test, training, and `unassigned` partitions are rejected by the
selector.

```bash
uv run carla-evaluate-detector \
  --config configs/evaluation/<validation-evaluation>.json \
  --dataset datasets/<released-dataset> \
  --model-package models/<model-id> \
  --device <device> \
  --run-id <validation-evaluation-run>
```

## 2. Select and seal the threshold

```bash
uv run carla-select-threshold \
  --config configs/thresholds/rtdetr_validation_f1_v1.json \
  --evaluation-run runs/<validation-evaluation-run> \
  --runs-root runs
```

Supported preregistered objectives are:

- `maximize_f1`;
- `minimum_recall`, which minimizes false positives subject to a recall floor;
- `minimum_precision`, which maximizes recall subject to a precision floor;
- `weighted_error`, which minimizes declared FP/FN cost.

The grid can be linear or logarithmic. Values are canonicalized to 15
significant digits to prevent floating-point boundary drift. The selector
retains:

- the full threshold sweep CSV;
- selected threshold and operating counts;
- episode-bootstrap selection stability and metric intervals;
- metric/error/stability plots in PNG and SVG;
- a Markdown report, descriptor, checksum index, and tracker manifest.

The bootstrap resamples complete episodes and reruns the selection rule in
each replicate. Constraint objectives disclose infeasible replicate count.

## 3. Mine validation failures

```bash
uv run carla-mine-failures \
  --config configs/failure_mining/validation_detector_failures_v1.json \
  --evaluation-run runs/<validation-evaluation-run> \
  --threshold-selection runs/<threshold-selection-id> \
  --runs-root runs
```

The matcher first removes class-correct IoU matches, then categorizes remaining
objects without overlap:

1. wrong-class matches above evaluation IoU become `misclassification`;
2. same-class matches between localization and evaluation IoU become
   `localization`;
3. remaining ground truth becomes `false_negative`;
4. remaining predictions become `false_positive`.

Every candidate receives a deterministic failure ID, image/episode lineage,
GT and prediction boxes/classes, score, IoU, context, severity, rarity factor,
priority, and pending review status. All candidates are retained. A separate
queue applies preregistered image/episode caps and deterministic tie seeds.

Outputs include all/selected JSONL, a flat queue CSV, an unmodified review
template, plots, a visual queue, summary, report, descriptor, and checksums.

## 4. Review without mutating the mining release

Copy the released template:

```bash
cp \
  runs/<failure-mining-id>/review/review_template.csv \
  reviews/<completed-review>.csv
```

Fill every row. Allowed decisions are:

- `confirmed`;
- `rejected`;
- `label_issue`;
- `duplicate`;
- `needs_more_context`.

The reviewer must be listed in the review configuration. Notes are mandatory.
`confirmed` and `label_issue` decisions also require a proposed remedy. The
immutable queue fields—ID, rank, original type, and category—must not change.

Finalize:

```bash
uv run carla-finalize-failure-review \
  --config configs/failure_review/validation_detector_review_v1.json \
  --mining-run runs/<failure-mining-id> \
  --review-csv reviews/<completed-review>.csv \
  --runs-root runs
```

The finalizer archives the exact completed CSV, writes normalized JSONL/CSV,
confirmed and label-issue catalogs, decision/type/reviewer plots, a summary,
report, descriptor, and checksum index.

## Training boundary

The reviewed catalog references validation images. It is not a training
dataset, and it never copies validation RGB into training. A remedy must:

1. define a new scenario or label correction;
2. use a new scenario/episode identity and appropriate seed;
3. create a new immutable dataset release;
4. preserve the original dataset, model, evaluation, mining, and review;
5. retrain under a new experiment ID;
6. re-evaluate the original validation slice plus general validation metrics.

Locked test data is excluded from threshold selection, failure mining,
qualitative selection, ontology design, and remediation.
