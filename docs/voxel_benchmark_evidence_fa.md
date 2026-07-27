# Benchmark و Evidence مدل Voxel

این مرحله فاصله بین آموزش Issue #7 و readiness gate در Issue #8 را پر می‌کند. دو فرمان جدید اضافه شده‌اند:

```text
carla-benchmark-voxel
carla-build-voxel-readiness-evidence
```

هیچ‌کدام فرمان خودرو را اعمال نمی‌کنند.

## ۱. تولید Run ارزیابی Teacher + Predictor

خودروی `hero` و ترافیک را اجرا کن، سپس مدل را همراه سنسورهای privileged teacher ارزیابی کن:

```bash
uv run carla-voxel-test \
  --mode teacher \
  --role-name hero \
  --predictor-factory carla_vision.voxel.model_examples.temporal_baseline:create_predictor \
  --predictor-checkpoint models/temporal-voxel-v001/checkpoint_best.pt \
  --predictor-device cuda \
  --history-frames 4 \
  --forecast-horizons 0,0.5,1,2 \
  --frames 600 \
  --output runs/voxel-eval-town10-route-a-s17
```

همین کار را روی route، seed، ترافیک و آب‌وهوای متفاوت تکرار کن. Depth و Semantic فقط Teacher هستند؛ predictor همچنان فقط RGB history دریافت می‌کند.

## ۲. Benchmark فعلی و آینده

```bash
uv run carla-benchmark-voxel \
  --run runs/voxel-eval-town10-route-a-s17 \
  --run runs/voxel-eval-town10-route-b-s23 \
  --run runs/voxel-eval-town10-route-c-s31 \
  --threshold 0.5 \
  --uncertainty-band 0.08 \
  --output reports/voxel-benchmark-v001.json
```

Benchmark prediction هر anchor را با teacher frame متناظر همان horizon مقایسه می‌کند. برای مثال prediction یک‌ثانیه آینده با teacher occupancy فریم آینده مقایسه می‌شود، نه occupancy فعلی.

### Metricهای هر Horizon

- occupied IoU؛
- free-space precision؛
- free-space recall؛
- Brier score؛
- uncertain voxel fraction؛
- semantic mIoU، در صورت وجود semantic logits؛
- empty-space occupied IoU؛
- persistence occupied IoU؛
- اختلاف IoU مدل با empty-space و persistence.

### Baseline Persistence

برای horizon آینده، occupancy Teacher فریم anchor بدون حرکت تکرار می‌شود و با Teacher آینده مقایسه می‌شود. این baseline ساده بررسی می‌کند آیا مدل واقعاً تغییرات زمانی را یاد گرفته یا فقط جهان فعلی را کپی می‌کند.

برای پذیرش Issue #7 حداقل باید این مقدار مثبت باشد و بهتر است حاشیه‌ی معنادار داشته باشد:

```json
{
  "future_model_minus_persistence_iou_1s": 0.02
}
```

مقدار دقیق threshold باید پس از مشاهده Dataset واقعی و failure review تعیین شود.

## ۳. اجرای Shadow

مدل را روی چند run کاملاً read-only اجرا کن:

```bash
uv run carla-voxel-shadow \
  --role-name hero \
  --predictor-factory carla_vision.voxel.model_examples.temporal_baseline:create_predictor \
  --predictor-checkpoint models/temporal-voxel-v001/checkpoint_best.pt \
  --predictor-device cuda \
  --history-frames 4 \
  --frames 600 \
  --output runs/voxel-shadow-town10-s17
```

Evidence builder از `summary.json` و `records.jsonl` این موارد را استخراج می‌کند:

- تعداد run و record؛
- error rate؛
- p95 prediction latency؛
- بیشترین uncertain voxel fraction؛
- `actuation_enabled=false`؛
- `control_calls=0`؛
- `actuation_applied=false` در هر record.

## ۴. ساخت Evidence

ابتدا بدون اعلام تکمیل Issue #7 و بدون operator review:

```bash
uv run carla-build-voxel-readiness-evidence \
  --checkpoint models/temporal-voxel-v001/checkpoint_best.pt \
  --training-summary models/temporal-voxel-v001/summary.json \
  --training-dataset runs/voxel-train-route-a-s17 \
  --training-dataset runs/voxel-train-route-b-s23 \
  --training-dataset runs/voxel-train-route-c-s31 \
  --benchmark-report reports/voxel-benchmark-v001.json \
  --shadow-run runs/voxel-shadow-town10-s17 \
  --shadow-run runs/voxel-shadow-town10-s23 \
  --shadow-run runs/voxel-shadow-town10-s31 \
  --output runs/voxel-readiness-evidence-v001.json
```

در این حالت خروجی عمداً شامل این مقادیر است:

```json
{
  "issue_7_acceptance_complete": false,
  "operator_review_complete": false
}
```

Evidence builder این موارد را خودکار استخراج یا verify می‌کند:

- SHA-256 checkpoint؛
- ساختار checkpoint و horizons؛
- وجود RGB و teacher voxelهای Dataset؛
- dataset count و route-group count؛
- overlap بین train/val/test؛
- metricهای benchmark؛
- latency، uncertainty و خطاهای Shadow؛
- عدم actuation در Shadow.

## ۵. اعلام صریح تکمیل

فقط بعد از اینکه معیارهای Issue #7 واقعاً پاس شدند و failure caseها توسط انسان بررسی شدند، دو flag زیر را اضافه کن:

```text
--issue-7-acceptance-complete
--operator-review-complete
```

مثال:

```bash
uv run carla-build-voxel-readiness-evidence \
  ... \
  --issue-7-acceptance-complete \
  --operator-review-complete \
  --output runs/voxel-readiness-evidence-reviewed-v001.json
```

این flagها نتیجه را خودکار معتبر نمی‌کنند؛ readiness verifier تمام thresholdها را جداگانه بررسی می‌کند.

## ۶. اجرای Readiness Gate

```bash
uv run carla-verify-voxel-actuation-readiness \
  --evidence runs/voxel-readiness-evidence-reviewed-v001.json \
  --policy configs/voxel_actuation_policy.json \
  --output reports/voxel-actuation-readiness-reviewed-v001.json
```

حتی گزارش `passed` نیز actuation را فعال نمی‌کند:

```json
{
  "actuation_enabled_by_this_report": false
}
```

## محدودیت Flow

Benchmark فعلی occupancy و semantics را ارزیابی می‌کند، اما `voxel flow` هنوز در دسترس نیست؛ Teacher artifacts فعلی motion vector پایدار برای هر voxel ندارند. گزارش این محدودیت را صریح ثبت می‌کند:

```json
{
  "flow_evaluation_available": false
}
```

برای تکمیل بخش Flow در Issue #7 باید ego-motion alignment و association اشیای dynamic بین فریم‌ها به Dataset اضافه شود.

## اصول تفسیر

- IoU خوب روی یک route برای پذیرش کافی نیست؛
- future IoU باید روی چند route/seed و سناریوی dynamic گزارش شود؛
- Brier score و uncertainty باید همراه IoU بررسی شوند؛
- مدل باید از persistence بهتر باشد؛
- p95 latency روی سخت‌افزار هدف مهم است؛
- Shadow باید کاملاً read-only باقی بماند؛
- هیچ Evidence یا report به‌تنهایی اجازه اعمال فرمان به خودرو نیست.
