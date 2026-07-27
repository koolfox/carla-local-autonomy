# ارزیابی Closed-loop رانندگی در CARLA

فرمان `carla-evaluate-drive` یک observer جداگانه است که به خودروی موجود با نقش `hero` متصل می‌شود. این ابزار کنترل خودرو را تغییر نمی‌دهد و مالک `world.tick()` نیست؛ فقط دو سنسور event به خودرو اضافه می‌کند و state موجود را می‌خواند.

## Metricهای ثبت‌شده

- route completion؛
- رسیدن یا نرسیدن به مقصد؛
- مسافت طی‌شده؛
- collision count و collisions/km؛
- lane-invasion count و lane invasions/km؛
- red-light violation تقریبی؛
- transition به full brake هنگام حرکت؛
- throttle، steer، brake، سرعت و موقعیت در هر frame.

## مرز Read-only

این observer هیچ فراخوانی `apply_control` ندارد. CI کد را با AST بررسی می‌کند. تمام summaryها نیز باید شامل این مقادیر باشند:

```json
{
  "read_only_vehicle_control": true,
  "control_calls": 0
}
```

ابزار فقط Actorهای سنسور collision و lane invasion را ایجاد و هنگام خروج پاک می‌کند.

## Dry run

```bash
uv run carla-evaluate-drive \
  --dry-run \
  --driver-label behavior \
  --run-label behavior-seed-17 \
  --destination-index 42
```

Dry run به CARLA نیاز ندارد.

## اجرای BehaviorAgent

ترمینال اول:

```bash
uv run carla-local-drive \
  --host 127.0.0.1 \
  --driver behavior \
  --behavior cautious \
  --ego-spawn-index 3 \
  --destination-index 42 \
  --seed 17 \
  --vehicles 60 \
  --walkers 30 \
  --duration 180
```

ترمینال دوم، بعد از ساخته‌شدن خودرو:

```bash
uv run carla-evaluate-drive \
  --host 127.0.0.1 \
  --role-name hero \
  --driver-label behavior \
  --run-label behavior-town10-s17 \
  --seed 17 \
  --destination-index 42 \
  --duration 170 \
  --output runs/eval-behavior-town10-s17
```

## اجرای مدل Imitation

ترمینال اول:

```bash
uv run carla-local-drive \
  --host 127.0.0.1 \
  --driver model \
  --ego-spawn-index 3 \
  --model-factory carla_vision.imitation.predictor:create_driver \
  --model-checkpoint models/imitation-behavior-v001/checkpoint_best.pt \
  --model-device cuda \
  --max-model-speed-kmh 25 \
  --max-steer-rate 1.5 \
  --seed 17 \
  --vehicles 60 \
  --walkers 30 \
  --duration 180
```

ترمینال دوم:

```bash
uv run carla-evaluate-drive \
  --host 127.0.0.1 \
  --role-name hero \
  --driver-label imitation \
  --run-label imitation-town10-s17 \
  --seed 17 \
  --destination-index 42 \
  --duration 170 \
  --output runs/eval-imitation-town10-s17
```

مدل مسیر یا مقصد را به‌عنوان ورودی دریافت نمی‌کند. مقصد فقط یک معیار privileged برای evaluator است. به همین دلیل نتیجه route completion باید با محدودیت مدل RGB+speed تفسیر شود.

## خروجی هر run

```text
runs/eval-imitation-town10-s17/
├── route.json
├── telemetry.jsonl
└── summary.json
```

`telemetry.jsonl` در هر tick وضعیت فعلی و event count تجمعی را ذخیره می‌کند. `summary.json` metricهای نهایی را دارد.

## ساخت گزارش چند seed

پس از اجرای BehaviorAgent، lane-center و imitation روی seedهای مشابه:

```bash
uv run carla-summarize-drive-evaluations \
  runs/eval-behavior-town10-s17 \
  runs/eval-behavior-town10-s23 \
  runs/eval-lane-center-town10-s17 \
  runs/eval-lane-center-town10-s23 \
  runs/eval-imitation-town10-s17 \
  runs/eval-imitation-town10-s23 \
  --output reports/closed-loop-comparison-v001.json
```

گزارش بر اساس `driver_label` گروه‌بندی می‌شود و collision/lane rate را از مجموع eventها تقسیم بر مجموع مسافت محاسبه می‌کند؛ بنابراین runهای کوتاه و بلند وزن نادرست یکسان نمی‌گیرند.

## محدودیت Red-light

CARLA برای این observer event مستقیم عبور از چراغ قرمز نمی‌فرستد. heuristic فعلی یک ورود به trigger volume چراغ قرمز را دنبال می‌کند و اگر خودرو پیش از خروج سرعتش به کمتر از آستانه نرسد، یک violation ثبت می‌کند.

این metric تقریبی است و ممکن است در هندسه‌های خاص false positive یا false negative داشته باشد. Summary متن این محدودیت را همراه نتیجه نگه می‌دارد.

## محدودیت Full-brake

observer فقط `vehicle.get_control()` را می‌خواند و دلیل داخلی کنترل را نمی‌داند. بنابراین transition به brake شدید می‌تواند یکی از این موارد باشد:

- safety brake مدل؛
- ترمز عمدی برای مانع یا چراغ؛
- توقف دستی.

این مقدار «intervention-like full-brake event» است و نباید بدون telemetry تکمیلی intervention قطعی نامیده شود.

## پروتکل مقایسه

برای مقایسه معتبر:

1. map، spawn شروع، destination و seed یکسان باشند؛
2. تراکم خودرو و عابر یکسان باشد؛
3. نسخه CARLA و fixed delta یکسان باشد؛
4. هر driver روی چند seed اجرا شود؛
5. observer بعد از شروع راننده اجرا و پیش از پایان آن متوقف شود؛
6. runهای ناقص یا دارای timeout جداگانه گزارش شوند؛
7. BehaviorAgent به‌عنوان Teacher، lane-center به‌عنوان demo و imitation به‌عنوان مدل آموزش‌دیده مقایسه شوند.

اجرای زنده این پروتکل روی سرور لوکال برای بستن Issue #3 لازم است.
