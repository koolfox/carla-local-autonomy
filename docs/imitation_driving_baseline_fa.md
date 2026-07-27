# آموزش baseline رانندگی Imitation

این baseline از Episodeهای ثبت‌شده با `carla-record-behavior-teacher` استفاده می‌کند و فقط دو ورودی قابل‌استفاده در زمان اجرا دارد:

- تصویر RGB دوربین جلو؛
- سرعت فعلی خودروی Ego.

مقصد، Waypoint، نقشه، Actorهای CARLA، instance mask و وضعیت شبیه‌ساز به مدل inference داده نمی‌شوند. این اطلاعات فقط برای ساخت Dataset، Teacher و ارزیابی نگه داشته می‌شوند.

## خروجی مدل

شبکه دو مقدار bounded تولید می‌کند:

```text
steer        در بازه [-1, 1]
longitudinal در بازه [-1, 1]
```

فرمان طولی مثبت به throttle و مقدار منفی به brake تبدیل می‌شود. این طراحی مانع درخواست هم‌زمان throttle و brake می‌شود و با قرارداد `ModelControl` سازگار است.

## بررسی Dry run

```bash
uv run carla-train-imitation \
  --dry-run \
  --device cpu \
  --image-size 64x96 \
  --encoder-channels 8 \
  --hidden-dim 16
```

Dry run یک forward/backward مصنوعی اجرا می‌کند و به Dataset یا CARLA نیاز ندارد.

## آماده‌سازی Dataset

ابتدا Episodeهای BehaviorAgent را ثبت و verify کن:

```bash
uv run carla-record-behavior-teacher \
  --scenario-plan runs/scenario-plan-native-integration-pilot-v1 \
  --dataset-id ds-behavior-teacher-train-v001 \
  --datasets-root datasets \
  --partition train \
  --behavior cautious \
  --target-speed-kmh 30 \
  --acknowledge-exclusive-tick-owner

uv run carla-verify-teacher-episodes \
  datasets/ds-behavior-teacher-train-v001
```

برای آموزش واقعی به چند route/seed group نیاز است. یک Episode یا یک route برای ساخت train/validation/test کافی نیست.

## آموزش

```bash
uv run carla-train-imitation \
  --dataset datasets/ds-behavior-teacher-train-v001 \
  --dataset datasets/ds-behavior-teacher-train-v002 \
  --output models/imitation-behavior-v001 \
  --device cuda \
  --image-size 192x320 \
  --epochs 20 \
  --batch-size 16 \
  --learning-rate 3e-4 \
  --val-fraction 0.20 \
  --test-fraction 0.10 \
  --brightness 0.12 \
  --contrast 0.12 \
  --horizontal-flip-probability 0.0
```

خروجی‌ها:

```text
models/imitation-behavior-v001/
├── checkpoint_best.pt
├── checkpoint_last.pt
├── history.json
└── summary.json
```

پوشه `models/` در `.gitignore` قرار دارد و checkpoint نباید وارد Git شود.

## جلوگیری از Leakage

تقسیم train/validation/test براساس یک کلید گروهی پایدار انجام می‌شود که شامل موارد زیر است:

- map family؛
- Scenario recipe؛
- route seed؛
- spawn شروع و مقصد؛
- static-layout seed.

تمام نمونه‌های یک route/seed group فقط در یک partition قرار می‌گیرند. `summary.json` باید این مقدار را خالی نشان دهد:

```json
{
  "split_summary": {
    "route_group_overlap": {}
  }
}
```

اگر overlap مشاهده شود، آموزش با خطا متوقف می‌شود.

## Augmentation بازتولیدپذیر

augmentation براساس ترکیب `augmentation_seed` و شناسه ثابت هر نمونه مشتق می‌شود. در نتیجه همان نمونه با همان seed همیشه دقیقاً همان تغییر را می‌گیرد.

augmentationهای فعلی محدود هستند:

- brightness؛
- contrast؛
- horizontal flip اختیاری همراه با معکوس‌کردن علامت steer.

horizontal flip پیش‌فرض خاموش است، چون تابلوها، جهت ترافیک و ساختار جاده ممکن است پس از flip غیرواقعی شوند.

## Loss و Metric

Loss اصلی از Smooth L1 برای موارد زیر ساخته می‌شود:

- steer؛
- longitudinal command.

نمونه‌های دارای steer شدید و braking وزن بیشتری می‌گیرند تا Datasetهای عمدتاً مستقیم/حرکت یکنواخت مدل را منحرف نکنند.

Metricهای آفلاین:

- steer MAE و RMSE؛
- longitudinal MAE و RMSE؛
- brake precision و recall؛
- steer-direction accuracy.

Metric آفلاین برای اثبات رانندگی ایمن کافی نیست. route completion و collision فقط در closed-loop CARLA معتبر هستند.

## اجرای checkpoint در CARLA

```bash
uv run carla-local-drive \
  --host 127.0.0.1 \
  --driver model \
  --model-factory carla_vision.imitation.predictor:create_driver \
  --model-checkpoint models/imitation-behavior-v001/checkpoint_best.pt \
  --model-device cuda \
  --model-options '{"steer_gain":1.0,"throttle_gain":0.8,"brake_gain":1.0,"longitudinal_deadband":0.03}' \
  --max-model-speed-kmh 25 \
  --max-steer-rate 1.5 \
  --vehicles 40 \
  --walkers 20 \
  --duration 120
```

لایه‌های ایمنی موجود در `carla-local-drive` همچنان فعال‌اند:

- محدودکننده سرعت؛
- محدودکننده نرخ تغییر steer؛
- ترمز کامل هنگام خطای مدل یا timeout دوربین؛
- توقف پس از تعداد مشخص خطای متوالی.

## مراحل قبل از Actuation جدی

1. ابتدا checkpoint را روی Dataset test آفلاین ارزیابی کن.
2. سپس در ترافیک کم و سرعت پایین اجرا کن.
3. collision sensor، lane invasion، red-light و route completion را ثبت کن.
4. چند seed و تراکم متفاوت را اجرا کن.
5. نتیجه را با BehaviorAgent و demo مدل lane-center مقایسه کن.

تا زمانی که این ارزیابی closed-loop انجام نشده، checkpoint فقط یک baseline تحقیقاتی است و نباید به‌عنوان مدل رانندگی معتبر تلقی شود.
