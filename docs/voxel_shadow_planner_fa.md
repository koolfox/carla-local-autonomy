# Voxel Planner Shadow

این ابزار یک فرآیند همراه و کاملاً **read-only** برای خودروی `hero` است. مدل camera-only از توالی RGB، occupancy فعلی و آینده را پیش‌بینی می‌کند؛ سپس چند مسیر کاندید امتیازدهی می‌شوند. خروجی planner فقط ثبت می‌شود و هیچ فرمانی به خودرو اعمال نمی‌گردد.

## مرز ایمنی

ماژول `carla_vision.voxel.shadow` هیچ فراخوانی `apply_control` ندارد.

- کنترل خودرو همچنان با `BehaviorAgent`، Traffic Manager یا راننده اصلی است.
- Shadow فقط دوربین RGB خودش را به خودرو متصل می‌کند.
- `vehicle.get_control()` فقط برای مقایسه با steering پیشنهادی خوانده می‌شود.
- در manifest همیشه این دو مقدار ثبت می‌شوند:

```json
{
  "actuation_enabled": false,
  "control_calls": 0
}
```

این ابزار نباید به‌عنوان کنترل‌کننده خودرو اجرا یا معرفی شود.

## نصب و Dry Run

```bash
uv sync --all-groups
uv run carla-voxel-shadow --dry-run
```

Dry run بدون import کردن CARLA، Grid، وزن‌های planner و غیرفعال بودن actuation را نمایش می‌دهد.

## اجرای Shadow کنار BehaviorAgent

ترمینال اول:

```bash
uv run carla-local-drive \
  --host 127.0.0.1 \
  --driver behavior \
  --behavior cautious \
  --vehicles 60 \
  --walkers 30 \
  --duration 300
```

ترمینال دوم:

```bash
uv run carla-voxel-shadow \
  --host 127.0.0.1 \
  --role-name hero \
  --predictor-factory \
    carla_vision.voxel.model_examples.temporal_baseline:create_predictor \
  --predictor-checkpoint models/temporal-voxel-v001/checkpoint_best.pt \
  --predictor-device cuda \
  --history-frames 4 \
  --frames 300 \
  --output runs/voxel-shadow-town10-v001
```

اگر Actor ID خودرو مشخص است:

```bash
uv run carla-voxel-shadow \
  --actor-id 24 \
  --predictor-factory \
    carla_vision.voxel.model_examples.temporal_baseline:create_predictor \
  --predictor-checkpoint models/temporal-voxel-v001/checkpoint_best.pt
```

## Grid و مسیرهای کاندید

Grid زمان اجرا باید دقیقاً با Grid checkpoint یکسان باشد. مقادیر پیش‌فرض:

```text
x:   0 تا 50 متر
 y: -25 تا 25 متر
 z:  -2 تا 5 متر
resolution: 0.5 متر
```

مسیرها با مدل ساده دوچرخه و steeringهای ثابت تولید می‌شوند:

```bash
--steering-values=-0.6,-0.3,0,0.3,0.6
--trajectory-steps 12
--trajectory-dt 0.25
```

هزینه هر مسیر شامل این بخش‌ها است:

- احتمال برخورد در footprint خودرو
- جریمه فضای ناشناخته
- جریمه curvature
- پاداش پیشرفت رو به جلو

نمونه تغییر وزن‌ها:

```bash
uv run carla-voxel-shadow \
  ... \
  --collision-weight 120 \
  --unknown-weight 8 \
  --curvature-weight 1.5 \
  --progress-weight 0.8
```

## عدم قطعیت

احتمال‌های نزدیک `0.5` به‌عنوان unknown در نظر گرفته می‌شوند:

```bash
--uncertainty-band 0.08
```

در این حالت بازه `0.42` تا `0.58` به planner به‌صورت فضای ناشناخته داده می‌شود. افزایش این مقدار planner را محافظه‌کارتر می‌کند.

## خروجی

```text
runs/voxel-shadow-town10-v001/
├── manifest.json
├── records.jsonl
└── summary.json
```

هر رکورد شامل موارد زیر است:

- frame و timestamp
- سرعت خودرو
- latency مدل و planner
- horizonهای occupancy
- steering و cost مسیر انتخاب‌شده
- cost همه مسیرهای کاندید
- سهم voxelهای uncertain
- کنترل واقعی مشاهده‌شده خودرو
- موقعیت خودرو برای تحلیل آفلاین
- `actuation_applied: false`

## تحلیل مقایسه‌ای

برای مقایسه Shadow با BehaviorAgent، route، seed، ترافیک و مدل را ثابت نگه دارید. سپس این موارد را از `records.jsonl` بررسی کنید:

- اختلاف steering پیشنهادی و steering واقعی
- collision cost مسیر واقعی تقریبی در برابر مسیر منتخب
- نرخ انتخاب مسیرهای چپ، مستقیم و راست
- latency و prediction errorها
- سهم فضای unknown

## محدودیت فاز اول

این PR فقط فاز Shadow از Issue #8 است. Actuation، emergency brake مستقل و fallback کنترل‌شده در این ابزار وجود ندارند. فعال‌سازی فرمان باید در تحویل جداگانه، با acknowledgement صریح و پس از ارزیابی مدل واقعی انجام شود.
