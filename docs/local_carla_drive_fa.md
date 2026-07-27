# اجرای رانندگی لوکال در CARLA

این مسیر اجرایی برای اتصال مستقیم به سرور لوکال CARLA، ساخت خودروی اصلی، ایجاد ترافیک و عابر، و رانندگی در محیط شلوغ اضافه شده است.

## پیش‌نیازها

- سرور **CARLA 0.9.16** در حال اجرا باشد.
- PythonAPI دقیقاً با نسخه سرور هماهنگ باشد.
- پوشه `PythonAPI/carla` نیز در `PYTHONPATH` قرار بگیرد تا `BehaviorAgent` قابل import باشد.
- برای حالت مدل، وزن‌ها و وابستگی‌های مدل به‌صورت محلی نصب شده باشند.

نمونه متغیر محیطی در Linux/macOS:

```bash
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$PYTHONPATH"
```

در PowerShell:

```powershell
$env:PYTHONPATH="$env:CARLA_ROOT\PythonAPI\carla;$env:PYTHONPATH"
```

## بررسی بدون اتصال

```bash
uv sync --all-groups
uv run carla-local-drive --dry-run
```

این دستور CARLA را import نمی‌کند و فقط تنظیمات و عملیات مورد انتظار را نمایش می‌دهد.

## رانندگی با BehaviorAgent در ترافیک

```bash
uv run carla-local-drive \
  --host 127.0.0.1 \
  --port 2000 \
  --tm-port 8000 \
  --map Town10HD_Opt \
  --driver behavior \
  --behavior cautious \
  --ego-target-speed-kmh 30 \
  --vehicles 80 \
  --walkers 40 \
  --traffic-distance 1.5 \
  --traffic-speed-difference 18 \
  --pedestrian-crossing 0.30 \
  --duration 300
```

پس از رسیدن به مقصد، به‌صورت پیش‌فرض مقصد جدید انتخاب می‌شود. برای توقف در مقصد اول از `--no-loop-destinations` استفاده کنید.

## تست سریع با Traffic Manager

```bash
uv run carla-local-drive \
  --driver traffic-manager \
  --vehicles 70 \
  --walkers 35 \
  --duration 180
```

در این حالت Traffic Manager هم خودروی اصلی و هم خودروهای NPC را کنترل می‌کند.

## رانندگی با مدل سفارشی

Factory مدل باید با قالب `package.module:callable` معرفی شود. callable یک `ModelDriverConfig` دریافت می‌کند و شیئی با متدهای زیر برمی‌گرداند:

```python
class DrivingModel:
    def reset(self) -> None: ...
    def predict(self, observation: ModelObservation) -> ModelControl: ...
    def close(self) -> None: ...
```

مشاهده مدل شامل تصویر BGR دوربین جلو، سرعت خودرو، شماره فریم، timestamp و فاصله زمانی است. مدل به map، waypoint، actor ID یا Traffic Manager دسترسی مستقیم ندارد.

نمونه اجرای رابط مدل:

```bash
uv run carla-local-drive \
  --driver model \
  --model-factory carla_vision.model_examples.lane_center:create_driver \
  --model-options '{"target_speed_mps":3.0,"steer_gain":0.7}' \
  --max-model-speed-kmh 18 \
  --vehicles 35 \
  --walkers 15 \
  --duration 60
```

`lane_center` فقط یک دموی تصویری برای آزمایش حلقه کنترل است و مدل آموزش‌دیده یا ایمن محسوب نمی‌شود.

## قرارداد خروجی مدل

مدل می‌تواند `ModelControl`، یک mapping یا یک sequence سه‌عضوی برگرداند:

```python
return {
    "throttle": 0.25,
    "steer": -0.1,
    "brake": 0.0,
}
```

محدوده‌ها:

- `throttle`: از ۰ تا ۱
- `steer`: از منفی ۱ تا ۱
- `brake`: از ۰ تا ۱

درخواست هم‌زمان throttle و brake معتبر نیست.

## کنترل‌های ایمنی

در حالت مدل موارد زیر اعمال می‌شوند:

- قطع یا تأخیر تصویر باعث ترمز کامل می‌شود.
- خروجی نامعتبر مدل باعث ترمز کامل می‌شود.
- پس از چند خطای متوالی اجرا متوقف می‌شود.
- سرعت از `--max-model-speed-kmh` محدود می‌شود.
- نرخ تغییر فرمان با `--max-steer-rate` محدود می‌شود.
- هنگام خروج، Actorها پاک و تنظیمات synchronous جهان بازگردانده می‌شوند.

## نکات عیب‌یابی

### PythonAPI پیدا نمی‌شود

نسخه PythonAPI را از همان نصب CARLA استفاده کنید و مسیر wheel/egg یا پوشه PythonAPI را به محیط اضافه کنید.

### BehaviorAgent پیدا نمی‌شود

پوشه‌ای که package به نام `agents` دارد باید در `PYTHONPATH` باشد.

### نسخه‌ها متفاوت‌اند

نسخه client و server باید 0.9.16 باشد. فقط برای آزمایش آگاهانه می‌توان از `--allow-version-mismatch` استفاده کرد.

### خودرو یا عابر کمتر از مقدار درخواستی ساخته شد

Spawn point یا فضای navigation کافی نبوده است. تعداد را کاهش دهید یا نقشه بزرگ‌تری انتخاب کنید.
