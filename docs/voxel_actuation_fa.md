# Actuation واقعی Voxel در `carla-local-drive`

این قابلیت خروجی voxel planner را به رانندگی واقعی Ego در CARLA وصل می‌کند، اما به‌صورت پیش‌فرض خاموش است.

## رفتار پیش‌فرض

فرمان زیر هیچ actuation وکسل فعال نمی‌کند و همان مسیر قبلی `carla-local-drive` را اجرا می‌کند:

```bash
uv run carla-local-drive --driver behavior
```

## فعال‌سازی صریح

برای فعال‌کردن steering مبتنی بر voxel باید هر دو flag زیر داده شوند:

```text
--enable-voxel-actuation
--acknowledge-voxel-actuation
```

همچنین predictor لازم است:

```bash
uv run carla-local-drive \
  --driver behavior \
  --enable-voxel-actuation \
  --acknowledge-voxel-actuation \
  --voxel-predictor-factory carla_vision.voxel.model_examples.temporal_flow:create_predictor \
  --voxel-predictor-checkpoint models/temporal-voxel-flow/checkpoint_best.pt \
  --voxel-predictor-device cuda \
  --vehicles 60 \
  --walkers 30
```

## تقسیم کنترل

در این حالت:

- `BehaviorAgent` همچنان throttle و brake پایه را تولید می‌کند.
- Voxel planner فقط steering را پیشنهاد می‌دهد.
- Supervisor قبل از اعمال steering proposal را بررسی می‌کند.
- اگر proposal رد شود، prediction نامعتبر باشد، timeout رخ دهد یا exception ایجاد شود، خودرو `brake=1.0` می‌گیرد.
- هنگام پرشدن RGB history نیز خودرو در حالت full brake می‌ماند.

بنابراین voxel planner بدون دو flag صریح هیچ فرمانی به خودرو نمی‌دهد.

## Readiness report اختیاری

در صورت داشتن report قبلی می‌توان آن را نیز وارد runtime کرد:

```bash
--voxel-readiness-report runs/voxel-readiness/report.json
```

اگر report داده شود، همان report عیناً توسط supervisor بررسی می‌شود و report نامعتبر/blocked باعث full brake می‌شود. اگر report داده نشود، acknowledgement صریح runtime برای ورود به مسیر actuation کافی است؛ safety checkهای runtime همچنان اجرا می‌شوند.

## Safety checkهای runtime

Supervisor موارد زیر را کنترل می‌کند:

- سن prediction؛
- latency؛
- قرارداد shape/range prediction؛
- horizon کافی؛
- uncertainty؛
- collision risk؛
- سرعت خودرو؛
- steering مطلق؛
- steering-rate.

مقادیر مهم از CLI قابل تنظیم‌اند:

```text
--voxel-max-collision-risk
--voxel-max-uncertain-fraction
--voxel-max-latency-ms
--voxel-max-prediction-age
--voxel-max-speed-kmh
--voxel-max-absolute-steering
--max-steer-rate
--voxel-minimum-horizon
```

## Grid و planner

Grid پیش‌فرض همان grid دوربین‌محور پروژه است:

```text
x: 0 .. 50 m
y: -25 .. +25 m
z: -2 .. +5 m
resolution: 0.5 m
```

قابل تغییر با:

```text
--voxel-x-min / --voxel-x-max
--voxel-y-min / --voxel-y-max
--voxel-z-min / --voxel-z-max
--voxel-resolution
```

Candidate steeringها نیز با `--voxel-steering-values` قابل تنظیم‌اند.

## Dry run

برای بررسی config بدون CARLA:

```bash
uv run carla-local-drive \
  --dry-run \
  --driver behavior \
  --enable-voxel-actuation \
  --acknowledge-voxel-actuation \
  --voxel-predictor-factory example.module:create_predictor
```

در dry-run predictor import نمی‌شود و هیچ اتصال CARLA انجام نمی‌شود.

## محدوده این تغییر

این integration فقط مسیر CLI مربوط به `carla-local-drive` را wrap می‌کند. اگر actuation voxel فعال نباشد، اجرا به `carla_vision.local_drive.run()` فعلی واگذار می‌شود. Drive Console، operator UI، bridge و runtime مرورگر تغییر نمی‌کنند.
