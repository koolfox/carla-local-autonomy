# ثبت Episodeهای Teacher با BehaviorAgent

برای ثبت هم‌زمان چند دوربین RGB و بازپخش آن‌ها، راهنمای
[Multi-camera episodes](multicamera_episodes.md) را ببین. حالت پیش‌فرض این صفحه
همچنان تک‌دوربینه است؛ ورودی مدل‌های فعلی خودکار چنددوربینه نمی‌شود.

این مسیر برای ساخت Dataset آموزش imitation learning طراحی شده است. خودرو با `BehaviorAgent` رانده می‌شود، اما ورودی قابل استفاده مدل نهایی همچنان فقط دوربین RGB و سرعت Ego است. اطلاعات نقشه، Actorها، دوربین instance و وضعیت شبیه‌ساز فقط برای Teacher، برچسب و ارزیابی ذخیره می‌شوند.

## تفاوت با `carla-native-collect`

`carla-native-collect` خودروی Ego را با Traffic Manager کنترل می‌کند. فرمان جدید زیر یک مسیر deterministic را از seed هر Episode انتخاب می‌کند و فرمان `BehaviorAgent.run_step()` را پیش از همان `world.tick()` که تصویر را تولید می‌کند اعمال و ثبت می‌کند:

```bash
uv run carla-record-behavior-teacher --help
```

هر نمونه شامل موارد زیر است:

- RGB جلو و instance mask هم‌فریم؛
- `carla_frame` و simulation timestamp؛
- سرعت، شتاب، وضعیت چراغ راهنمایی و pose خودروی Ego؛
- throttle، steer و brake تولیدشده توسط BehaviorAgent؛
- `route_id`، شماره leg، seed مسیر، spawn شروع و مقصد؛
- transform کامل مقصد و فاصله باقی‌مانده تا آن؛
- شناسه Scenario، Episode، split، آب‌وهوا و تراکم ترافیک.

## پیش‌نیاز مهم

برای اجرای واقعی، هم ماژول `carla` و هم پوشه `agents` متعلق به PythonAPI نسخه **CARLA 0.9.16** باید import شوند. در بسیاری از نصب‌ها wheel فقط ماژول `carla` را دارد؛ در این حالت مسیر زیر را به `PYTHONPATH` اضافه کن:

```bash
export PYTHONPATH="/path/to/CARLA_0.9.16/PythonAPI/carla:${PYTHONPATH}"
```

بررسی:

```bash
uv run python -c "import carla; from agents.navigation.behavior_agent import BehaviorAgent; print(carla.__version__)"
```

## Dry run

Dry run هیچ تغییری در شبیه‌ساز ایجاد نمی‌کند و فقط Scenario Plan و تعداد Episodeها را بررسی می‌کند:

```bash
uv run carla-record-behavior-teacher \
  --scenario-plan runs/scenario-plan-native-integration-pilot-v1 \
  --dataset-id ds-behavior-teacher-dry-v001 \
  --partition train \
  --max-episodes 1 \
  --behavior cautious \
  --target-speed-kmh 30 \
  --dry-run
```

## اولین اجرای واقعی

این Collector نقشه را reload می‌کند، Actorهای قبلی را از بین می‌برد و باید تنها مالک `world.tick()` باشد. قبل از اجرا تمام clientهای دیگر مانند `carla-local-drive`، manual control و traffic generator را متوقف کن.

```bash
uv run carla-record-behavior-teacher \
  --scenario-plan runs/scenario-plan-native-integration-pilot-v1 \
  --dataset-id ds-behavior-teacher-pilot-v001 \
  --datasets-root datasets \
  --host 127.0.0.1 \
  --port 2000 \
  --partition train \
  --max-episodes 1 \
  --behavior cautious \
  --target-speed-kmh 30 \
  --minimum-route-distance-m 40 \
  --timeout 30 \
  --sensor-timeout 10 \
  --acknowledge-exclusive-tick-owner
```

Verifier پس از Collection به‌صورت پیش‌فرض اجرا می‌شود. برای غیرفعال‌کردن موقت آن:

```text
--no-verify-after-collection
```

## بررسی مستقل Dataset

```bash
uv run carla-verify-teacher-episodes \
  datasets/ds-behavior-teacher-pilot-v001 \
  --report reports/ds-behavior-teacher-pilot-v001.json
```

Verifier موارد زیر را کنترل می‌کند:

- وجود و SHA-256 تمام Artifactهای checksum-index؛
- قابل decode بودن RGB و تطابق ابعاد با metadata؛
- تطابق frame و timestamp در `dataset.json` و metadata؛
- افزایش strict شماره frameها در هر Episode؛
- هم‌فریم بودن RGB و instance teacher؛
- وجود route ID و مقصد کامل برای هر نمونه؛
- معتبر بودن throttle، steer و brake؛
- وجود سرعت Ego و release metadata مخصوص BehaviorAgent.

## مقایسه دو اجرای تکراری

همان Scenario Plan و Episode را با دو Dataset ID متفاوت اجرا کن، سپس شواهد تکرارپذیری را به‌شکل خودکار مقایسه کن:

```bash
uv run carla-compare-teacher-episodes \
  datasets/ds-behavior-teacher-repeat-a \
  datasets/ds-behavior-teacher-repeat-b \
  --control-tolerance 1e-6 \
  --timestamp-tolerance 1e-6 \
  --state-tolerance 1e-4 \
  --report reports/behavior-teacher-repeat-comparison.json
```

این ابزار ابتدا هر دو Dataset را verify می‌کند و سپس موارد زیر را براساس Episode و ترتیب نمونه مقایسه می‌کند:

- cadence نسبی frameها؛
- timestamp نسبی نسبت به اولین نمونه هر Episode؛
- route ID، شماره leg و spawn مقصد؛
- throttle، steer و brake؛
- سرعت Ego؛
- تعداد Episodeها و نمونه‌ها.

شماره absolute فریم و timestamp شروع می‌تواند پس از reload نقشه متفاوت باشد؛ بنابراین مقایسه روی offsetهای نسبی انجام می‌شود. برای آزمون سخت‌گیرانه‌ی خروجی renderer نیز می‌توان این گزینه را اضافه کرد:

```text
--require-identical-rgb
```

یکسان‌نبودن byte دقیق RGB الزاماً به معنی غیردeterministic بودن route/control نیست و ممکن است از renderer یا سخت‌افزار گرافیکی ناشی شود؛ به همین دلیل این gate پیش‌فرض خاموش است.

## ساختار route

نمونه metadata:

```json
{
  "control_mode": "behavior_agent_teacher",
  "route": {
    "route_id": "route-ep-example-r000-leg-000-d042",
    "route_seed": 123456,
    "leg_index": 0,
    "start_spawn_index": 3,
    "destination_spawn_index": 42,
    "destination_world_transform": {
      "x": 12.0,
      "y": -34.0,
      "z": 0.5,
      "pitch": 0.0,
      "yaw": 90.0,
      "roll": 0.0
    },
    "remaining_straight_line_distance_m": 51.2
  },
  "privileged_teacher_control": {
    "source": "BehaviorAgent.run_step",
    "applied_before_world_tick": true,
    "carla_frame": 880,
    "throttle": 0.35,
    "steer": -0.08,
    "brake": 0.0
  }
}
```

اگر BehaviorAgent یک مقصد را تمام کند، مقصد بعدی با همان seed مسیر و شماره leg بعدی به‌شکل deterministic انتخاب می‌شود. این کار تعداد نمونه برنامه‌ریزی‌شده را ثابت نگه می‌دارد و از tail طولانی توقف در انتهای مسیر جلوگیری می‌کند.

## تکرارپذیری

برای بازپخش یک Episode، این موارد باید ثابت بمانند:

- Scenario Plan و hashهای آن؛
- CARLA client/server نسخه 0.9.16؛
- نقشه و fixed delta؛
- seedهای Episode، به‌ویژه `route`؛
- behavior profile و target speed؛
- نسخه کد و PythonAPI؛
- تنها بودن tick owner.

Datasetها، تصاویر، checkpointها و خروجی‌های `runs/` و `reports/` توسط `.gitignore` از Git خارج نگه داشته می‌شوند.
