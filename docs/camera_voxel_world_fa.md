# جهان سه‌بعدی Voxel فقط با دوربین

این قابلیت یک مسیر آزمایشی برای تبدیل توالی تصاویر RGB به نمایش سه‌بعدی
**occupancy voxel** فراهم می‌کند. هدف نهایی این است که مدل بدون LiDAR، رادار یا
Depth camera عملیاتی، فضای آزاد، موانع، معنای اجسام و وضعیت آینده محیط را پیش‌بینی
کند و برنامه‌ریز مسیر از این پیش‌بینی استفاده کند.

## اصل جداسازی Teacher و Student

### Teacher در CARLA

در مرحله جمع‌آوری داده، سه دوربین کاملاً هم‌مکان و هم‌فریم به خودروی ego متصل
می‌شوند:

- RGB؛
- Depth؛
- Semantic segmentation.

Depth و Semantic فقط **منبع privileged برای برچسب آموزشی و ارزیابی** هستند. خروجی
آن‌ها به مختصات ego با قرارداد CARLA تبدیل می‌شود:

- `x`: جلو؛
- `y`: راست؛
- `z`: بالا.

سپس شبکه‌ای با ترتیب آرایه `(Z, Y, X)` ساخته می‌شود. هر voxel یکی از وضعیت‌های زیر
را دارد:

- `-1`: ناشناخته یا پشت ناحیه قابل مشاهده؛
- `0`: فضای مشاهده‌شده و آزاد؛
- `1`: سطح اشغال‌شده.

برای voxelهای اشغال‌شده، شناسه semantic کلاس CARLA نیز نگهداری می‌شود.

### Student هنگام اجرا

قرارداد مدل deployable فقط این ورودی‌ها را می‌پذیرد:

```text
rgb_history: tuple[uint8 HxWx3]
voxel_grid_spec
```

مدل باید خروجی زیر را تولید کند:

```text
occupancy_probability: float32 [T, Z, Y, X]
horizons_s:            [T]
semantic_logits:       optional [T, C, Z, Y, X]
```

هیچ depth، semantic image، actor ID، waypoint، map یا شیء CARLA وارد قرارداد مدل
نمی‌شود. بنابراین امکان آموزش با Teacher و اجرای camera-only بدون نشت اطلاعات
privileged وجود دارد.

## آماده‌سازی

```bash
uv sync --all-groups
uv run carla-voxel-test --dry-run
```

## تست Teacher voxel روی خودروی hero

ابتدا CARLA و خودروی اصلی را اجرا کنید:

```bash
uv run carla-local-drive \
  --driver behavior \
  --vehicles 50 \
  --walkers 25 \
  --duration 300
```

در ترمینال دیگر:

```bash
uv run carla-voxel-test \
  --host 127.0.0.1 \
  --role-name hero \
  --mode teacher \
  --frames 120 \
  --output runs/voxel-town10-teacher-001 \
  --x-min 0 \
  --x-max 50 \
  --y-min -25 \
  --y-max 25 \
  --z-min -2 \
  --z-max 5 \
  --voxel-resolution 0.5 \
  --pixel-stride 8 \
  --forecast-horizons 0,0.5,1,2
```

برای اتصال مستقیم به actor مشخص:

```bash
uv run carla-voxel-test --actor-id 24 --frames 30
```

## تست مسیر RGB-only

یک predictor تشخیصی وجود دارد که همه voxelها را آزاد اعلام می‌کند. این مدل فقط
برای بررسی قرارداد و فایل‌های خروجی است و برای رانندگی معتبر نیست:

```bash
uv run carla-voxel-test \
  --mode rgb-only \
  --predictor-factory \
    carla_vision.voxel.model_examples.empty_space:create_predictor \
  --predictor-options '{"horizons_s":[0,0.5,1,2]}' \
  --history-frames 4 \
  --frames 30 \
  --output runs/voxel-rgb-contract-001
```

برای ارزیابی مدل RGB واقعی در برابر Teacher، `--mode teacher` را همراه با
`--predictor-factory` اجرا کنید. در این حالت مدل فقط RGB می‌گیرد، ولی Depth و
Semantic جداگانه برای ساخت ground truth و محاسبه occupied IoU استفاده می‌شوند.

## ساخت مدل سفارشی

Factory باید `CameraVoxelModelConfig` را دریافت کند و شیئی با متد `predict` برگرداند:

```python
from carla_vision.voxel.contracts import CameraVoxelPrediction


class MyVoxelModel:
    def predict(self, rgb_history, spec):
        # فقط RGB و spec در دسترس هستند.
        occupancy_probability = model(rgb_history)
        return CameraVoxelPrediction(
            occupancy_probability=occupancy_probability,
            horizons_s=(0.0, 0.5, 1.0, 2.0),
        )


def create_predictor(config):
    return MyVoxelModel()
```

اجرای مدل:

```bash
uv run carla-voxel-test \
  --mode teacher \
  --predictor-factory my_package.voxel_model:create_predictor \
  --predictor-checkpoint models/my-voxel-model.pt \
  --predictor-device cuda \
  --history-frames 4 \
  --frames 200
```

## فایل‌های خروجی

```text
manifest.json
sequence.json
rgb/<frame>.png
teacher_voxels/<frame>.npz
predicted_voxels/<frame>.npz
bev/<frame>.png
metadata/<frame>.json
```

`sequence.json` برای هر فریم، فریم هدف نزدیک به horizonهای آینده را ثبت می‌کند.
این ساختار می‌تواند مستقیماً برای آموزش current occupancy و future occupancy
استفاده شود.

## برنامه‌ریز مسیر مبتنی بر occupancy

ماژول `carla_vision.voxel.planner` شامل این اجزا است:

- تولید مسیرهای کاندید با مدل دوچرخه‌ای و curvature ثابت؛
- انتخاب occupancy slice متناسب با زمان هر نقطه مسیر؛
- محاسبه هزینه برخورد داخل footprint خودرو؛
- جریمه فضای ناشناخته؛
- جریمه curvature؛
- پاداش پیشرفت رو به جلو؛
- انتخاب کم‌هزینه‌ترین مسیر.

`persistence_forecast` یک baseline غیرآموزشی است که occupancy فعلی را برای آینده
تکرار می‌کند. این baseline باید با مدل temporal واقعی جایگزین شود، اما برای تست
زنجیره‌ی planner مفید است.

## پیشنهاد آموزش مرحله‌ای

### مرحله ۱: هندسه فعلی

ورودی: یک یا چند RGB گذشته.

خروجی:

- occupancy فعلی؛
- semantic occupancy فعلی.

Lossهای پیشنهادی:

- binary/focal occupancy loss روی voxelهای known؛
- semantic cross entropy روی voxelهای occupied؛
- visibility-aware weighting برای unknown/free/occupied imbalance.

### مرحله ۲: حافظه زمانی

از ۴ تا ۸ فریم RGB استفاده کنید. ویژگی‌های فریم‌های قبلی باید با حرکت ego به
مختصات فریم فعلی منتقل شوند. شروع مناسب:

- image backbone؛
- depth-aware lift یا voxel queries؛
- temporal attention/GRU در BEV یا voxel space؛
- 3D decoder.

### مرحله ۳: آینده و flow

مدل علاوه بر occupancy در horizonهای `0.5، 1، 2` ثانیه، flow هر voxel یا
جابجایی voxelهای dynamic را پیش‌بینی کند. این بخش برای خودروها و عابران مهم است.

### مرحله ۴: برنامه‌ریزی

برای چند فرمان steering/trajectory، occupancy آینده با action candidate شرطی شود
و trajectory کم‌هزینه انتخاب گردد. در شروع، occupancy forecasting و planning را
جدا آموزش دهید؛ بعد می‌توان آن‌ها را end-to-end fine-tune کرد.

## محدودیت‌های مهم

- یک دوربین monocular عمق قطعی نمی‌دهد؛ مدل باید ambiguity، occlusion و scale را
  یاد بگیرد.
- voxelهای پشت مانع باید unknown بمانند، نه free.
- شبکه ۰٫۵ متری در محدوده ۵۰×۵۰×۷ متر حدود ۱۴۰ هزار voxel دارد؛ کاهش resolution
  به ۰٫۲۵ متر حافظه و محاسبات را تقریباً هشت برابر می‌کند.
- camera-only occupancy می‌تواند جایگزین ادراکی مهمی برای LiDAR باشد، اما حذف
  کامل سنسورهای مکمل در سامانه ایمنی واقعی نیازمند ارزیابی سخت‌گیرانه، uncertainty
  و fail-safe مستقل است.
- مدل نمونه `empty_space` فقط smoke test است.

## مسیر پیشنهادی مدل

برای پروژه فعلی، معماری کم‌ریسک‌تر این است:

```text
RGB history
  -> image backbone
  -> monocular depth distribution / sparse voxel queries
  -> temporal voxel fusion
  -> current semantic occupancy
  -> future occupancy + voxel flow
  -> occupancy-aware candidate trajectory scoring
  -> steering / target trajectory
```

این طراحی با ایده‌های رایج VoxFormer، SurroundOcc، Occ3D، UniOcc و occupancy world
models هم‌راستا است، ولی Dataset و قراردادهای آن برای CARLA و محدودیت camera-only
این پروژه مستقل نگه داشته شده‌اند.
