# آموزش Voxel Flow از RGB در CARLA

این قابلیت برای Issue #7 است و به بخش‌های Drive Console، bridge و runtime دست نمی‌زند.

## ایده

مدل نهایی فقط توالی RGB را می‌بیند. برای ساخت label در شبیه‌ساز از سه منبع privileged استفاده می‌شود:

- Depth camera برای هندسه سه‌بعدی
- Semantic camera برای محدودکردن supervision به خودرو و عابر
- Optical Flow camera برای correspondence بین دو فریم متوالی

جابه‌جایی دو نقطه بعد از تبدیل هر دو فریم به مختصات world محاسبه می‌شود؛ بنابراین حرکت خود ego تا حد زیادی حذف می‌شود. سپس بردار حرکت دوباره در دستگاه مختصات دوربین/ego فریم مبدا بیان و بر `dt` تقسیم می‌شود. خروجی teacher یک velocity سه‌بعدی با واحد متر بر ثانیه است.

Teacher دو جهت علامت optical-flow را امتحان می‌کند و جهتی را انتخاب می‌کند که روی پیکسل‌های غیر-dynamic کمترین residual سه‌بعدی را داشته باشد. این انتخاب در metadata ثبت می‌شود و جای smoke test واقعی CARLA را نمی‌گیرد.

## جمع‌آوری Dataset

ابتدا یک ego با `role_name=hero` اجرا کنید. سپس در ترمینال دیگر:

```bash
uv run carla-voxel-flow-capture \
  --host 127.0.0.1 \
  --role-name hero \
  --frames 600 \
  --output runs/voxel-flow-town10-seed17 \
  --forecast-horizons 0,0.5,1,2 \
  --flow-pixel-stride 4
```

خروجی همان ساختار Dataset voxel قبلی را نگه می‌دارد و فقط `teacher_flow/` و فیلدهای مربوط به Flow را اضافه می‌کند:

```text
runs/voxel-flow-town10-seed17/
├── manifest.json
├── sequence.json
├── rgb/
├── teacher_voxels/
├── teacher_flow/
├── bev/
└── metadata/
```

هر فایل Flow شامل موارد زیر است:

```python
payload = np.load("teacher_flow/00000123.npz")
velocity_mps = payload["velocity_mps"]  # [3, Z, Y, X]
valid = payload["valid"]                # [Z, Y, X]
```

فریم آخر Flow ندارد چون depth فریم بعد برای correspondence موجود نیست.

## آموزش

برای split معنادار باید چند route/seed مستقل داشته باشید:

```bash
uv run carla-train-voxel-flow \
  --dataset runs/voxel-flow-town10-seed17 \
  --dataset runs/voxel-flow-town10-seed23 \
  --dataset runs/voxel-flow-town10-seed41 \
  --output models/temporal-voxel-flow-v001 \
  --device cuda \
  --history-frames 4 \
  --horizons 0,0.5,1,2 \
  --epochs 20 \
  --batch-size 2
```

Loss نهایی از occupancy، semantic، temporal consistency و sparse flow Smooth-L1 ساخته می‌شود. Flow فقط روی voxelهایی supervise می‌شود که Teacher آن‌ها را dynamic و معتبر شناخته است.

خروجی `summary.json` شامل این موارد است:

- occupied IoU و semantic mIoU
- dynamic-object IoU در هر horizon
- Flow EPE و MAE بر حسب m/s
- inference mean/p95 latency
- CUDA peak memory در صورت اجرای CUDA
- split summary

## استفاده checkpoint در camera-only evaluation

Flow head یک auxiliary training head است. planner هنوز فقط occupancy/semantics را از contract استاندارد می‌گیرد:

```bash
uv run carla-voxel-test \
  --mode teacher \
  --predictor-factory carla_vision.voxel.model_examples.temporal_flow:create_predictor \
  --predictor-checkpoint models/temporal-voxel-flow-v001/checkpoint_best.pt \
  --predictor-device cuda \
  --history-frames 4 \
  --frames 300 \
  --output runs/voxel-flow-eval-v001
```

سپس benchmark occupancy آینده در برابر empty-space و persistence مثل قبل اجرا می‌شود:

```bash
uv run carla-benchmark-voxel \
  --run runs/voxel-flow-eval-v001 \
  --output reports/voxel-flow-v001.json
```

## شرط تکمیل Issue #7

کد Flow به‌تنهایی برای بستن Issue کافی نیست. قبل از بستن باید روی CARLA واقعی:

1. چند route و seed مستقل جمع‌آوری شوند.
2. checkpoint واقعی آموزش داده شود.
3. مدل در horizon آینده از persistence بهتر باشد.
4. dynamic IoU و Flow EPE گزارش شوند.
5. latency، uncertainty و peak-memory روی سخت‌افزار هدف ثبت شوند.
6. failure caseها بررسی شوند.

Dataset، checkpoint و run artifactها نباید داخل Git commit شوند.
