# آموزش Temporal Voxel از RGB

این baseline از چند فریم RGB گذشته استفاده می‌کند و occupancy سه‌بعدی فعلی و آینده را پیش‌بینی می‌کند. در مسیر inference هیچ Depth، Semantic، LiDAR، actor ID، map یا waypoint به مدل داده نمی‌شود.

## خروجی‌های مدل

- `occupancy_logits`: شکل `(B, T, Z, Y, X)`
- `semantic_logits`: شکل `(B, T, C, Z, Y, X)`
- horizonهای پیش‌فرض: `0، 0.5، 1 و 2 ثانیه`

Semantic خام CARLA برای کاهش حافظه به هفت گروه فشرده می‌شود:

1. building
2. road
3. sidewalk
4. vehicle
5. pedestrian
6. vegetation
7. other

فضای `unknown` و voxelهای free در semantic loss نادیده گرفته می‌شوند.

## آماده‌سازی داده

ابتدا چند Episode با route و seedهای متفاوت بسازید:

```bash
uv run carla-voxel-test \
  --mode teacher \
  --frames 600 \
  --history-frames 4 \
  --forecast-horizons 0,0.5,1,2 \
  --output runs/voxel-town10-route01-seed001
```

برای جلوگیری از leakage، split در سطح گروه `route_id + seed` انجام می‌شود. اگر manifest هنوز این فیلدها را ندارد، نام Episode به‌عنوان group استفاده می‌شود. برای ارزیابی معتبر، در manifest هر run مقدارهای `route_id` و `seed` ثبت شوند.

## Dry run بدون Dataset

```bash
uv run carla-train-voxel \
  --dry-run \
  --device cpu \
  --history-frames 4 \
  --horizons 0,0.5,1,2
```

این فرمان یک batch مصنوعی می‌سازد و forward، loss و metricها را بررسی می‌کند.

## بررسی Dataset واقعی بدون آموزش

```bash
uv run carla-train-voxel \
  --dataset runs/voxel-town10-route01-seed001 \
  --dataset runs/voxel-town10-route02-seed002 \
  --dry-run \
  --device cpu
```

## آموزش

```bash
uv run carla-train-voxel \
  --dataset runs/voxel-town10-route01-seed001 \
  --dataset runs/voxel-town10-route02-seed002 \
  --dataset runs/voxel-town10-route03-seed003 \
  --output models/temporal-voxel-v001 \
  --device cuda \
  --history-frames 4 \
  --horizons 0,0.5,1,2 \
  --image-size 192x320 \
  --epochs 20 \
  --batch-size 2
```

خروجی شامل این فایل‌ها است:

```text
models/temporal-voxel-v001/
├── checkpoint_best.pt
├── checkpoint_last.pt
├── config.json
├── history.json
└── summary.json
```

مسیر `models/` در `.gitignore` قرار دارد و checkpoint نباید وارد Git شود.

## استفاده در `carla-voxel-test`

```bash
uv run carla-voxel-test \
  --mode teacher \
  --predictor-factory \
    carla_vision.voxel.model_examples.temporal_baseline:create_predictor \
  --predictor-checkpoint models/temporal-voxel-v001/checkpoint_best.pt \
  --predictor-device cuda \
  --history-frames 4 \
  --frames 300 \
  --output runs/temporal-voxel-eval-v001
```

Factory checkpoint فقط `rgb_history` و `VoxelGridSpec` را دریافت می‌کند. اگر Grid زمان اجرا با Grid checkpoint یکسان نباشد، inference متوقف می‌شود.

## Loss و ارزیابی

- focal BCE فقط روی voxelهای known
- semantic cross entropy فقط روی voxelهای occupied
- temporal consistency با وزن کم
- occupied IoU برای هر horizon
- free-space precision/recall
- Brier score برای calibration
- compact semantic mIoU

## محدودیت فعلی

این baseline برای ایجاد یک مسیر آموزش واقعی و قابل بازتولید است، نه معماری نهایی پژوهشی. مدل فعلی از encoder سبک، GRU و decoder سه‌بعدی استفاده می‌کند. هنوز voxel flow ندارد، زیرا Dataset فعلی بردار حرکت voxel پایدار تولید نمی‌کند. گام بعدی افزودن ego-motion alignment و flow target برای خودروها و عابران است.
