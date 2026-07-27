# Gate آمادگی Actuation برای Voxel Planner

این قابلیت عمداً **فرمان خودرو را فعال نمی‌کند**. هدف آن جلوگیری از اتصال زودهنگام Voxel Planner به `carla.VehicleControl` است.

فاز فعلی دو جزء دارد:

1. `carla-verify-voxel-actuation-readiness`: بررسی Evidence آموزش، validation و Shadow؛
2. `VoxelActuationSupervisor`: اعتبارسنجی fail-closed هر proposal در حافظه، بدون CARLA و بدون `apply_control`.

حتی گزارش `passed` نیز شامل این مقدار است:

```json
{
  "actuation_enabled_by_this_report": false
}
```

برای فعال‌شدن واقعی فرمان، یک PR جدا، acknowledgement صریح اپراتور و اجرای زنده‌ی کنترل‌شده لازم است.

## Dry run

```bash
uv run carla-verify-voxel-actuation-readiness --dry-run
```

این دستور policy پیش‌فرض و بخش‌های Evidence موردنیاز را چاپ می‌کند و همیشه actuation را خاموش نشان می‌دهد.

## فایل‌های نمونه

```text
configs/voxel_actuation_readiness.example.json
configs/voxel_actuation_policy.json
```

فایل Evidence نمونه عمداً قابل پاس‌شدن نیست؛ مقادیر Issue #7، تعداد Dataset و Shadow run در آن صفر یا false هستند.

برای ساخت Evidence واقعی، آن را خارج از Git کپی کن:

```bash
cp configs/voxel_actuation_readiness.example.json \
  runs/voxel-actuation-readiness-v001.json
```

## محاسبه SHA-256 مدل

Linux/macOS:

```bash
sha256sum models/temporal-voxel-v001/checkpoint_best.pt
```

PowerShell:

```powershell
Get-FileHash models/temporal-voxel-v001/checkpoint_best.pt -Algorithm SHA256
```

SHA باید در این فیلد ثبت شود:

```json
{
  "model": {
    "checkpoint_sha256": "..."
  }
}
```

این کار Evidence را به یک checkpoint مشخص متصل می‌کند.

## Evidence الزامی

### پذیرش Issue #7

```json
{
  "issue_7_acceptance_complete": true,
  "operator_review_complete": true
}
```

این دو مقدار نباید صرفاً برای عبور از gate به‌صورت دستی true شوند. `issue_7_acceptance_complete` یعنی Dataset واقعی، آموزش، validation و مقایسه با baselineها انجام شده‌اند. `operator_review_complete` یعنی failure caseها و telemetry توسط انسان بازبینی شده‌اند.

### قرارداد مدل

```json
{
  "model": {
    "rgb_only_runtime_contract": true,
    "checkpoint_sha256": "64-character-sha256"
  }
}
```

### آموزش و split

```json
{
  "training": {
    "dataset_verification_status": "passed",
    "route_group_leakage": false,
    "dataset_count": 3,
    "route_group_count": 20
  }
}
```

### Validation

```json
{
  "validation": {
    "current_occupied_iou": 0.35,
    "future_occupied_iou_1s": 0.20,
    "brier_score": 0.20,
    "p95_latency_ms": 100.0
  }
}
```

اعداد بالا حد پیش‌فرض policy هستند، نه ادعای اینکه این thresholdها برای رانندگی واقعی کافی‌اند. بعد از جمع‌آوری داده باید براساس risk review سخت‌گیرانه‌تر شوند.

### Shadow

```json
{
  "shadow": {
    "run_count": 3,
    "record_count": 300,
    "error_rate": 0.01,
    "p95_latency_ms": 120.0,
    "maximum_uncertain_voxel_fraction": 0.45,
    "actuation_enabled": false,
    "control_calls": 0
  }
}
```

Evidence فقط از اجرای read-only Shadow پذیرفته می‌شود.

## اجرای Verifier

```bash
uv run carla-verify-voxel-actuation-readiness \
  --evidence runs/voxel-actuation-readiness-v001.json \
  --policy configs/voxel_actuation_policy.json \
  --output reports/voxel-actuation-readiness-v001.json
```

Exit code:

- `0`: تمام checkها پاس شده‌اند؛ actuation هنوز فعال نشده است؛
- `2`: حداقل یک check شکست خورده و actuation باید unavailable بماند؛
- خطای Python: Evidence یا policy از نظر ساختاری نامعتبر است.

گزارش شامل hash canonical خود Evidence است تا تغییر فایل پس از review قابل تشخیص باشد.

## Safety Supervisor

کلاس `VoxelActuationSupervisor` به‌صورت تابع pure پیاده‌سازی شده و proposal زیر را ارزیابی می‌کند:

- frame؛
- زمان تولید prediction؛
- latency؛
- steering پیشنهادی؛
- collision risk؛
- uncertain voxel fraction؛
- `CameraVoxelPrediction` کامل.

ورودی runtime تکمیلی:

- سرعت فعلی خودرو؛
- steering قبلی؛
- `dt`؛
- readiness report؛
- `VoxelGridSpec`.

در این حالت‌ها خروجی همیشه safe stop است:

- readiness report پاس نشده؛
- frame یا scalar نامعتبر؛
- prediction قدیمی یا timestamp آینده؛
- latency بیش از حد؛
- shape، NaN یا probability نامعتبر؛
- horizon ناکافی؛
- uncertainty بالا؛
- collision risk بالا؛
- سرعت بیش از limit؛
- steering خارج از limit.

Safe stop انتزاعی:

```json
{
  "actuation_authorized": false,
  "emergency_brake": true,
  "steering": 0.0
}
```

برای proposal معتبر، steering با نرخ مجاز محدود می‌شود. این تصمیم هنوز به CARLA ارسال نمی‌شود.

## مرز مهم

این PR عمداً موارد زیر را ندارد:

- `carla.VehicleControl`؛
- `vehicle.apply_control`؛
- throttle controller؛
- اجرای actuating در `carla-local-drive`؛
- acknowledgement flag برای اجرای زنده.

CI با AST نبودن `apply_control` را کنترل می‌کند. مرحله‌ی اتصال واقعی فقط بعد از پاس‌شدن Issue #7، اجرای Shadow و review جداگانه قابل بررسی است.
