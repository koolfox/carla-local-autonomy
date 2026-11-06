# کارگاه CARLA Vision

## مسئله و محدودهٔ کار

- مسئله: دریافت تصویر دوربین شبیه‌ساز، اجرای مدل بینایی و ثبت نتیجه برای بررسی بعدی.
- سامانهٔ موجود، ابزار پژوهشی برای CARLA 0.9.16 است؛ نمایش مدل و کنترل خودرو دو مسیر جدا دارند.
- این کارگاه پیاده‌سازی، نمونهٔ داده و روش ارزیابی را بررسی می‌کند؛ ادعای عملکرد خودروی خودران ندارد.

> منبع: [معرفی فارسی پروژه](https://github.com/koolfox/carla-local-autonomy/blob/main/README.fa.md) و [مقالهٔ CARLA](https://proceedings.mlr.press/v78/dosovitskiy17a.html).

## معماری اجرا

[رسانهٔ واقعی منتخب برای معماری و اجرای پروژه](#slot-project)

- CARLA و World Worker روی میزبان شبیه‌ساز، صحنه و حسگرها را مدیریت می‌کنند.
- Operator روی رایانهٔ پژوهشی رابط وب، اجرای مدل و ثبت خروجی را هماهنگ می‌کند.
- در حالت دو رایانه، تصویر و درخواست‌ها بین Worker و Operator جابه‌جا می‌شوند.

> مسیرها و نقش‌ها: [راهنمای توسعه](https://github.com/koolfox/carla-local-autonomy/blob/main/docs/development.md). این اسلاید نتیجهٔ آزمایش شبکه نیست.

## قرارداد آشکارساز

- `DetectorConfig` نام backend، مسیر وزن، دستگاه، اندازهٔ ورودی و آستانه را مشخص می‌کند.
- `create_detector` بر اساس تنظیم، YOLO، RT-DETR، M9 یا کارخانهٔ سفارشی را انتخاب می‌کند.
- قرارداد مشترک شامل `infer` و `close` است؛ backend از مصرف‌کننده پنهان می‌ماند.

> کد: [contracts.py](https://github.com/koolfox/carla-local-autonomy/blob/main/carla_vision/contracts.py) و [detectors/factory.py](https://github.com/koolfox/carla-local-autonomy/blob/main/carla_vision/detectors/factory.py).

## M9: بارگذاری و اعتبار فایل وزن

[تصویر یا ویدیوی تأییدشدهٔ خروجی آشکارساز M9](#slot-detection)

- آداپتور M9 اندازهٔ ورودی **۸۰۰×۸۰۰** و checkpoint مشخص‌شده را می‌پذیرد.
- پیش از بارگذاری checkpoint قدیمی، SHA-256 فایل و نسخهٔ پشتیبانی‌شدهٔ Ultralytics بررسی می‌شود.
- خروجی شامل پیش‌بینی‌های ریز و درشت است؛ امتیاز هرکدام جداگانه گزارش می‌شود.

> کد: [m9_hierarchical.py](https://github.com/koolfox/carla-local-autonomy/blob/main/carla_vision/detectors/m9_hierarchical.py) و [راهنمای M9](https://github.com/koolfox/carla-local-autonomy/blob/main/docs/m9_detector.md). تطابق هش، دقت مدل را ثابت نمی‌کند.

## DeiT: تشخیص نوع علامت

- DeiT یک مرحلهٔ اختیاری پس از M9 برای دسته‌بندی تصویر بریده‌شدهٔ علامت است.
- checkpointهای موجود، سر خروجی **۶۴ یا ۶۸ کلاس** دارند و باید با جدول کلاس متناظر استفاده شوند.
- بارگذار تطابق شمار کلاس‌ها و وزن‌های سر مدل را بررسی می‌کند.

> کد: [deit64.py](https://github.com/koolfox/carla-local-autonomy/blob/main/carla_vision/detectors/deit64.py) و [جدول‌های کلاس](https://github.com/koolfox/carla-local-autonomy/tree/main/models).

## بخش‌بندی جاده و خط

[تصویر یا ویدیوی تأییدشدهٔ خروجی بخش‌بندی](#slot-segmentation)

- YOLOP با ONNX و YOLOPv2 با TorchScript از مسیرهای پشتیبانی‌شدهٔ پروژه‌اند.
- در YOLOPv2، تصویر به RGB تبدیل و به بازهٔ صفر تا یک مقیاس می‌شود؛ خروجی جاده و خط جداست.
- ماسک خروجی، پیش‌بینی مدل است و جای برچسب ارزیابی را نمی‌گیرد.

> کد و منبع: [yolopv2.py](https://github.com/koolfox/carla-local-autonomy/blob/main/carla_vision/segmentation/yolopv2.py)، [راهنمای مدل‌ها](https://github.com/koolfox/carla-local-autonomy/blob/main/docs/road_models.md) و [مقالهٔ YOLOP](https://arxiv.org/abs/2108.11250).

## صف فریم در بخش‌بندی

- Worker بخش‌بندی یک فریم در انتظار نگه می‌دارد؛ فریم جدید می‌تواند فریم منتظر قبلی را جایگزین کند.
- شمار فریم‌های ارسال‌شده، پردازش‌شده و کنارگذاشته‌شده ثبت می‌شود.
- این سیاست از رشد صف جلوگیری می‌کند؛ خودِ آن تضمین نرخ یا تأخیر مشخص نیست.

> کد: [segmentation/worker.py](https://github.com/koolfox/carla-local-autonomy/blob/main/carla_vision/segmentation/worker.py). اندازه‌گیری تأخیر باید در اجرای هدف انجام شود.

## تخمین عمق و نمایش وکسل

- آداپتور RGB-depth، مدل Depth Anything V2 متریک را با ورودی تصویر بارگذاری می‌کند.
- عمق تخمینی با پارامترهای دوربین به نقاط سطحی و سپس نمایش وکسل تبدیل می‌شود.
- این مسیر، عمق مرجع CARLA یا برآورد کامل فضای آزاد نیست.

> کد: [voxel/rgb_depth.py](https://github.com/koolfox/carla-local-autonomy/blob/main/carla_vision/voxel/rgb_depth.py)؛ منبع روش: [Depth Anything V2](https://arxiv.org/abs/2406.09414).

## یک نمونهٔ دادهٔ واقعی در مخزن

[تصویر یا ویدیوی منتخب از نمونهٔ دادهٔ واقعی](#slot-dataset)

- نمونهٔ ثبت‌شده شامل **۳ فریم RGB** و **۲۷ برچسب شیء** است.
- هر سه فریم از یک اجرای ثابت‌اند و در بخش `unassigned` باقی مانده‌اند.
- داده برای بازبینی ساختار و مسیر تولید مفید است؛ مجموعهٔ آزمون مدل نیست.

> مدرک: [dataset.json](https://github.com/koolfox/carla-local-autonomy/blob/main/samples/data/ds-carla0916-town10-pilot-v001/dataset.json) و [راهنمای نمونه‌ها](https://github.com/koolfox/carla-local-autonomy/blob/main/samples/README.md).

## RGB و برچسب مرجع را تفکیک کنیم

- نمونهٔ ذخیره‌شده برای هر تصویر، فایل RGB، ماسک instance و فرادادهٔ مرتبط دارد.
- شناسهٔ فریم CARLA و مسیر و SHA-256 هر فایل در descriptor نمونه ثبت شده است.
- ماسک instance از حسگر معلم شبیه‌ساز می‌آید؛ ورودی معمول مدل، تصویر RGB است.

> مدرک: [نمونهٔ ذخیره‌شده](https://github.com/koolfox/carla-local-autonomy/tree/main/samples/data/ds-carla0916-town10-pilot-v001) و [پروتکل فریم](https://github.com/koolfox/carla-local-autonomy/blob/main/docs/experiment_protocol.md#62-camera-and-frame-contract).

## از خروجی مدل تا سنجش دقت

- جعبهٔ پیش‌بینی‌شده با برچسب مرجع و آستانهٔ مشخص مقایسه می‌شود.
- AP، precision و recall به نوع خطا و تعداد نمونه حساس‌اند؛ یک امتیاز روی تصویر، دقت مدل نیست.
- آستانه را باید روی validation انتخاب و آزمون نهایی را جدا نگه داشت.

> روش پیشنهادی پروژه: [پروتکل ارزیابی](https://github.com/koolfox/carla-local-autonomy/blob/main/docs/experiment_protocol.md#9-evaluation-metrics). این اسلاید عدد عملکردی گزارش نمی‌کند.

## تأخیر و هم‌زمانی

- زمان inference، سن فریم هنگام نمایش و نرخ واقعی فریم، اندازه‌گیری‌های متفاوتی هستند.
- اگر خروجی متعلق به فریم قدیمی روی صحنهٔ جدید قرار گیرد، مشاهده می‌تواند گمراه‌کننده باشد.
- پروتکل، توزیع‌های p50، p95 و p99 و تعداد فریم‌های کنارگذاشته‌شده را پیشنهاد می‌کند.

> منبع: [معیارهای زمانی پروژه](https://github.com/koolfox/carla-local-autonomy/blob/main/docs/experiment_protocol.md#92-temporal-and-live-runtime). مقدار عددی بدون اجرای ثبت‌شده ارائه نمی‌شود.

## مشاهده با کنترل خودرو فرق دارد

[ویدیوی تأییدشده با مشخص‌بودن شیوهٔ کنترل خودرو](#slot-control)

- نمایش تشخیص و بخش‌بندی به‌تنهایی نشان نمی‌دهد چه مؤلفه‌ای فرمان می‌دهد.
- Traffic Manager متعلق به CARLA است؛ مسیرهای پژوهشی دیگر باید جداگانه معرفی شوند.
- برای ادعای کنترل مدل، ثبت حالت کنترل، فرمان‌ها و آزمون حلقه‌بسته لازم است.

> منبع: [معرفی پروژه](https://github.com/koolfox/carla-local-autonomy/blob/main/README.fa.md) و [محدودهٔ پذیرش زنده](https://github.com/koolfox/carla-local-autonomy/issues/27).

## Operator و رابط کارگاهی

[ضبط صفحه یا تصویر منتخب از کنسول پروژه](#slot-console)

- رابط Garage امکان تنظیم صحنه و انتخاب مسیرهای بینایی را ارائه می‌کند.
- مدل‌ها و ضبط در Operator مدیریت می‌شوند؛ Worker کارهای مربوط به CARLA را انجام می‌دهد.
- عکس رابط، به‌تنهایی شاهد اتصال سالم یا اجرای زندهٔ مدل نیست.

> مالک قابلیت‌ها و آزمون‌ها: [نقشهٔ توسعه](https://github.com/koolfox/carla-local-autonomy/blob/main/docs/README.md).

## ثبت و بازبینی اجرا

- پروژه مسیرهایی برای ویدیوی خام یا حاشیه‌نویسی‌شده، تشخیص‌ها و فرادادهٔ اجرا دارد.
- مانیفست و checksum به شناسایی فایل کمک می‌کنند؛ اعتبار آزمایش به روش جمع‌آوری هم وابسته است.
- فایل تاریخیِ بدون هویت کامل مدل یا زمان‌بندی فریم را نباید نتیجهٔ نهایی تلقی کرد.

> منبع: [خروجی‌های پروژه](https://github.com/koolfox/carla-local-autonomy/blob/main/README.fa.md) و [محدودیت ضبط‌های تاریخی](https://github.com/koolfox/carla-local-autonomy/blob/main/samples/README.md).

## چه شواهدی برای بررسی داریم؟

- descriptor نمونه، تعداد فریم‌ها، برچسب‌ها، شناسهٔ فریم و هش فایل‌ها را ثبت کرده است.
- بررسی‌های خودکار PR شامل آزمون Operator، رابط و مجموعهٔ پژوهشی‌اند.
- نتیجهٔ هر اجرا باید همراه نسخهٔ کد، تنظیمات و شرایط آن بررسی شود.

> شواهد: [descriptor نمونه](https://github.com/koolfox/carla-local-autonomy/blob/main/samples/data/ds-carla0916-town10-pilot-v001/dataset.json) و [CI مربوط به PR #149](https://github.com/koolfox/carla-local-autonomy/actions/runs/42286765138). CI، دقت مدل یا ایمنی رانندگی را ثابت نمی‌کند.

## تمرین: یک خروجی را بررسی کنیم

[فریم یا ویدیوی منتخب همراه با برچسب و مشخصات اجرا](#slot-exercise)

1. خروجی مدل و برچسب مرجع را از هم جدا کنید؛ هرکدام چگونه تولید شده‌اند؟
2. شناسهٔ فریم، نسخهٔ وزن و آستانهٔ تشخیص را پیدا کنید.
3. برای سنجش یک خطا، چه نمونه‌ها و چه معیاری لازم است؟

> تا دریافت رسانهٔ مناسب، از [descriptor نمونهٔ واقعی](https://github.com/koolfox/carla-local-autonomy/blob/main/samples/data/ds-carla0916-town10-pilot-v001/dataset.json) برای تمرین استفاده کنید.

## محدودیت‌ها و گام بعد

- نمونهٔ سه‌فریمی، کوچک و از یک اجرای ثابت است؛ برای سنجش تعمیم کافی نیست.
- آزمون زندهٔ تکرارپذیر، بازیابی خطا و پاک‌سازی actorها نیازمند شواهد ثبت‌شده‌اند.
- برای مقایسهٔ مدل‌ها، دادهٔ مستقل، آستانهٔ ثابت و گزارش خطا و تأخیر لازم است.

> معیارهای باز: [پذیرش CARLA زنده #27](https://github.com/koolfox/carla-local-autonomy/issues/27) و [آزمون صحنهٔ سنگین #56](https://github.com/koolfox/carla-local-autonomy/issues/56).

## منابع اصلی

- [مقالهٔ CARLA، ۲۰۱۷](https://proceedings.mlr.press/v78/dosovitskiy17a.html)
- [RT-DETR، کنفرانس CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html)
- [YOLOP، ۲۰۲۱/۲۰۲۲](https://arxiv.org/abs/2108.11250) و [Depth Anything V2، ۲۰۲۴](https://arxiv.org/abs/2406.09414)
- [کد و مستندات CARLA Vision Research Console](https://github.com/koolfox/carla-local-autonomy)

> برای ادعای مربوط به این پیاده‌سازی، کد و شواهد ثبت‌شدهٔ پروژه را از ادعاهای مقالات اصلی جدا بخوانید.
