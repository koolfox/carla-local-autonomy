# یکپارچه‌سازی Garage، Cockpit و Research

این تغییر `carla-operator-ui` را از طریق یک wrapper افزایشی اجرا می‌کند. موتور Browser Drive، HTTP API پایه، Research Tools موجود، Voxel/Imitation runners و فایل‌های static اصلی بازنویسی نشده‌اند.

## Garage

Garage علاوه بر انتخاب خودرو، رنگ، آب‌وهوا و صحنه، این حالت‌های کنترل را نمایش می‌دهد:

- `Manual`: همان کنترل Browser با deadman فعلی.
- `BehaviorAgent`: کنترل کامل توسط BehaviorAgent با Emergency Brake مستقل Browser.
- `Imitation`: مدل RGB + speed موجود، با دوربین RGB مستقیم CARLA و fail-closed.
- `Voxel Planner`: BehaviorAgent برای longitudinal و Voxel supervisor/planner برای steering؛ rejection یا stale prediction به full brake منجر می‌شود.

حالت‌های autonomous فقط با acknowledgement صریح قابل شروع هستند. Browser control در زمان مالکیت autonomous رد می‌شود.

در صورت در دسترس بودن CARLA PythonAPI، Traffic و Pedestrians از Garage قابل انتخاب هستند و actorهای ایجادشده متعلق به همان session بوده و در پایان پاک می‌شوند.

## Cockpit

Cockpit، camera stream، recording، telemetry، SafeActuator و Emergency Brake موجود را حفظ می‌کند. تغییر mode فقط منبع command را عوض می‌کند؛ actuator و cleanup فعلی بازنویسی نشده‌اند.

## Saved Run → Research

پس از Save همان Run بدون وارد کردن دوباره path قابل:

- Inspect
- Analyze
- Verify

است.

Research launcher همچنین commandهای allow-listed زیر را اجرا می‌کند و هیچ shell/module دلخواهی از Browser قبول نمی‌کند:

- BehaviorAgent teacher capture
- Imitation training
- Voxel teacher/RGB prediction capture
- Voxel Flow teacher capture
- Voxel occupancy training
- Voxel + Flow training
- Voxel Shadow
- Voxel benchmark
- Closed-loop observer

Teacher capture هنگام Drive فعال مسدود است، چون map reload و ownership انحصاری tick دارد. Live capture/shadow/evaluation بدون ego فعال فقط در حالت dry-run مجاز است.

## مرز اعتبارسنجی

CI این موارد را بررسی می‌کند:

- contract و path safety
- acknowledgement و control ownership
- fail-closed قبل از policy initialization
- عدم دسترسی Browser integration به `/api/drive/control`
- عدم پذیرش arbitrary shell command
- regression تست‌های Drive Console فعلی
- JavaScript syntax و operator entrypoint

CI جای اجرای زنده CARLA را نمی‌گیرد. BehaviorAgent/Imitation/Voxel actuation زمانی از نظر رفتار simulator تأیید می‌شوند که روی CARLA 0.9.16 زنده اجرا شوند. Closed-loop observer داخل خود Garage برای ثبت route completion، collision، lane invasion و brake behavior قابل اجرا است و هیچ vehicle controlی اعمال نمی‌کند.
