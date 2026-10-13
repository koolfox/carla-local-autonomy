# جلسه‌ی تست از راه دور

**فارسی** · [English](remote_test_session.md)

CARLA را روی یک سرور ویندوزی اجرا کنید و از یک کامپیوتر دیگر با هم تستش کنید: شما در مرورگر، و یک عامل برنامه‌نویسی (یا هر اسکریپتی) با فرمان `carla-remote`. دو راه وجود دارد. هر وقت هر دو کامپیوتر در یک شبکه هستند از مسیر شبکه‌ی محلی استفاده کنید؛ وقتی نیستند، از مسیر Cloudflare.

| | مسیر شبکه‌ی محلی (LAN) | مسیر Cloudflare |
| --- | --- | --- |
| وب‌اپ کجا اجرا می‌شود | کامپیوتر دیگر (مثلاً مک) | سرور ویندوزی |
| چه چیزی از شبکه عبور می‌کند | ورکرِ احرازهویت‌شده، فقط در شبکه‌ی محلی | HTTPS به دامنه‌ی شما در Cloudflare، پشت Cloudflare Access |
| چه چیزی روی اینترنت باز است | هیچ چیز | فقط وب‌اپ، و فقط برای هویت‌هایی که شما اجازه می‌دهید |
| کیفیت تصویر | MJPEG زنده‌ی کامل | عکس لحظه‌ای و ویدیوی بدون تضمین (Cloudflare جریان را بافر می‌کند) |

در هیچ‌کدام از دو مسیر، ورکر (8766) و CARLA (2000 تا 2002) روی اینترنت باز نمی‌شوند.

## ۱. آماده کردن سرور ویندوزی (هر دو مسیر)

۱. طبق [راهنمای کاربر](user_guide_fa.md#ویندوز) نصب و راه‌اندازی کنید: `restore.ps1 … --role local`، سپس **Setup ← Install / Repair**.

۲. در **Run** دکمه‌ی **Start all** را بزنید و صبر کنید تا CARLA، Worker و Web وضعیت **Ready** را نشان دهند.

## ۲-الف. مسیر شبکه‌ی محلی

روی سرور ویندوزی:

۱. در **Setup** گزینه‌ی **Share Worker with another computer on this LAN** را فعال کنید، آدرس شبکه‌ی سرور را انتخاب کنید، **Allow LAN in Windows Firewall…** و **Save settings** را بزنید، سپس **Export connection**.

۲. فایل اتصال را به‌صورت خصوصی به کامپیوتر دیگر منتقل کنید (توکن ورکر داخل آن است).

روی کامپیوتر دیگر (مثلاً مک):

۱. لانچر را باز کنید، در **Setup** گزینه‌ی *Web app — connect to another CARLA computer* را انتخاب کنید، مسیر فایل را وارد کنید و **Import connection** و بعد **Check connection** را بزنید.

۲. **Run ← Start web app**. کنسول در `http://127.0.0.1:8765/` است.

در این حالت `carla-remote` بدون تنظیم اضافه کار می‌کند:

```bash
uv run carla-remote status
```

## ۲-ب. مسیر Cloudflare

به یک دامنه‌ی مدیریت‌شده در Cloudflare نیاز دارید. همه‌ی مراحل زیر روی سرور ویندوزی انجام می‌شود، مگر جایی که گفته شده.

### ساخت تونل

```powershell
winget install --id Cloudflare.cloudflared
cloudflared tunnel login
cloudflared tunnel create carla-garage
```

فایل `%USERPROFILE%\.cloudflared\config.yml` را بسازید (شناسه‌ی تونل، نام کاربری و دامنه را جایگزین کنید):

```yaml
tunnel: <TUNNEL-UUID>
credentials-file: C:\Users\<YOU>\.cloudflared\<TUNNEL-UUID>.json

ingress:
  - hostname: carla.example.com
    service: http://127.0.0.1:8765
  - service: http_status:404
```

```powershell
cloudflared tunnel route dns carla-garage carla.example.com
cloudflared tunnel run carla-garage
```

وب‌اپ هنگام شروع این فایل را می‌خواند و `carla.example.com` را خودکار می‌پذیرد، چون به پورت loopback خود وب‌اپ اشاره می‌کند. بعد از تغییر دامنه، وب‌اپ را دوباره راه‌اندازی کنید (**Stop** و بعد **Start all**). برای اجرای تونل به‌صورت سرویس ویندوز، `cloudflared service install` را اجرا کنید و `config.yml` و فایل JSON اعتبارنامه را به `C:\Windows\System32\config\systemprofile\.cloudflared\` کپی کنید؛ وب‌اپ این مسیر را هم می‌خواند.

### محافظت با Cloudflare Access (الزامی)

در داشبورد Cloudflare به **Zero Trust ← Access controls ← Applications ← Create new application ← Self-hosted** بروید، دامنه را `carla.example.com` بگذارید و دو policy تعریف کنید:

۱. **Allow**: گزینه‌ی *Emails* برابر با آدرس ایمیل خودتان. این برای خود شما در مرورگر است.

۲. **Service Auth**: توکن سرویسی را که در **Zero Trust ← Access controls ← Service credentials ← Service Tokens** ساخته‌اید (مثلاً `carla-remote-test`) اضافه کنید. Client ID و Client Secret را همان لحظه کپی کنید؛ Secret فقط یک بار نمایش داده می‌شود.

هرگز policy از نوع **Bypass** اضافه نکنید. بدون Access، هر کسی که دامنه را پیدا کند می‌تواند شبیه‌ساز را براند و کارهای پژوهشی اجرا کند.

### دادن توکن به `carla-remote` (روی کامپیوتر تست)

یک فایل خصوصی بسازید؛ Secret را هرگز در چت، issue یا commit قرار ندهید:

```bash
mkdir -p ~/.config/carla-vision
cat > ~/.config/carla-vision/remote.env <<'EOF'
CARLA_REMOTE_URL=https://carla.example.com
CF_ACCESS_CLIENT_ID=<client id>.access
CF_ACCESS_CLIENT_SECRET=<client secret>
EOF
chmod 600 ~/.config/carla-vision/remote.env
```

`carla-remote` فایلی را که دیگر کاربران بتوانند بخوانند نمی‌پذیرد، اعتبارنامه را فقط به آدرس `https://` می‌فرستد و هرگز توکن داخلی وب‌اپ را چاپ نمی‌کند.

## ۳. تست با هم

به عامل بگویید سرور آماده است. سپس می‌تواند این فرمان‌ها را اجرا کند:

| فرمان | کاربرد |
| --- | --- |
| `uv run carla-remote status` | نسخه و نقشه‌ی CARLA، وضعیت ورکر، گاراژ، رانندگی و آخرین بررسی زنده |
| `uv run carla-remote snapshot --view garage --out garage.jpg` | ذخیره‌ی فریم فعلی دوربین گاراژ |
| `uv run carla-remote snapshot --view drive --out drive.jpg` | ذخیره‌ی فریم فعلی دوربین رانندگی |
| `uv run carla-remote live-check` | اجرای بررسی زنده‌ی CARLA و نمایش نتیجه (نقشه‌ی Town10HD را دوباره بارگذاری می‌کند و حدود یک دقیقه یک خودروی آزمایشی را می‌راند) |

شما می‌توانید همان جلسه را هم‌زمان در مرورگر و از آدرس کنسول ببینید. رانندگی از راه دور از طریق Cloudflare فقط برای تست است و هنوز مسیر کنترلِ تأییدشده برای انتشار نیست؛ ویدیو بافر می‌شود و ممکن است تأخیر داشته باشد.

## ۴. بعد از تست

- تونل را متوقف کنید (`Ctrl+C` یا توقف سرویس `cloudflared`).
- وقتی دوره‌ی تست تمام شد، توکن سرویس را در Cloudflare باطل یا عوض کنید.
- اگر دیگر لازم نیست، فایل `~/.config/carla-vision/remote.env` را روی کامپیوتر تست پاک کنید.
