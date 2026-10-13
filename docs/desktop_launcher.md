# Install and control CARLA Vision through the TUI

The launcher preview runs CARLA 0.9.16, its World Worker and the existing web app
on one machine. Windows x86-64 is the primary target; Linux x86-64 is supported
by the same Python controller and its own bootstrap. The same TUI also runs the
web app alone on macOS, Linux and Windows, connected to a separate CARLA host.
The `main` branch includes
the launcher, published models, historical samples and [recovery tools](restore_project.md).

**Validation status:** source review and package integrity, plus offline recovery
checks. CI exercises the existing regression suite and recovery file-preservation
contracts without CARLA. Headless TUI and launcher contracts also run without
CARLA. Native Windows/Linux installer/TUI execution, full fresh-machine
installation, inference and live acceptance remain pending.

For a current installation, use the [curl bootstrap](../README.md#quick-start-with-curl).
The rebuilt ZIP/TAR previews below include the cross-platform TUI and pairing.
Each archive retains the exact source revision recorded in its manifest; use
the curl bootstrap for the latest main after updates.

## macOS and web-only Linux/Windows

Restore with `--role web`, then open `CARLA-Vision.command` on macOS,
`deploy/desktop/launch.sh` on Linux, or `deploy\desktop\CARLA-Vision.cmd` on Windows.
A checkout's macOS entry point prepares private Python and the TUI if missing;
choose **Install / Repair** to prepare the selected application's packages.
No Python commands, virtual-environment activation or hand-editing `.env.local`
are needed for ordinary use. Intel macOS defaults to **Web / recording** because
the locked vision wheels target Apple Silicon; choose that profile on Intel.

In **Setup**, choose **Web app**, then **Import connection** using a file exported
by the CARLA host TUI. Alternatively enter its CARLA host, Worker URL and matching
token. **Check connection** saves the settings and verifies authenticated health
and CARLA 0.9.16 without changing the world. Use **Run → Start web app**, **Open
web app** and **Stop owned services** for daily use. With URL/token both empty,
the web server can open for saved recordings while the simulator is unavailable.
The launcher supplies FFmpeg's PATH automatically.

Stopping this profile cleans up its local scene owners and stops only its own
web process. It never terminates the remote Worker or CARLA server. Quit also
performs this cleanup; if cleanup cannot be confirmed, services are preserved
and the error explains how to recover.

## Share a Windows/Linux Worker with another web computer

1. In the host TUI choose **CARLA + Worker + web app** and select its CARLA folder.
   **Install / Repair** installs the matching official SDK, NumPy, OpenCV,
   application dependencies, encoder and model files.
2. Enable **Share Worker with another computer on this LAN**. The TUI selects
   an available private IPv4 address; confirm the network you want to use.
   Worker binds to that address instead of loopback. Wildcard/public bindings
   are rejected. Stop owned services before changing host settings.
3. On Windows use **Allow LAN in Windows Firewall…** when required and accept
   UAC. The rule allows the selected Worker port and, when found, the CARLA
   executable's RPC/streaming ports from the local subnet on **Private** networks.
   It does not open the Public profile. On Linux, the distribution's firewall
   must permit these same ports; see its administrator's network policy.
4. Enter a new connection-file path and choose **Export connection**, then
   **Run → Start all**. Transfer the file privately to the web computer and import
   it in that computer's TUI. The file includes a credential; never commit it.
5. Use **Check connection** on the client and **Start web app**. Keep one Worker
   per port. When the LAN address changes, stop the host services, select the new
   address, update the firewall and export/import a fresh file.

Sharing is opt-in. Same-computer use keeps generated credentials and loopback
defaults without firewall changes.

## Prerequisites and recovery in the TUI

- Missing application packages, SDK or camera dependencies: **Install / Repair**.
  Setup checks that NumPy/OpenCV and, in local mode, the CARLA SDK can load.
- Windows native-library/runtime errors: select the official CARLA folder and
  use **Install CARLA prerequisites…**. This runs
  `Engine/Extras/Redist/en-us/UE4PrereqSetup_x64.exe` from that package with UAC,
  without requesting an automatic reboot. If Windows reports a restart is needed,
  restart and retry **Install / Repair**. A stripped package may lack this installer.
- Worker authentication failed: import a new host connection file or paste the
  matching token; it stays masked and is omitted from public status/command lines.
- Connection refused/timed out: check the host **Run** ledger, LAN address,
  network profile and scoped firewall rule. Connection checks do not change CARLA.
- Port occupied: choose a free app/Worker port in **Setup**. The TUI preserves
  services it did not start.
- Linux native imports failed: install the OS libraries required by CARLA/OpenCV
  (commonly `libgl1` and `libglib2.0-0` on Ubuntu), then retry repair. GPU drivers,
  host OS libraries and administrator-owned network policy remain OS prerequisites.

Manual SDK/Worker commands remain available in the [developer guide](manual_setup.md).

## Install on Windows

For a Mac web client, extract the current ZIP completely and open its top-level
`CARLA-Vision.command`. It prepares private Python/TUI; finish through Setup.


1. Download [CARLA-Vision-setup-preview.zip](../installers/CARLA-Vision-setup-preview.zip)
   and extract the complete archive to a writable local folder.
2. Double-click its top-level **Install.cmd**. Python, Git and Node need not
   already be installed. Setup verifies the pinned uv download and installs
   managed Python 3.12 plus the terminal UI in a private environment.
3. The full-screen launcher opens. In **Setup**, choose the CARLA folder containing
   `CarlaUE4.exe`. Use **Install / Repair** to install the remaining locked vision
   dependencies, the official `carla==0.9.16` PythonAPI, the private recording
   encoder and all ten model weights.
4. Open **Run → Start all**. The launcher starts the server, checks its real
   version/world/map APIs, starts Worker and web, then opens the local browser.

The bundle installs to `%LOCALAPPDATA%\CARLA-Vision\app` and creates a
**CARLA Vision** desktop shortcut. A custom location can be supplied in PowerShell:

```powershell
.\app\deploy\desktop\install.ps1 -Destination 'D:\CARLA-Vision'
```

To install directly in a Git checkout, run `deploy\desktop\Install.cmd`; it
keeps the application in that checkout. The bootstrap uses process-scoped
PowerShell execution policy; it does not change the machine-wide policy or PATH.

## Install on Linux

1. Download [CARLA-Vision-setup-preview.tar.gz](../installers/CARLA-Vision-setup-preview.tar.gz)
   and extract the archive.
2. From the extracted setup folder, run `sh install.sh`.
3. In **Setup**, select the CARLA folder containing `CarlaUE4.sh`, then use
   **Install / Repair** and **Run → Start all**.

The bundle installs to `${XDG_DATA_HOME:-~/.local/share}/carla-vision/app` and
adds **CARLA Vision** to the applications menu with a terminal window. A custom
location is supported with `sh install.sh --destination '/path/CARLA-Vision'`.
From a Git checkout, run `sh deploy/desktop/install.sh` to install in place.
Use `sh deploy/desktop/launch.sh` for daily startup from that checkout.

Linux bootstrap prerequisites are `curl`, `tar` and `sha256sum` (coreutils).
The OS must provide the libraries required by CARLA and OpenCV; Ubuntu installations
typically need `libgl1` and `libglib2.0-0`. Hardware/graphics drivers and any CARLA
system prerequisites remain properties of the host OS. The installer manages
application Python packages, models and FFmpeg without replacing system Python.
CARLA's [official quickstart](https://carla.readthedocs.io/en/0.9.16/start_quickstart/)
describes its host requirements.

## First-time Setup

Setup autodetects common local CARLA paths and accepts an explicit path. If the
server package is missing, select **Download official CARLA if missing** before
**Install / Repair**. This is a multi-gigabyte download with a minimum 35 GiB
free-space check; allow additional space for dependencies, extraction and data.
Existing simulator directories are preserved. Downloads use the URLs published
in the [CARLA 0.9.16 release](https://github.com/carla-simulator/carla/releases/tag/0.9.16).
The download record stores its locally computed archive SHA-256; an upstream
archive checksum was not published, so it is not represented as publisher verification.

The **Vision** profile installs the existing detector, DeiT, segmentation and
voxel runtime dependencies. **Research** adds the existing research/experimental
dependencies; it does not enable experimental actuation by itself. Existing
experimental features retain their explicit opt-in configuration.

The application supplies all ten weights, class tables, processors, configurations
and model notices. The small setup archives carry exact LFS pointers for weights;
**Install / Repair** downloads about 699 MiB from commit
`2c600765e34862becc4f236d99291b295604f973` and verifies each size and SHA-256 from
`models/assets.json`. Git LFS is not needed for this installer path. An ordinary
LFS-enabled clone downloads the same full files. Original sample movies, images,
labels and historical reports are also included in the setup payload.
See [model notices and licenses](../models/README.md) before reusing or distributing assets.

Internet is required for the initial setup. Later startup uses the installed
environment and local model snapshots, including `HF_HUB_OFFLINE=1` for the
pinned Hugging Face caches. Native targets use managed Python 3.12 and the
official Windows/Linux CARLA 0.9.16 wheel. macOS uses the same TUI in web-only
mode; manual entry points remain available for developers.

## Daily use

The full-screen TUI has **Run**, **Setup** and **Logs** tabs. Run shows real
service readiness, external server ownership and actionable failures. Setup
runs dependency installation/downloads outside the UI thread. Logs includes
bounded new CARLA, Worker and web process output with the underlying file paths.

- `Ctrl+R`: Start all.
- `Ctrl+S`: Stop launcher-owned services.
- `Ctrl+B`: Open the ready web app.
- `Ctrl+Q`: Stop owned services and exit; wait for an active operation first.
- `Tab` / `Shift+Tab`: move focus through native terminal controls.

Use a terminal of at least 100 columns by 30 rows for initial acceptance.
Actual resizing, focus, contrast and rendering still need native verification.

Ports default to CARLA `2000`, Worker `8766`, web `8765` and Traffic Manager
`8000`. CARLA streaming/secondary ports are also reserved when validating port
choices. All application connections use `127.0.0.1`; the browser server and
authenticated Worker are bound to loopback.

If CARLA is already running with the correct version, the launcher reuses it
and preserves its process on Stop. It records only processes it starts, with
PID and creation time to avoid terminating a reused PID. Stop asks the existing
Drive, Preview and capture owners to finish; it waits for recording/import and
scene-cleanup confirmation before shutting down owned processes. Uncertain
cleanup preserves services and gives a specific recovery error. Worker preparing
or busy state is not accepted as an idle scene.

## Files, upgrades and recovery

Settings and process receipts live in `.launcher/`. A fresh random Worker
credential is generated in `.launcher/worker.token` and is passed through the
child environment, not command arguments. It is not included in Git or bundles.
The application uses its private `.venv`, `.launcher/python`, `.launcher/uv`,
`.launcher/cache` and `.launcher/bin`. Automatically installed CARLA goes to
`simulator/CARLA_0.9.16` inside the application workspace.

Close the TUI and its services before reinstalling a bundle. Setup updates only
indexed application files; it preserves `.launcher`, `.env.local`, simulator,
runs, datasets, sessions and already hydrated weights. Model repair verifies
the weights; a changed full weight is backed up before replacement. Bootstrap
and model download failures are visible and can be retried. Large server
downloads currently restart after interruption.

Locally edited or previously unindexed files under `configs/` and
`operator_configs/` are preserved. The incoming version is placed under
`.launcher/incoming-defaults/<source-commit>/` for manual comparison. Unchanged
versioned defaults can be updated normally. Every TUI entry point holds a
lifetime lock; bootstrap holds that lock and the operation lock across file
copy and environment synchronization. Crash recovery releases the OS locks,
and service checks compare both PID and creation time.

- Missing folder: select a local CARLA path or request the official download.
- Occupied Worker/web port: stop the other instance or select a free port in Setup.
- Version mismatch: use the matching official CARLA 0.9.16 server and PythonAPI.
- Process exit or timeout: read **Logs** and the reported file path.
- Failed cleanup: use **Stop & Save** in the web app, inspect the owner error,
  wait for the operation to finish, then retry Stop. A busy native owner is preserved.
- Broken dependency installation: rerun **Install / Repair** after resolving the reported error.

## CLI and installer build

The same controller is available as `carla-app`. Inside the installed environment:

```text
carla-app tui
carla-app setup --carla-root <CARLA-folder>
carla-app setup --download-carla
carla-app start --no-browser
carla-app status
carla-app doctor
carla-app stop
```

On Windows, use `.venv\Scripts\carla-app.exe`; on Linux, `.venv/bin/carla-app`.
Global `--workspace <path>` precedes the command. `doctor` reports the same
read-only readiness and returns a nonzero exit code until all services are ready.
The platform bootstrap accepts `-NoLaunch` / `--no-launch` for environment
preparation without opening the TUI; **Install / Repair** still prepares the full runtime.

Maintainers build the distributable payload from a clean committed checkout:

```bash
python -m carla_vision.launcher.bundle --output dist/desktop
```

The builder includes tracked runtime code, compiled console, configurations,
model metadata/notices, sample material and docs. It excludes private credentials,
runtime directories, environments, caches and simulator binaries. Weight entries
are reproducible LFS pointers; the fixed model source and hashes drive hydration.
Both archives have deterministic metadata and an indexed source SHA. The
published preview payloads and checksums are under [installers/](../installers/README.md).

## Live check

After **Start all** (or **Start web app** with a paired Worker), press **Live
check** on the Run tab, then press it again within six seconds to confirm. The
TUI asks the running web app to run its live CARLA check and shows progress and
the result on the Run tab; the same check and result appear in the web app under
**System → Live check**. It reloads Town10HD and drives a test car for about a
minute, so run it before a session, not during one.

## Deferred acceptance

Tomorrow's validation needs actual Windows/Linux hosts and a reachable CARLA:
fresh installation in a path with spaces; model/dependency availability;
server reuse and owned-server startup; real version/map readiness; process
identity and Windows bootstrap descendants; partial startup failure/recovery;
duplicate launchers; recording/capture finalization; clean Stop/Exit; keyboard,
terminal resizing and contrast. This checklist records pending work and does
not schedule or claim any completed acceptance.
