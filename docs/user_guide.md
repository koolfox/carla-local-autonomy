# User guide

**English** · [فارسی](user_guide_fa.md)

This guide takes you from a fresh computer to a saved driving session. Pick
your operating system first; everything after installation is the same on all
of them.

| Your computer | What runs on it | Go to |
| --- | --- | --- |
| Windows x86-64 | CARLA, the World Worker, the web app and the launcher | [Windows](#windows) |
| Linux x86-64 | CARLA, the World Worker, the web app and the launcher | [Linux](#linux) |
| macOS 14+ (Apple Silicon or Intel) | The web app only; CARLA runs on a Windows/Linux computer | [macOS](#macos) |

You need CARLA 0.9.16 on a Windows or Linux computer with a suitable graphics
card. The installer can download CARLA for you.

## The two windows you use

**The launcher (TUI)** is a full-screen terminal window. It installs
everything, then starts and stops CARLA, the World Worker and the web app.
Its tabs are **Run**, **Setup** and **Logs**.

<p align="center"><img src="images/tui-run.webp" alt="Launcher Run tab with CARLA, Worker and web app ready" width="760"></p>

**The web app** is the console you use day to day: Garage, settings, driving
and recordings. The launcher opens it in your browser at
`http://127.0.0.1:8765/`.

<p align="center"><img src="images/console-garage.webp" alt="Web app Garage with the Scene settings panel" width="760"></p>

---

## Windows

### 1. Install

Open **PowerShell** and run:

```powershell
curl.exe -fL https://raw.githubusercontent.com/koolfox/carla-local-autonomy/refs/heads/main/deploy/restore/restore.ps1 -o restore.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\restore.ps1 --destination "C:\CARLA-Vision" --role local --download-carla
```

If CARLA 0.9.16 is already installed, replace `--download-carla` with
`--carla-root "C:\CARLA_0.9.16"`. Use an empty destination folder; if the
download is interrupted, run the same command again with `--resume`.

### 2. Open the launcher

```powershell
cd C:\CARLA-Vision
.\deploy\desktop\CARLA-Vision.cmd
```

After the first setup you can also use the **CARLA Vision** shortcut.

### 3. Set up once

In **Setup**, check that **Run on this computer** is *CARLA + Worker + web app*
and that the CARLA folder is correct, then press **Install / Repair**.

<p align="center"><img src="images/tui-setup.webp" alt="Launcher Setup tab" width="760"></p>

If CARLA cannot load its native libraries, press **Install CARLA
prerequisites…** (Windows asks for administrator permission), restart if
asked, and run **Install / Repair** again.

### 4. Start

Go to **Run** and press **Start all** (or `Ctrl+R`). When CARLA, Worker and Web
show **Ready**, press **Open web app** (`Ctrl+B`). Continue with
[Using the web app](#using-the-web-app).

**Stop** (`Ctrl+S`) cleans up the scene and stops what the launcher started.
A CARLA server you started yourself is left running.

### Using another computer as the screen (optional)

To drive from a Mac or a second PC, enable **Share Worker with another computer
on this LAN** in Setup, choose this computer's LAN address, press **Allow LAN
in Windows Firewall…**, then **Export connection**. Copy the exported file to
the other computer; it contains a password, so keep it private.

---

## Linux

### 1. Install

```bash
curl -fsSL https://raw.githubusercontent.com/koolfox/carla-local-autonomy/refs/heads/main/restore.sh -o restore.sh
sh restore.sh --destination "$HOME/CARLA-Vision" --role local --download-carla
```

To use an existing CARLA, replace `--download-carla` with
`--carla-root /path/to/CARLA_0.9.16`. Graphics drivers are not installed for
you; make sure CARLA itself runs on this machine.

### 2. Open the launcher

```bash
cd "$HOME/CARLA-Vision"
sh deploy/desktop/launch.sh
```

After setup, the **CARLA Vision** entry in your applications menu does the
same. Use a terminal of at least 100 × 30 characters.

### 3. Set up and start

Same as Windows: **Setup → Install / Repair**, then **Run → Start all** and
**Open web app**. For LAN sharing, your distribution's firewall must allow TCP
8766 (Worker) and 2000–2002 (CARLA) from your local network only, for example
`sudo ufw allow from 192.168.1.0/24 to any port 2000:2002,8766 proto tcp`.

---

## macOS

A Mac runs the web app, recordings and vision. CARLA and the World Worker run
on a Windows or Linux computer on the same network.

### 1. Prepare the CARLA computer

On the Windows/Linux computer, follow its section above, enable **Share Worker
with another computer on this LAN**, **Start all**, and **Export connection**.
Copy the connection file to the Mac (AirDrop, USB, or a private share).

### 2. Install on the Mac

Open **Terminal**:

```bash
curl -fsSL https://raw.githubusercontent.com/koolfox/carla-local-autonomy/refs/heads/main/restore.sh -o restore.sh
sh restore.sh --destination "$HOME/CARLA-Vision" --role web
cd "$HOME/CARLA-Vision"
./CARLA-Vision.command
```

### 3. Connect and start

In **Setup**, choose **Web app — connect to another CARLA computer**, enter the
path of the connection file and press **Import connection**, then **Check
connection**. In **Run**, press **Start web app**, then **Open web app**.

Saved recordings can be watched on the Mac even when the CARLA computer is off.
On Intel Macs the model files are restored, but live detection needs a
supported vision host.

---

## Using the web app

The screen works like a design tool: the live CARLA camera is in the middle,
sections are on the left, their settings are on the right, and the main
actions are in the floating bar at the bottom.

<p align="center"><img src="images/console-vision.webp" alt="Vision settings beside the Garage preview" width="760"></p>

| Area | What it does |
| --- | --- |
| Top bar | Current map and run ID, and the status of CARLA, the Worker bridge and the session. A **solid dot** means OK or idle, a **ring** offline or failed, a **struck ring** not configured, a **dashed ring** waiting, a **half-filled dot** syncing. Click the status to open **System**. |
| Left rail | **Experiment**, **Scene** (map, weather, traffic, pedestrians, start point, route), **Vehicle** (car, control owner, models), **Vision** (camera, detection, road and voxel overlays, recording), **Research**, **Recordings**, **System**. |
| Right panel | Settings of the selected section. Changes save immediately and sync to the live Garage after a short pause; the panel says when changes are still syncing. Hide it with the button at the top right for a larger camera. |
| Camera | The live Garage view. Drag to orbit, scroll to zoom. The applied resolution and FPS are above it (amber when CARLA capped your request), and the applied map, weather and population below it. |
| Bottom bar | Previous/next vehicle, paint, camera angle (orbit, front, rear, top, cockpit) and **Start session**. |

### Drive

Press **Start session**. The camera fills the canvas, with speed, gear, RPM and
elapsed time at the bottom left and **Emergency Brake** and **Stop & Save** in
the bottom bar.

<p align="center"><img src="images/console-drive.webp" alt="Driving with HUD, Emergency Brake and Stop & Save" width="760"></p>

Click the camera, then drive with the keyboard:

| Key | Action |
| --- | --- |
| `W` / `↑` | Throttle |
| `S` / `↓` | Brake |
| `A` / `←`, `D` / `→` | Steer |
| `Space` | Handbrake |
| `Shift` + `W` | Reverse |

On a phone or tablet, use the round touch keys instead. If input stops or the
camera freezes, the car brakes by itself. **Stop & Save** ends the run and
writes it to `runs/<run-id>/`.

### Phone and tablet

The same app works on a phone or tablet on your network. Settings open as a
sheet from the bottom (the **Settings** button at the top right), and the camera
stays visible above it.

<p align="center">
  <img src="images/mobile-settings.webp" alt="Settings sheet on a phone" width="260">
  &nbsp;
  <img src="images/mobile-drive.webp" alt="Touch driving on a phone" width="260">
</p>

### Watch recordings

Open **Recordings** in the left rail, search or filter runs, and play them. You
can switch between the original camera and recorded overlays, and download the
original files under **Run files**. Older videos that the browser cannot play
can be converted with **Prepare playable copy**; the original is never changed.

### Check the setup against real CARLA

Before a session, run the live check once to confirm that CARLA, the Worker and
the console work together. In the web app open **System → Live check**, tick
the confirmation and press **Run live check**; in the launcher press **Live
check** on the Run tab twice. The check loads Town10HD, spawns a test car with
five cars and five walkers, drives it, then stops and cleans up (about a
minute). The Garage preview pauses meanwhile. The result shows how many checks
passed, names any that failed, and saves a report under `runs/`.

### Light and dark

The app follows your system's light or dark setting automatically.

<p align="center"><img src="images/console-dark.webp" alt="Dark theme" width="760"></p>

---

## Troubleshooting

| Problem | Try |
| --- | --- |
| "CARLA and the World Worker bridge are offline" | Start the stack from the launcher (**Run → Start all**), or on a Mac check the connection in **Setup**. |
| A service shows **Unavailable** in the launcher | Open **Logs** for that service. On Windows, run **Install CARLA prerequisites…** and restart. |
| "Web port is occupied" | Another program uses port 8765. Close it, or choose another port in **Setup**. |
| The Mac cannot reach the Worker | Both computers must be on the same network; check the firewall step on the CARLA computer and press **Check connection**. |
| A recording does not play | Use **Prepare playable copy** in **Recordings**. |
| "Host … is not allowed" | Open the app through `127.0.0.1` or `localhost`. A Cloudflare Tunnel hostname routed to the web app in `~/.cloudflared/config.yml` is accepted automatically after a restart; see [Cloudflare Tunnel](cloudflare_tunnel.md). |
| Live check fails | Read the failed check names in **System → Live check** and the report in `runs/`. Make sure no drive or capture is running. |

More detail: [launcher guide](desktop_launcher.md),
[recovery guide](restore_project.md), [manual setup](manual_setup.md).

Screenshots show the console without a simulator connected, except the driving
images, which replay a raw camera frame recorded from CARLA in the sample
dataset (`samples/data/`).
