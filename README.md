# CARLA Vision Research Console

A local research console for CARLA 0.9.16. It provides a full-screen Garage,
manual and Traffic Manager driving, live monocular vision inference, recording,
and reproducible dataset/model workflows without requiring researchers to write
Python for every session.

This repository is a LAN research tool, not an internet-facing or multi-user
service. The browser server deliberately binds to loopback only.

New contributors should start with the
[`developer navigation hub`](docs/README.md): find a feature, its owner, and
its first test without reading the whole codebase. Branch/PR rules and the current
source map are documented in [`CONTRIBUTING.md`](CONTRIBUTING.md) and
[`docs/project_structure.md`](docs/project_structure.md).

## Product boundary

The production surface contains:

- a real CARLA Garage with vehicle, paint, map, weather, traffic, pedestrians,
  props, and camera views;
- keyboard and touch driving with deadman braking and emergency stop;
- optional CARLA Traffic Manager autopilot with immediate manual takeover;
- RT-DETR, YOLO, or custom detector overlays;
- raw and annotated video, controls, detections, events, summaries, and
  checksum manifests;
- LAN discovery for the CARLA simulator; and
- saved-run inspection and verification.

BehaviorAgent actuation, imitation driving, voxel planning, and closed-loop
research jobs are experimental. They are disabled by default and require the
explicit `--enable-experimental` server flag. CARLA Traffic Manager autopilot
is a simulator feature and is not presented as a model developed by this
project.

## Watch saved recordings

Open **Garage → Menu → Recordings** to search saved runs, switch between original
camera and recorded overlays, and play them with Video.js. The player includes
seeking, five-second skip buttons, playback speed, and fullscreen. Expand **Run
files** to download the original artifacts; browsing does not verify their integrity.

New recordings (raw, overlays, voxel, and dataset replay) are encoded directly as
**H.264 MP4** with `yuv420p` and fast-start indexing. Install FFmpeg with `libx264`
on the recording/WebUI computer (`brew install ffmpeg` on macOS), then restart
the WebUI. No subsequent conversion is needed. Lossless dataset images and the
live MJPEG feed are unchanged.

Some older MP4 files use a codec browsers cannot decode. **Prepare playable copy**
creates a temporary H.264 version using FFmpeg on the WebUI computer (on macOS,
`brew install ffmpeg` if needed). Conversion runs in the background, one at a time;
long recordings can take longer. Copies are reused until the WebUI stops and never
replace the original recording or change its evidence hashes. No CARLA or Windows
World Worker update is required for playback.

## Architecture

```text
Windows CARLA host                         Research computer
------------------                        -----------------
CARLA 0.9.16 :2000  <-------------------> Garage / Drive server :8765
World Worker :8766  <-------------------> browser + vision runtime
official PythonAPI                         RT-DETR / YOLO / recorder
```

The Windows World Worker is intentionally thin. It owns official-PythonAPI
operations such as map reload, weather, Traffic Manager, pedestrians, props,
and the persistent in-memory MJPEG relay. It does not install Torch,
Ultralytics, or the research environment.

The research computer owns the UI, detector inference, video recording,
artifacts, datasets, and reports.

## Install profiles

Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/) are required.

Lean Garage and Operator:

```bash
uv sync --group dev
```

Garage with RT-DETR/YOLO inference:

```bash
uv sync --extra vision --group dev
```

Dataset, training, evaluation, plotting, and model release workflows:

```bash
uv sync --extra research --group dev
```

Everything, including experimental modules:

```bash
uv sync --all-extras --all-groups
```

The installed package already contains the production Svelte console. Node.js
is needed only when changing the frontend; contributors rebuild the reviewed
bundle with `npm ci && npm run build` from `web/`.

## Run the Windows World Worker

Use the Python environment in which the matching CARLA 0.9.16 PythonAPI can be
imported. The `ue5-dev` PythonAPI currently targets a newer CARLA generation;
do not load its binary extension into a 0.9.16 server process.

From the cloned repository in PowerShell:

```powershell
$env:CARLA_WORLD_WORKER_TOKEN = "replace-with-a-new-random-secret"

.\.worker-venv\Scripts\python.exe .\carla_vision\native\observable_world_worker.py `
  --bind 172.20.10.7 `
  --port 8766 `
  --allow-lan `
  --carla-host 127.0.0.1 `
  --carla-port 2000 `
  --traffic-manager-port 8000
```

The Worker file is standalone. Map/control operations import only the standard
library plus the official `carla` module. Live camera encoding lazily requires
two camera-only runtime packages in the same Worker interpreter:

```powershell
.\.worker-venv\Scripts\python.exe -m pip install numpy opencv-python-headless
```

Running the Worker by file path still avoids installing this project, Torch,
Ultralytics, or the research environment on the CARLA computer.

Restrict inbound TCP 8766 in Windows Firewall to the research computer's LAN
address. Never expose the Worker to the public internet.

## Run the research console

Copy the environment template and place a fresh Worker token in `.env.local`:

```bash
cp .env.example .env.local
set -a
source .env.local
set +a
```

Start with automatic CARLA and colocated-Worker resolution:

```bash
uv run carla-operator-ui \
  --carla-host auto \
  --world-worker-url auto \
  --open-browser
```

`--carla-host auto` validates CARLA RPC on directly connected private LANs.
`--world-worker-url auto` uses port 8766 on that same discovered host, avoiding
stale Worker addresses after changing networks.
If `.env.local` contains `CARLA_WORLD_WORKER_TOKEN` and no Worker URL is given,
the packaged entrypoint selects this colocated `auto` behavior automatically.

If only CARLA is running and the Worker is unavailable, omit
`--world-worker-url`. Manual raw-bridge driving and vision remain available;
map reload, traffic, pedestrians, props, and TM autopilot stay unavailable.

The console opens at:

```text
http://127.0.0.1:8765/
```

## Remote camera performance

The default release profile is `1280 x 720 @ 30 FPS`. `1280 x 720 @ 60 FPS` is
selectable, but it remains a live-unverified candidate until the real Windows
CARLA host and LAN pass the fixed acceptance matrix. Camera frames are JPEG
encoded in memory beside CARLA and forwarded through persistent MJPEG streams;
the normal browser path does not write per-frame temporary files or issue one
HTTP request per frame.

The live MJPEG feed is for driving feedback. Training data must come from the
synchronized native dataset-capture workflow; dropped/newest-only preview frames
and review MP4s are not canonical training samples. See
[docs/realtime_streaming.md](docs/realtime_streaming.md) for exact profiles,
measurements, hard pass/fail thresholds, and the H.264/WebRTC fallback decision.

For synchronized multi-camera teacher collection initiated from the Mac, see
[native research jobs](docs/native_research_jobs.md). Use **Research → Record
teacher dataset**: the normal Worker runs the existing collector on Windows;
the WebUI retrieves the verified dataset and adds review videos to Recordings.
No separate service, task export, or extra Worker startup flags are needed.

## Garage and Drive

The normal operator journey is:

1. choose the vehicle and paint;
2. select map, weather, traffic, pedestrians, props, and starting mode;
3. inspect the live CARLA Garage and camera angle;
4. start the drive;
5. use keyboard or touch controls, optionally with detector overlays;
6. mark useful human-observed moments; and
7. end and save the run.

Keyboard controls:

```text
W / Up       throttle
S / Down     brake
A / Left     steer left
D / Right    steer right
Space        handbrake
Shift + W    reverse transition
```

Browser input, camera freshness, and the actuator watchdog are independent
safety gates. Stale input or camera loss applies service braking. Model overlay
output never blends silently with human input.

## Outputs

Each retained drive is written under `runs/<run-id>/` and may include:

```text
config.json
controls.jsonl
detections.jsonl
events.jsonl
human_markers.jsonl
raw.mp4
annotated.mp4
summary.json
manifest.json
```

Generated datasets, runs, models, reports, bundles, native kits, and operator
sessions are ignored by Git. Publish frozen artifacts through a release or
external artifact store rather than committing generated ZIPs and videos to
the source repository.

## Research workflows

The Research surface retains the useful artifact-producing pipeline:

- synchronized RGB plus privileged instance-label collection;
- canonical COCO and derived YOLO exports;
- automated dataset QA;
- deterministic scenario planning and native capture;
- detector training for RT-DETR, YOLO, or a custom adapter;
- evaluation, paired replay, threshold selection, and failure review;
- analysis, plots, reports, model packaging, and reproduction bundles; and
- manifest/checksum verification.

Install the `research` profile before launching compute-heavy jobs. Passing a
unit test or dry run proves the software contract, not model quality. A model
claim requires a retained dataset, checkpoint, evaluation, and live acceptance
run.

## Experimental research

To expose the retained BehaviorAgent/imitation/voxel/closed-loop endpoints:

```bash
uv run carla-operator-ui \
  --carla-host auto \
  --world-worker-url auto \
  --enable-experimental \
  --open-browser
```

These paths are not production capabilities. In particular, the current
imitation and voxel-control contracts consume simulator-derived ego speed and
therefore do not satisfy the repository's strict monocular-only policy claim.

## Detector adapter

The built-in adapter supports Ultralytics RT-DETR and YOLO. A custom detector
implements the small `Detector` protocol: metadata, `infer(image_bgr)`, and
`close()`. The core runtime supplies only a copied `uint8 HxWx3` image to the
detector.

## Verification

Run the production checks:

```bash
uv run ruff check carla_vision tests
node --check carla_vision/operator/static/app.js
uv run pytest -q
cd web
npm ci --no-audit --no-fund
npm run check
npm run build
cd ..
git status --porcelain --untracked-files=all -- carla_vision/operator/console_static
```

The final `git status` command must be empty when the committed Svelte source
and deployable bundle are synchronized. CI has three jobs: a lean Operator
profile, the SvelteKit console, and the complete research/experimental suite.
Live CARLA acceptance remains a separate manual gate because unit tests use
fake RPC/PythonAPI objects and do not prove simulator behavior.

## Production acceptance gate

Before tagging a release, perform one real LAN session that verifies:

- Worker health and exact CARLA client/server version 0.9.16;
- two map transitions without losing the Worker listener;
- weather, traffic, pedestrians, and each fixed prop preset;
- Garage vehicle replacement and all camera views;
- ten minutes of manual driving and TM-autopilot takeover;
- the 720p30 remote-stream rows A-D in the real-time acceptance matrix;
- raw and annotated review recording without truncated duration;
- detector latency/FPS, source FPS, frame-age, and Worker-encode summaries; and
- `carla-verify` success on the retained run.

Detailed contracts remain in [docs/architecture.md](docs/architecture.md),
[docs/operator_ui.md](docs/operator_ui.md), and
[docs/artifact_policy.md](docs/artifact_policy.md). Remote-video acceptance is
defined in [docs/realtime_streaming.md](docs/realtime_streaming.md).
