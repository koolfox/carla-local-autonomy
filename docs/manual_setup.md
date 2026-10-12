# Developer and manual setup

End users should use the [full-screen TUI](desktop_launcher.md). This guide is
for development, diagnosing an existing custom environment or a dedicated lean
Worker deployment. It is not required for ordinary installation or daily use.

## Manual install profiles

For manual installation, Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/)
are required. The bootstrap manages Python 3.12 itself; the native launcher
uses that version for the matching CARLA SDK.

Lean Garage and Operator:

```bash
uv sync --frozen --group dev
```

Garage with RT-DETR/YOLO inference:

```bash
uv sync --frozen --extra vision --group dev
```

Dataset, training, evaluation, plotting, and model release workflows:

```bash
uv sync --frozen --extra research --group dev
```

Everything, including experimental modules:

```bash
uv sync --frozen --all-extras --all-groups
```

The installed package already contains the production Svelte console. Node.js
is needed only when changing the frontend; contributors rebuild the reviewed
bundle with `npm ci && npm run build` from `web/`.

## Manual Windows World Worker

The same-computer TUI starts this Worker automatically on loopback. For a
separate research computer, use the following LAN command. Use an environment
that imports the official CARLA 0.9.16 PythonAPI matching the server. The local
bootstrap prepares `.venv` with that SDK and the camera dependencies; a separately
prepared lean Worker environment can be used by replacing the Python path.

From the cloned repository in PowerShell:

```powershell
$env:CARLA_WORLD_WORKER_TOKEN = "replace-with-a-new-random-secret"
$WorkerBind = "192.168.1.100" # Replace with this CARLA computer's LAN address.

.\.venv\Scripts\python.exe .\carla_vision\native\observable_world_worker.py `
  --bind $WorkerBind `
  --port 8766 `
  --allow-lan `
  --carla-host 127.0.0.1 `
  --carla-port 2000 `
  --traffic-manager-port 8000
```

The Worker file is standalone. Map/control operations import only the standard
library plus the official `carla` module. Live camera encoding lazily requires
two camera-only runtime packages in the same Worker interpreter. These are
already installed by the local bootstrap. For a manually prepared Worker:

```powershell
.\.worker-venv\Scripts\python.exe -m pip install numpy opencv-python-headless
```

Running the Worker by file path allows a dedicated lean Worker deployment
without installing the full application on that host. Keep one Worker per port;
do not start a second LAN Worker over the TUI's existing listener.

Restrict inbound TCP 8766 in Windows Firewall to the research computer's LAN
address. Never expose the Worker to the public internet.

## Manual research console

For a manual checkout, copy the full environment template, then edit the CARLA
host, Worker URL and matching token in `.env.local`. The bootstrap creates this
file automatically. The console loads it directly; it is not a shell script.

```bash
cp .env.local.example .env.local
```

Start with automatic CARLA and colocated-Worker resolution:

```bash
uv run --no-sync carla-operator-ui \
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
