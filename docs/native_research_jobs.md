# Run native research from the Mac through the Windows bridge

The optional native task host gives trusted Python tasks access to the **real
CARLA PythonAPI on Windows**. The Mac submits JSON, reads status/logs, cancels,
and downloads checked artifacts. It does not receive Python actor objects or a
transparent remote `import carla`.

The first task is `teacher_capture`: the existing BehaviorAgent collector with
the `front`, `front-three`, or custom RGB rig. No replacement collector, driving
controller, UI configuration store, or video transport is introduced.

```text
Mac: native-jobs CLI / NativeResearchClient
  -> authenticated World Worker task API
       -> separate Python process, installed task package
            -> official CARLA PythonAPI + BehaviorAgent
       <- status, phase logs, cleanup receipt, artifact archive
  <- size + SHA-256 checked download
```

## One-time installation

This needs Worker revision **7** or later. Normal Garage/Drive use remains
unchanged and task hosting is disabled unless explicitly enabled.

### 1. Export the collector on the research machine

From the project root, using its existing environment:

```bash
uv run carla-build-native-task \
  --destination native_tasks/teacher_capture \
  --version YOUR_SOURCE_COMMIT
```

Use the commit you actually built. The destination must not exist. Transfer the
whole `native_tasks/teacher_capture/` directory to `native_tasks/teacher_capture/`
under the Windows checkout. It contains a manifest and Python source snapshot;
no model weights, datasets, credentials, frontend bundle, or dependency installer.
The snapshot currently includes the package's Python sources to preserve existing
imports, but does not install the project or import its ML runtimes.

### 2. Prepare the task interpreter on Windows

Choose a Python 3.11–3.13 environment compatible with the installed CARLA 0.9.16
wheel. It must provide `carla`, the **matching official** `agents` package, and
the collector's dependencies: `numpy`, `opencv-python`, `msgpack`, plus the
dependencies of those CARLA agents (including `networkx` and `shapely`). No Torch,
Ultralytics, Node, WebUI or model installation is required for this collector.

This can be a separate `.collector-venv`; do not install the full project merely
to host jobs. Use the matching CARLA wheel/agents from your simulator installation.
For example, after creating that environment and installing the matching wheel:

```powershell
.\.collector-venv\Scripts\python.exe -m pip install numpy opencv-python msgpack networkx shapely
$env:PYTHONPATH = "C:\PATH_TO_CARLA\PythonAPI\carla"
.\.collector-venv\Scripts\python.exe -c "import carla, cv2, numpy, msgpack; from agents.navigation.behavior_agent import BehaviorAgent; print('native task imports OK')"
```

Replace `C:\PATH_TO_CARLA` with the actual installation, and preserve any existing
`PYTHONPATH` entries you need. The task inherits this host-selected path; HTTP
requests cannot choose an interpreter, environment, executable, or Python path.

If you prefer **pull-and-build on Windows** instead of copying a package from
the Mac, the prepared collector interpreter can export from the updated checkout:

```powershell
$taskVersion = git rev-parse --short HEAD
.\.collector-venv\Scripts\python.exe -m carla_vision.native.task_package `
  --destination .\native_tasks\teacher_capture --version $taskVersion
```

Run this from the repository root. It copies source; no project installation is
needed. Pick either this approach or the Mac export, not both into the same folder.

### 3. Enable the host on your existing Worker command

After updating the Windows checkout, keep the normal token environment variable
and restart the existing bridge with these three additional options. Example
from the Windows repository root:

```powershell
.\.worker-venv\Scripts\python.exe .\carla_vision\native\observable_world_worker.py `
  --bind 0.0.0.0 --allow-lan --port 8766 `
  --carla-host 127.0.0.1 --carla-port 2000 --traffic-manager-port 8000 `
  --research-tasks-dir .\native_tasks `
  --research-jobs-root .\native_jobs `
  --research-python .\.collector-venv\Scripts\python.exe
```

Use your usual bind address/firewall rules if different. Keep
`CARLA_WORLD_WORKER_TOKEN` set; never paste it into a committed command or request
file. The bridge's optional runner uses only the Python standard library. A
standalone-file deployment also needs `research_jobs.py` beside `world_worker.py`
when enabling this option; it remains unnecessary for the ordinary bridge.

## First capture from the Mac

Use the connected Windows bridge URL, not the Mac WebUI URL. Commands below use
`CARLA_WORLD_WORKER_URL` and the existing token from the environment or `.env.local`.
You can instead place `--worker-url http://WINDOWS_IP:8766` **before** the subcommand.

```bash
export CARLA_WORLD_WORKER_URL=http://WINDOWS_IP:8766
uv run carla-native-jobs tasks
uv run carla-native-jobs submit \
  --job-id multicamera-smoke-001 \
  --parameters configs/capture/remote_teacher_smoke.json \
  --acknowledge-world-reload
uv run carla-native-jobs status multicamera-smoke-001
uv run carla-native-jobs log multicamera-smoke-001
```

**Before submitting, Stop & Save Drive, close Garage preview, and close the
Garage browser tab so it cannot automatically reopen preview.** Stop any other
simulation/ticking clients too. The collector reloads the map, destroys the old
world's actors, and moves the ego using BehaviorAgent. It is not a continuation
of the Garage's current vehicle. A retained preview lease returns `409` rather
than being silently stolen; close it through its owning Operator and wait for
cleanup. Native jobs and Garage may not own CARLA simultaneously.

The checked-in smoke request has `dry_run: true`: it validates the plan and rig
without connecting to CARLA. It still uses the exclusive job gate, deliberately.
Inspect its result first, then make your own request copy, set `dry_run` to
`false`, choose an installed map, and submit with a **new job ID**. The sample
requests a 10-second episode at 5 captured bundles/second: three 1280×720 RGB
views, two seconds of warmup, no background population or props. These are
simulation timings, not a promise of ten seconds wall time. Increase population
only after verifying the small capture; map capacity and rendering speed remain
CARLA/hardware constraints.

Reusing an ID with the identical request returns the existing job; it never
starts another capture after an HTTP timeout. Reusing it with different settings
returns `409`. Request bodies are subject to the existing Worker JSON size limit.

To cancel, then inspect final cleanup status:

```bash
uv run carla-native-jobs cancel multicamera-smoke-002
uv run carla-native-jobs status multicamera-smoke-002
```

Collection checks cancellation between controlled ticks. A blocked native RPC
may not respond immediately. Jobs have a package-declared wall-clock deadline
(900 seconds for this collector); cancellation allows 15 seconds for cleanup
before terminating the process. Forced termination never counts as confirmed
cleanup. Logs expose phases and native errors, not a fabricated completion
percentage. The host retains up to 8 MiB of log data; each log response is at
most 32 KiB. Completed job records are kept across bridge restarts.

## Bring the dataset back and inspect it

After a **real** job reports `succeeded` and `cleanup_confirmed: true`:

```bash
uv run carla-native-jobs fetch multicamera-smoke-002 \
  --destination downloads/multicamera-smoke-002.zip
```

Fetch checks declared size and SHA-256 and refuses to overwrite an existing
destination. It does not execute or automatically extract anything. Extract into
a **new empty staging directory**, then move the entire resulting
`datasets/multicamera-smoke-002/` into your research workspace's `datasets/`.
Preserve its relative paths and retain `inputs/` and `runs/` as provenance.
The archive's `inputs/native_task.json` records the installed task version/hash.

```bash
uv run carla-verify-teacher-episodes datasets/multicamera-smoke-002
uv run carla-replay-teacher-episode \
  --dataset datasets/multicamera-smoke-002 \
  --run-id multicamera-review-002
```

Open **Garage → Menu → Recordings** for the generated replay run. A dry-run
archive contains a plan, **not** recorded camera images. Failed/cancelled outputs
remain on Windows for diagnosis; the download helper intentionally accepts only
successful archives. See [multi-camera data boundaries](multicamera_episodes.md):
front teacher labels do not automatically annotate the side cameras.

## Add future native features without another bridge endpoint

Install a reviewed task folder under the Windows task registry. Its `task.json`
declares `schema_version`, `id`, `version`, `world_access: "exclusive"`,
`max_seconds` (1–3600), a Python `entrypoint`, and a relative-file SHA-256 inventory
including that entrypoint. Use the generated collector manifest as the example.

The executable receives `--request PATH --output DIRECTORY`. The request supplies
JSON `parameters`, the host's CARLA `endpoint`, task identity, and a `cancel_file`
path. The task may use the native PythonAPI directly, but owns its actors/ticks
and must release them and restore asynchronous mode before writing a terminal
`result.json` with `status` and `cleanup_confirmed`. Optional `artifacts.zip`
metadata declares its SHA-256 and byte size. Never claim cleanup before it has
actually completed, and do not leave background child processes running.

Registry discovery and checksums are refreshed for new submissions. Update a
task **only while no job is active**, then refresh the catalog; a stale manifest
hash is rejected. Future task code updates require deploying that package to
Windows, but no bridge route change or restart. Host/protocol fixes still need a
bridge update; new task dependencies still need installation in its interpreter.
This is not a promise of a permanently frozen bridge.

The narrow stable API is:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/v1/research/tasks` | Installed tasks and versions |
| POST | `/v1/research/jobs` | Submit one acknowledged, exclusive task |
| GET | `/v1/research/jobs/{id}` | Status and terminal receipt |
| GET | `/v1/research/jobs/{id}/log` | Bounded log tail |
| POST | `/v1/research/jobs/{id}/cancel` | Request cooperative cancellation |
| GET | `/v1/research/jobs/{id}/files/artifacts.zip` | Completed artifact archive |
| GET | `/v1/research/jobs/{id}/files/result.json` | Completed receipt |

No Python upload, arbitrary shell, remote `pip`, or arbitrary-file download route
exists. This is **trusted process isolation, not a security sandbox**. An installed
task executes as the host user. Checksums establish identity/integrity, not code
trust. Restrict registry write access and use the bridge only on a trusted LAN or
protected tunnel; bearer authentication over plain HTTP is not encryption.

## Failure recovery and acceptance

A task crash leaves the HTTP listener alive. If cleanup cannot be confirmed,
Garage and new native jobs are blocked with `recovery_required`. A bridge restart
does not clear this: unfinished records are marked `interrupted`, and
`native_jobs/recovery-required.json` retains the reason. On Windows, stop the
bridge, confirm the old job PID/process has exited, reconcile/reset CARLA and
other tick owners, then remove **only that recovery marker** and restart. Do not
delete datasets or job records. This deliberate host check prevents a recovered
UI from driving into an unreconciled simulation.

Current automated coverage uses real local HTTP and subprocesses with fixture
tasks, a source-exported collector dry run, and fake-CARLA sensor/cancellation
tests. **It does not certify real Windows capture or latency.** Live acceptance
still requires: install/enable on Windows, dry run, one real short capture,
verified download/replay, cooperative cancellation, and a subsequent normal
Garage session. Retain those artifacts before claiming live support is accepted.

Developer ownership: `native/research_jobs.py` owns scheduling/process state;
`native/tasks/teacher_capture.py` adapts the existing collector;
`native/task_package.py` exports source; `operator/native_research.py` owns the
SDK/CLI. No Svelte or live-driving changes are needed to add a native task.
