# Capture through the normal World Worker

**Pull the updated checkout on Windows and restart your usual World Worker
command.** Keep its existing port, token and Python environment. Built-in
research tasks load directly from that checkout: no package export, separate
collector environment, second server, or extra startup flags.

## Normal operator workflow

1. In Garage's **Research** menu, choose the shared Scene settings and capture
   duration/rate/repetitions. Choose a named weather preset for reproducibility.
2. Under **Record teacher dataset**, choose Front or Front + left + right.
3. Acknowledge scene reload and teacher motion, then select **Record dataset**.
4. The WebUI pauses Garage preview, starts the existing BehaviorAgent collector
   through the Worker, and shows progress and a Cancel action.
5. After confirmed cleanup, Garage is available again. The WebUI retrieves and
   verifies the dataset, then creates review videos in **Recordings** automatically.

Stop & Save an active drive first. Capture owns the simulator while collecting;
changing Scene fields during capture affects the next run, not the running job.
There is one collector at a time, at most 32 episodes and a 15-minute wall-clock
deadline (including map loads/cleanup). Begin with one short episode.

The first collector uses the official **BehaviorAgent**, not Traffic Manager
actions relabelled as a learned policy. Teacher labels and CARLA metadata remain
separate from deployable RGB model inputs. Camera calibration and exact-frame
matching use the existing dataset implementation.

## Requirements, in the environment you already use

The bridge itself still starts without ML dependencies. Collection needs the
matching CARLA PythonAPI and official `agents` module, plus NumPy, OpenCV,
msgpack and the agents' dependencies (`networkx`, `shapely`). Missing imports
produce a preflight error without taking the listener down. If needed, add only
the missing packages to **the existing `.worker-venv`**; do not install the whole
project, Torch, Node or a second WebUI on Windows. Keep the matching CARLA
`PythonAPI/carla` directory on that interpreter's Python path for `agents`.

Review video encoding needs FFmpeg with `libx264` on the **WebUI computer**
(`brew install ffmpeg` on macOS). Videos are encoded directly as H.264 MP4,
`yuv420p`, with fast-start indexing; no later conversion is needed for new
recordings. Original training images remain lossless PNGs. This does not
change the live MJPEG transport.

## Where the outputs go

- Windows: `native_jobs/<job-id>/`, including parameters, logs, cleanup receipt,
  source identity, original dataset and downloadable archive.
- Research computer: `operator_sessions/native/<job-id>/artifacts.zip`, then
  verified data in `datasets/<job-id>/` and review runs in `runs/<job-id>-N/`.
- Downloads are size/SHA-256 checked; extraction rejects escaping paths,
  symlinks and oversized contents. Existing output files are never overwritten.
- Closing the browser does not cancel capture. A WebUI restart resumes observing
  its saved job. Use **Cancel capture** to request cooperative cancellation.

These are collection/development episodes, not an automatically valid train/test
split. Use existing split/verification tools before training or evaluation.

## Advanced access uses that same connection

Inside the operator, use its configured Worker client:

```python
research = application.world_worker.research()
catalog = research.tasks()
job = research.submit(
    task_id="teacher_capture",
    job_id="my-capture",
    parameters=parameters,
    acknowledge_world_reload=True,
)
status = research.status(job["job_id"])
log = research.log(job["job_id"])
```

`research.fetch(job_id, destination)` and `research.cancel(job_id)` also use that
client's authenticated HTTP connection. The advanced caller must release its
existing scene before submitting; Garage handles this automatically. JSON task
requests, status, logs and file downloads all use `/v1/research/` on the usual
Worker port. There is no remote Python eval, arbitrary shell, or HTTP code
upload. Optional host-installed extensions are still supported, but are not part
of normal startup or built-in collection.

## Failure handling and acceptance

A collector crash does not terminate the Worker listener. Unknown cleanup blocks
world mutations: do not automatically erase that warning. Stop the Worker,
confirm the old child process has exited, reconcile/reset CARLA and other tick
owners, then remove only `native_jobs/recovery-required.json` and restart. If the
WebUI retains an unresolved capture guard, stop it and remove only
`operator_sessions/native_capture_state.json` after that host reconciliation.
Datasets and job records are retained. A lost network response is observed using
the same job ID, never resubmitted as another collector.

Automated tests cover local HTTP/subprocess execution, built-in collector dry
runs, capture handoff/cancellation/import, and actual H.264 encoding/decoding.
**Real Windows/CARLA acceptance is still required:** one short capture, cancel
another, verify the returned dataset/video, then start a normal Garage drive.

Developer owners: `native/research_jobs.py` hosts tasks;
`native/tasks/teacher_capture.py` adapts the existing collector;
`operator/garage_capture.py` coordinates Garage/import;
`operator/native_research.py` uses the existing Worker connection. Algorithm
changes belong in the task/collector, not new bridge routes.
