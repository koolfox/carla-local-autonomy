# Capture through the normal World Worker

**Pull the updated checkout on Windows and restart your usual World Worker
command.** Keep its existing port, token and Python environment. Built-in
research tasks load directly from that checkout: no package export, separate
collector environment, second server, or extra startup flags.

## Place cameras once, then choose what to record

In **Vision → Record camera rig during Drive**, or in **Research**, use the
same camera editor. Choose Front + rear, three front views, or four directions;
add/remove views up to eight RGB cameras. Select a camera to edit its X/Y/Z
position, yaw/pitch/roll and FOV. Distances are metres relative to the vehicle
origin: X forward, Y right, Z up. Yaw 180° faces backwards; negative yaw faces
left. A mount can be inside the body of a large vehicle: check a short recording
before a long run. This is configuration, not a live placement preview.

The shared layout is kept in the browser. Changes affect the **next** capture
or drive recording. They do not respawn the car or change the current live/model
camera. The main driving camera stays unchanged; optional rig perception uses
only each selected camera's RGB image and never controls the vehicle.

### Record during a normal drive

1. Under **Vision**, enable **Record video** and **Record camera rig during Drive**.
2. Set the rig and recording FPS (1/2/5/10); start with Front + rear at 5 FPS.
3. Select manual or Traffic Manager driving as usual, then **Start session**.
4. **Stop & Save**, open **Recordings**, select that run and choose **Camera · rear**,
   **Camera · front**, etc. Each view has a directly playable H.264 MP4.

No teacher job, BehaviorAgent dependency, second environment, or world reload
is needed. Recording cameras attach to the **existing ego** and share its Worker
lease. They are additional to the live/model front camera and cost GPU, LAN and
disk bandwidth. Resolution follows the session camera setting; the rig FPS is
independent of live preview FPS. Stop the current drive before editing its rig.

Files are in `runs/<run-id>/cameras/`: `<camera>.mp4`,
`<camera>.frames.jsonl`, and `rig.json`. The manifest registers/hashes them all.
The rig metadata contains mounts, actual CARLA mount matrices, intrinsics and
per-camera completion/error counts. Each index entry maps a **written video
frame** to its CARLA frame ID, simulation timestamp and world camera transform.

These are **asynchronous, lossy review recordings**, not an exact-frame training
dataset. Latest-frame streaming can skip frames; video playback uses the selected
fixed FPS, so consult timestamps rather than inferring simulation time from the
playback clock. Do not assume frame N from different videos is synchronized.
Use the Research workflow below for synchronized, lossless images and labels.

### Detections and sign reading on selected cameras

In **Vision**, enable **Detection overlay** and choose your detector. For sign
recognition, select **M9 hierarchical** and enable **Read traffic signs (DeiT-64)**
with its checkpoint and ontology. Enable **Record video** and **Record camera rig
during Drive**. Select each camera in the rig editor and choose its **Perception**:

- **Off**: raw video only (the default).
- **Detections**: your selected detector, without DeiT sign classification.
- **Detections + sign reading**: M9 detections and DeiT-64 sign classification.

These selections apply to the **next Start session**. In Drive, the camera picker
switches between the main driving camera and the configured rig views. **Raw**
and **Detections** toggle the selected camera's display immediately; choosing Raw
does not disable inference or stop saving annotated evidence. A camera configured
Off cannot enable inference midway through a run; select it in Vision before the
next run. Road segmentation and Voxel remain on the main driving camera.
The DeiT **Show unknown / unaccepted statuses** preference is shared by the main
camera and selected rig cameras, including their saved overlays. It changes only
captions, never the raw scores or acceptance flags retained in detection JSON.

One detector instance (and one optional DeiT classifier) serves all selected
cameras serially, with fair turns and at most one pending frame per camera.
More selected cameras reduce per-camera overlay FPS. Raw recording is independent
of inference; a failed rig inference/overlay reports its error without stopping
the other cameras or raw recording. This is not cross-camera tracking, 3D fusion,
or a claim of complete 360-degree scene understanding.

For selected cameras, the same run also saves:

- `cameras/<camera>.overlay.mp4`: annotated H.264 review video.
- `cameras/<camera>.detections.jsonl`: every written overlay frame's camera ID,
  CARLA frame, sequence, timestamp, pose/FOV, detections and sign attributes.
- `cameras/rig.json`: selected modes, completion/errors and scheduler counts.

In **Recordings**, choose **Detections · rear**, etc. for annotated video or
**Camera · rear** for untouched RGB. Annotation always uses the exact image
inferred by that camera, not a newer raw frame. Overlay videos contain sampled
inference frames at the configured rig video FPS and may be shorter/faster than
raw footage; use the JSON timestamps for timing, not the playback clock. All
videos and JSON files are registered in the existing run manifest. Detector and
classifier identity/hashes remain in the shared `detector-metadata.json`.

This feature needs the updated **Mac/operator backend and built WebUI**, not a
new Windows Worker API. Restart the Mac web process after updating; a previously
running process does not reload Python code when frontend files change.

### Record a teacher dataset

1. In Garage's **Research** menu, choose the shared Scene settings and capture
   duration/rate/repetitions. Choose a named weather preset for reproducibility.
2. Edit the shared camera layout under **Record teacher dataset**.
3. Acknowledge scene reload and teacher motion, then select **Record dataset**.
4. The WebUI pauses Garage preview, starts the existing BehaviorAgent collector
   through the Worker, and shows progress and a Cancel action.
5. When the job ends, Garage becomes available again. On success with confirmed
   cleanup, the WebUI retrieves and verifies the dataset, then creates review
   videos in **Recordings** automatically.

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
project, Torch, Node or a second WebUI on Windows. The checkout bundles the
minimal official CARLA 0.9.16 agents (with license/provenance) as a fallback
when `agents` is absent. An installed `agents` package takes precedence; its
missing dependencies are reported rather than silently replaced.

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

A failed/interrupted capture no longer permanently locks Garage. The failure and
unconfirmed-cleanup receipt remain visible; unconfirmed datasets are not imported.
The next normal Garage preparation restores asynchronous ticking when a failed
collector left the world synchronous. Old `recovery-required.json` records do
not block operation; there is no recovery-file deletion command to run.
An **actively running** capture still owns the world: finish or cancel it before
another drive. Datasets and job records are retained. A lost network response is
observed using the same job ID, never resubmitted as another collector. This
does not hide a genuinely offline simulator or certify unknown actor cleanup.

Automated tests cover local HTTP/subprocess execution, built-in collector dry
runs, capture handoff/cancellation/import, and actual H.264 encoding/decoding.
**Real Windows/CARLA acceptance is still required:** one short capture, cancel
another, verify the returned dataset/video, then start a normal Garage drive.

Developer owners: `operator/drive_cameras.py` records optional Drive camera rigs;
`perception.py` schedules the shared model, and `operator/camera_perception.py`
renders/saves per-camera results without owning models or vehicle controls.
`native/world_worker.py:recording_cameras` attaches their sensors and the existing
camera endpoints accept `X-Camera-View` to select a rig stream. Old requests
without that header continue to use the live front camera. The capability
`drive_recording_cameras` prevents attempting this with an older Windows Worker.
`web/src/lib/domain/capture.ts` defines the shared rig; `CameraRigEditor.svelte`
edits it; `driveRecordingSettings` snapshots it into the session request.

`native/research_jobs.py` hosts tasks;
`native/tasks/teacher_capture.py` adapts the existing collector;
`operator/garage_capture.py` coordinates Garage/import;
`operator/native_research.py` uses the existing Worker connection. Algorithm
changes belong in the task/collector, not new bridge routes.
