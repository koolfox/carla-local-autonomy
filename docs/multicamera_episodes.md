# Multi-camera RGB teacher episodes

This extends the existing BehaviorAgent research collector. It does **not**
change Garage camera settings, the thin World Worker protocol, or live vehicle
control. Collection uses the official PythonAPI and requires exclusive ownership
of CARLA. Offline replay needs neither CARLA nor model weights.

To launch this collector from a Mac that cannot import CARLA, use the optional
[native research task host](native_research_jobs.md). It runs this same collector
on Windows and returns an artifact archive; it does not change live Garage video.

## What is available

- Default `front`: unchanged single-camera capture and detector/imitation inputs.
- `front-three`: the scene's front RGB camera plus left/right RGB views at
  minus/plus 60 degrees relative to its yaw. Resolution, FOV and sensor period
  are inherited. These are three views, not a 360-degree surround rig.
- A custom JSON rig: one to seven additional RGB cameras with individual mounts
  and FOV; resolution and sensor period still come from the scene.
- Exact-frame view bundles, intrinsics and rigid camera-to-ego calibration.
- Lossless PNGs, source timestamps, checksum verification and indexed video replay.

The existing front instance-segmentation camera remains **teacher-label-only**.
COCO/YOLO annotations still describe the front image; side images are not
silently assigned front-camera labels. Per-view detection/segmentation ground
truth and multi-camera policy training are not implemented by this slice.

## Collect

Use a machine/environment with the matching CARLA 0.9.16 PythonAPI and official
`agents.navigation.behavior_agent`. This is the existing **research collector**
environment, not an instruction to install ML dependencies into the thin bridge.
See [teacher collection prerequisites](behavior_teacher_episodes_fa.md).

From the repository root, create a verified plan if you do not have one:

```bash
uv run carla-plan-scenarios \
  --suite configs/scenarios/thesis_pilot_v1.json \
  --split-plan configs/scenarios/split_plan_thesis_pilot_v1.json \
  --run-id multicamera-plan-v001
```

Inspect the planned cameras without connecting to CARLA:

```bash
uv run carla-record-behavior-teacher \
  --scenario-plan runs/multicamera-plan-v001 \
  --dataset-id multicamera-pilot-v001 \
  --camera-rig front-three \
  --max-episodes 1 \
  --dry-run
```

Before real collection, stop other driving/preview clients and synchronous tick
owners. **The collector reloads the map, destroys existing actors and moves the
ego using BehaviorAgent.** After checking the dry-run selection, use the same
command with `--dry-run` removed and these arguments added:

```text
--host 127.0.0.1 --acknowledge-exclusive-tick-owner
```

`127.0.0.1` assumes collection on the CARLA host; use its actual address when
running from another compatible PythonAPI host. The Mac thin-bridge connection
does not itself provide a native PythonAPI import for this collector.

To customize mounts/FOV, replace `--camera-rig front-three` with:

```text
--camera-rig-config configs/capture/front_sides_rgb_v1.json
```

Mounts are ego-relative metres and CARLA pitch/yaw/roll degrees. `front` and
`front_teacher` are reserved; `front` continues to use the scene camera. Adjust
the sample mounts for your vehicle; check for body occlusion before collecting
a large dataset. More cameras increase GPU, RAM, disk and capture time. The
eight-camera format limit is not a performance or hardware guarantee.

## Verify and replay

Verification runs after successful collection by default. To run it again:

```bash
uv run carla-verify-teacher-episodes datasets/multicamera-pilot-v001
uv run carla-replay-teacher-episode \
  --dataset datasets/multicamera-pilot-v001 \
  --run-id multicamera-preview-v001
```

If the dataset has multiple episodes, pass `--episode-id` from `dataset.json`.
Replay requires at least two uniformly sampled frames; it refuses to hide gaps
by inventing timing. This is **recorded-image replay**, not CARLA recorder replay
or a new simulation. It generates `runs/multicamera-preview-v001/` containing a
camera mosaic MP4, exact source-frame index, verification report and manifest.
Open **Garage → Menu → Recordings** and select that run. If the browser rejects
the OpenCV MP4 codec, use the existing **Prepare playable copy** option (FFmpeg
on the WebUI host). If collection is on Windows, transfer the entire dataset
directory to the research workspace before offline verification/replay; do not
copy only `dataset.json` or rewrite its relative paths.

## Data contract

Front image, labels and existing consumer paths stay intact:

```text
dataset.json                         samples + declared rgb_camera_ids
images/<split>/<sample>.png           original front image
cameras/<camera_id>/<split>/<sample>.png
metadata/<sample>.json                rgb_views + existing teacher context
episodes/<episode>.json               rig, actor and collection provenance
checksums.sha256                      indexes side images too
```

Each `rgb_views` entry contains the image reference/hash, CARLA frame, timestamp
and calibration. `camera_to_ego` maps camera **CARLA axes** into ego CARLA axes:
x forward, y right, z up. Its translation uses metres. Projection intrinsics
use optical axes x right, y down, z forward; `optical_from_camera` explicitly
performs that conversion. Matrices come from the actual configured CARLA mounts;
lens distortion is disabled for the calibrated rig. Intrinsics use the official
[CARLA projection convention](https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/).

World camera transforms are separately named `privileged_camera_world_transform`.
Teacher actions, navigation labels and simulator state remain evaluation/label
metadata, not automatic model inputs. Existing models still receive their old
front-image contract; additional views do not make a single-camera checkpoint
a multi-camera model.

All rig sensors spawn before ticking in one batch. Collection joins them by
CARLA frame under one `--sensor-timeout` budget per captured bundle, following
[CARLA synchronization](https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/).
Callback queues retain at most four frames per camera. Stale callbacks can be
discarded; a missing **required** frame fails the run rather than substituting
another image. Capture-time failures retain a camera/frame diagnostic under
`episodes/<episode>/capture_failure.json`; failed runs are not complete datasets.
Every successfully spawned camera is owned by episode cleanup, including partial
spawn failures.

## Developer entry points and acceptance

| Change | Owner | Test |
| --- | --- | --- |
| Rig names, validation, calibration | `dataset/camera_rig.py` | `tests/test_multicamera_dataset.py` |
| Spawn, frame deadline and cleanup registration | `native/camera_rig.py`, `native/behavior_teacher.py` | `tests/test_multicamera_teacher.py` |
| Additional images and checksum index | `dataset/writer.py`, `dataset/camera_views.py` | `tests/test_multicamera_dataset.py` |
| Recorded-image replay | `dataset/episode_replay.py` | `tests/test_multicamera_dataset.py` |

Source paths are under `carla_vision/`. Run both test files plus the existing
dataset/teacher tests before a PR. Fakes test alignment, dropped frames and
cleanup ownership; they do not establish real renderer latency or mount quality.
Live acceptance is still required: capture a short real episode, verify it,
inspect all camera angles in replay, confirm cleanup, and repeat with the default
single-camera mode. Retain hardware, simulator/PythonAPI versions, resolution,
capture FPS and elapsed time with that evidence.
