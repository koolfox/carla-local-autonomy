# Real-time remote camera contract

This document defines what the browser video path does, what the 60 FPS option
means, and how it must be accepted on the real Windows CARLA computer before a
performance claim is made.

## Current status

The implemented MVP path is:

```text
CARLA RGB sensor on Windows
  -> newest-only in-memory queue
  -> OpenCV JPEG encode on Windows
  -> one persistent authenticated MJPEG response from World Worker
  -> newest-only frame cache on the Operator computer
  -> one persistent MJPEG response to the browser
```

The camera callback never writes a temporary image to disk. A slow encoder or
consumer replaces an old pending frame with the newest frame instead of building
an ever-growing latency queue. The previous one-HTTP-request-per-frame endpoints
remain compatibility and diagnostic paths; the canonical browser path uses the
persistent streams.

`1280 x 720 @ 60 FPS` is selectable and covered by code/contract tests. It is
**not yet live-verified** on the Windows simulator and LAN. Until the acceptance
matrix below passes, the release profile is `1280 x 720 @ 30 FPS`. This is an
explicit evidence boundary, not an estimate of the Windows GPU.

This MVP uses MJPEG, not H.264/WebRTC. MJPEG is simple, low-state, and adequate
for establishing whether removing disk I/O and per-frame HTTP round trips fixes
the current driving experience. It consumes substantially more bandwidth than
H.264. The decision rule for a codec upgrade is defined below.

## Runtime requirements

The World Worker still runs directly from `world_worker.py`; the project and ML
stack are not installed on the Windows CARLA host. Its map, actor, weather, and
control features need only the matching official `carla` module. Starting an
in-memory camera additionally requires NumPy and OpenCV in that same Worker
interpreter:

```powershell
.\.worker-venv\Scripts\python.exe -m pip install numpy opencv-python-headless
```

The imports are lazy. If they are absent, non-camera Worker features continue to
work and a camera request fails explicitly with `camera_encoder_unavailable`.

Before testing video, authenticated Worker health/catalog output must report:

```text
persistent_mjpeg_camera_relay: true
in_memory_jpeg_encoder_available: true
camera_60_fps: true
```

The Worker-to-Operator stream is:

```text
GET /v1/scenes/<scene-id>/camera/stream.mjpg
Authorization: Bearer <worker-token>
X-Scene-Lease: <scene-lease>
Content-Type: multipart/x-mixed-replace; boundary=carla-frame
```

The browser never receives the Worker URL, bearer token, or scene lease. It uses
the local Operator endpoints:

```text
GET /api/garage/preview/stream.mjpg
GET /api/drive/stream.mjpg?view=raw
GET /api/drive/stream.mjpg?view=overlay
```

Raw and overlay are selectable views of one Drive session. They are not separate
CARLA camera sensors.

## Supported profiles

| Profile | Resolution | Requested FPS | Release status | Intended use |
| --- | ---: | ---: | --- | --- |
| Production | 1280 x 720 | 30 | Default until live acceptance | Normal remote Garage and manual Drive |
| High refresh | 1280 x 720 | 60 | Selectable, live-unverified | Low-latency manual Drive candidate |
| Low bandwidth | 640 x 384 | 10 | Fallback | Weak Wi-Fi or diagnosis only |
| 1080p preview | 1920 x 1080 | 30 | Optional | Visual inspection, not the release latency gate |

If the connected Worker does not advertise the in-memory persistent encoder,
the UI forces Compatibility and the server rejects higher profiles. This guard
prevents an API client from accidentally sending 720p/1080p uncompressed BGRA
over the LAN.

The requested camera rate is not a promise that CARLA rendered, encoded,
transported, or displayed that many unique frames. A real run must measure all
layers. The detector also has no obligation to run at camera FPS: raw video can
remain at 30/60 FPS while RT-DETR publishes a slower overlay.

## Preview, review video, and training data are different products

Do not train from the MJPEG browser stream. It is JPEG-compressed, newest-only,
and intentionally allowed to drop intermediate frames to protect control
latency.

Three artifact lanes remain separate:

| Lane | Purpose | Retention rule |
| --- | --- | --- |
| Live MJPEG | Human driving and visual feedback | Ephemeral; newest frame wins |
| Drive `raw.mp4` / `annotated.mp4` | Session review and demonstrations | Retain with the Drive manifest; compression must be disclosed |
| Native dataset capture | Training/evaluation RGB plus synchronized labels and metadata | Retain as a verified dataset with exact frame identity |

Turning on a detector or review-video recorder must not block the raw browser
stream. Detector throughput is reported separately as overlay FPS. A review MP4
is not promoted to a training dataset merely because its resolution is high.

## Exact Windows/LAN acceptance setup

Use the same fixed setup for every candidate so results are comparable:

- CARLA 0.9.16 and the matching PythonAPI/Worker on the Windows host;
- Windows power mode set to Best performance and CARLA in the foreground;
- Operator computer and Windows host on the same LAN;
- wired Gigabit Ethernet for the release measurement; Wi-Fi is recorded only as
  an additional environment, never substituted silently;
- `Town10HD_Opt`, Clear Noon, one ego vehicle, fixed seed `20260809`;
- browser zoom 100%, one Operator tab, developer tools closed;
- a 30-second warm-up excluded from every measurement; and
- no unrelated downloads, cloud sync, or GPU workloads during the run.

Record the Windows CPU/GPU model, CARLA quality preset, screen resolution,
connection type/link speed, Operator computer/browser versions, commit hash,
Worker commit hash, and whether spectator mirroring is enabled. A result without
this inventory is diagnostic only.

During a Drive, `GET /api/drive/state` exposes `stream.target_fps`,
`stream.source_fps`, `stream.frame_age_seconds`, `stream.stale`, and
`stream.overlay_fps`. During Garage, `GET /api/garage/preview/state` exposes the
same target/source FPS, age, and stale fields under `stream`. The authenticated
Worker current-scene response exposes camera telemetry:

```text
frames_received
frames_encoded
frames_dropped_pending
frames_replaced
average_encode_ms
actual_fps_5s
latest_jpeg_bytes
encoded_bytes_total
```

When saving Worker telemetry, extract only the camera and capability fields; do
not retain `lease_token` or authorization headers in an artifact.

## Acceptance matrix

Run the rows in order. Restart the Garage or Drive session between rows. Each
timed row follows the 30-second warm-up and uses raw view unless the row
explicitly says overlay.

| ID | Camera | Scene load | Detector / recording | Duration | Required outcome |
| --- | --- | --- | --- | ---: | --- |
| A | Garage 720p30 | 40 traffic, 30 walkers, construction props | Off / off | 5 min | Orbit and every camera preset work; replace the vehicle once; all applicable production thresholds pass |
| B | Drive 720p30 | 0 traffic, 0 walkers, no props | Off / off | 10 min | All production thresholds pass |
| C | Drive 720p30 | 40 traffic, 30 walkers, construction props | Off / off | 10 min | All production thresholds pass |
| D | Drive 720p30 | Same as C | RT-DETR overlay / review MP4 on | 10 min | Raw thresholds pass; overlay stays responsive; saved video duration is within 1 s of session duration |
| E | Garage 720p60 | Same as A | Off / off | 5 min | All applicable high-refresh thresholds pass |
| F | Drive 720p60 | Same as B | Off / off | 5 min | All high-refresh thresholds pass |
| G | Drive 720p60 | Same as C | Off / off | 5 min | Stress row passes before an unqualified 60 FPS claim |

Rows A-D are the release gate. Row F promotes Drive 720p60 from “selectable” to
“accepted for an empty/light scene.” Rows E-G together are required before
making an unqualified `720p60` Garage/Drive claim. RT-DETR at 60 inference FPS
is not a goal and is not part of rows E-G.

## Hard pass/fail thresholds

For every production row (A-D):

- Worker `actual_fps_5s` and Operator `stream.source_fps` are at least `28.5`
  in at least 95% of one-second samples;
- neither metric is below `24.0` for five consecutive samples;
- `frames_dropped_pending / frames_received <= 0.05` after warm-up;
- `average_encode_ms <= 25.0`;
- Operator `stream.frame_age_seconds <= 0.150` in at least 95% of samples and
  never exceeds `0.500` for two consecutive samples;
- `stream.stale` is never true after warm-up;
- the persistent browser stream does not reconnect, terminate, or display a
  frozen frame for one second or longer;
- input-to-visible-response p95 is at most `150 ms` over 30 deliberate steering
  transitions measured with a 120 FPS or faster external recording; and
- no unrequested deadman brake, emergency brake, control-mode change, Worker
  camera error, or truncated output artifact occurs.

For high-refresh rows (E-G), replace the frame and encoder limits with:

- Worker and Operator FPS at least `57.0` in at least 95% of samples;
- neither FPS metric below `48.0` for five consecutive samples;
- pending-drop ratio at most `0.05`;
- average encode time at most `12.5 ms`; and
- the same 150 ms age/response, stale, reconnect, safety, and artifact limits.

A single hard-threshold failure fails that row. Report the observed value; do
not average a failed loaded scene together with an easy scene.

For row D, overlay FPS is recorded but has no camera-rate threshold. It must be
greater than zero, the latest overlay must continue updating, and raw FPS/age
must still pass. This keeps model speed from being misreported as video speed.

## Bottleneck decision tree and final solution

Use the first failing boundary; do not tune several layers at once.

1. If the per-second delta of Worker `frames_received` or `actual_fps_5s` is
   below threshold, test the same scene locally on Windows. If CARLA itself
   cannot maintain the target, reduce CARLA quality/traffic or keep the
   production profile at 720p30. A network codec cannot create real simulator
   frames.
2. If `frames_received` meets target but `frames_encoded` does not, and pending
   drops or encode time fails, JPEG encoding on the Windows Worker is the
   bottleneck. Keep 720p30 and replace the encoder with hardware H.264 before
   promoting 60 FPS.
3. If Worker FPS passes but Operator `source_fps` fails, the Worker-to-Operator
   LAN/MJPEG path is the bottleneck. Verify wired Gigabit and no packet loss. If
   it still fails, the final transport is hardware H.264 on Windows plus WebRTC
   to the browser; adding more JPEG polling is explicitly rejected.
4. If Operator source FPS passes but frame age or input-to-visible latency fails,
   browser MJPEG decode/render is the bottleneck. The final solution is again
   H.264/WebRTC with the control POST/heartbeat path kept independent.
5. If raw passes but overlay fails, lower detector cadence or use a faster model.
   Do not lower the raw camera rate or couple steering to inference completion.

Therefore the release answer is deterministic: ship 720p30 only after A-D pass;
enable 720p60 as an accepted profile only after E-G pass; and implement
Windows-side hardware H.264/WebRTC if the encoded Worker stream is healthy but
the remote MJPEG boundary still fails. Moving the complete research UI to the
Windows machine is not required.
