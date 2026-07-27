# CARLA Vision Framework Validation Report v0.5

Date: 2026-07-26  
Status: development validation  
Primary sensor boundary: one forward-facing monocular RGB camera  
Primary detector family: RT-DETR  
Comparison adapter exercised: YOLO26n  
CARLA target: 0.9.16 at `172.20.10.7:2000`  
Live map gate: `Town10HD_Opt`

## 1. Executive conclusion

The framework now includes a model-neutral vision-policy interface and a
live, sequential multi-model shadow workflow. A policy can inspect only the
front RGB image and detections derived from that image. It returns a bounded
control proposal, but the runtime does not send that proposal to CARLA.
Privileged teacher control remains a separate, disclosed actuation lane.

A real two-cell matrix executed RT-DETR-L and YOLO26n for twelve seconds each
against the configured CARLA vehicle and RGB camera. Both child runs completed,
stopped the actuator, produced videos and machine-readable logs, and passed
dedicated plus generic semantic verification. The parent matrix completed
2/2 runs with zero failures. All policy-input audits passed and every retained
proposal records `actuation_applied: false`.

This establishes the live policy-shadow and auditing infrastructure. It does
not establish a safe driver, learned policy, detector ranking, or thesis
benchmark. The two models observed different sequential scenes, used
pretrained COCO weights and different development confidence floors, and were
moved only by the privileged simulator teacher.

## 2. Implemented safety boundary

The runtime path is:

```text
front RGB frame
      |
      v
detector adapter -> normalized RGB-derived detections
      |
      v
VisionObservation
  sequence
  source timestamp
  read-only copied RGB
  detections
  camera FOV
      |
      v
VisionPolicy -> bounded proposal -> log + overlay only
                                      |
                                      +-- NOT sent to CARLA

CARLA pose/speed -> privileged teacher -> watchdog actuator -> vehicle
```

`VisionObservation` excludes pose, velocity, map, route, actor IDs, CARLA
frame, depth, instance/semantic segmentation, LiDAR, GNSS, IMU, lane invasion,
ground truth, waypoints, and direct CARLA objects. The policy runs before the
runtime computes privileged pose and speed for the teacher and evaluation
logs.

The audit is structural isolation against accidental privileged inputs. It is
not a process sandbox for arbitrary malicious custom Python. Confirmatory
experiments must additionally control and review the custom policy package.

`--control vision` remains rejected. Valid live shadow combinations are:

- `--control none`: stationary or externally controlled perception shadow;
- `--control teacher`: disclosed privileged teacher motion;
- `--shadow-policy hazard-stop`: built-in contract/systems baseline;
- `--shadow-policy custom --policy-factory module:callable`: modular custom
  policy under the same observation/proposal contract.

## 3. Live matrix execution

Configuration:
`configs/shadow/rtdetr_yolo26_live_shadow_v1.json`

Parent artifact:
`runs/shadow-rtdetr-yolo26-live-v1`

The matrix preregistered:

| Field | Value |
|---|---|
| Endpoint | `172.20.10.7:2000` |
| Expected map | `Town10HD_Opt` |
| Vehicle actor | `24` |
| RGB camera actor | `25` |
| Camera resolution | `640x384` |
| Requested camera rate | 10 FPS |
| Camera FOV | 90 degrees |
| Duration | 12 seconds per child |
| Control source | privileged teacher |
| View | live split view |
| Recording | annotated MP4 |
| Policy | built-in hazard-stop shadow |
| Policy actuation authorized | false |

Execution required both `--execute` and
`--acknowledge-teacher-motion`. The parent ran children sequentially so the
models did not compete for MPS resources and so only one runtime controlled
the vehicle at a time.

The successful children were:

```text
shadow-rtdetr-yolo26-live-v1--rtdetr-l-hazard-stop-r00
shadow-rtdetr-yolo26-live-v1--yolo26n-hazard-stop-r00
```

The parent semantic verifier reconstructed the commands and their digests,
control mode, plan JSONL/CSV, execution results, child manifest references,
checksums, and no-actuation invariant.

## 4. Observed systems measurements

The following values were extracted from immutable child logs by separate
post-hoc analysis runs.

| Measurement | RT-DETR-L | YOLO26n |
|---|---:|---:|
| Requested duration | 12.0 s | 12.0 s |
| Actual elapsed | 12.1169 s | 12.0148 s |
| Perception submitted | 78 | 104 |
| Perception processed | 74 | 104 |
| Dropped before inference | 3 | 0 |
| Recorded/proposal frames | 73 | 103 |
| Effective source FPS | 5.9476 | 8.6415 |
| Model latency median | 152.8615 ms | 25.4605 ms |
| Model latency p95 | 164.3319 ms | 30.3080 ms |
| Pipeline latency median | 202.2025 ms | 42.8230 ms |
| Pipeline latency p95 | 286.2107 ms | 59.5245 ms |
| Estimated analyzed distance | 15.7348 m | 20.5400 m |
| Runtime distance summary | 15.9660 m | 20.7486 m |
| Maximum simulator speed | 2.4612 m/s | 2.4994 m/s |
| Retained detections | 520 | 129 |
| Frames with detections | 72 | 81 |
| Policy proposals | 73 | 103 |
| Throttle proposals | 73 | 103 |
| Brake proposals | 0 | 0 |
| Policy latency median | 0.0150 ms | 0.0128 ms |
| Policy latency p95 | 0.0290 ms | 0.0145 ms |
| Vision-policy actuation applied | no | no |

RT-DETR-L had a mean of 7.12 retained detections per processed frame at a
0.20 detector floor. YOLO26n had a mean of 1.25 at a 0.05 floor. These counts
must not be compared as accuracy: route position, scene content, frame
selection, model family, and operating floor differ. The paired offline replay
remains the correct mechanism for identical-input model comparison.

The built-in policy found no object satisfying all of its close,
driving-corridor, hazard-class, and confidence conditions, so it proposed low
throttle rather than braking for every recorded frame. Because proposals were
not applied, this result measures policy invocation and logging only.

## 5. Visual artifacts

Each child retains:

- `images/latest_overlay.png`;
- `video/overlay.mp4`;
- `logs/detections.jsonl`;
- `logs/policy_shadow.jsonl`;
- `policy_input_audit.json`;
- `summary.json`;
- `manifest.json`.

The RT-DETR video contains 73 frames at 10 FPS and is approximately 2.1 MiB.
The YOLO26 video contains 103 frames at 10 FPS and is approximately 2.6 MiB.
The terminal overlays visibly identify the simulator-teacher control source
and show `SHADOW PROPOSAL ... (NOT APPLIED)`.

Post-hoc analyses:

```text
runs/shadow-rtdetr-yolo26-live-v1--rtdetr-l-hazard-stop-r00-analysis
runs/shadow-rtdetr-yolo26-live-v1--yolo26n-hazard-stop-r00-analysis
```

Each analysis retains a canonical per-frame CSV, summary JSON, class-frequency
plot, detection/hazard timeline, latency/cadence plot, and simulator-speed
plot, with source-manifest and source-log hashes.

## 6. Manifest-generated validation report

`reports/rpt-framework-validation-20260726-v003` aggregates seven verified
sources:

1. the v0.4 framework report;
2. the v0.4 reproduction bundle;
3. the live shadow matrix parent;
4. the RT-DETR child;
5. the YOLO26 child;
6. the RT-DETR post-hoc analysis;
7. the YOLO26 post-hoc analysis.

The release contains:

| Field | Value |
|---|---:|
| Verified source objects | 7 |
| Canonical metric rows | 54 |
| Source-artifact inventory rows | 69 |
| Registered report artifacts | 11 |
| Payload checksum entries | 10 |
| External source references | 7 |
| Unregistered files | 0 |

Its plots include a dedicated live throughput/latency figure whose title
explicitly states that sequential systems measurements are not detector
accuracy.

## 7. Verification evidence

The real objects passed:

```bash
uv run carla-verify-shadow-matrix \
  runs/shadow-rtdetr-yolo26-live-v1

uv run carla-verify-shadow \
  runs/shadow-rtdetr-yolo26-live-v1--rtdetr-l-hazard-stop-r00

uv run carla-verify-shadow \
  runs/shadow-rtdetr-yolo26-live-v1--yolo26n-hazard-stop-r00

uv run carla-verify \
  runs/shadow-rtdetr-yolo26-live-v1 \
  runs/shadow-rtdetr-yolo26-live-v1--rtdetr-l-hazard-stop-r00 \
  runs/shadow-rtdetr-yolo26-live-v1--yolo26n-hazard-stop-r00 \
  --reject-unregistered

uv run carla-verify \
  reports/rpt-framework-validation-20260726-v003 \
  --reject-unregistered
```

The parent manifest SHA-256 is:

```text
e0f1351856ea7c1d3b735de2e4c413718977cf17e882d1cc86f522cfb4816f73
```

The child manifest SHA-256 values are:

```text
RT-DETR-L  1d30b86aa4b27f207ecd9273e0346352047ac854bd06402f3e58d6aabc4b91b3
YOLO26n    db70ebf8b919380eeb352fbaf9582cf5f7d037313ea3c7d8b9ec8e393a4e292e
```

The v0.5 report manifest SHA-256 is:

```text
40fcb6fdf902009ab6780c865ee2f966dabdc698b241eaf7e301f2e6c5b95879
```

The code regression at this stage completed 149 tests plus 44 subtests. Ruff,
format checking, lock resolution, and byte-code compilation are rerun after
the final documentation and reproduction bundle are added.

## 8. Defensible claim boundary

The new evidence supports this claim:

> The development framework can invoke interchangeable detector and
> vision-policy adapters on a live CARLA front-RGB stream, display and record
> exact-frame proposals, keep privileged teacher actuation separate, and
> independently verify that policy proposals were never applied.

It does not support these claims:

- the hazard-stop baseline is a safe or competent driver;
- a learned vision-only policy exists;
- RT-DETR is more accurate than YOLO26 on the live sequence;
- detector performance generalizes across maps, weather, traffic, or seeds;
- teacher movement is vision-only autonomy;
- structural Python isolation is a security sandbox;
- CARLA performance implies public-road safety.

## 9. Remaining acceptance sequence

1. Create and tag a clean Git baseline.
2. Generate a confirmatory reproduction bundle from that exact commit.
3. Install the matching official CARLA 0.9.16 PythonAPI on an isolated native
   host.
4. Execute and audit one short synchronous collection episode.
5. Scale to a frozen, episode-grouped train/validation/locked-test dataset.
6. Pass tiny-overfit, interruption/resume, and one-epoch training gates.
7. Train at least three RT-DETR seeds and fair comparison baselines.
8. Select thresholds on validation only and complete blinded failure review.
9. Evaluate once on the locked multi-episode test release with episode
   bootstrap and target-hardware latency.
10. Extend live shadow to the frozen scenario matrix and model packages.
11. Specify temporal monocular state, route-intent policy, safety supervisor,
    and intervention accounting before any vision actuation is enabled.
12. Add registry/signoff and thesis PDF/LaTeX rendering.

Until those gates pass, `--control vision` remains disabled and all current
results remain development evidence.
