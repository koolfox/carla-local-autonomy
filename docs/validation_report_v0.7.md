# CARLA Vision Framework Validation Report v0.7

Date: 2026-07-26  
Status: development validation  
Primary sensor boundary: one forward-facing monocular RGB camera  
CARLA endpoint: `172.20.10.7:2000`  
CARLA target: 0.9.16  
New gate: read-only native-collection readiness

## 1. Executive conclusion

The framework now has an artifact-producing native preflight between scenario
planning and destructive dataset collection. A successful preflight command
means the assessment completed; readiness is a separate boolean that is true
only when all automated and manual gates pass.

Two preflights ran against the real endpoint. Both verified their scenario
plans and selected train episode, opened the CARLA RPC endpoint, read server
version `0.9.16` and map `Town10HD_Opt`, and recorded
`simulator_mutated: false`. Both correctly reported `not_ready`: the current
Apple arm64 host has no importable official CARLA PythonAPI, and the operator
has not confirmed world reload or exclusive `world.tick()` ownership.

No attempt was made to reinterpret endpoint reachability as collection
readiness. No map was loaded, actor created or destroyed, setting changed,
world ticked, control applied, or Dataset directory created.

A separate one-episode integration plan reduces the first destructive test
from the 3,450 captures in the thesis pilot to 50 captures. This is the next
authorized unit of work once a compatible PythonAPI execution path exists and
the two manual gates are explicitly confirmed.

## 2. Read-only preflight contract

The command is:

```bash
uv run carla-native-preflight \
  --scenario-plan runs/<scenario-plan> \
  --dataset-id <unused-dataset-id> \
  --runs-root runs \
  --run-id <unique-preflight-id> \
  --host 172.20.10.7 \
  --port 2000 \
  --partition train \
  --max-episodes 1
```

It creates:

```text
runs/<preflight-id>/
├── manifest.json
├── summary.json
├── checks.json
├── checks.csv
├── selected_episodes.json
└── report.md
```

The 14 checks cover:

- scenario-plan hash and deterministic recomputation;
- non-empty episode selection;
- unused Dataset output ID;
- project Python version;
- official binary platform advisory;
- TCP endpoint reachability;
- read-only version/map RPC;
- server version agreement with the plan;
- native PythonAPI import;
- native read-only connection;
- native client version;
- server version observed through the native client;
- world-reload confirmation;
- exclusive-tick-owner confirmation.

The semantic verifier independently recomputes automated, manual, and overall
readiness, check counts, episode selection, and planned capture count. It
rejects any preflight claiming mutation.

## 3. Real endpoint evidence

The small pilot assessment is:

```text
runs/native-preflight-pilot-20260726-v1
```

| Field | Observed |
|---|---:|
| Status | `not_ready` |
| TCP reachable | true |
| TCP connect latency | 7.5085 ms |
| Read-only RPC latency | 22.1502 ms |
| Server version | `0.9.16` |
| Current map | `Carla/Maps/Town10HD_Opt` |
| PythonAPI importable | false |
| Automated ready | false |
| Manual ready | false |
| Overall ready | false |
| Simulator contacted | true |
| Simulator mutated | false |
| Checks passed | 7 |
| Checks failed | 1 |
| Checks pending | 2 |
| Checks skipped | 3 |
| Advisory warnings | 1 |

The only direct automated failure is the missing version-matched official
PythonAPI. The three downstream native connection/version checks are skipped,
not misreported as independent failures. The platform warning records Darwin
arm64; CARLA 0.9.16 officially distributes Python wheels for supported
Windows/Linux x86-64 combinations rather than this platform.

The official 0.9.16 guidance lists Windows and Ubuntu binary targets and
Python 3.7–3.12. PyPI publishes 0.9.16 wheels for Windows x86-64 and
manylinux x86-64. See the versioned
[CARLA quick start](https://carla.readthedocs.io/en/0.9.16/start_quickstart/)
and [CARLA package files](https://pypi.org/project/carla/0.9.16/).

## 4. Minimal first native plan

Configuration:

```text
configs/scenarios/native_integration_pilot_v1.json
configs/scenarios/split_plan_native_integration_pilot_v1.json
```

Verified plan:

```text
runs/scenario-plan-native-integration-pilot-v1
```

| Field | Value |
|---|---:|
| Map | `Town10HD_Opt` |
| Partition | `train` |
| Episodes | 1 |
| Planned captures | 50 |
| Duration | 10 simulation seconds |
| Warm-up | 40 ticks |
| Traffic vehicles | 12 |
| Walkers | 8 |
| Crossing factor | 0.10 |
| Weather | clear day |
| Props | construction sign and traffic cone |
| RGB camera | 1280x720, 90 degree FOV |
| Fixed delta | 0.05 s |
| Sensor tick | 0.10 s |

The plan retains the same exact-frame, synchronous-world, synchronous-Traffic
Manager, one-tick-owner, reload-per-episode, front-RGB runtime, and privileged
teacher disclosure contracts as the full thesis plan.

The intended output is:

```text
datasets/ds-carla0916-native-pilot-v001
```

The Dataset ID remains unused.

## 5. Operator UI execution

The operator panel now has a **Run read-only readiness preflight** action in
the native workflow card. It uses the same selected plan, Dataset ID, endpoint,
partition, episode limit, optional PythonAPI path, and visible manual
acknowledgement as native capture.

The real HTTP API created:

```text
operator_sessions/op-20260726t203533z-native-preflight-34a7877d
runs/native-preflight-ui-20260726-v1
```

The operator job completed with:

```text
returncode             0
destructive            false
motion_authorized      false
expected_output_exists true
```

The child assessed one train episode and 150 planned captures from the thesis
plan. Its not-ready result matches the 50-frame pilot assessment. The operator
session fingerprints the child manifest as an external reference.

## 6. Multi-episode dataset handoff

A new integration test executes this chain without contacting CARLA:

```text
verified thesis scenario plan
  -> fake native session for one train + one val_seen episode
  -> real DatasetWriter release
  -> generic and dataset semantic verification
  -> real Dataset QA tables, plots, and montage
  -> real training dry-run
```

The fake boundary is only actor/sensor delivery. Dataset files, instance-mask
labels, COCO/YOLO exports, checksums, QA, split resolution, seed schedule,
trainer data configuration, and artifact manifests use production code.

This test exposed and fixed a real defect: the Dataset verifier previously
rejected the repeated `native_episode_provenance` role required by
multi-episode collection. It now permits that repeatable role while checking:

- exactly one provenance artifact per Dataset episode;
- episode IDs and partitions;
- per-episode sample and annotation counts;
- complete episode status;
- successful actor cleanup;
- release episode count;
- restoration to asynchronous world mode.

The handoff test also proves that `train` and `val_seen` resolve correctly,
that no locked-test partition reaches the trainer, that training is not
executed during dry-run, and that QA/training leave the source Dataset
manifest unchanged.

## 7. Manifest-generated report

`reports/rpt-framework-validation-20260726-v005` aggregates:

1. framework report v0.6;
2. reproduction bundle v0.6;
3. the minimal native scenario plan;
4. the 50-frame native pilot preflight;
5. the operator-launched thesis-plan preflight;
6. the operator session.

| Field | Value |
|---|---:|
| Verified source objects | 6 |
| Canonical metric rows | 54 |
| Source-artifact inventory rows | 43 |
| Registered report artifacts | 9 |
| Payload checksum entries | 8 |
| External source references | 6 |
| Unregistered files | 0 |

The report manifest SHA-256 is:

```text
d6e3366bdfdc5f35fedc4c18f1ca6362025a210e6fe69d7f55948002dee314ee
```

## 8. Verification evidence

Artifact manifest SHA-256 values:

```text
pilot scenario plan  4b1b7414f8083c53f48261bab2d14b5fff97cd602d83c09c826ec22a21712d5e
pilot preflight      2bf0ad29ba006760b3fbaced137a9df2e35f9cf140d6fc497c10573d789b9bdf
UI child preflight   3b4202f87d229069a5b7256c4cf686553d53d2a52085638636edb5e02a321123
operator session     6be2eef2ab87aa278d2e20b972d5211b8fe48f250b924d784cdc2254d905a217
v0.7 report          d6e3366bdfdc5f35fedc4c18f1ca6362025a210e6fe69d7f55948002dee314ee
```

The final pre-bundle regression completed:

```text
160 passed, 44 subtests passed
```

Ruff lint, format over 169 files, byte-code compilation, JavaScript syntax,
HTML parsing, lock consistency, source-reference verification, deep semantic
verification, and unregistered-file rejection all passed.

## 9. Defensible claim boundary

The evidence supports:

> The framework can create a read-only, independently verifiable native
> readiness assessment against the real CARLA endpoint, distinguish endpoint
> reachability from execution readiness, prepare a bounded first pilot, and
> carry a multi-episode native-shaped Dataset through QA and training dry-run.

It does not support:

- that the current Mac can execute the official native worker;
- that the simulator host is available over SSH;
- that the manual world/tick-owner gates have been satisfied;
- that the 50-frame pilot has been collected;
- that real actor spawning, sensor phase, labels, cleanup, or restoration pass;
- that a CARLA-specific RT-DETR checkpoint has been trained;
- that any detector or driving policy meets a thesis benchmark.

## 10. Next acceptance sequence

1. Make the repository available on a Linux or Windows x86-64 PythonAPI host.
2. Install the exact CARLA 0.9.16 wheel for its Python interpreter.
3. Rerun preflight until automated readiness is true.
4. Stop all competing clients and explicitly confirm both manual gates.
5. Execute only the one-episode, 50-frame native pilot.
6. Verify and audit the Dataset; manually inspect RGB/instance alignment,
   props, actors, cleanup, and restoration.
7. Fix or reject the pilot before any scale-up.
8. Build a multi-episode A2 Dataset with train/validation/locked-test groups.
9. Pass tiny-overfit, resume, and one-epoch gates.
10. Begin three-seed RT-DETR training and fair paired evaluation.

Vision control remains disabled throughout these stages.
