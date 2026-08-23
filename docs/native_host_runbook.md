# Native CARLA 0.9.16 Collection Runbook

Status: execution handoff for the first real synchronous dataset episode  
Risk class: simulator-destructive map reload; synthetic artifacts are
recoverable, existing world actors are not

## 1. Current readiness evidence

The checksum-tracked preflights
`native-preflight-pilot-20260726-v1` and
`native-preflight-ui-20260726-v1` establish that:

- CARLA RPC ports `2000` and `2001` are reachable;
- the read-only `version` and `get_map_info` calls succeed;
- the server reports `0.9.16`;
- the current map is `Town10HD_Opt`;
- the selected train episodes and Dataset IDs are valid;
- the Apple arm64 project environment has no importable official `carla`
  module;
- SSH connection to `172.20.10.7:22` times out;
- world-reload and exclusive-tick-owner confirmations remain pending;
- no simulator mutation occurred.

Consequently the native worker cannot yet execute on either side:

- the Mac can run scenario verification and `--dry-run`, but not import the
  official PythonAPI;
- the simulator host cannot currently be reached as a command-execution host.

Do not replace the official PythonAPI worker with extra undocumented raw RPC
calls for thesis capture. Synchronous stepping, Traffic Manager ownership,
batch actor creation, and cleanup should stay on the version-matched official
API path.

## 2. Required execution host

Prepare one Linux or Windows machine that:

1. can reach the configured CARLA 0.9.16 endpoint;
2. has a Python version accepted by this project and by the exact CARLA 0.9.16
   wheel/egg;
3. can import `carla` and reports client version `0.9.16`;
4. has the extracted portable native-host kit;
5. has enough storage for lossless RGB, retained privileged instance masks,
   metadata, COCO, and derived YOLO labels;
6. is the only client allowed to call `world.tick()` during capture.

The API can be installed in the environment or supplied as a directory,
wheel, or egg through `--carla-python-api`/`PYTHONPATH`.

Build and transfer the current handoff as an on-demand artifact:

```text
native_kits/<kit-id>/payload/native-host-kit.zip
```

It carries the exact project source required by this workflow. Follow
[Portable native-host kit](native_host_kit.md) for extraction, bootstrap, the
two preflights, guarded collection, and transfer.

## 3. Non-destructive preflight

Run these before stopping any live demo:

```bash
uv sync --extra research --group dev

uv run python -c \
  "import carla; print(carla.__file__); print(carla.__version__)"

uv run carla-verify \
  runs/scenario-plan-native-integration-pilot-v1 \
  --reject-unregistered

uv run carla-native-preflight \
  --scenario-plan runs/scenario-plan-native-integration-pilot-v1 \
  --dataset-id ds-carla0916-native-pilot-v001 \
  --runs-root runs \
  --run-id native-preflight-pilot-<unique-id> \
  --host 172.20.10.7 \
  --port 2000 \
  --partition train \
  --max-episodes 1
```

If the PythonAPI is not installed in the environment:

```bash
uv run carla-native-preflight \
  --scenario-plan runs/scenario-plan-native-integration-pilot-v1 \
  --dataset-id ds-carla0916-native-pilot-v001 \
  --runs-root runs \
  --run-id native-preflight-pilot-<unique-id> \
  --carla-python-api /absolute/path/to/carla-0.9.16.egg \
  --partition train \
  --max-episodes 1
```

The preflight creates `summary.json`, `checks.json`, `checks.csv`,
`selected_episodes.json`, `report.md`, and `manifest.json`. It may connect and
issue only the read-only version/map/world/settings queries listed in the
report. It must record `read_only: true` and `simulator_mutated: false`.

On the compatible execution host, repeat it with:

```text
--confirm-world-reload
--confirm-exclusive-tick-owner
```

Do not continue unless automated, manual, and overall readiness are all true.
These flags record operator confirmations; they do not mutate the simulator.

## 4. Exclusive-ownership checklist

Before the real command:

- stop `carla-vision`, manual control, traffic generators, and every other
  tick-producing client;
- confirm no valuable actor or unsaved world state remains;
- record the simulator build/version and current map;
- ensure CARLA is responsive and not already in synchronous mode owned by
  another client;
- reserve a unique development dataset ID;
- use a `train` episode for the first integration test, not a locked test
  partition;
- keep the simulator console visible for crash diagnostics.

The acknowledgement flag is authorization for this selected run only. It does
not authorize unrelated deletion or a full 23-episode capture.

## 5. First real episode

The first run uses the 50-frame integration plan, not the 3,450-frame thesis
plan. Run exactly one train episode:

```bash
uv run carla-native-collect \
  --scenario-plan runs/scenario-plan-native-integration-pilot-v1 \
  --dataset-id ds-carla0916-native-pilot-v001 \
  --datasets-root datasets \
  --host 172.20.10.7 \
  --port 2000 \
  --partition train \
  --max-episodes 1 \
  --timeout 30 \
  --sensor-timeout 10 \
  --acknowledge-exclusive-tick-owner
```

Expected behavior:

1. verify every scenario-plan hash and recompute its episodes;
2. fail unless PythonAPI client and server versions are both exactly 0.9.16;
3. load/reload the selected map, which destroys actors from the old world;
4. become the sole synchronous fixed-delta tick owner;
5. seed Traffic Manager, pedestrian generation, and all framework namespaces;
6. spawn ego, traffic, walkers/controllers, declared props, RGB, and
   co-located privileged instance camera;
7. discover the real sensor phase and retain only exact target frame IDs;
8. write per-episode actor/spawn/sample/cleanup provenance;
9. destroy actors created by the worker;
10. restore asynchronous world and Traffic Manager settings before finalizing
    the checksum-indexed dataset.

If the process or simulator crashes, restart CARLA before another attempt; do
not assume synchronous settings or actors were restored.

## 6. Post-run gates

Do not train immediately. First run:

```bash
uv run carla-verify \
  datasets/ds-carla0916-native-pilot-v001 \
  --reject-unregistered

uv run carla-audit-dataset \
  --dataset datasets/ds-carla0916-native-pilot-v001 \
  --runs-root runs \
  --run-id ds-carla0916-native-pilot-v001-qa

uv run carla-verify \
  runs/ds-carla0916-native-pilot-v001-qa \
  --reject-unregistered
```

Then inspect:

- the QA montage, especially small signs, lights, pedestrians, riders, props,
  truncation, and instance fragmentation;
- retained frame IDs versus planned capture cadence;
- RGB/instance transforms, dimensions, FOV, timestamps, and sensor phase;
- spawned versus destroyed actor inventories;
- world/Traffic Manager asynchronous restoration;
- split and episode IDs;
- exact duplicate/leakage output;
- map, weather, traffic, walker, prop, and seed provenance;
- any failed or skipped spawn record.

Only after this 50-frame review should collection scale to the 1,000-frame A2
pilot. The full 3,450-capture scenario plan remains a later explicitly
authorized execution. See
[`native_pilot_checklist.md`](native_pilot_checklist.md) for the compact
operator handoff.
