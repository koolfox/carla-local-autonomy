# First Native Dataset Pilot Checklist

Status: execution gate  
Scope: one Town10HD episode, 50 planned RGB/instance pairs  
Dataset target: `ds-carla0916-native-pilot-v001`

## Why this pilot exists

The 23-episode thesis plan contains 3,450 planned captures and is too large for
the first destructive integration test. The first native run instead uses:

- one `Town10HD_Opt` episode;
- 12 traffic vehicles and 8 walkers;
- clear daylight;
- one construction sign and one traffic cone;
- 40 warm-up ticks;
- 200 capture-duration ticks;
- 50 planned captures at one capture every four world ticks.

This is large enough to test spawning, Traffic Manager, walkers, props,
autopilot motion, exact RGB/instance synchronization, labels, cleanup, and QA
without turning an integration check into a dataset campaign.

## Read-only gate

Run:

```bash
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

Do not continue unless the preflight report says:

```text
automated_ready           true
manual_ready              true
ready_for_native_execution true
simulator_mutated         false
```

The manual flags may be assessed with:

```text
--confirm-world-reload
--confirm-exclusive-tick-owner
```

They are recorded confirmations, not simulator operations.

## Operator confirmation

Before checking the two manual gates:

- stop every live runtime, manual-control client, traffic generator, and other
  tick owner;
- confirm the current CARLA world and actors are disposable;
- keep the simulator console visible;
- confirm at least several hundred MiB of output space;
- verify the selected Dataset ID does not exist;
- use only the `train` pilot episode;
- do not run a second collector concurrently.

## One authorized native command

Only after a ready preflight:

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

This command is destructive to the current simulator world. The
acknowledgement applies only to this one pilot.

## Mandatory post-run gates

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

Manually inspect the QA montage, exact frame IDs, actor spawn failures,
construction props, small objects, RGB/instance alignment, COCO/YOLO exports,
cleanup inventory, and restoration to asynchronous mode.

Do not scale collection or train a model until this pilot is accepted.
