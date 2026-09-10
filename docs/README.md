# Start here: maintain the console

You do not need to understand every research module to fix the Garage. Start
with one user action, its owner, and its test. This page is the navigation hub;
[development.md](development.md) explains the request paths in detail.

## Choose your route

| I want to… | Read |
| --- | --- |
| Install and use the product | [Main README](../README.md) |
| Make my first change | The walkthrough below, then [contribution rules](../CONTRIBUTING.md) |
| Find a feature or trace a bug | The feature map below, then [developer guide](development.md) |
| Understand directories | [Structure and ownership](project_structure.md) |
| Use my own detector / M9 | [M9 adapter](m9_detector.md), [detector contract](../carla_vision/contracts.py), [factory](../carla_vision/detectors/factory.py) |
| Change the road/lane model | [Road models](road_models.md) |
| Understand executable driving-policy packages | [Runtime model packages](runtime_model_packages.md) — not the same as detector weights |
| Understand voxel research | [RGB voxel viewer (Persian)](voxel_live_view_fa.md), [voxel research flow (Persian)](voxel_flow_fa.md) |
| Build datasets / evaluate models | [Experiment protocol](experiment_protocol.md), [reproducibility](reproducibility.md), [paired replay](paired_replay.md) |
| Prepare a public release | [Release assessment and gates](release_readiness.md) |

## The mental model

```text
web/                         browser: intent, presentation, input
  -> carla_vision/operator/  Mac/research host: sessions, models, files
       -> native/           Windows: CARLA API, actors, camera transport
       -> model/research packages: inference, datasets, evaluation
  <- exact source frames, status, recorded evidence
```

These are responsibilities inside one repository, not a proposal for new
services. Keep PythonAPI work on the CARLA host and ML work on the research
host in the normal topology. Experimental native research has separate
requirements, documented in its runbooks.

- **Edit:** `web/src/`, `carla_vision/`, focused `tests/`, versioned `configs/`.
- **Build, do not hand-edit:** `carla_vision/operator/console_static/`.
- **Do not extend:** `carla_vision/operator/static/` is the legacy fallback.
- **Do not commit:** `.env.local`, `models/`, `runs/`, `datasets/`, `reports/`,
  `operator_sessions/`, caches. Keep real recordings and weights safe during cleanup.

## Find the feature, then its test

Paths below are relative to the repository root. A Windows update is needed
only when native code/protocol or native dependencies change, not for every UI edit.

| User-visible change | Start here | First regression check | Windows? |
| --- | --- | --- | --- |
| Paint names / icons | [paint.ts](../web/src/lib/ui/paint.ts), [VehiclePicker](../web/src/lib/components/VehiclePicker.svelte) | `web/tests/paint.test.mjs` | No |
| Menu, layout, touch input | [GarageMenu](../web/src/lib/components/GarageMenu.svelte), [DriveCockpit](../web/src/lib/components/DriveCockpit.svelte), [ManualControlPad](../web/src/lib/components/ManualControlPad.svelte) | Svelte check/build; `tests/test_garage_visual_contract.py`; manual browser check | No |
| A shared Scene/Research setting | [SceneWorldFields](../web/src/lib/components/SceneWorldFields.svelte), [configuration store](../web/src/lib/stores/configuration.ts), [Python mapping](../carla_vision/operator/configuration.py) | `tests/test_operator_configuration.py`, `npm test` | Only if its native implementation changes |
| Rapid edits / Apply stuck | [garagePreview.ts](../web/src/lib/domain/garagePreview.ts), [Operator API](../web/src/lib/api/operator.ts), [preview manager](../carla_vision/operator/garage_preview.py) | `web/tests/garage-preview.test.mjs`, `tests/test_garage_reconfigure.py` | Depends on failing layer |
| Camera disappears / stream stalls | [garagePreviewStream.ts](../web/src/lib/domain/garagePreviewStream.ts), [Worker client](../carla_vision/operator/world_worker_client.py) | `web/tests/garage-preview-stream.test.mjs`, `tests/test_world_worker_stream_lifecycle.py` | If camera relay changes |
| Start location / preview-to-Drive handoff | [garage_drive.py](../carla_vision/operator/garage_drive.py), [garage_preview.py](../carla_vision/operator/garage_preview.py) | `tests/test_garage_scene_handoff.py` | If lease/native behavior changes |
| Traffic, crossing, actor replacement | [world_worker.py](../carla_vision/native/world_worker.py), [observable worker](../carla_vision/native/observable_world_worker.py) | `tests/test_world_worker_population.py`, `tests/test_world_worker_reconfigure.py`; retained live run | Yes |
| Detection adapter / custom heads | [detector factory](../carla_vision/detectors/factory.py), [M9](../carla_vision/detectors/m9_hierarchical.py), [contracts](../carla_vision/contracts.py) | `tests/test_perception.py`, `tests/test_m9_notebook.py` | No |
| Detection text and confidences | [display.py](../carla_vision/display.py), [drive.py](../carla_vision/operator/drive.py) | `tests/test_overlay_labels.py` | No |
| Road / lane perception | [segmentation factory](../carla_vision/segmentation/factory.py), [overlay](../carla_vision/segmentation/overlay.py) | `tests/test_segmentation.py`, `tests/test_yolop.py`, `tests/test_yolopv2.py` | No |
| RGB voxel display / geometry | [live_view.py](../carla_vision/voxel/live_view.py), [geometry.py](../carla_vision/voxel/geometry.py), [rendering.py](../carla_vision/voxel/rendering.py) | `tests/test_voxel_live_view.py`, `tests/test_voxel_geometry.py` | Not for RGB-only rendering |
| Saved runs and player | [SavedRuns](../web/src/lib/components/SavedRuns.svelte), [artifacts.py](../carla_vision/operator/artifacts.py), [recording_preview.py](../carla_vision/operator/recording_preview.py), [catalog.py](../carla_vision/operator/catalog.py) | `tests/test_operator_artifacts.py`, `tests/test_operator.py`, `tests/test_recording_preview.py`; actual playback/seek check | No |
| Dataset / training / evaluation | [dataset](../carla_vision/dataset/), [training](../carla_vision/training/), [evaluation](../carla_vision/evaluation/) | `tests/test_dataset_collector.py`, `tests/test_training.py`, `tests/test_evaluation.py` | Native collection only |
| Research-job buttons / execution | [commands.py](../carla_vision/operator/commands.py), [jobs.py](../carla_vision/operator/jobs.py), [installed commands](../pyproject.toml) | `tests/test_operator.py`; producer's own test | Depends on job |

The table finds owners; it does not certify a feature's real-CARLA performance.
For a new API field, read the Python contract and TypeScript mirror together.
Do not add a second configuration store or another retry loop around an
existing owner.

## Your first independent change

Use a clean checkout and one branch. This example changes a paint display
label without touching CARLA, RGB payloads, or the world lifecycle.

1. Follow [branch setup](../CONTRIBUTING.md#start-one-task). Install with
   `uv sync --group dev --frozen`; then run `npm ci` inside `web/`.
2. Read [paint.ts](../web/src/lib/ui/paint.ts) and its
   [test](../web/tests/paint.test.mjs). Pick one display-only improvement.
3. Add a failing test for the chosen RGB value/name. Run from `web/`:

   ```bash
   node --test tests/paint.test.mjs
   ```

4. Change only the presentation mapping. The CARLA RGB value must remain
   unchanged. Run the same test again, then:

   ```bash
   npm test
   npm run check
   npm run build
   ```

5. Review the source diff and generated bundle together. In the PR, explain
   the expected label, test result, and why no Windows update is needed.

For browser work, use the two-terminal setup in
[development.md](development.md#start-here). An offline Operator is **not** a
simulator fixture: live controls remain unavailable. A deterministic developer
mode is still [#68](https://github.com/koolfox/carla-local-autonomy/issues/68).
Do not use a live driving session to experiment with untested control changes.

## A repeatable debugging routine

1. Write the exact action and expected outcome: e.g. change paint twice,
   expected final paint B, camera recovers, ego unchanged except paint.
2. Capture the request/response, operation ID, and relevant Operator/Worker
   logs. Redact both Operator and Worker tokens before sharing them.
3. Use the feature map. Identify whether the first failure is browser state,
   HTTP translation, Operator lifecycle, native CARLA work, or model output.
4. Reproduce at that boundary with a focused test. A timeout alone does not
   establish whether the remote mutation happened; inspect operation state.
5. Fix the owning layer. Re-run its tests and one adjacent boundary test.
6. For actors, camera timing, controls, or population: retain a real CARLA
   acceptance run. Fakes cannot certify those behaviors.

## Keep this understandable

One PR should deliver one outcome: implementation, its regression test, and
the relevant documentation update. Keep existing public APIs and model/frame
contracts stable during extraction. No new framework, service, package, or
folder is required merely to make a file shorter.

Update this map when an entry point moves; update the focused guide when its
behavior changes. Link to the [live milestones](https://github.com/koolfox/carla-local-autonomy/milestones)
instead of copying a second backlog into every document. Older delivery notes
may describe their original scope; current code/tests and maintained guides
take precedence when they disagree.
