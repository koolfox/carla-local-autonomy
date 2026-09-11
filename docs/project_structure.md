# Project structure and ownership

This document describes current source ownership and the incremental target
while the Operator completes product and developer convergence.

For a concrete bug or feature, start with the [feature-to-test map](README.md).

## Source layout

| Path | Responsibility |
| --- | --- |
| `carla_vision/native/` | Thin CARLA-host processes, Worker protocol, preflight, and native capture |
| `carla_vision/operator/` | Browser transport, Operator application/session ownership, jobs, and Worker/model adapters |
| `carla_vision/detectors/` | Model-neutral detector adapters and canonical detections |
| `carla_vision/segmentation/` | Independent road/lane segmentation adapters, worker and overlays |
| `carla_vision/model_registry.py`, `model_package_contracts.py` | Runtime model discovery, executable package identity, and pure adapter contracts |
| `carla_vision/dataset/` | Exact-frame dataset capture, labels, export, and QA |
| `carla_vision/training/` | Detector training orchestration |
| `carla_vision/evaluation/` | Offline detector evaluation |
| `carla_vision/replay/` | Paired model replay |
| `carla_vision/failure_mining/` | Threshold selection follow-up and human review inputs |
| `carla_vision/evidence/`, `reporting/`, `reproduction/` | Artifact registry, reports, bundles, and verification |
| `carla_vision/imitation/`, `voxel/` | Explicitly experimental driving research |
| `web/` | The only long-term product frontend source |
| `configs/` | Versioned example and experiment configuration, never machine-local secrets |
| `tests/` | Contract, component, workflow, and fake-runtime regression tests |
| `docs/` | Architecture, operator runbooks, protocols, and evidence boundaries |

Multi-camera research capture extends `native/behavior_teacher.py` using
`native/camera_rig.py`. Rig contracts, additional-image verification and offline
replay live under `dataset/`; see [the focused runbook](multicamera_episodes.md).
It does not add a second live-session configuration or change the Garage camera.

Optional [native research jobs](native_research_jobs.md) launch installed trusted
tasks in a separate Windows process. `native/research_jobs.py` owns the stable
host protocol, `native/tasks/` contains task adapters, and
`operator/native_research.py` is the Mac SDK/CLI. Update a collector task package
to change its algorithm; change the bridge only for hosting/protocol behavior.
The existing `operator/jobs.py` still owns local research jobs; it is not replaced.

## Where should my next change go?

Choose the existing feature owner, not a new generic `utils/` or `services/`
directory. Most work needs only one implementation and its focused test:

- **Display and interaction:** Svelte components; pure presentation helpers
  under `web/src/lib/ui/`. Keep CARLA and model work out of the browser.
- **Shared setting:** the TypeScript configuration store and Python
  `operator/configuration.py` mapping together. No per-menu copy of state.
- **A new detector or road model:** its existing adapter package/factory, not
  another model-specific branch inside the Drive loop.
- **A user workflow:** its Operator owner. HTTP routes validate/dispatch and
  return the result; they should not implement dataset or model algorithms.
- **CARLA actors, ticks or camera relay:** `native/`, with the matching Worker
  client/protocol tests when the Worker is affected. Worker changes require an
  update on its host. Standalone research-collector changes only require updating
  the collector's PythonAPI environment. Both require appropriate live evidence.
- **Research algorithms:** the dataset, training, evaluation, imitation or voxel
  package. Keep their outputs behind existing contracts; mark experimental scope.

For example, the saved-recordings feature has these distinct owners:

| Responsibility | Source owner |
| --- | --- |
| Library list, selection, player UI | `web/src/lib/components/SavedRuns.svelte` |
| Workspace catalogue | `operator/catalog.py` |
| Manifest inspection and safe artifact resolution | `operator/artifacts.py` |
| Browser-compatible video conversion | `operator/recording_preview.py` |
| HTTP authorization, file transfer and seeking | `operator/server.py` |

The Python paths in this table are under `carla_vision/`. This is a map of
existing modules, not five services. `ArtifactStore(workspace)` can be used
without constructing the Operator; its tests are a small starting point for
backend contributors. Preserve the separation between manifest declarations,
file availability, and explicit research-object verification.

A recording label or player layout belongs to this saved-results boundary; it
does not justify edits to spawning, training, or the Windows bridge. Adding a new
artifact producer also should not require a new recording-library implementation:
register its video and metadata in the existing run manifest. Only extend a
shared contract when the feature needs new data, and test its existing consumers.
For example, multi-camera collection needs additional sensor ownership and image
references; browsing the resulting recording needs neither CARLA nor model loading.

## Migration boundaries

- `web/` is the official frontend source. The reviewed release bundle under
  `carla_vision/operator/console_static/` is generated from the locked frontend
  dependencies, committed with its source change, and included in the wheel.
  It must never be edited by hand.
- The packaged Svelte console is the primary product at `/`.
  `carla_vision/operator/static/` is available only at `/legacy/` as a
  temporary rollback surface. It is not a second product to extend and can be
  removed only after parity is verified.
- Shared map, weather, vehicle, traffic, pedestrian, perception, recording,
  and policy values belong to one `SessionConfig`. Garage preview, Drive, and
  the Research Scene Builder all consume that object through canonical
  adapters; recipe-only capture fields remain local to Scene Builder.
- Experiment presets are named patches over `SessionConfig`, not independent
  configuration stores; #55 owns this boundary.
- The Windows Worker remains thin and does not install Torch, Ultralytics, or
  the research stack. It owns CARLA world mutation, actor cleanup, and camera
  relay through a versioned capability contract.
- Research jobs consume a session reference plus job-specific values. They do
  not redefine shared world settings.
- Executable driving-policy packages use the manifest contract documented in
  [`runtime_model_packages.md`](runtime_model_packages.md). Built-in/custom
  detector selection uses `DetectorConfig` and its adapter; road models use
  `SegmentationConfig`. These are distinct current interfaces, not a universal
  loader for any `.pt` or `.onnx` file.
- The current HTTP request-handler inheritance is transport, not the desired
  home for feature logic. Issue #67 extracts application use cases and stable
  contracts incrementally before any backend-framework decision.
- Architecture, control, privileged-data, topology, and artifact decisions use
  the lightweight ADR process in [`adr/`](adr/).

## Runtime data

The following are workspace state, not source ownership:

```text
.env.local
datasets/
models/
operator_sessions/
output/
reports/
runs/
*.pt
*.onnx
*.engine
*.safetensors
web/node_modules/
web/.svelte-kit/
```

These paths are ignored by Git. A public sample artifact must be introduced
explicitly and must include provenance, size, license, and verification data.

## Convergence without a rewrite

Use the [live milestones](https://github.com/koolfox/carla-local-autonomy/milestones)
for issue status and the [release assessment](release_readiness.md) for gates.
First make the core workflow maintainable and reproducible, then extract a
small tested lifecycle boundary when a feature needs it. Preserve public
imports, Worker startup, APIs and artifact contracts during extraction.

Do not move all modules into new folders in one change. Do not delete a
working capability before its replacement is understandable, usable and
verified. Research/voxel work and public-console readiness are separate tracks,
so a useful console need not wait for every trained-driving experiment.
