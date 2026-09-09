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
