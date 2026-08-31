# Project structure and ownership

This document describes current source ownership and the incremental target
while the Operator completes product and developer convergence.

## Source layout

| Path | Responsibility |
| --- | --- |
| `carla_vision/native/` | Thin CARLA-host processes, Worker protocol, preflight, and native capture |
| `carla_vision/operator/` | Browser transport, Operator application/session ownership, jobs, and Worker/model adapters |
| `carla_vision/detectors/` | Model-neutral detector adapters and canonical detections |
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
- Executable external models enter through the manifest contract documented in
  [`runtime_model_packages.md`](runtime_model_packages.md); a loose checkpoint
  filename is never a runnable model identity.
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

## Convergence order

1. #66 — human developer map and feature-slice playbook.
2. #54 — one shared SessionConfig for Garage and Research (implemented; live
   acceptance remains part of closing the issue).
3. #67 and #68 — stable Operator application seam and deterministic no-CARLA
   development.
4. #55, #56, and #57 — executable presets, dense-scene evidence, and stable
   preview/control presentation.
5. #69, #62, #70, and #63 — route intent, canonical scene perception,
   ego-centric driver scene, and supervised Vision/Voxel control.
6. #27, #29-#32, and #64 — real CARLA data, checkpoints, and closed-loop
   acceptance.
7. #58 and #33 — final structural cleanup and release-candidate audit.

The milestones and issue bodies are authoritative if this summary becomes
stale. This order prevents cleanup from deleting a working capability before
its replacement is understandable, usable, and verified.
