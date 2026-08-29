# Project structure and ownership

This document describes the intended source layout while the Operator console
converges on one product surface.

## Source layout

| Path | Responsibility |
| --- | --- |
| `carla_vision/native/` | Thin CARLA-host processes, Worker protocol, preflight, and native capture |
| `carla_vision/operator/` | Browser API, Garage and Drive session ownership, jobs, and Worker client |
| `carla_vision/detectors/` | Model-neutral detector adapters and canonical detections |
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
- `carla_vision/operator/static/` is a temporary rollback surface. It is not a
  second product to extend. It can be removed only after parity is verified.
- Shared map, weather, vehicle, traffic, pedestrian, perception, recording,
  and policy values belong to one `SessionConfig`; #54 completes that mapping.
- Experiment presets are named patches over `SessionConfig`, not independent
  configuration stores; #55 owns this boundary.
- The Windows Worker remains thin and does not install Torch, Ultralytics, or
  the research stack. It owns CARLA world mutation, actor cleanup, and camera
  relay through a versioned capability contract.
- Research jobs consume a session reference plus job-specific values. They do
  not redefine shared world settings.

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

1. #53 — deterministic Svelte packaging from a clean checkout.
2. #54 and #55 — one session configuration and real experiment presets.
3. #56 — responsive dense-scene preparation and maximum-setting evidence.
4. #57 — stable preview quality, camera fit, and responsive controls.
5. #48 and #27 — model-runtime hardening and live-CARLA acceptance.
6. #58 and #33 — final structural cleanup and release-candidate audit.

This order prevents final cleanup from deleting a legacy capability before its
replacement is usable and verified.
