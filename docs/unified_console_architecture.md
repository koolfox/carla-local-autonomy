# Unified Console Architecture

Status: active product architecture; remaining convergence tracked by Issue #45

## Why this exists

The Operator/Garage grew through several working experiments. The packaged
Svelte console is now the primary product, but some configuration and
application behavior is still represented in several places: machine-local
startup values, Drive start fields, experiment presets, research job
parameters, and World Worker capabilities. The remaining goal is not to
discard those capabilities. It is to finish one product model and one
application boundary.

## Configuration ownership

### SystemSettings

Machine/service configuration. It is resolved at process startup and is not a per-drive experiment setting.

Examples:

- workspace
- CARLA host/port
- World Worker configured/connected state
- local-only HTTP boundary
- experimental feature gate
- runtime capabilities

`.env.local`, process environment and explicit CLI arguments remain inputs to this layer. Secrets remain server-side and are never copied into browser configuration.

### SessionConfig

The single editable configuration for one interactive session/experiment:

- identity and seed
- map/weather/props
- traffic and pedestrians
- vehicle
- route
- control owner
- camera
- perception
- recording
- experiment preset identity

A preset is a named patch over this object. It must never own a parallel hidden copy of scene, perception or control fields.

### JobConfig

Research workflows add only parameters unique to that job: dataset ID, epochs, output ID, benchmark input, checkpoint selection, and similar values. Shared scene/session context should be referenced from SessionConfig or a retained SessionConfig artifact rather than re-entered in unrelated forms.

## Frontend

`web/` is the authoritative SvelteKit frontend. Its deterministic static build
is packaged and served at `/` by the Python application. In development, Vite
uses an `/api` proxy to the local Operator server. The old static shell remains
only at `/legacy/` as a temporary rollback path until the remaining parity and
live-acceptance issues pass.

The current foundation establishes:

- a component-framework build boundary;
- typed API reads for bootstrap and Drive catalog;
- a central Svelte store for SystemSettings and SessionConfig;
- browser persistence for the editable SessionConfig;
- visible preset patches instead of independent experiment state; and
- a Garage-first full-screen experience with responsibility-based modals;
- shared Scene and Research controls backed by the same store; and
- requested, resolved, and applied configuration evidence at execution
  boundaries.

Issue #54 establishes shared Research/Garage configuration. Issues #55-#57
complete executable presets, dense preparation, and preview/control quality. Issue #68 adds a
deterministic no-CARLA developer surface against the same contracts.

## Backend convergence

The frontend depends on stable JSON contracts, not a Python web framework.
Issue #67 first extracts explicit application use cases and adapters from the
current HTTP request-handler hierarchy. Preview, Drive, research jobs, model
execution, and artifact verification must be testable without starting an HTTP
server.

Do not combine that work with a framework rewrite. A later ASGI/FastAPI change
is justified only if it reduces transport code while preserving the same
contracts, simulator ownership, safety gates, and artifact semantics.

## World Worker boundary

The Windows World Worker remains a narrow remote execution adapter next to the official CARLA PythonAPI. It owns world mutation and camera relay operations that require the native simulator environment. The Operator owns user intent, research state, artifacts and policy/inference workflows.

The target protocol is one versioned capability document plus explicit scene lifecycle operations. Frontend controls must be enabled from resolved capabilities, not from duplicated assumptions in multiple screens.

## Migration rule

No legacy surface is deleted until the replacement slice is feature-complete, tested, and—where simulator behavior is involved—validated against live CARLA. This refactor is a consolidation, not a feature reset.
