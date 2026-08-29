# Unified Console Architecture

Status: migration foundation for Issue #45

## Why this exists

The current Operator/Garage grew through several working experiments. That produced useful capability, but configuration is now represented in several places: machine-local startup values, Drive start fields, experiment presets, Garage experimental controls, research job parameters, and World Worker capabilities. The migration goal is not to discard those capabilities. It is to give them one product model and one frontend.

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

## Frontend migration

`web/` is the new SvelteKit frontend. During migration it runs beside the existing Operator server and uses Vite's `/api` proxy in development. The legacy static shell remains the production UI until the new frontend reaches feature parity and its production build is served by the Python package.

The first slice deliberately does not rewrite Drive endpoints. It establishes:

- a component-framework build boundary;
- typed API reads for bootstrap and Drive catalog;
- a central Svelte store for SystemSettings and SessionConfig;
- browser persistence for the editable SessionConfig;
- visible preset patches instead of independent experiment state; and
- one visual settings surface for scene, drive, perception and recording.

## Backend migration

The frontend must depend on stable JSON contracts, not a Python web framework. Once the SvelteKit migration has captured the real API surface, the backend can move from `http.server` to a typed ASGI application (FastAPI is the current preferred candidate) without changing frontend semantics.

The backend migration should introduce versioned request/response models and OpenAPI, but should not change simulator ownership, safety gates or artifact semantics merely to fit the framework.

## World Worker boundary

The Windows World Worker remains a narrow remote execution adapter next to the official CARLA PythonAPI. It owns world mutation and camera relay operations that require the native simulator environment. The Operator owns user intent, research state, artifacts and policy/inference workflows.

The target protocol is one versioned capability document plus explicit scene lifecycle operations. Frontend controls must be enabled from resolved capabilities, not from duplicated assumptions in multiple screens.

## Migration rule

No legacy surface is deleted until the replacement slice is feature-complete, tested, and—where simulator behavior is involved—validated against live CARLA. This refactor is a consolidation, not a feature reset.
