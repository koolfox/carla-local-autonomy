# Local operator configuration

The normal root command is:

```bash
uv run carla-operator-ui --open-browser
```

If a `.env.local` file exists in the current project root, the operator entry point
loads the supported machine-local settings before starting the normal Garage
server. `.env.local` is ignored by Git; `.env.local.example` is the safe template.

## Precedence

```text
explicit CLI > process environment > .env.local > built-in defaults
```

For example, `CARLA_PORT=2000` in `.env.local` can be overridden by a process
`CARLA_PORT`, and an explicit `--carla-port 3000` overrides both.

## Supported keys

```text
CARLA_HOST
CARLA_PORT
CARLA_OPERATOR_BIND
CARLA_OPERATOR_PORT
CARLA_WORKSPACE
CARLA_OPERATOR_SESSIONS_ROOT
CARLA_WORLD_WORKER_URL
CARLA_WORLD_WORKER_PORT
CARLA_WORLD_WORKER_TOKEN
CARLA_ENABLE_EXPERIMENTAL
detector.enabled
```

`CARLA_WORLD_WORKER_URL` must be a raw `http://` or `https://` URL. Markdown link
syntax is rejected. The worker token remains process-local and is never added to
command-line arguments or browser-visible startup output.

`detector.enabled` controls the initial Garage detector checkbox. It accepts
`true/false`, `1/0`, `yes/no`, or `on/off`. A user action in the browser can still
change the checkbox after startup.

To explicitly disable experimental Garage features when an environment or file
enables them, use:

```bash
uv run carla-operator-ui --no-enable-experimental --open-browser
```

## Runtime state is not configuration

Values such as `overlay_frame_sequence`, current speed, actor IDs, current run ID,
and telemetry are produced by the running system. They must not be placed in
`.env.local`. `overlay_frame_sequence` is explicitly rejected so a stale value
cannot masquerade as live state.

## Example

```dotenv
CARLA_WORLD_WORKER_TOKEN='replace-with-local-secret'
CARLA_WORLD_WORKER_URL='http://192.168.1.108:8766'
CARLA_HOST='192.168.1.108'
CARLA_PORT='2000'
detector.enabled=true
```
