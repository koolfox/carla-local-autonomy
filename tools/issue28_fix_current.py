from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}: {old[:100]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "carla_vision/native/world_worker.py",
    "            vehicles: list[dict[str, Any]] = []\n"
    "            for blueprint in sorted(definitions, key=lambda value: str(value.id)):",
    "            spawn_points = list(world.get_map().get_spawn_points())\n"
    "            vehicles: list[dict[str, Any]] = []\n"
    "            for blueprint in sorted(definitions, key=lambda value: str(value.id)):",
)

replace_once(
    "web/src/lib/components/SceneWorldFields.svelte",
    "  import { fieldNumber, fieldValue } from '$lib/ui/events';\n\n"
    "  export let includeRoute = false;",
    "  import type { RouteMode } from '$lib/domain/config';\n"
    "  import { fieldNumber, fieldValue } from '$lib/ui/events';\n\n"
    "  export let includeRoute = false;\n\n"
    "  $: selectedMap = $sessionConfig.scene.mapName === 'current'\n"
    "    ? $systemSettings?.currentMap ?? ''\n"
    "    : $sessionConfig.scene.mapName;\n"
    "  $: spawnCatalogReady = Boolean(\n"
    "    $systemSettings?.workerConfigured\n"
    "      && $systemSettings?.capabilities.spawn_point_selection\n"
    "      && $workspaceOptions.spawnPointMap\n"
    "      && selectedMap === $workspaceOptions.spawnPointMap\n"
    "  );\n"
    "  $: selectedRouteAvailable = Boolean($systemSettings?.capabilities.selected_route);\n\n"
    "  function spawnValue(value: string): number | null {\n"
    "    return value === '' ? null : Number(value);\n"
    "  }",
)

old_route = '''  {#if includeRoute}
    <label class="field">
      <span>Route</span>
      <select
        value={$sessionConfig.route.mode}
        onchange={(event) =>
          patchSessionSection('route', {
            mode: fieldValue(event) as 'free' | 'random_destination'
          })}
      >
        <option value="free">Free drive</option>
        <option value="random_destination">Random destination</option>
      </select>
      {#if !$systemSettings?.workerConfigured}<small>Saved locally; route generation needs the World Worker.</small>{/if}
    </label>
  {/if}'''

new_route = '''  {#if includeRoute}
    <label class="field">
      <span>Start point</span>
      <select
        value={$sessionConfig.route.startSpawnIndex ?? ''}
        disabled={!spawnCatalogReady}
        onchange={(event) => patchSessionSection('route', {
          startSpawnIndex: spawnValue(event.currentTarget.value)
        })}
      >
        <option value="">Automatic from seed</option>
        {#each $workspaceOptions.spawnPoints as point}
          <option value={point.index}>#{point.index} · {point.label}</option>
        {/each}
      </select>
      <small>{spawnCatalogReady ? 'Exact CARLA map spawn point; no fallback if occupied.' : 'Load the selected map in Garage to enumerate its CARLA spawn points.'}</small>
    </label>

    <label class="field">
      <span>Route</span>
      <select
        value={$sessionConfig.route.mode}
        disabled={!$systemSettings?.workerConfigured}
        onchange={(event) => {
          const mode = fieldValue(event) as RouteMode;
          patchSessionSection('route', {
            mode,
            destinationSpawnIndex: mode === 'selected_destination'
              ? $sessionConfig.route.destinationSpawnIndex
              : null
          });
        }}
      >
        <option value="free">Free drive</option>
        <option value="random_destination" disabled={!$systemSettings?.capabilities.random_route}>Random destination</option>
        <option value="selected_destination" disabled={!selectedRouteAvailable || !spawnCatalogReady}>Selected destination</option>
      </select>
      <small>Planned routes use CARLA GlobalRoutePlanner; the active control owner consumes the selected route.</small>
    </label>

    {#if $sessionConfig.route.mode === 'selected_destination'}
      <label class="field">
        <span>Destination</span>
        <select
          value={$sessionConfig.route.destinationSpawnIndex ?? ''}
          disabled={!spawnCatalogReady}
          onchange={(event) => patchSessionSection('route', {
            destinationSpawnIndex: spawnValue(event.currentTarget.value)
          })}
        >
          <option value="">Choose destination…</option>
          {#each $workspaceOptions.spawnPoints as point}
            <option value={point.index} disabled={point.index === $sessionConfig.route.startSpawnIndex}>
              #{point.index} · {point.label}
            </option>
          {/each}
        </select>
        <small>The exact CARLA spawn-point destination is retained in session evidence.</small>
      </label>
    {/if}
  {/if}'''

replace_once("web/src/lib/components/SceneWorldFields.svelte", old_route, new_route)

replace_once(
    "tests/test_operator_configuration.py",
    '    assert defaults["route"] == {"mode": "free"}',
    '    assert defaults["route"] == {\n        "mode": "free",\n        "startSpawnIndex": None,\n        "destinationSpawnIndex": None,\n    }',
)
replace_once(
    "tests/test_operator_configuration.py",
    '        "route_mode": "random_destination",\n        "pedestrian_crossing_factor": 0.85,',
    '        "route_mode": "random_destination",\n        "start_spawn_index": None,\n        "destination_spawn_index": None,\n        "pedestrian_crossing_factor": 0.85,',
)

print("current-shape fixups applied")
