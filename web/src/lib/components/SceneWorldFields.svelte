<script lang="ts">
  import {
    patchSessionSection,
    sessionConfig,
    systemSettings,
    workspaceOptions
  } from '$lib/stores/configuration';
  import type { RouteMode } from '$lib/domain/config';
  import { fieldNumber, fieldValue } from '$lib/ui/events';

  export let includeRoute = false;

  $: selectedMap = $sessionConfig.scene.mapName === 'current'
    ? $systemSettings?.currentMap ?? ''
    : $sessionConfig.scene.mapName;
  $: spawnCatalogReady = Boolean(
    $systemSettings?.workerConfigured
      && $systemSettings?.capabilities.spawn_point_selection
      && $workspaceOptions.spawnPointMap
      && selectedMap === $workspaceOptions.spawnPointMap
  );
  $: selectedRouteAvailable = Boolean($systemSettings?.capabilities.selected_route);

  function spawnValue(value: string): number | null {
    return value === '' ? null : Number(value);
  }
</script>

<div class="field-grid two-columns">
  <label class="field">
    <span>Map</span>
    <select
      value={$sessionConfig.scene.mapName}
      onchange={(event) => patchSessionSection('scene', { mapName: fieldValue(event) })}
    >
      <option value="current">Keep current map</option>
      {#each $workspaceOptions.maps as map}
        <option value={map.id}>{map.label}</option>
      {/each}
    </select>
    {#if !$systemSettings?.workerConfigured}<small>Saved locally; live map reload needs the World Worker.</small>{/if}
  </label>

  <label class="field">
    <span>Weather</span>
    <select
      value={$sessionConfig.scene.weatherPreset}
      onchange={(event) => patchSessionSection('scene', { weatherPreset: fieldValue(event) })}
    >
      {#each $workspaceOptions.weatherPresets as preset}
        <option value={preset.id}>{preset.label}</option>
      {/each}
    </select>
  </label>

  <label class="field">
    <span>Traffic vehicles</span>
    <input
      type="number"
      min="0"
      max="250"
      value={$sessionConfig.scene.trafficCount}
      oninput={(event) => patchSessionSection('scene', { trafficCount: fieldNumber(event) })}
    />
  </label>

  <label class="field">
    <span>Pedestrians</span>
    <input
      type="number"
      min="0"
      max="250"
      value={$sessionConfig.scene.walkerCount}
      oninput={(event) => patchSessionSection('scene', { walkerCount: fieldNumber(event) })}
    />
  </label>

  <label class="field">
    <span>Road props</span>
    <select
      value={$sessionConfig.scene.propPreset}
      onchange={(event) => patchSessionSection('scene', { propPreset: fieldValue(event) })}
    >
      {#each $workspaceOptions.propPresets as preset}
        <option value={preset.id}>{preset.label}</option>
      {/each}
    </select>
  </label>

  {#if includeRoute}
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
  {/if}
</div>

<details class="advanced-block">
  <summary>Traffic behaviour</summary>
  <div class="field-grid three-columns">
    <label class="field">
      <span>Pedestrian crossing chance</span>
      <input
        type="number"
        min="0"
        max="1"
        step="0.05"
        value={$sessionConfig.scene.pedestrianCrossingFactor}
        oninput={(event) =>
          patchSessionSection('scene', { pedestrianCrossingFactor: fieldNumber(event) })}
      />
      <small>0 never crosses · 1 all walkers may cross</small>
    </label>
    <label class="field">
      <span>Traffic speed difference %</span>
      <input
        type="number"
        min="-100"
        max="100"
        value={$sessionConfig.scene.speedDifferencePercent}
        oninput={(event) =>
          patchSessionSection('scene', { speedDifferencePercent: fieldNumber(event) })}
      />
    </label>
    <label class="field">
      <span>Following distance m</span>
      <input
        type="number"
        min="0.1"
        max="20"
        step="0.1"
        value={$sessionConfig.scene.followingDistanceMetres}
        oninput={(event) =>
          patchSessionSection('scene', { followingDistanceMetres: fieldNumber(event) })}
      />
    </label>
  </div>
  {#if !$systemSettings?.workerConfigured}
    <p class="inline-note">These values remain in the shared session and sync when the World Worker connects.</p>
  {/if}
</details>
