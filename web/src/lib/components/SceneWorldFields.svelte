<script lang="ts">
  import {
    patchSessionSection,
    sessionConfig,
    systemSettings,
    workspaceOptions
  } from '$lib/stores/configuration';
  import { fieldNumber, fieldValue } from '$lib/ui/events';

  export let includeRoute = false;
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
    {#if !$systemSettings?.workerConnected}<small>Saved locally; live map reload needs the World Worker.</small>{/if}
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
      {#if !$systemSettings?.workerConnected}<small>Saved locally; route generation needs the World Worker.</small>{/if}
    </label>
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
  {#if !$systemSettings?.workerConnected}
    <p class="inline-note">These values remain in the shared session. A connected World Worker is required to apply them to live CARLA.</p>
  {/if}
</details>
