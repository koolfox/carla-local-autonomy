<script lang="ts">
  import {
    patchSessionSection,
    sessionConfig,
    systemSettings,
    workspaceOptions
  } from '$lib/stores/configuration';
  import { fieldNumber, fieldValue } from '$lib/ui/events';
</script>

<section id="scene" class="config-card scroll-section">
  <div class="section-heading">
    <div>
      <span class="eyebrow">02 · Scene</span>
      <h2>World & population</h2>
    </div>
    {#if $systemSettings?.workerConnected}
      <span class="capability ok-text">World Worker owns world mutation</span>
    {:else if $systemSettings?.capabilities.garage_traffic_population}
      <span class="capability">PythonAPI population fallback available</span>
    {:else}
      <span class="capability">World controls are capability-gated</span>
    {/if}
  </div>

  <div class="field-grid two-columns">
    <label class="field">
      <span>Map</span>
      <select
        value={$sessionConfig.scene.mapName}
        disabled={!$systemSettings?.workerConnected}
        onchange={(event) => patchSessionSection('scene', { mapName: fieldValue(event) })}
      >
        <option value="current">Keep current map</option>
        {#each $workspaceOptions.maps as map}
          <option value={map}>{map.split('/').at(-1) ?? map}</option>
        {/each}
      </select>
      {#if !$systemSettings?.workerConnected}<small>Map reload requires the World Worker.</small>{/if}
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

    <label class="field">
      <span>Route</span>
      <select
        value={$sessionConfig.route.mode}
        disabled={!$systemSettings?.workerConnected}
        onchange={(event) =>
          patchSessionSection('route', {
            mode: fieldValue(event) as 'free' | 'random_destination'
          })}
      >
        <option value="free">Free drive</option>
        <option value="random_destination">Random destination</option>
      </select>
    </label>
  </div>

  <details class="advanced-block">
    <summary>Traffic dynamics</summary>
    <div class="field-grid three-columns">
      <label class="field">
        <span>Pedestrian crossing factor</span>
        <input
          type="number"
          min="0"
          max="1"
          step="0.05"
          disabled={!$systemSettings?.workerConnected}
          value={$sessionConfig.scene.pedestrianCrossingFactor}
          oninput={(event) =>
            patchSessionSection('scene', { pedestrianCrossingFactor: fieldNumber(event) })}
        />
      </label>
      <label class="field">
        <span>TM speed difference %</span>
        <input
          type="number"
          min="-100"
          max="100"
          disabled={!$systemSettings?.workerConnected}
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
          disabled={!$systemSettings?.workerConnected}
          value={$sessionConfig.scene.followingDistanceMetres}
          oninput={(event) =>
            patchSessionSection('scene', { followingDistanceMetres: fieldNumber(event) })}
        />
      </label>
    </div>
  </details>

  <p class="section-footnote">Editing this section does not mutate CARLA. Use the explicit Garage preview action or Start session to apply it.</p>
</section>
