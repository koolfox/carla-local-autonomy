<script lang="ts">
  import { validateSession } from '$lib/domain/sessionValidation';
  import {
    resetSession,
    sessionConfig,
    systemSettings,
    workspaceOptions
  } from '$lib/stores/configuration';
  import { garageRuntime } from '$lib/stores/runtime';

  $: populationOwner = $systemSettings?.workerConnected
    ? 'World Worker'
    : $systemSettings?.capabilities.garage_traffic_population
      ? 'Garage PythonAPI fallback'
      : 'Unavailable';
  $: issues = validateSession($sessionConfig, $systemSettings, $workspaceOptions);
  $: blockers = issues.filter((issue) => issue.severity === 'error');
  $: runtimeActive = ['starting', 'running', 'stopping'].includes($garageRuntime.drive.status);
</script>

<aside class="resolved-panel">
  <div class="resolved-header">
    <div>
      <span class="eyebrow">Session brief</span>
      <h2>What will run</h2>
    </div>
    <button type="button" class="ghost-button" disabled={runtimeActive} onclick={resetSession}>Reset</button>
  </div>

  <div class:ready={!blockers.length} class:blocked={blockers.length > 0} class="readiness-banner">
    <span class="readiness-dot"></span>
    <div>
      <strong>{blockers.length ? `${blockers.length} readiness ${blockers.length === 1 ? 'issue' : 'issues'}` : 'Ready to start'}</strong>
      <small>{blockers[0]?.message ?? 'The frontend preflight is clear; the backend remains authoritative at Start.'}</small>
    </div>
  </div>

  <div class="resolved-summary">
    <div><span>Preset</span><strong>{$sessionConfig.experiment.preset.replaceAll('_', ' ')}</strong></div>
    <div><span>Control</span><strong>{$sessionConfig.control.mode}</strong></div>
    <div><span>Vehicle</span><strong>{$sessionConfig.vehicle.blueprint.split('.').at(-1) || 'Not selected'}</strong></div>
    <div><span>Map</span><strong>{$sessionConfig.scene.mapName.split('/').at(-1)}</strong></div>
    <div>
      <span>Population</span>
      <strong>{$sessionConfig.scene.trafficCount} cars · {$sessionConfig.scene.walkerCount} walkers</strong>
    </div>
    <div><span>Population owner</span><strong>{populationOwner}</strong></div>
    <div>
      <span>Camera</span>
      <strong>{$sessionConfig.camera.resolution} · {$sessionConfig.camera.fps} FPS</strong>
    </div>
    <div>
      <span>Perception</span>
      <strong>{$sessionConfig.perception.enabled ? $sessionConfig.perception.detector.toUpperCase() : 'Off'}</strong>
    </div>
    <div><span>Recording</span><strong>{$sessionConfig.recording.video ? 'Video on' : 'Video off'}</strong></div>
    <div><span>Run ID</span><strong title={$sessionConfig.identity.runId}>{$sessionConfig.identity.runId}</strong></div>
  </div>

  {#if blockers.length > 1}
    <details class="readiness-details">
      <summary>All readiness issues</summary>
      <ul>
        {#each blockers as issue}<li>{issue.message}</li>{/each}
      </ul>
    </details>
  {/if}

  <div class="architecture-note">
    <strong>One SessionConfig.</strong>
    <p>
      The backend adapter resolves this exact object onto the World Worker or Garage fallback. The preview is explicit and does not create a second configuration source.
    </p>
  </div>
</aside>
