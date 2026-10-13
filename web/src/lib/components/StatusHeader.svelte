<script lang="ts">
  import { driveStatusLabel, isDriveActive } from '$lib/domain/runtime';
  import { sessionConfig, systemSettings } from '$lib/stores/configuration';
  import { inspectorRequest } from '$lib/stores/garageSync';
  import { garageRuntime } from '$lib/stores/runtime';

  $: active = isDriveActive($garageRuntime.drive);
  $: mapName = $sessionConfig.scene.mapName === 'current'
    ? ($systemSettings?.currentMap?.split('/').at(-1) ?? 'Current map')
    : $sessionConfig.scene.mapName.split('/').at(-1);
  $: runId = active ? ($garageRuntime.drive.run_id ?? $sessionConfig.identity.runId) : $sessionConfig.identity.runId;
  $: bridgeLabel = $systemSettings?.workerConnected
    ? 'Bridge ready'
    : $systemSettings?.workerConfigured
      ? 'Bridge busy · retrying'
      : 'Bridge not configured';
</script>

<header class="topbar">
  <a class="brand-lockup" href="#garage" aria-label="CARLA Garage home">
    <span class="brand-mark" aria-hidden="true">CV</span>
    <strong>CARLA Garage</strong>
  </a>

  <nav class="breadcrumb" aria-label="Current session">
    <span>{mapName}</span>
    <span class="run-id" title={runId}>{runId}</span>
  </nav>

  {#if $systemSettings}
    <button
      type="button"
      class="topbar-status"
      aria-label="Runtime status · open System details"
      title="Open connection and runtime details"
      onclick={() => inspectorRequest.set('system')}
    >
      <span
        class="status-pill"
        class:ok={$systemSettings.connected}
        class:bad={!$systemSettings.connected}
        title={$systemSettings.connected ? 'CARLA online' : 'CARLA offline'}
      ><i></i><span>{$systemSettings.connected ? 'CARLA online' : 'CARLA offline'}</span></span>
      <span
        class="status-pill"
        class:ok={$systemSettings.workerConnected}
        class:pending={!$systemSettings.workerConnected && $systemSettings.workerConfigured}
        class:off={!$systemSettings.workerConfigured}
        title={bridgeLabel}
      ><i></i><span>{bridgeLabel}</span></span>
      <span
        class="status-pill"
        class:running={active}
        class:ok={$garageRuntime.drive.status === 'success'}
        class:bad={$garageRuntime.drive.status === 'failed'}
        class:idle={!active && !['success', 'failed'].includes($garageRuntime.drive.status)}
        title={driveStatusLabel($garageRuntime.drive.status)}
      ><i></i><span>{driveStatusLabel($garageRuntime.drive.status)}</span></span>
    </button>
  {/if}

</header>
