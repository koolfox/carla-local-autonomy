<script lang="ts">
  import { driveStatusLabel, isDriveActive } from '$lib/domain/runtime';
  import { systemSettings } from '$lib/stores/configuration';
  import { garageRuntime } from '$lib/stores/runtime';

  export let compact = false;

  $: active = isDriveActive($garageRuntime.drive);
</script>

<header class="topbar" class:compact>
  <a class="brand-lockup" href="#garage" aria-label="CARLA Garage home">
    <div class="brand-mark" aria-hidden="true">CV</div>
    <div>
      <strong>CARLA Garage</strong>
      <span>Vision Research Console</span>
    </div>
  </a>

  {#if $systemSettings}
    <div class="topbar-status" aria-label="runtime status">
      <span class:ok={$systemSettings.connected} class:bad={!$systemSettings.connected} class="status-pill">
        <i></i>{$systemSettings.connected ? 'CARLA online' : 'CARLA offline'}
      </span>
      <span class:ok={$systemSettings.workerConnected} class="status-pill">
        <i></i>{$systemSettings.workerConnected
          ? 'Bridge ready'
          : $systemSettings.workerConfigured
            ? 'Bridge busy · retrying'
            : 'Bridge not configured'}
      </span>
      <span class:running={active} class:ok={$garageRuntime.drive.status === 'success'} class:bad={$garageRuntime.drive.status === 'failed'} class="status-pill">
        <i></i>{driveStatusLabel($garageRuntime.drive.status)}
      </span>
    </div>
  {/if}
</header>
