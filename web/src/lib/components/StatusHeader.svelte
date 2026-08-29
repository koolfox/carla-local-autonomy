<script lang="ts">
  import { systemSettings } from '$lib/stores/configuration';
</script>

<header class="topbar">
  <div class="brand-lockup">
    <div class="brand-mark">CV</div>
    <div>
      <strong>CARLA Vision Console</strong>
      <span>Unified research workspace</span>
    </div>
  </div>

  {#if $systemSettings}
    <div class="topbar-status" aria-label="runtime status">
      <span class:ok={$systemSettings.connected} class="status-pill">
        <i></i>{$systemSettings.connected ? 'CARLA connected' : 'CARLA offline'}
      </span>
      <span class:ok={$systemSettings.workerConnected} class="status-pill">
        <i></i>{$systemSettings.workerConnected ? 'Worker connected' : 'Worker unavailable'}
      </span>
    </div>
  {/if}
</header>
