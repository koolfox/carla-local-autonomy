<script lang="ts">
  import { systemSettings } from '$lib/stores/configuration';
</script>

<aside class="context-rail">
  <div class="eyebrow">Configuration</div>
  <h1>One session.<br />One source of truth.</h1>
  <p>
    Scene, vehicle, control, perception and recording belong to one session configuration.
    Presets modify this object instead of maintaining hidden parallel state.
  </p>

  {#if $systemSettings}
    <dl class="system-facts">
      <div>
        <dt>CARLA</dt>
        <dd>{$systemSettings.carlaHost}:{$systemSettings.carlaPort}</dd>
      </div>
      <div>
        <dt>Map</dt>
        <dd>{$systemSettings.currentMap ?? 'Unknown'}</dd>
      </div>
      <div>
        <dt>Server</dt>
        <dd>{$systemSettings.serverVersion ?? 'Unknown'}</dd>
      </div>
      <div>
        <dt>World Worker</dt>
        <dd title={$systemSettings.workerUrl ?? ''}>
          {$systemSettings.workerConnected
            ? $systemSettings.workerUrl ?? 'Connected'
            : $systemSettings.workerConfigured
              ? 'Configured · offline'
              : 'Not configured'}
        </dd>
      </div>
      <div>
        <dt>Workspace</dt>
        <dd title={$systemSettings.workspace}>{$systemSettings.workspace}</dd>
      </div>
    </dl>
  {/if}
</aside>
