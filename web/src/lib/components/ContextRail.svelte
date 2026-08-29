<script lang="ts">
  import { systemSettings } from '$lib/stores/configuration';
</script>

<aside class="context-rail">
  <div class="eyebrow">Garage workspace</div>
  <h1>Build it.<br />Drive it.<br />Keep the evidence.</h1>
  <p>
    One session configuration owns scene, vehicle, control, vision and recording. The live Garage is applied explicitly, so editing the form never creates hidden CARLA mutations.
  </p>

  <nav class="garage-nav" aria-label="Garage sections">
    <a href="#garage"><span>00</span><strong>Garage</strong><small>Live vehicle bay</small></a>
    <a href="#intent"><span>01</span><strong>Intent</strong><small>Experiment preset</small></a>
    <a href="#scene"><span>02</span><strong>Scene</strong><small>World & population</small></a>
    <a href="#drive-config"><span>03</span><strong>Drive</strong><small>Vehicle & control</small></a>
    <a href="#vision"><span>04</span><strong>Vision</strong><small>Camera & evidence</small></a>
    <a href="#policy"><span>05</span><strong>Policy</strong><small>Autonomy settings</small></a>
  </nav>

  {#if $systemSettings}
    <dl class="system-facts">
      <div>
        <dt>CARLA endpoint</dt>
        <dd>{$systemSettings.carlaHost}:{$systemSettings.carlaPort}</dd>
      </div>
      <div>
        <dt>Current map</dt>
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
            ? 'Connected'
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
