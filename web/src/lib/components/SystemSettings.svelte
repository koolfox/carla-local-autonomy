<script lang="ts">
  import { systemSettings } from '$lib/stores/configuration';
</script>

<div class="system-settings">
  <section class="builder-section">
    <div class="section-heading">
      <div>
        <span class="eyebrow">System</span>
        <h3>Connection and runtime</h3>
      </div>
      <span class="section-note">Machine-level facts only. Scene and traffic live in Scene.</span>
    </div>
    {#if $systemSettings}
      <dl class="system-facts">
        <div>
          <dt>CARLA endpoint</dt>
          <dd>{$systemSettings.carlaHost}:{$systemSettings.carlaPort}</dd>
        </div>
        <div>
          <dt>CARLA server</dt>
          <dd>{$systemSettings.connected ? ($systemSettings.serverVersion ?? 'Connected') : 'Offline'}</dd>
        </div>
        <div>
          <dt>Current map</dt>
          <dd>{$systemSettings.currentMap ?? 'Unknown'}</dd>
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
        <div>
          <dt>Vision runtime</dt>
          <dd>{$systemSettings.visionRuntimeAvailable ? 'Available' : 'Unavailable'}</dd>
        </div>
      </dl>
    {/if}
  </section>
</div>
