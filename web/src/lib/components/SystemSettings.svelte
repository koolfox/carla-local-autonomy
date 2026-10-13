<script lang="ts">
  import { isDriveActive } from '$lib/domain/runtime';
  import { systemSettings } from '$lib/stores/configuration';
  import { captureRuntime, garageRuntime, liveCheckRuntime, startLiveCheck } from '$lib/stores/runtime';

  let acknowledge = false;
  let starting = false;
  let startError = '';

  $: check = $liveCheckRuntime;
  $: blocked = !check.available
    ? 'Connect the World Worker to run the live check.'
    : isDriveActive($garageRuntime.drive)
      ? 'Stop & Save the active drive first.'
      : $captureRuntime.holds_world
        ? 'Wait for teacher capture to finish.'
        : '';

  async function run(): Promise<void> {
    starting = true;
    startError = '';
    try {
      await startLiveCheck();
      acknowledge = false;
    } catch (error) {
      startError = error instanceof Error ? error.message : String(error);
    } finally {
      starting = false;
    }
  }
</script>

<div class="system-settings">
  <section class="builder-section">
    <div class="section-heading">
      <div>
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
                ? 'Configured · busy or reconnecting'
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

  <section class="builder-section" aria-label="Live CARLA check">
    <div class="section-heading">
      <h3>Live check</h3>
      <span class="section-note">Runs the Garage acceptance against the connected CARLA: loads Town10HD, spawns a test car with 5 cars and 5 walkers, drives it, then stops and cleans up. Takes about a minute; the Garage preview pauses meanwhile.</span>
    </div>

    {#if check.active}
      <div class="notice" role="status">
        <strong><span class="loading-ring" aria-hidden="true"></span> Live check · {check.phase}</strong>
        <span>Driving is blocked until the check releases CARLA.</span>
      </div>
    {:else}
      <label class="field">
        <span><input type="checkbox" bind:checked={acknowledge} disabled={Boolean(blocked)} />
          Reload the map and let the check drive a test vehicle.</span>
      </label>
      <div class="builder-actions">
        <button
          type="button"
          class="button primary-button"
          disabled={!acknowledge || starting || Boolean(blocked)}
          onclick={run}
        >{starting ? 'Starting…' : 'Run live check'}</button>
      </div>
      {#if blocked}<p class="section-note">{blocked}</p>{/if}
    {/if}

    {#if startError}<p class="inline-error" role="alert">{startError}</p>{/if}

    {#if !check.active && check.summary}
      <div class:ok={check.phase === 'passed'} class:bad={check.phase !== 'passed'} class="job-result">
        <strong>{check.phase === 'passed' ? 'Live check passed' : 'Live check failed'}</strong>
        <span>{check.summary.passed_checks} of {check.summary.total_checks} checks passed · {check.summary.modes.join(' + ')} driving{check.summary.duration_seconds ? ` · ${Math.round(check.summary.duration_seconds)} s` : ''}</span>
        {#if check.summary.failed_checks.length}<small>Failed: {check.summary.failed_checks.join(', ')}</small>{/if}
        {#if check.error}<small>{check.error}</small>{/if}
      </div>
      {#if check.report}<p class="saved-path">{check.report}</p>{/if}
    {:else if !check.active && check.phase === 'failed' && check.error}
      <p class="inline-error" role="alert">{check.error}</p>
    {/if}
  </section>
</div>
