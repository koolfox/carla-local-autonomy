<script lang="ts">
  import { newRunId } from '$lib/domain/config';
  import { isDriveActive } from '$lib/domain/runtime';
  import { blockingIssues, validateSession } from '$lib/domain/sessionValidation';
  import {
    patchSessionSection,
    sessionConfig,
    systemSettings,
    workspaceOptions
  } from '$lib/stores/configuration';
  import {
    emergencyStopDrive,
    garageRuntime,
    startDrive,
    stopDrive
  } from '$lib/stores/runtime';

  export let compact = false;

  $: issues = validateSession($sessionConfig, $systemSettings, $workspaceOptions);
  $: blockers = blockingIssues(issues);
  $: active = isDriveActive($garageRuntime.drive);
  $: running = $garageRuntime.drive.status === 'running';
  $: terminal = ['success', 'failed'].includes($garageRuntime.drive.status);
  $: freshRunId = !terminal || $sessionConfig.identity.runId !== $garageRuntime.drive.run_id;
  $: disabled = blockers.length > 0 || active || $garageRuntime.action !== null;
  $: hint = $garageRuntime.error
    ?? blockers[0]?.message
    ?? (running
      ? 'Emergency braking and Stop & Save remain available while driving.'
      : terminal && !freshRunId
        ? 'Start another session with a fresh run ID.'
        : `Start ${$sessionConfig.identity.runId}.`);

  async function launch(): Promise<void> {
    if (disabled) return;
    let config = $sessionConfig;
    if (terminal && !freshRunId) {
      const runId = newRunId();
      patchSessionSection('identity', { runId });
      config = {
        ...config,
        identity: { ...config.identity, runId }
      };
    }
    try {
      await startDrive(config);
    } catch {
      // The runtime store exposes the backend error in the action bar.
    }
  }

  async function stop(): Promise<void> {
    try {
      await stopDrive();
    } catch {
      // The runtime store exposes the backend error in the action bar.
    }
  }

  async function emergency(): Promise<void> {
    try {
      await emergencyStopDrive();
    } catch {
      // The runtime store exposes the backend error in the action bar.
    }
  }

</script>

<section class:compact class="launch-bar" aria-label={`Session actions. ${hint}`} title={hint}>
  {#if !compact}<div class="launch-copy">
    <span class="launch-kicker">Session control</span>
    {#if $garageRuntime.error}
      <strong class="launch-error" role="alert">{$garageRuntime.error}</strong>
    {:else if blockers.length}
      <strong>{blockers[0].message}</strong>
      {#if blockers.length > 1}<small>{blockers.length - 1} more readiness checks need attention.</small>{/if}
    {:else if active}
      <strong>{$garageRuntime.drive.status === 'stopping' ? 'Saving and releasing the session…' : 'Garage session is active.'}</strong>
      <small>{$garageRuntime.drive.run_id ?? $sessionConfig.identity.runId}</small>
    {:else if terminal && !freshRunId}
      <strong>{$garageRuntime.drive.status === 'success' ? 'Run saved.' : 'The previous run failed.'}</strong>
      <small>The next start creates a fresh run ID automatically.</small>
    {:else if terminal}
      <strong>Next run is ready to start.</strong>
      <small>{$sessionConfig.identity.runId}</small>
    {:else}
      <strong>Configuration is ready to start.</strong>
      <small>{$sessionConfig.identity.runId}</small>
    {/if}
  </div>{/if}

  <div class="launch-actions">
    {#if running}
      <button
        type="button"
        class="button danger-button"
        title={hint}
        disabled={$garageRuntime.action !== null}
        onclick={emergency}
      >
        {$garageRuntime.action === 'emergency' ? 'Applying brake…' : 'Emergency Brake'}
      </button>
      <button
        type="button"
        class="button secondary-button"
        title={hint}
        disabled={$garageRuntime.action !== null}
        onclick={stop}
      >
        {$garageRuntime.action === 'stop' ? 'Saving…' : 'Stop & Save'}
      </button>
    {:else}
      <button type="button" class="button primary-button" title={hint} disabled={disabled} onclick={launch}>
        {$garageRuntime.action === 'start' ? 'Starting Garage…' : terminal ? 'Start next run' : 'Start session'}
      </button>
    {/if}
  </div>
</section>
