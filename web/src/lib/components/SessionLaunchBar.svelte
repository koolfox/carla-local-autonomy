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

  $: issues = validateSession($sessionConfig, $systemSettings, $workspaceOptions);
  $: blockers = blockingIssues(issues);
  $: active = isDriveActive($garageRuntime.drive);
  $: running = $garageRuntime.drive.status === 'running';
  $: terminal = ['success', 'failed'].includes($garageRuntime.drive.status);
  $: disabled = blockers.length > 0 || active || $garageRuntime.action !== null;

  async function launch(): Promise<void> {
    if (disabled) return;
    try {
      await startDrive($sessionConfig);
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

  function prepareAnotherRun(): void {
    patchSessionSection('identity', { runId: newRunId() });
  }
</script>

<section class="launch-bar" aria-label="session actions">
  <div class="launch-copy">
    <span class="launch-kicker">Session control</span>
    {#if $garageRuntime.error}
      <strong class="launch-error" role="alert">{$garageRuntime.error}</strong>
    {:else if blockers.length}
      <strong>{blockers[0].message}</strong>
      {#if blockers.length > 1}<small>{blockers.length - 1} more readiness checks need attention.</small>{/if}
    {:else if active}
      <strong>{$garageRuntime.drive.status === 'stopping' ? 'Saving and releasing the session…' : 'Garage session is active.'}</strong>
      <small>{$garageRuntime.drive.run_id ?? $sessionConfig.identity.runId}</small>
    {:else if terminal}
      <strong>{$garageRuntime.drive.status === 'success' ? 'Run saved.' : 'The previous run failed.'}</strong>
      <small>{$garageRuntime.drive.output_path ?? $garageRuntime.drive.error ?? 'Review the runtime panel for details.'}</small>
    {:else}
      <strong>Configuration is ready to start.</strong>
      <small>{$sessionConfig.identity.runId}</small>
    {/if}
  </div>

  <div class="launch-actions">
    {#if running}
      <button
        type="button"
        class="button danger-button"
        disabled={$garageRuntime.action !== null}
        onclick={emergency}
      >
        {$garageRuntime.action === 'emergency' ? 'Applying brake…' : 'Emergency Brake'}
      </button>
      <button
        type="button"
        class="button secondary-button"
        disabled={$garageRuntime.action !== null}
        onclick={stop}
      >
        {$garageRuntime.action === 'stop' ? 'Saving…' : 'Stop & Save'}
      </button>
    {:else if terminal}
      <button type="button" class="button secondary-button" onclick={prepareAnotherRun}>New run ID</button>
      <button type="button" class="button primary-button" disabled={disabled} onclick={launch}>Start again</button>
    {:else}
      <button type="button" class="button primary-button" disabled={disabled} onclick={launch}>
        {$garageRuntime.action === 'start' ? 'Starting Garage…' : 'Start session'}
      </button>
    {/if}
  </div>
</section>
