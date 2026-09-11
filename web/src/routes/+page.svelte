<script lang="ts">
  import { onMount } from 'svelte';

  import { loadWorkspaceSnapshot } from '$lib/api/operator';
  import DriveCockpit from '$lib/components/DriveCockpit.svelte';
  import GarageMenu from '$lib/components/GarageMenu.svelte';
  import GaragePreview from '$lib/components/GaragePreview.svelte';
  import SessionLaunchBar from '$lib/components/SessionLaunchBar.svelte';
  import StatusHeader from '$lib/components/StatusHeader.svelte';
  import { isDriveActive } from '$lib/domain/runtime';
  import { hydrateWorkspace } from '$lib/stores/configuration';
  import { hydrateCaptureSettings } from '$lib/stores/capture';
  import {
    beginRuntimePolling,
    endRuntimePolling,
    garageRuntime,
    initializeRuntime
  } from '$lib/stores/runtime';
  import '$lib/styles/app.css';

  let loading = true;
  let error = '';

  onMount(() => {
    let cancelled = false;
    void (async () => {
      try {
        const snapshot = await loadWorkspaceSnapshot();
        if (cancelled) return;
        hydrateWorkspace(snapshot);
        hydrateCaptureSettings();
        initializeRuntime(snapshot.token, snapshot.driveState, snapshot.captureState);
        beginRuntimePolling();
      } catch (caught) {
        if (!cancelled) error = caught instanceof Error ? caught.message : String(caught);
      } finally {
        if (!cancelled) loading = false;
      }
    })();

    return () => {
      cancelled = true;
      endRuntimePolling();
    };
  });

  $: sessionLocked = isDriveActive($garageRuntime.drive);
  $: showCockpit = sessionLocked;
</script>

<svelte:head>
  <title>CARLA Garage · Vision Console</title>
  <meta
    name="description"
    content="Garage, cockpit and session configuration for the CARLA Vision Research Console"
  />
</svelte:head>

<div class="app-shell">
  <StatusHeader compact={showCockpit} />

  {#if loading}
    <main class="loading-shell" aria-busy="true">
      <div class="loading-ring"></div>
      <strong>Opening CARLA Garage…</strong>
      <span>Loading runtime capabilities and the canonical session contract.</span>
    </main>
  {:else if error}
    <main class="loading-shell error-shell">
      <strong>Could not open the Garage</strong>
      <span>{error}</span>
      <small>Start <code>uv run carla-operator-ui --open-browser</code> and reload this page.</small>
    </main>
  {:else}
    <main class="garage-workspace" class:driving={showCockpit}>
      <GarageMenu locked={sessionLocked} />
      {#if showCockpit}
        <DriveCockpit />
        <SessionLaunchBar compact={true} />
      {:else}
        <GaragePreview />
      {/if}
    </main>
  {/if}
</div>
