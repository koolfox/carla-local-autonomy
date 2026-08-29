<script lang="ts">
  import { onMount } from 'svelte';

  import { loadWorkspaceSnapshot } from '$lib/api/operator';
  import ContextRail from '$lib/components/ContextRail.svelte';
  import DriveCockpit from '$lib/components/DriveCockpit.svelte';
  import DriveSettings from '$lib/components/DriveSettings.svelte';
  import GaragePreview from '$lib/components/GaragePreview.svelte';
  import PolicySettings from '$lib/components/PolicySettings.svelte';
  import PresetSelector from '$lib/components/PresetSelector.svelte';
  import ResolvedSession from '$lib/components/ResolvedSession.svelte';
  import SceneSettings from '$lib/components/SceneSettings.svelte';
  import SessionLaunchBar from '$lib/components/SessionLaunchBar.svelte';
  import StatusHeader from '$lib/components/StatusHeader.svelte';
  import VisionSettings from '$lib/components/VisionSettings.svelte';
  import { isDriveActive } from '$lib/domain/runtime';
  import { hydrateWorkspace } from '$lib/stores/configuration';
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
        initializeRuntime(snapshot.token, snapshot.driveState);
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
  $: showCockpit = $garageRuntime.drive.status !== 'idle';
</script>

<svelte:head>
  <title>CARLA Garage · Vision Console</title>
  <meta
    name="description"
    content="Garage, cockpit and session configuration for the CARLA Vision Research Console"
  />
</svelte:head>

<div class="app-shell">
  <StatusHeader />

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
    <main class="workspace">
      <ContextRail />

      <section class="configuration-column">
        <GaragePreview />

        {#if showCockpit}
          <DriveCockpit />
        {/if}

        <div class="setup-heading">
          <div>
            <span class="eyebrow">Session setup</span>
            <h2>Build the next drive</h2>
          </div>
          {#if sessionLocked}<span class="status-pill pending"><i></i>Locked while session is active</span>{/if}
        </div>

        <div class:configuration-locked={sessionLocked} aria-disabled={sessionLocked}>
          <PresetSelector />
          <SceneSettings />
          <DriveSettings />
          <VisionSettings />
          <PolicySettings />
        </div>
      </section>

      <ResolvedSession />
    </main>

    <SessionLaunchBar />
  {/if}
</div>
