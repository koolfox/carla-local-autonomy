<script lang="ts">
  import { onMount } from 'svelte';

  import { loadWorkspaceSnapshot } from '$lib/api/operator';
  import ContextRail from '$lib/components/ContextRail.svelte';
  import DriveSettings from '$lib/components/DriveSettings.svelte';
  import ModelLibrary from '$lib/components/ModelLibrary.svelte';
  import PolicySettings from '$lib/components/PolicySettings.svelte';
  import PresetSelector from '$lib/components/PresetSelector.svelte';
  import ResolvedSession from '$lib/components/ResolvedSession.svelte';
  import SceneSettings from '$lib/components/SceneSettings.svelte';
  import StatusHeader from '$lib/components/StatusHeader.svelte';
  import VisionSettings from '$lib/components/VisionSettings.svelte';
  import { hydrateWorkspace } from '$lib/stores/configuration';
  import '$lib/styles/app.css';

  let loading = true;
  let error = '';

  onMount(async () => {
    try {
      hydrateWorkspace(await loadWorkspaceSnapshot());
    } catch (caught) {
      error = caught instanceof Error ? caught.message : String(caught);
    } finally {
      loading = false;
    }
  });
</script>

<svelte:head>
  <title>CARLA Vision Console</title>
  <meta
    name="description"
    content="Unified configuration surface for the CARLA Vision Research Console"
  />
</svelte:head>

<div class="app-shell">
  <StatusHeader />

  <main class="workspace">
    <ContextRail />

    <section class="configuration-column" aria-busy={loading}>
      {#if loading}
        <div class="notice">Loading the current Operator state…</div>
      {:else if error}
        <div class="notice error-notice">
          <strong>Could not load Operator API</strong>
          <span>{error}</span>
          <small>Start <code>uv run carla-operator-ui --open-browser</code> on port 8765.</small>
        </div>
      {:else}
        <PresetSelector />
        <SceneSettings />
        <DriveSettings />
        <VisionSettings />
        <ModelLibrary />
        <PolicySettings />
      {/if}
    </section>

    <ResolvedSession />
  </main>
</div>
