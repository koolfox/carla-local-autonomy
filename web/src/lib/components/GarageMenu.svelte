<script lang="ts">
  import { onMount } from 'svelte';

  import DriveSettings from '$lib/components/DriveSettings.svelte';
  import Icon, { type IconName } from '$lib/components/Icon.svelte';
  import ModelLibrary from '$lib/components/ModelLibrary.svelte';
  import PolicySettings from '$lib/components/PolicySettings.svelte';
  import PresetSelector from '$lib/components/PresetSelector.svelte';
  import ResolvedSession from '$lib/components/ResolvedSession.svelte';
  import SceneBuilder from '$lib/components/SceneBuilder.svelte';
  import SceneSettings from '$lib/components/SceneSettings.svelte';
  import SystemSettings from '$lib/components/SystemSettings.svelte';
  import VisionSettings from '$lib/components/VisionSettings.svelte';
  import SavedRuns from '$lib/components/SavedRuns.svelte';
  import { garageSyncPending, inspectorRequest } from '$lib/stores/garageSync';

  export let locked = false;

  type Panel = 'experiment' | 'scene' | 'drive' | 'vision' | 'research' | 'recordings' | 'system';
  const items: Array<{ id: Panel; label: string; title: string; icon: IconName }> = [
    { id: 'experiment', label: 'Experiment', title: 'Experiment purpose and preset', icon: 'experiment' },
    { id: 'scene', label: 'Scene', title: 'Map, weather, traffic and pedestrians', icon: 'scene' },
    { id: 'drive', label: 'Vehicle', title: 'Ego vehicle, control and models', icon: 'vehicle' },
    { id: 'vision', label: 'Vision', title: 'Camera, perception and recording', icon: 'vision' },
    { id: 'research', label: 'Research', title: 'Situation recipe and scenario plan', icon: 'research' },
    { id: 'recordings', label: 'Recordings', title: 'Saved runs and recordings', icon: 'recordings' },
    { id: 'system', label: 'System', title: 'Connection and runtime', icon: 'system' }
  ];
  const widePanels = new Set<Panel>(['research', 'recordings']);
  const narrowQuery = '(max-width: 860px)';

  let open = true;
  let active: Panel = 'scene';
  let wasLocked = false;
  let reopenAfterDrive = false;

  $: if (locked && active !== 'system') active = 'system';
  $: current = items.find((item) => item.id === active) ?? items[1];
  // The cockpit needs the whole canvas: fold the inspector away while driving, restore it after.
  $: if (locked !== wasLocked) {
    if (locked) {
      reopenAfterDrive = open;
      open = false;
    } else {
      open = reopenAfterDrive && !isNarrow();
    }
    wasLocked = locked;
  }

  $: if ($inspectorRequest) {
    const requested = items.find((item) => item.id === $inspectorRequest);
    if (requested) choose(requested.id);
    inspectorRequest.set(null);
  }

  function isNarrow(): boolean {
    return typeof window !== 'undefined' && window.matchMedia(narrowQuery).matches;
  }

  onMount(() => {
    if (isNarrow()) open = false;
  });

  function close(): void {
    open = false;
  }

  function choose(panel: Panel): void {
    active = locked ? 'system' : panel;
    open = true;
  }

  function useInGarage(): void {
    if (isNarrow()) close();
    requestAnimationFrame(() => document.querySelector('#garage')?.scrollIntoView({ block: 'start' }));
  }

  function keydown(event: KeyboardEvent): void {
    if (event.key === 'Escape' && open && isNarrow()) close();
  }
</script>

<svelte:window onkeydown={keydown} />

<button
  type="button"
  class="panel-toggle"
  aria-expanded={open}
  aria-controls="garage-inspector"
  title={open ? 'Hide settings panel' : 'Show settings panel'}
  onclick={() => (open = !open)}
>
  <Icon name="sliders" />
  <span>Settings</span>
</button>

<div class="panels" class:open class:wide={widePanels.has(active)}>
  <button type="button" class="sheet-backdrop" tabindex="-1" aria-label="Close settings" onclick={close}></button>

  <nav class="section-rail" aria-label="Garage tools">
    <div id="garage-control-tabs" role="tablist" aria-label="Garage control sections" aria-orientation="vertical">
      {#each items as item}
        <button
          type="button"
          role="tab"
          aria-selected={active === item.id}
          aria-controls="garage-inspector"
          class:active={active === item.id && open}
          disabled={locked && item.id !== 'system'}
          title={item.title}
          onclick={() => (active === item.id && open && !isNarrow() ? close() : choose(item.id))}
        >
          <Icon name={item.icon} size={18} />
          <span>{item.label}</span>
        </button>
      {/each}
    </div>
  </nav>

  <div id="garage-inspector" class="inspector" role="tabpanel" aria-label={current.title} hidden={!open}>
    <header class="inspector-header">
      <div>
        <h2>{current.label}</h2>
        <p>{current.title}</p>
        {#if $garageSyncPending && ['scene', 'drive', 'vision'].includes(active)}
          <span class="sync-pending" role="status">Changes pending · syncing to Garage</span>
        {/if}
      </div>
      <button type="button" class="icon-button" aria-label="Close settings" title="Close settings" onclick={close}>
        <Icon name="close" />
      </button>
    </header>

    <div class="inspector-body">
      {#key active}
        <div class="inspector-panel">
          {#if active === 'experiment'}
            <PresetSelector />
            <ResolvedSession />
          {:else if active === 'scene'}
            <SceneSettings />
          {:else if active === 'drive'}
            <DriveSettings />
            <PolicySettings />
            <ModelLibrary />
          {:else if active === 'vision'}
            <VisionSettings />
          {:else if active === 'research'}
            <SceneBuilder useInGarage={useInGarage} />
          {:else if active === 'recordings'}
            {#if open}<SavedRuns />{/if}
          {:else}
            <SystemSettings />
          {/if}
        </div>
      {/key}
    </div>
  </div>
</div>
