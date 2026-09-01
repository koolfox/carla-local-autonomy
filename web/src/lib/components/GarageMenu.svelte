<script lang="ts">
  import DriveSettings from '$lib/components/DriveSettings.svelte';
  import Modal from '$lib/components/Modal.svelte';
  import ModelLibrary from '$lib/components/ModelLibrary.svelte';
  import PolicySettings from '$lib/components/PolicySettings.svelte';
  import PresetSelector from '$lib/components/PresetSelector.svelte';
  import ResolvedSession from '$lib/components/ResolvedSession.svelte';
  import SceneBuilder from '$lib/components/SceneBuilder.svelte';
  import SceneSettings from '$lib/components/SceneSettings.svelte';
  import SystemSettings from '$lib/components/SystemSettings.svelte';
  import VisionSettings from '$lib/components/VisionSettings.svelte';

  export let locked = false;

  type Panel = 'experiment' | 'scene' | 'drive' | 'vision' | 'research' | 'system';
  const items: Array<{ id: Panel; label: string; title: string }> = [
    { id: 'experiment', label: 'Experiment', title: 'Experiment purpose and preset' },
    { id: 'scene', label: 'Scene', title: 'Map, weather, traffic and pedestrians' },
    { id: 'drive', label: 'Vehicle', title: 'Vehicle and control owner' },
    { id: 'vision', label: 'Vision', title: 'Camera, perception and recording' },
    { id: 'research', label: 'Research', title: 'Situation recipe and scenario plan' },
    { id: 'system', label: 'System', title: 'Connection and runtime' }
  ];
  let open = false;
  let active: Panel = 'scene';

  $: if (locked && active !== 'system') active = 'system';

  function close(): void {
    open = false;
  }

  function show(panel?: Panel): void {
    active = locked ? 'system' : panel ?? active;
    open = true;
  }

  function useInGarage(): void {
    close();
    requestAnimationFrame(() => document.querySelector('#garage')?.scrollIntoView({ block: 'start' }));
  }
</script>

<nav class="garage-command-menu" aria-label="Garage tools">
  <button
    type="button"
    class="garage-menu-trigger"
    aria-expanded={open}
    aria-controls="garage-control-tabs"
    title="Open Garage controls"
    onclick={() => show()}
  >
    <span aria-hidden="true">☰</span>
    <strong>Menu</strong>
  </button>
</nav>

<Modal
  {open}
  title="Garage controls"
  close={close}
>
  <div id="garage-control-tabs" class="garage-modal-tabs" role="tablist" aria-label="Garage control sections">
    {#each items as item}
      <button
        type="button"
        role="tab"
        aria-selected={active === item.id}
        class:active={active === item.id}
        disabled={locked && item.id !== 'system'}
        title={item.title}
        onclick={() => (active = item.id)}
      >{item.label}</button>
    {/each}
  </div>

  <div class="garage-modal-panel" role="tabpanel" aria-label={items.find((item) => item.id === active)?.title}>
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
    {:else}
      <SystemSettings />
    {/if}
  </div>
</Modal>
