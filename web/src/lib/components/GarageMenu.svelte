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
  const items: Array<{ id: Panel; label: string; detail: string }> = [
    { id: 'experiment', label: 'Experiment', detail: 'Purpose' },
    { id: 'scene', label: 'Scene', detail: 'World & traffic' },
    { id: 'drive', label: 'Vehicle', detail: 'Car & control' },
    { id: 'vision', label: 'Vision', detail: 'Camera & models' },
    { id: 'research', label: 'Research', detail: 'Recipe & plan' },
    { id: 'system', label: 'Settings', detail: 'Connection & runtime' }
  ];
  let active: Panel | null = null;

  function close(): void {
    active = null;
  }

  function useInGarage(): void {
    close();
    requestAnimationFrame(() => document.querySelector('#garage')?.scrollIntoView({ block: 'start' }));
  }
</script>

<nav class="garage-command-menu" aria-label="Garage tools">
  {#each items as item}
    <button
      type="button"
      disabled={locked && item.id !== 'system'}
      class:active={active === item.id}
      onclick={() => (active = item.id)}
    >
      <strong>{item.label}</strong>
      <span>{item.detail}</span>
    </button>
  {/each}
</nav>

<Modal
  open={active === 'experiment'}
  title="Choose an experiment"
  description="Presets update the same visible session. You can refine every value afterward."
  close={close}
>
  <PresetSelector />
  <ResolvedSession />
</Modal>

<Modal
  open={active === 'scene'}
  title="Scene"
  description="Map, weather, traffic, pedestrians and fixed road items."
  close={close}
>
  <SceneSettings />
</Modal>

<Modal
  open={active === 'drive'}
  title="Vehicle and drive"
  description="Choose the ego vehicle and the single owner of its controls."
  close={close}
>
  <DriveSettings />
  <PolicySettings />
  <ModelLibrary />
</Modal>

<Modal
  open={active === 'vision'}
  title="Vision and recording"
  description="Live camera, detection overlay and retained video evidence."
  close={close}
>
  <VisionSettings />
</Modal>

<Modal
  open={active === 'research'}
  title="Research scene builder"
  description="Turn the shared Garage scene into a reproducible situation recipe and scenario plan."
  close={close}
>
  <SceneBuilder useInGarage={useInGarage} />
</Modal>

<Modal
  open={active === 'system'}
  title="Settings"
  description="Machine connection, runtime availability and workspace facts."
  close={close}
>
  <SystemSettings />
</Modal>
