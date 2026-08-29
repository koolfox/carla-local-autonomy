<script lang="ts">
  import {
    applyExperimentPreset,
    experimentPresets,
    sessionConfig
  } from '$lib/stores/configuration';
</script>

<section id="intent" class="config-card preset-card scroll-section">
  <div class="section-heading">
    <div>
      <span class="eyebrow">01 · Intent</span>
      <h2>Experiment preset</h2>
    </div>
    <span class="section-note">A preset patches the visible SessionConfig; it never creates parallel state.</span>
  </div>

  <div class="preset-grid">
    {#each $experimentPresets as preset}
      <button
        type="button"
        class:active={$sessionConfig.experiment.preset === preset.id}
        class="preset-button"
        onclick={() => applyExperimentPreset(preset.id)}
      >
        <strong>{preset.label}</strong>
        <span>{preset.description}</span>
      </button>
    {/each}
  </div>
</section>
