<script lang="ts">
  import SceneWorldFields from '$lib/components/SceneWorldFields.svelte';
  import { systemSettings } from '$lib/stores/configuration';
</script>

<section id="scene" class="config-card scroll-section">
  <div class="section-heading">
    <div>
      <span class="eyebrow">02 · Scene</span>
      <h2>World & population</h2>
    </div>
    {#if $systemSettings?.workerConfigured}
      <span class="capability ok-text">Live sync</span>
    {:else if $systemSettings?.capabilities.garage_traffic_population}
      <span class="capability">Traffic sync available · map needs Worker</span>
    {:else}
      <span class="capability">Edits stay saved until Worker connects</span>
    {/if}
  </div>

  <SceneWorldFields includeRoute={true} />

  <p class="section-footnote">Changes save immediately and sync to the live Garage automatically after a short pause.</p>
</section>
