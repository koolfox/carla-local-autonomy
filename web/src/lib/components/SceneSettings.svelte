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
    {#if $systemSettings?.workerConnected}
      <span class="capability ok-text">Ready to apply in live CARLA</span>
    {:else if $systemSettings?.capabilities.garage_traffic_population}
      <span class="capability">Traffic can apply · map reload needs Worker</span>
    {:else}
      <span class="capability">Edits stay saved until Worker connects</span>
    {/if}
  </div>

  <SceneWorldFields includeRoute={true} />

  <p class="section-footnote">Changes update the shared session immediately. Open or refresh the Garage preview, or start a session, to apply them to CARLA.</p>
</section>
