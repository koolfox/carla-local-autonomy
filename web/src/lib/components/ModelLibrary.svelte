<script lang="ts">
  import { workspaceOptions } from '$lib/stores/configuration';
</script>

<section class="config-card">
  <div class="section-heading">
    <div>
      <span class="eyebrow">Models</span>
      <h2>Registered model library</h2>
    </div>
    <span class="section-note">Only manifest-backed packages are runnable</span>
  </div>

  {#if !$workspaceOptions.models.length}
    <div class="notice">
      <strong>No registered model packages</strong>
      <span>Add <code>models/&lt;model-id&gt;/model.json</code> beside the model artifact.</span>
    </div>
  {:else}
    <div class="preset-grid">
      {#each $workspaceOptions.models as model}
        <article class="preset-card">
          <strong>{model.name}</strong>
          <span>{model.id} · {model.version}</span>
          <small>{model.role} · {model.runtime} · {model.devices.join(', ')}</small>
          {#if model.requiresTrustedCode}
            <small>Executable package · explicit trust required</small>
          {/if}
        </article>
      {/each}
    </div>
  {/if}

  {#if $workspaceOptions.invalidModels.length}
    <details class="advanced-block">
      <summary>{$workspaceOptions.invalidModels.length} invalid model package(s)</summary>
      {#each $workspaceOptions.invalidModels as model}
        <div class="notice error-notice">
          <strong>{model.path}</strong>
          <span>{model.errorType}: {model.message}</span>
        </div>
      {/each}
    </details>
  {/if}
</section>
