<script lang="ts">
  import VehiclePicker from '$lib/components/VehiclePicker.svelte';
  import type { ControlMode } from '$lib/domain/config';
  import {
    patchSessionSection,
    sessionConfig,
    systemSettings
  } from '$lib/stores/configuration';
  import { fieldNumber, fieldValue } from '$lib/ui/events';

  function setControlMode(mode: ControlMode): void {
    patchSessionSection('control', { mode });
  }
</script>

<section id="drive-config" class="config-card scroll-section">
  <div class="section-heading">
    <div>
      <span class="eyebrow">03 · Drive</span>
      <h2>Vehicle & control owner</h2>
    </div>
    <span class="section-note">Exactly one owner reaches the ego control path for a session.</span>
  </div>

  <VehiclePicker />

  <div class="subsection-heading">
    <span>Control owner</span>
    <small>Manual and Traffic Manager are production paths; research policies remain explicitly gated.</small>
  </div>

  <div class="control-mode-grid">
    <button
      type="button"
      class:active={$sessionConfig.control.mode === 'manual'}
      onclick={() => setControlMode('manual')}
    >
      <span class="mode-icon">M</span><strong>Manual</strong><small>Browser keyboard / touch</small>
    </button>
    <button
      type="button"
      class:active={$sessionConfig.control.mode === 'autopilot'}
      disabled={!$systemSettings?.capabilities.autopilot}
      onclick={() => setControlMode('autopilot')}
    >
      <span class="mode-icon">TM</span><strong>Traffic Manager</strong><small>CARLA autopilot baseline</small>
    </button>
    {#if $systemSettings?.experimentalEnabled}
      <button
        type="button"
        class:active={$sessionConfig.control.mode === 'behavior'}
        disabled={!$systemSettings.capabilities.garage_behavior_drive}
        onclick={() => setControlMode('behavior')}
      >
        <span class="mode-icon">B</span><strong>BehaviorAgent</strong><small>Experimental actuation</small>
      </button>
      <button
        type="button"
        class:active={$sessionConfig.control.mode === 'imitation'}
        disabled={!$systemSettings.capabilities.garage_imitation_drive}
        onclick={() => setControlMode('imitation')}
      >
        <span class="mode-icon">I</span><strong>Imitation</strong><small>RGB + speed policy</small>
      </button>
      <button
        type="button"
        class:active={$sessionConfig.control.mode === 'voxel'}
        disabled={!$systemSettings.capabilities.garage_voxel_drive}
        onclick={() => setControlMode('voxel')}
      >
        <span class="mode-icon">V</span><strong>Voxel Planner</strong><small>Supervised voxel steering</small>
      </button>
      <button
        type="button"
        class:active={$sessionConfig.control.mode === 'model'}
        disabled={!$systemSettings.capabilities.garage_model_drive}
        onclick={() => setControlMode('model')}
      >
        <span class="mode-icon">P</span><strong>PyTorch Model</strong><small>Registered driving policy</small>
      </button>
    {/if}
  </div>

  <details class="advanced-block">
    <summary>Run identity</summary>
    <div class="field-grid two-columns">
      <label class="field">
        <span>Run ID</span>
        <input
          value={$sessionConfig.identity.runId}
          oninput={(event) => patchSessionSection('identity', { runId: fieldValue(event) })}
        />
      </label>
      <label class="field">
        <span>Seed</span>
        <input
          type="number"
          min="0"
          value={$sessionConfig.identity.seed}
          oninput={(event) => patchSessionSection('identity', { seed: fieldNumber(event) })}
        />
      </label>
    </div>
  </details>
</section>
