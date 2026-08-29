<script lang="ts">
  import type { ControlMode } from '$lib/domain/config';
  import { patchSessionSection, sessionConfig, systemSettings, workspaceOptions } from '$lib/stores/configuration';
  import { fieldNumber, fieldValue } from '$lib/ui/events';

  $: selectedVehicle = $workspaceOptions.vehicles.find(
    (vehicle) => vehicle.id === $sessionConfig.vehicle.blueprint
  );

  function setControlMode(event: Event): void {
    patchSessionSection('control', { mode: fieldValue(event) as ControlMode });
  }
</script>

<section class="config-card">
  <div class="section-heading">
    <div>
      <span class="eyebrow">02 · Drive</span>
      <h2>Vehicle & control owner</h2>
    </div>
    <span class="section-note">One visible mode maps to the correct backend owner</span>
  </div>

  <div class="field-grid two-columns">
    <label class="field">
      <span>Vehicle</span>
      <select
        value={$sessionConfig.vehicle.blueprint}
        onchange={(event) => patchSessionSection('vehicle', { blueprint: fieldValue(event), color: '' })}
      >
        {#if !$workspaceOptions.vehicles.length}
          <option value="">No vehicle catalog</option>
        {/if}
        {#each $workspaceOptions.vehicles as vehicle}
          <option value={vehicle.id}>{vehicle.label ?? vehicle.id}</option>
        {/each}
      </select>
    </label>

    <label class="field">
      <span>Paint</span>
      <select
        value={$sessionConfig.vehicle.color}
        disabled={!selectedVehicle?.colors?.length}
        onchange={(event) => patchSessionSection('vehicle', { color: fieldValue(event) })}
      >
        <option value="">Default</option>
        {#each selectedVehicle?.colors ?? [] as color}
          <option value={color}>{color}</option>
        {/each}
      </select>
    </label>

    <label class="field">
      <span>Control owner</span>
      <select value={$sessionConfig.control.mode} onchange={setControlMode}>
        <option value="manual">Manual · Browser</option>
        <option value="autopilot" disabled={!$systemSettings?.capabilities.autopilot}>
          Traffic Manager Autopilot
        </option>
        {#if $systemSettings?.experimentalEnabled}
          <option value="behavior" disabled={!$systemSettings.capabilities.garage_behavior_drive}>
            BehaviorAgent · Experimental
          </option>
          <option value="imitation" disabled={!$systemSettings.capabilities.garage_imitation_drive}>
            Imitation · Experimental
          </option>
          <option value="voxel" disabled={!$systemSettings.capabilities.garage_voxel_drive}>
            Voxel Planner · Experimental
          </option>
        {/if}
      </select>
    </label>

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
</section>
