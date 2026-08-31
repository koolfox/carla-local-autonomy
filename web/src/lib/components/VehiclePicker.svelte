<script lang="ts">
  import { patchSessionSection, sessionConfig, workspaceOptions } from '$lib/stores/configuration';
  import { fieldValue } from '$lib/ui/events';

  export let compact = false;

  $: vehicles = $workspaceOptions.vehicles;
  $: selectedIndex = vehicles.findIndex(
    (vehicle) => vehicle.id === $sessionConfig.vehicle.blueprint
  );
  $: selectedVehicle = vehicles[selectedIndex];

  function chooseVehicle(blueprint: string): void {
    patchSessionSection('vehicle', { blueprint, color: '' });
  }

  function stepVehicle(direction: number): void {
    if (!vehicles.length) return;
    const base = selectedIndex < 0 ? 0 : selectedIndex;
    const next = (base + direction + vehicles.length) % vehicles.length;
    chooseVehicle(vehicles[next].id);
  }
</script>

{#if compact}
  <div class="vehicle-carousel" aria-label="Garage vehicle selector">
    <button type="button" aria-label="Previous vehicle" disabled={!vehicles.length} onclick={() => stepVehicle(-1)}>‹</button>
    <label>
      <span>Vehicle</span>
      <select
        value={$sessionConfig.vehicle.blueprint}
        onchange={(event) => chooseVehicle(fieldValue(event))}
      >
        {#if !vehicles.length}<option value="">No vehicle catalog</option>{/if}
        {#each vehicles as vehicle}
          <option value={vehicle.id}>{vehicle.label ?? vehicle.id}</option>
        {/each}
      </select>
    </label>
    <button type="button" aria-label="Next vehicle" disabled={!vehicles.length} onclick={() => stepVehicle(1)}>›</button>
    <label class="paint-picker">
      <span>Paint</span>
      <select
        aria-label="Vehicle paint"
        value={$sessionConfig.vehicle.color}
        disabled={!selectedVehicle?.colors?.length}
        onchange={(event) => patchSessionSection('vehicle', { color: fieldValue(event) })}
      >
        <option value="">Default paint</option>
        {#each selectedVehicle?.colors ?? [] as color}<option value={color}>{color}</option>{/each}
      </select>
    </label>
  </div>
{:else}
  <div class="field-grid two-columns">
    <label class="field">
      <span>Vehicle</span>
      <select
        value={$sessionConfig.vehicle.blueprint}
        onchange={(event) => chooseVehicle(fieldValue(event))}
      >
        {#if !vehicles.length}<option value="">No vehicle catalog</option>{/if}
        {#each vehicles as vehicle}
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
        <option value="">Blueprint default</option>
        {#each selectedVehicle?.colors ?? [] as color}<option value={color}>{color}</option>{/each}
      </select>
    </label>
  </div>
{/if}
