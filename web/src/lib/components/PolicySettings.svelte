<script lang="ts">
  import type { BehaviorStyle } from '$lib/domain/config';
  import {
    patchSessionSection,
    sessionConfig,
    workspaceOptions
  } from '$lib/stores/configuration';
  import { fieldChecked, fieldNumber, fieldValue } from '$lib/ui/events';

  $: autonomous = ['behavior', 'imitation', 'voxel', 'model'].includes($sessionConfig.control.mode);
  $: checkpointRequired = ['imitation', 'voxel'].includes($sessionConfig.control.mode);
  $: drivingModels = $workspaceOptions.models.filter((model) => model.role === 'driving_policy');
  $: selectedModel = drivingModels.find((model) => model.id === $sessionConfig.policy.modelId);

  function setBehavior(event: Event): void {
    patchSessionSection('policy', { behavior: fieldValue(event) as BehaviorStyle });
  }

  function setModel(event: Event): void {
    const modelId = fieldValue(event);
    const model = drivingModels.find((candidate) => candidate.id === modelId);
    patchSessionSection('policy', {
      modelId,
      device: model?.devices[0] ?? 'cpu',
      acknowledgeTrustedCode: false
    });
  }
</script>

<section id="policy" class="config-card scroll-section">
  <div class="section-heading">
    <div>
      <span class="eyebrow">05 · Policy</span>
      <h2>Autonomy runtime</h2>
    </div>
    {#if autonomous}
      <span class="capability">Explicit opt-in · fail-closed runtime</span>
    {:else}
      <span class="capability">No research policy owns control</span>
    {/if}
  </div>

  {#if autonomous}
    <div class="field-grid three-columns">
      {#if $sessionConfig.control.mode === 'behavior' || $sessionConfig.control.mode === 'voxel'}
        <label class="field">
          <span>Behavior style</span>
          <select value={$sessionConfig.policy.behavior} onchange={setBehavior}>
            <option value="cautious">Cautious</option>
            <option value="normal">Normal</option>
            <option value="aggressive">Aggressive</option>
          </select>
        </label>
      {/if}

      {#if checkpointRequired}
        <label class="field">
          <span>Policy checkpoint</span>
          <select
            value={$sessionConfig.policy.checkpoint}
            onchange={(event) => patchSessionSection('policy', { checkpoint: fieldValue(event) })}
          >
            <option value="">Select checkpoint</option>
            {#each $workspaceOptions.checkpoints as checkpoint}<option value={checkpoint}>{checkpoint}</option>{/each}
          </select>
        </label>
      {/if}

      {#if $sessionConfig.control.mode === 'model'}
        <label class="field">
          <span>Registered model</span>
          <select value={$sessionConfig.policy.modelId} onchange={setModel}>
            <option value="">Select driving model</option>
            {#each drivingModels as model}
              <option value={model.id}>{model.name} · {model.version} · {model.runtime}</option>
            {/each}
          </select>
        </label>
      {/if}

      {#if checkpointRequired || $sessionConfig.control.mode === 'model'}
        <label class="field">
          <span>Policy device</span>
          <select
            value={$sessionConfig.policy.device}
            onchange={(event) => patchSessionSection('policy', { device: fieldValue(event) })}
          >
            {#if selectedModel && $sessionConfig.control.mode === 'model'}
              {#each selectedModel.devices as device}
                <option value={device}>{device.toUpperCase()}</option>
              {/each}
            {:else}
              <option value="cpu">CPU</option>
              <option value="mps">MPS</option>
              <option value="cuda">CUDA</option>
            {/if}
          </select>
        </label>
      {/if}

      <label class="field">
        <span>Target speed km/h</span>
        <input
          type="number"
          min="5"
          max="120"
          value={$sessionConfig.policy.targetSpeedKmh}
          oninput={(event) => patchSessionSection('policy', { targetSpeedKmh: fieldNumber(event) })}
        />
      </label>
    </div>

    {#if $sessionConfig.control.mode === 'model' && selectedModel}
      <div class="notice">
        <strong>{selectedModel.name}</strong>
        <span>
          {selectedModel.role} · {selectedModel.runtime} · artifact {selectedModel.artifact}
        </span>
        <small>
          This package is selected by manifest identity. The backend verifies its artifact hash before actuation.
        </small>
      </div>
      {#if selectedModel.requiresTrustedCode}
        <label class="switch-field autonomy-acknowledgement">
          <input
            type="checkbox"
            checked={$sessionConfig.policy.acknowledgeTrustedCode}
            onchange={(event) =>
              patchSessionSection('policy', { acknowledgeTrustedCode: fieldChecked(event) })}
          />
          <span>
            <strong>I trust this executable model package and its adapter</strong>
            <small>PyTorch checkpoints, TorchScript files, and Python adapters may execute code while loading or running.</small>
          </span>
        </label>
      {/if}
    {/if}

    {#if $sessionConfig.control.mode === 'voxel'}
      <details class="advanced-block">
        <summary>Voxel supervisor inputs</summary>
        <div class="field-grid three-columns">
          <label class="field">
            <span>Readiness report</span>
            <input
              value={$sessionConfig.policy.voxelReadinessReport}
              placeholder="Optional workspace JSON"
              oninput={(event) =>
                patchSessionSection('policy', { voxelReadinessReport: fieldValue(event) })}
            />
          </label>
          <label class="field">
            <span>Max model speed km/h</span>
            <input
              type="number"
              min="5"
              max="160"
              value={$sessionConfig.policy.maxModelSpeedKmh}
              oninput={(event) =>
                patchSessionSection('policy', { maxModelSpeedKmh: fieldNumber(event) })}
            />
          </label>
          <label class="field">
            <span>Max steer rate</span>
            <input
              type="number"
              min="0.1"
              max="10"
              step="0.1"
              value={$sessionConfig.policy.maxSteerRate}
              oninput={(event) => patchSessionSection('policy', { maxSteerRate: fieldNumber(event) })}
            />
          </label>
          <label class="field">
            <span>Max policy errors</span>
            <input
              type="number"
              min="1"
              max="20"
              value={$sessionConfig.policy.maxPolicyErrors}
              oninput={(event) =>
                patchSessionSection('policy', { maxPolicyErrors: fieldNumber(event) })}
            />
          </label>
        </div>
      </details>
    {/if}

    <label class="switch-field autonomy-acknowledgement">
      <input
        type="checkbox"
        checked={$sessionConfig.policy.acknowledgeAutonomy}
        onchange={(event) =>
          patchSessionSection('policy', { acknowledgeAutonomy: fieldChecked(event) })}
      />
      <span>
        <strong>I acknowledge that this mode actuates the CARLA ego vehicle</strong>
        <small>Required by the Garage safety contract before an autonomous session can start.</small>
      </span>
    </label>
  {:else}
    <div class="policy-empty">
      <span class="policy-empty-icon">✓</span>
      <div>
        <strong>No experimental driving policy is selected.</strong>
        <p>Manual Browser control and Traffic Manager use their own runtime paths. Choose an experimental control owner above to expose policy-specific settings here.</p>
      </div>
    </div>
  {/if}
</section>
