<script lang="ts">
  import type { DetectorKind } from '$lib/domain/config';
  import { patchSessionSection, sessionConfig, systemSettings, workspaceOptions } from '$lib/stores/configuration';
  import { fieldChecked, fieldNumber, fieldValue } from '$lib/ui/events';

  function setDetector(event: Event): void {
    patchSessionSection('perception', { detector: fieldValue(event) as DetectorKind });
  }
</script>

<section class="config-card">
  <div class="section-heading">
    <div>
      <span class="eyebrow">03 · Vision</span>
      <h2>Camera, perception & evidence</h2>
    </div>
    <span class:ok-text={$systemSettings?.visionRuntimeAvailable} class="capability">
      {$systemSettings?.visionRuntimeAvailable ? 'Vision runtime ready' : 'Vision runtime unavailable'}
    </span>
  </div>

  <div class="toggle-row">
    <label class="switch-field">
      <input
        type="checkbox"
        checked={$sessionConfig.perception.enabled}
        disabled={!$systemSettings?.visionRuntimeAvailable}
        onchange={(event) => patchSessionSection('perception', { enabled: fieldChecked(event) })}
      />
      <span><strong>Detection overlay</strong><small>RT-DETR / YOLO advisory output</small></span>
    </label>
    <label class="switch-field">
      <input
        type="checkbox"
        checked={$sessionConfig.recording.video}
        onchange={(event) => patchSessionSection('recording', { video: fieldChecked(event) })}
      />
      <span><strong>Record video</strong><small>Retain review media with the run</small></span>
    </label>
    <label class="switch-field">
      <input
        type="checkbox"
        checked={$sessionConfig.camera.spectatorFollow}
        onchange={(event) => patchSessionSection('camera', { spectatorFollow: fieldChecked(event) })}
      />
      <span><strong>Spectator follow</strong><small>Server-side observer camera</small></span>
    </label>
  </div>

  <div class="field-grid three-columns">
    <label class="field">
      <span>Resolution</span>
      <select
        value={$sessionConfig.camera.resolution}
        onchange={(event) => patchSessionSection('camera', { resolution: fieldValue(event) })}
      >
        <option value="640x384">640 × 384</option>
        <option value="1280x720">1280 × 720</option>
        <option value="1920x1080">1920 × 1080</option>
      </select>
    </label>
    <label class="field">
      <span>Camera FPS</span>
      <select
        value={$sessionConfig.camera.fps}
        onchange={(event) => patchSessionSection('camera', { fps: fieldNumber(event) })}
      >
        <option value="10">10 FPS</option>
        <option value="30">30 FPS</option>
        <option value="60">60 FPS</option>
      </select>
    </label>
    <label class="field">
      <span>FOV</span>
      <input
        type="number"
        min="30"
        max="150"
        value={$sessionConfig.camera.fov}
        oninput={(event) => patchSessionSection('camera', { fov: fieldNumber(event) })}
      />
    </label>
  </div>

  {#if $sessionConfig.perception.enabled}
    <div class="field-grid three-columns nested-fields">
      <label class="field">
        <span>Detector</span>
        <select value={$sessionConfig.perception.detector} onchange={setDetector}>
          <option value="rtdetr">RT-DETR</option>
          <option value="yolo">YOLO</option>
        </select>
      </label>
      <label class="field">
        <span>Weights</span>
        <select
          value={$sessionConfig.perception.weights}
          onchange={(event) => patchSessionSection('perception', { weights: fieldValue(event) })}
        >
          <option value="">Select a workspace .pt file</option>
          {#each $workspaceOptions.detectorWeights as weights}
            <option value={weights}>{weights}</option>
          {/each}
        </select>
      </label>
      <label class="field">
        <span>Device</span>
        <select
          value={$sessionConfig.perception.device}
          onchange={(event) => patchSessionSection('perception', { device: fieldValue(event) })}
        >
          <option value="cpu">CPU</option>
          <option value="mps">MPS</option>
          <option value="cuda">CUDA</option>
        </select>
      </label>
      <label class="field">
        <span>Image size</span>
        <input
          type="number"
          min="64"
          max="4096"
          step="32"
          value={$sessionConfig.perception.imageSize}
          oninput={(event) => patchSessionSection('perception', { imageSize: fieldNumber(event) })}
        />
      </label>
      <label class="field">
        <span>Confidence</span>
        <input
          type="number"
          min="0"
          max="1"
          step="0.05"
          value={$sessionConfig.perception.confidence}
          oninput={(event) => patchSessionSection('perception', { confidence: fieldNumber(event) })}
        />
      </label>
    </div>
  {/if}
</section>
