<script lang="ts">
  import type { DetectorKind } from '$lib/domain/config';
  import {
    patchSessionSection,
    sessionConfig,
    systemSettings,
    workspaceOptions
  } from '$lib/stores/configuration';
  import { fieldChecked, fieldNumber, fieldValue } from '$lib/ui/events';

  function setDetector(event: Event): void {
    const detector = fieldValue(event) as DetectorKind;
    if (detector === 'm9-hierarchical') {
      patchSessionSection('perception', {
        detector,
        imageSize: 800,
        confidence: 0
      });
      return;
    }
    patchSessionSection('perception', { detector });
  }
</script>

<section id="vision" class="config-card scroll-section">
  <div class="section-heading">
    <div>
      <span class="eyebrow">04 · Vision</span>
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
      <span><strong>Detection overlay</strong><small>Advisory detector output · no vehicle control</small></span>
    </label>
    <label class="switch-field">
      <input
        type="checkbox"
        checked={$sessionConfig.perception.voxelEnabled}
        onchange={(event) => patchSessionSection('perception', { voxelEnabled: fieldChecked(event) })}
      />
      <span><strong>Voxel (RGB depth)</strong><small>Estimated spatial view · no vehicle control</small></span>
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
      <span><strong>Spectator follow</strong><small>Mirror the ego from CARLA</small></span>
    </label>
  </div>

  <label class="switch-field">
    <input type="checkbox" checked={$sessionConfig.perception.roadEnabled}
      onchange={(event) => patchSessionSection('perception', { roadEnabled: fieldChecked(event) })} />
    <span><strong>Road & lane overlay</strong><small>RGB model estimates · first use downloads weights</small></span>
  </label>
  {#if $sessionConfig.perception.roadEnabled}
    <label class="field">
      <span>Road model</span>
      <select value={$sessionConfig.perception.roadBackend}
        onchange={(event) => patchSessionSection('perception', { roadBackend: fieldValue(event), roadCheckpoint: '', roadDevice: 'cpu' })}>
        <option value="segformer">SegFormer · road & sidewalk</option>
        <option value="yolop">YOLOP · road & lane markings</option>
        <option value="yolopv2">YOLOPv2 · road & lane markings</option>
      </select>
    </label>
    <label class="field">
      <span>Checkpoint (optional)</span>
      <input value={$sessionConfig.perception.roadCheckpoint} placeholder="Blank uses the official model"
        onchange={(event) => patchSessionSection('perception', { roadCheckpoint: fieldValue(event) })} />
      <small>{$sessionConfig.perception.roadBackend === 'yolopv2' ? 'Official YOLOPv2 .pt only; blank downloads and verifies it on first use. CPU or CUDA.' : $sessionConfig.perception.roadBackend === 'yolop' ? 'Workspace .onnx file, or blank for official YOLOP 640.' : 'Workspace SegFormer directory, or blank for Cityscapes (no lane markings).'}</small>
    </label>
    <label class="field">
      <span>Road model device</span>
      <select value={$sessionConfig.perception.roadDevice}
        onchange={(event) => patchSessionSection('perception', { roadDevice: fieldValue(event) })}>
        <option value="cpu">CPU</option>
        {#if $sessionConfig.perception.roadBackend === 'segformer'}<option value="mps">MPS</option>{/if}
        <option value="cuda">CUDA</option>
      </select>
    </label>
  {/if}

  {#if $sessionConfig.perception.voxelEnabled}
    <p class="section-note">
      Metric Outdoor Small uses RGB only. First use downloads approximately 100 MB.
      Distances are estimates; this view does not infer lanes or steer the vehicle.
      Manual and Traffic Manager control remain unchanged.
    </p>
    {#if !$sessionConfig.perception.enabled}
      <div class="field-grid three-columns">
        <label class="field">
          <span>Perception device</span>
          <select
            value={$sessionConfig.perception.device}
            onchange={(event) => patchSessionSection('perception', { device: fieldValue(event) })}
          >
            <option value="cpu">CPU</option>
            <option value="mps">MPS</option>
            <option value="cuda">CUDA</option>
          </select>
        </label>
      </div>
    {/if}
  {/if}

  <div class="subsection-heading">
    <span>Drive camera</span>
    <small>The Operator may cap unsupported profiles to the runtime capability.</small>
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
      <span>Field of view</span>
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
    <div class="detector-settings nested-fields">
      <div class="subsection-heading compact-subsection">
        <span>Detector</span>
        <small>Overlay remains advisory in manual and Traffic Manager sessions.</small>
      </div>
      {#if $sessionConfig.perception.detector === 'm9-hierarchical'}
        <p class="section-note">
          M9 uses the certified notebook checkpoint only, direct 800 × 800 RGB resize, and
          reports four coarse classes. Confidence here is a post-fusion display filter;
          0 keeps the notebook candidate set while the model's internal fine gate remains 0.10.
        </p>
      {/if}
      <div class="field-grid three-columns">
        <label class="field">
          <span>Backend</span>
          <select value={$sessionConfig.perception.detector} onchange={setDetector}>
            <option value="rtdetr">RT-DETR</option>
            <option value="yolo">YOLO</option>
            <option value="m9-hierarchical">M9 Hierarchical RT-DETR</option>
          </select>
        </label>
        <label class="field">
          <span>Weights</span>
          <select
            value={$sessionConfig.perception.weights}
            onchange={(event) => patchSessionSection('perception', { weights: fieldValue(event) })}
          >
            <option value="">Select workspace weights</option>
            {#each $workspaceOptions.detectorWeights as weights}<option value={weights}>{weights}</option>{/each}
          </select>
          {#if !$workspaceOptions.detectorWeights.length}<small>No detector .pt files are currently catalogued.</small>{/if}
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
            disabled={$sessionConfig.perception.detector === 'm9-hierarchical'}
            oninput={(event) => patchSessionSection('perception', { imageSize: fieldNumber(event) })}
          />
        </label>
        <label class="field confidence-field">
          <span class="confidence-label">
            Minimum confidence
            <output>{Math.round($sessionConfig.perception.confidence * 100)}%</output>
          </span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            aria-label="Minimum detection confidence"
            aria-valuetext={`${Math.round($sessionConfig.perception.confidence * 100)} percent`}
            value={$sessionConfig.perception.confidence}
            oninput={(event) => patchSessionSection('perception', { confidence: fieldNumber(event) })}
          />
          <small>Hide detections below this score. Applies when the session starts.</small>
        </label>
      </div>
    </div>
  {/if}
</section>

<style>
  .confidence-label {
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
  }

  .confidence-label output {
    font-variant-numeric: tabular-nums;
  }

  .confidence-field input[type='range'] {
    width: 100%;
    min-height: 24px;
    padding: 0;
    cursor: pointer;
  }
</style>
