<script lang="ts">
  import { onMount } from 'svelte';

  import { loadWorkspaceSnapshot } from '$lib/api/operator';
  import type { ExperimentPreset } from '$lib/domain/config';
  import {
    applyExperimentPreset,
    hydrateWorkspace,
    patchSessionSection,
    resetSession,
    sessionConfig,
    systemSettings,
    workspaceOptions
  } from '$lib/stores/configuration';
  import '$lib/styles/app.css';

  let loading = true;
  let error = '';

  const presets: Array<{ id: ExperimentPreset; label: string; description: string }> = [
    { id: 'free_drive', label: 'Free Drive', description: 'Clean manual baseline' },
    { id: 'manual_handling', label: 'Manual Handling', description: 'Human control, no overlay' },
    { id: 'autopilot_takeover', label: 'Autopilot Takeover', description: 'TM baseline with takeover' },
    { id: 'perception_review', label: 'Perception Review', description: 'Detection + recording' },
    { id: 'traffic_stress', label: 'Traffic Stress', description: 'Dense vehicles and walkers' },
    { id: 'adverse_weather', label: 'Adverse Weather', description: 'Weather-focused recording' }
  ];

  onMount(async () => {
    try {
      hydrateWorkspace(await loadWorkspaceSnapshot());
    } catch (caught) {
      error = caught instanceof Error ? caught.message : String(caught);
    } finally {
      loading = false;
    }
  });

  function inputValue(event: Event): string {
    return (event.currentTarget as HTMLInputElement | HTMLSelectElement).value;
  }

  function inputNumber(event: Event): number {
    return Number(inputValue(event));
  }

  function inputChecked(event: Event): boolean {
    return (event.currentTarget as HTMLInputElement).checked;
  }
</script>

<svelte:head>
  <title>CARLA Vision Console</title>
  <meta
    name="description"
    content="Unified configuration surface for the CARLA Vision Research Console"
  />
</svelte:head>

<div class="app-shell">
  <header class="topbar">
    <div class="brand-lockup">
      <div class="brand-mark">CV</div>
      <div>
        <strong>CARLA Vision Console</strong>
        <span>Unified research workspace</span>
      </div>
    </div>

    {#if $systemSettings}
      <div class="topbar-status" aria-label="runtime status">
        <span class:ok={$systemSettings.connected} class="status-pill">
          <i></i>{$systemSettings.connected ? 'CARLA connected' : 'CARLA offline'}
        </span>
        <span class:ok={$systemSettings.workerConnected} class="status-pill">
          <i></i>{$systemSettings.workerConnected ? 'Worker connected' : 'Worker unavailable'}
        </span>
      </div>
    {/if}
  </header>

  <main class="workspace">
    <aside class="context-rail">
      <div class="eyebrow">Configuration</div>
      <h1>One session.<br />One source of truth.</h1>
      <p>
        Scene, vehicle, control, perception and recording now belong to one session configuration.
        Presets modify this same object instead of maintaining hidden parallel state.
      </p>

      {#if $systemSettings}
        <dl class="system-facts">
          <div>
            <dt>CARLA</dt>
            <dd>{$systemSettings.carlaHost}:{$systemSettings.carlaPort}</dd>
          </div>
          <div>
            <dt>Map</dt>
            <dd>{$systemSettings.currentMap ?? 'Unknown'}</dd>
          </div>
          <div>
            <dt>Server</dt>
            <dd>{$systemSettings.serverVersion ?? 'Unknown'}</dd>
          </div>
          <div>
            <dt>Workspace</dt>
            <dd title={$systemSettings.workspace}>{$systemSettings.workspace}</dd>
          </div>
        </dl>
      {/if}
    </aside>

    <section class="configuration-column" aria-busy={loading}>
      {#if loading}
        <div class="notice">Loading the current Operator state…</div>
      {:else if error}
        <div class="notice error-notice">
          <strong>Could not load Operator API</strong>
          <span>{error}</span>
          <small>Start <code>uv run carla-operator-ui --open-browser</code> on port 8765.</small>
        </div>
      {:else}
        <section class="config-card preset-card">
          <div class="section-heading">
            <div>
              <span class="eyebrow">Intent</span>
              <h2>Experiment preset</h2>
            </div>
            <span class="section-note">Preset = visible patch, never separate state</span>
          </div>
          <div class="preset-grid">
            {#each presets as preset}
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

        <section class="config-card">
          <div class="section-heading">
            <div>
              <span class="eyebrow">01 · Scene</span>
              <h2>World & traffic</h2>
            </div>
            {#if $systemSettings?.workerConnected}
              <span class="capability ok-text">World Worker ready</span>
            {:else}
              <span class="capability">Worker-gated controls may be unavailable</span>
            {/if}
          </div>

          <div class="field-grid two-columns">
            <label class="field">
              <span>Map</span>
              <select
                value={$sessionConfig.scene.mapName}
                onchange={(event) => patchSessionSection('scene', { mapName: inputValue(event) })}
              >
                <option value="current">Keep current map</option>
                {#each $workspaceOptions.maps as map}
                  <option value={map}>{map.split('/').at(-1) ?? map}</option>
                {/each}
              </select>
            </label>

            <label class="field">
              <span>Weather</span>
              <select
                value={$sessionConfig.scene.weatherPreset}
                onchange={(event) =>
                  patchSessionSection('scene', { weatherPreset: inputValue(event) })}
              >
                {#each $workspaceOptions.weatherPresets as preset}
                  <option value={preset.id}>{preset.label}</option>
                {/each}
              </select>
            </label>

            <label class="field">
              <span>Traffic vehicles</span>
              <input
                type="number"
                min="0"
                max="250"
                value={$sessionConfig.scene.trafficCount}
                oninput={(event) =>
                  patchSessionSection('scene', { trafficCount: inputNumber(event) })}
              />
            </label>

            <label class="field">
              <span>Pedestrians</span>
              <input
                type="number"
                min="0"
                max="250"
                value={$sessionConfig.scene.walkerCount}
                oninput={(event) =>
                  patchSessionSection('scene', { walkerCount: inputNumber(event) })}
              />
            </label>

            <label class="field">
              <span>Props</span>
              <select
                value={$sessionConfig.scene.propPreset}
                onchange={(event) => patchSessionSection('scene', { propPreset: inputValue(event) })}
              >
                {#each $workspaceOptions.propPresets as preset}
                  <option value={preset.id}>{preset.label}</option>
                {/each}
              </select>
            </label>

            <label class="field">
              <span>Route</span>
              <select
                value={$sessionConfig.route.mode}
                onchange={(event) =>
                  patchSessionSection('route', {
                    mode: inputValue(event) as 'free' | 'random_destination'
                  })}
              >
                <option value="free">Free drive</option>
                <option value="random_destination">Random destination</option>
              </select>
            </label>
          </div>

          <details class="advanced-block">
            <summary>Traffic dynamics</summary>
            <div class="field-grid three-columns">
              <label class="field">
                <span>Crossing factor</span>
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.05"
                  value={$sessionConfig.scene.pedestrianCrossingFactor}
                  oninput={(event) =>
                    patchSessionSection('scene', {
                      pedestrianCrossingFactor: inputNumber(event)
                    })}
                />
              </label>
              <label class="field">
                <span>TM speed difference %</span>
                <input
                  type="number"
                  min="-100"
                  max="100"
                  value={$sessionConfig.scene.speedDifferencePercent}
                  oninput={(event) =>
                    patchSessionSection('scene', {
                      speedDifferencePercent: inputNumber(event)
                    })}
                />
              </label>
              <label class="field">
                <span>Following distance m</span>
                <input
                  type="number"
                  min="0.1"
                  max="20"
                  step="0.1"
                  value={$sessionConfig.scene.followingDistanceMetres}
                  oninput={(event) =>
                    patchSessionSection('scene', {
                      followingDistanceMetres: inputNumber(event)
                    })}
                />
              </label>
            </div>
          </details>
        </section>

        <section class="config-card">
          <div class="section-heading">
            <div>
              <span class="eyebrow">02 · Drive</span>
              <h2>Vehicle & control</h2>
            </div>
          </div>

          <div class="field-grid two-columns">
            <label class="field">
              <span>Vehicle</span>
              <select
                value={$sessionConfig.vehicle.blueprint}
                onchange={(event) =>
                  patchSessionSection('vehicle', { blueprint: inputValue(event) })}
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
              <span>Control owner</span>
              <select
                value={$sessionConfig.control.mode}
                onchange={(event) =>
                  patchSessionSection('control', {
                    mode: inputValue(event) as typeof $sessionConfig.control.mode
                  })}
              >
                <option value="manual">Manual · Browser</option>
                <option value="autopilot" disabled={!$systemSettings?.capabilities.autopilot}>
                  Traffic Manager Autopilot
                </option>
                {#if $systemSettings?.experimentalEnabled}
                  <option value="behavior">BehaviorAgent · Experimental</option>
                  <option value="imitation">Imitation · Experimental</option>
                  <option value="voxel">Voxel Planner · Experimental</option>
                {/if}
              </select>
            </label>

            <label class="field">
              <span>Run ID</span>
              <input
                value={$sessionConfig.identity.runId}
                oninput={(event) => patchSessionSection('identity', { runId: inputValue(event) })}
              />
            </label>

            <label class="field">
              <span>Seed</span>
              <input
                type="number"
                min="0"
                value={$sessionConfig.identity.seed}
                oninput={(event) => patchSessionSection('identity', { seed: inputNumber(event) })}
              />
            </label>
          </div>
        </section>

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
                onchange={(event) =>
                  patchSessionSection('perception', { enabled: inputChecked(event) })}
              />
              <span><strong>Detection overlay</strong><small>RT-DETR / YOLO advisory output</small></span>
            </label>
            <label class="switch-field">
              <input
                type="checkbox"
                checked={$sessionConfig.recording.video}
                onchange={(event) =>
                  patchSessionSection('recording', { video: inputChecked(event) })}
              />
              <span><strong>Record video</strong><small>Retain review media with the run</small></span>
            </label>
            <label class="switch-field">
              <input
                type="checkbox"
                checked={$sessionConfig.camera.spectatorFollow}
                onchange={(event) =>
                  patchSessionSection('camera', { spectatorFollow: inputChecked(event) })}
              />
              <span><strong>Spectator follow</strong><small>Server-side observer camera</small></span>
            </label>
          </div>

          <div class="field-grid three-columns">
            <label class="field">
              <span>Resolution</span>
              <select
                value={$sessionConfig.camera.resolution}
                onchange={(event) =>
                  patchSessionSection('camera', { resolution: inputValue(event) })}
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
                onchange={(event) => patchSessionSection('camera', { fps: inputNumber(event) })}
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
                oninput={(event) => patchSessionSection('camera', { fov: inputNumber(event) })}
              />
            </label>
          </div>

          {#if $sessionConfig.perception.enabled}
            <div class="field-grid three-columns nested-fields">
              <label class="field">
                <span>Detector</span>
                <select
                  value={$sessionConfig.perception.detector}
                  onchange={(event) =>
                    patchSessionSection('perception', {
                      detector: inputValue(event) as 'rtdetr' | 'yolo'
                    })}
                >
                  <option value="rtdetr">RT-DETR</option>
                  <option value="yolo">YOLO</option>
                </select>
              </label>
              <label class="field">
                <span>Device</span>
                <select
                  value={$sessionConfig.perception.device}
                  onchange={(event) =>
                    patchSessionSection('perception', { device: inputValue(event) })}
                >
                  <option value="cpu">CPU</option>
                  <option value="mps">MPS</option>
                  <option value="cuda">CUDA</option>
                </select>
              </label>
              <label class="field">
                <span>Confidence</span>
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.05"
                  value={$sessionConfig.perception.confidence}
                  oninput={(event) =>
                    patchSessionSection('perception', { confidence: inputNumber(event) })}
                />
              </label>
            </div>
          {/if}
        </section>
      {/if}
    </section>

    <aside class="resolved-panel">
      <div class="resolved-header">
        <div>
          <span class="eyebrow">Resolved session</span>
          <h2>What will run</h2>
        </div>
        <button type="button" class="ghost-button" onclick={resetSession}>Reset</button>
      </div>

      <div class="resolved-summary">
        <div><span>Preset</span><strong>{$sessionConfig.experiment.preset.replaceAll('_', ' ')}</strong></div>
        <div><span>Control</span><strong>{$sessionConfig.control.mode}</strong></div>
        <div><span>Map</span><strong>{$sessionConfig.scene.mapName.split('/').at(-1)}</strong></div>
        <div>
          <span>Population</span>
          <strong>{$sessionConfig.scene.trafficCount} cars · {$sessionConfig.scene.walkerCount} walkers</strong>
        </div>
        <div><span>Camera</span><strong>{$sessionConfig.camera.resolution} · {$sessionConfig.camera.fps} FPS</strong></div>
        <div>
          <span>Perception</span>
          <strong>{$sessionConfig.perception.enabled ? $sessionConfig.perception.detector.toUpperCase() : 'Off'}</strong>
        </div>
        <div><span>Recording</span><strong>{$sessionConfig.recording.video ? 'Video on' : 'Video off'}</strong></div>
      </div>

      <div class="architecture-note">
        <strong>No duplicate settings.</strong>
        <p>
          Research launchers will reference this session context and add only job-specific fields in the next migration slice.
        </p>
      </div>
    </aside>
  </main>
</div>
