<script lang="ts">
  import type { GarageOrbitRequest, GaragePreviewConfig } from '$lib/domain/runtime';
  import { isDriveActive } from '$lib/domain/runtime';
  import { sessionConfig, systemSettings, workspaceOptions } from '$lib/stores/configuration';
  import { garageRuntime, runtimeOperatorApi } from '$lib/stores/runtime';

  type CameraPreset = GarageOrbitRequest['preset'];

  const presets: Record<CameraPreset, { yaw: number; pitch: number; distance: number }> = {
    orbit: { yaw: 325, pitch: -10, distance: 6.5 },
    front: { yaw: 0, pitch: -8, distance: 6 },
    rear: { yaw: 180, pitch: -8, distance: 6 },
    top: { yaw: 325, pitch: -25, distance: 8 },
    cockpit: { yaw: 0, pitch: 0, distance: 4 }
  };
  const presetNames: CameraPreset[] = ['orbit', 'front', 'rear', 'top', 'cockpit'];

  let active = false;
  let busy = false;
  let error = '';
  let appliedSignature = '';
  let streamNonce = 0;
  let streamReady = false;
  let sequence = 0;
  let preset: CameraPreset = 'orbit';
  let yaw = presets.orbit.yaw;
  let pitch = presets.orbit.pitch;
  let distance = presets.orbit.distance;
  let pointer: { id: number; x: number; y: number; yaw: number; pitch: number } | null = null;
  let orbitInFlight = false;
  let orbitPending = false;

  $: selectedVehicle = $workspaceOptions.vehicles.find(
    (vehicle) => vehicle.id === $sessionConfig.vehicle.blueprint
  );
  $: driveActive = isDriveActive($garageRuntime.drive);
  $: available = Boolean(
    $systemSettings?.workerConnected && $sessionConfig.vehicle.blueprint && !driveActive
  );
  $: previewConfig = buildPreviewConfig();
  $: signature = JSON.stringify(previewConfig);
  $: dirty = active && signature !== appliedSignature;
  $: streamSource = active && !driveActive
    ? `/api/garage/preview/stream.mjpg?t=${streamNonce}`
    : '';
  $: if (driveActive && active) {
    active = false;
    streamReady = false;
    appliedSignature = '';
    pointer = null;
  }

  function cameraProfile(): GaragePreviewConfig['profile'] {
    const resolution = $sessionConfig.camera.resolution;
    const fps = Number($sessionConfig.camera.fps);
    if (resolution === '640x384' && fps <= 10) return 'compatibility';
    if (resolution === '1280x720' && fps >= 60) return 'high-refresh';
    if (resolution === '1920x1080') return 'detail';
    return 'balanced';
  }

  function buildPreviewConfig(): GaragePreviewConfig {
    return {
      map_name: $sessionConfig.scene.mapName,
      weather_preset: $sessionConfig.scene.weatherPreset,
      vehicle_blueprint: $sessionConfig.vehicle.blueprint,
      color: $sessionConfig.vehicle.color,
      seed: $sessionConfig.identity.seed,
      traffic_count: $sessionConfig.scene.trafficCount,
      walker_count: $sessionConfig.scene.walkerCount,
      prop_preset: $sessionConfig.scene.propPreset,
      pedestrian_crossing_factor: $sessionConfig.scene.pedestrianCrossingFactor,
      speed_difference_percent: $sessionConfig.scene.speedDifferencePercent,
      following_distance_metres: $sessionConfig.scene.followingDistanceMetres,
      spectator_mirror: $sessionConfig.camera.spectatorFollow,
      profile: cameraProfile()
    };
  }

  async function configure(): Promise<void> {
    if (!available || busy) return;
    busy = true;
    error = '';
    streamReady = false;
    try {
      await runtimeOperatorApi().configureGaragePreview(previewConfig);
      active = true;
      appliedSignature = signature;
      streamNonce = Date.now();
      applyPreset('orbit', false);
    } catch (caught) {
      active = false;
      error = caught instanceof Error ? caught.message : String(caught);
    } finally {
      busy = false;
    }
  }

  async function close(): Promise<void> {
    if (busy) return;
    busy = true;
    error = '';
    try {
      await runtimeOperatorApi().stopGaragePreview();
    } catch (caught) {
      error = caught instanceof Error ? caught.message : String(caught);
    } finally {
      active = false;
      streamReady = false;
      appliedSignature = '';
      pointer = null;
      busy = false;
    }
  }

  function normalizeYaw(value: number): number {
    return ((value % 360) + 360) % 360;
  }

  function clamp(value: number, minimum: number, maximum: number): number {
    return Math.min(maximum, Math.max(minimum, value));
  }

  async function sendOrbit(): Promise<void> {
    if (!active || orbitInFlight) {
      if (active) orbitPending = true;
      return;
    }
    orbitInFlight = true;
    orbitPending = false;
    try {
      await runtimeOperatorApi().orbitGaragePreview({
        sequence: ++sequence,
        yaw,
        pitch,
        distance,
        preset
      });
    } catch (caught) {
      error = caught instanceof Error ? caught.message : String(caught);
    } finally {
      orbitInFlight = false;
      if (orbitPending) void sendOrbit();
    }
  }

  function applyPreset(name: CameraPreset, send = true): void {
    const value = presets[name];
    preset = name;
    yaw = value.yaw;
    pitch = value.pitch;
    distance = value.distance;
    if (send) void sendOrbit();
  }

  function pointerDown(event: PointerEvent): void {
    if (!active) return;
    const element = event.currentTarget as HTMLElement;
    try {
      element.setPointerCapture(event.pointerId);
    } catch {
      return;
    }
    pointer = { id: event.pointerId, x: event.clientX, y: event.clientY, yaw, pitch };
    event.preventDefault();
  }

  function pointerMove(event: PointerEvent): void {
    if (!pointer || pointer.id !== event.pointerId) return;
    yaw = normalizeYaw(pointer.yaw - (event.clientX - pointer.x) * 0.35);
    pitch = clamp(pointer.pitch + (event.clientY - pointer.y) * 0.22, -25, 15);
    preset = 'orbit';
    void sendOrbit();
    event.preventDefault();
  }

  function pointerUp(event: PointerEvent): void {
    if (pointer?.id !== event.pointerId) return;
    pointer = null;
  }

  function wheel(event: WheelEvent): void {
    if (!active) return;
    distance = clamp(distance + Math.sign(event.deltaY) * 0.4, 3.5, 10);
    preset = 'orbit';
    void sendOrbit();
    event.preventDefault();
  }
</script>

<section id="garage" class="garage-preview-card scroll-section">
  <div class="garage-preview-stage" class:live={active && streamReady}>
    {#if streamSource}
      <img
        src={streamSource}
        alt="Live CARLA Garage preview"
        draggable="false"
        onload={() => {
          streamReady = true;
          error = '';
        }}
        onerror={() => {
          streamReady = false;
          if (active) error = 'The live Garage stream stopped. Refresh the Garage preview.';
        }}
      />
    {/if}

    <div
      class="garage-orbit-surface"
      role="application"
      tabindex="0"
      aria-label="Garage orbit camera"
      onpointerdown={pointerDown}
      onpointermove={pointerMove}
      onpointerup={pointerUp}
      onpointercancel={pointerUp}
      onwheel={wheel}
    ></div>

    {#if !active}
      <div class="garage-preview-placeholder">
        <span class="garage-preview-mark">CV</span>
        <strong>{selectedVehicle?.label ?? 'Select a CARLA vehicle'}</strong>
        <p>Open the live Garage when the scene is ready. Configuration edits stay local until you explicitly refresh it.</p>
      </div>
    {:else if !streamReady}
      <div class="garage-preview-placeholder compact-placeholder">
        <span class="loading-ring"></span>
        <strong>Waiting for CARLA camera…</strong>
      </div>
    {/if}

    <div class="garage-stage-hud">
      <span>{$sessionConfig.scene.mapName.split('/').at(-1) ?? 'current'}</span>
      <span>{$sessionConfig.scene.weatherPreset.replaceAll('-', ' ')}</span>
      <span>{$sessionConfig.scene.trafficCount} cars · {$sessionConfig.scene.walkerCount} walkers</span>
    </div>
  </div>

  <div class="garage-preview-toolbar">
    <div class="garage-preview-identity">
      <span class="eyebrow">Live Garage</span>
      <strong>{selectedVehicle?.label ?? ($sessionConfig.vehicle.blueprint || 'No vehicle selected')}</strong>
      <small>
        {#if dirty}Settings changed · refresh explicitly{:else if active}CARLA preview matches the current form{:else}Preview is closed{/if}
      </small>
    </div>

    <div class="garage-camera-presets" aria-label="Garage camera presets">
      {#each presetNames as name}
        <button
          type="button"
          class:active={preset === name}
          disabled={!active}
          onclick={() => applyPreset(name)}
        >{name}</button>
      {/each}
    </div>

    <div class="garage-preview-actions">
      {#if active}
        <button type="button" class="button secondary-button" disabled={busy} onclick={close}>Close</button>
        <button type="button" class="button primary-button" disabled={busy || !dirty} onclick={configure}>
          {busy ? 'Updating…' : 'Refresh Garage'}
        </button>
      {:else}
        <button type="button" class="button primary-button" disabled={!available || busy} onclick={configure}>
          {busy ? 'Preparing…' : 'Open live Garage'}
        </button>
      {/if}
    </div>
  </div>

  {#if !$systemSettings?.workerConnected}
    <p class="garage-preview-note">Live preview requires the connected World Worker. Session configuration remains available without it.</p>
  {/if}
  {#if error}<p class="inline-error" role="alert">{error}</p>{/if}
</section>
