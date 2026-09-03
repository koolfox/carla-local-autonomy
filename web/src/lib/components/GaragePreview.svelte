<script lang="ts">
  import { onDestroy } from 'svelte';

  import type { ConfigurationEvidence } from '$lib/api/operator';
  import SessionLaunchBar from '$lib/components/SessionLaunchBar.svelte';
  import VehiclePicker from '$lib/components/VehiclePicker.svelte';
  import type { GarageOrbitRequest } from '$lib/domain/runtime';
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

  function recordNumber(
    record: Record<string, unknown>,
    key: string,
    fallback: number
  ): number {
    const value = Number(record[key]);
    return Number.isFinite(value) ? value : fallback;
  }

  function recordText(
    record: Record<string, unknown>,
    key: string,
    fallback: string
  ): string {
    const value = record[key];
    return typeof value === 'string' && value ? value : fallback;
  }

  let active = false;
  let busy = false;
  let error = '';
  let previewEvidence: ConfigurationEvidence | null = null;
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
  let attemptedAutoStart = '';
  let configureRetryDelay = 1000;
  let configureRetryTimer: ReturnType<typeof setTimeout> | null = null;
  let streamRetryDelay = 1000;
  let streamRetryTimer: ReturnType<typeof setTimeout> | null = null;

  $: selectedVehicle = $workspaceOptions.vehicles.find(
    (vehicle) => vehicle.id === $sessionConfig.vehicle.blueprint
  );
  $: driveActive = isDriveActive($garageRuntime.drive);
  $: available = Boolean(
    $systemSettings?.workerConfigured && $sessionConfig.vehicle.blueprint && !driveActive
  );
  $: signature = JSON.stringify({
    seed: $sessionConfig.identity.seed,
    scene: $sessionConfig.scene,
    vehicle: $sessionConfig.vehicle,
    route: $sessionConfig.route,
    control: $sessionConfig.control,
    camera: $sessionConfig.camera,
    // Voxel is a Drive-only observer and does not change the parked scene.
    perception: { ...$sessionConfig.perception, voxelEnabled: false },
    recording: $sessionConfig.recording,
    experiment: $sessionConfig.experiment,
    policy: $sessionConfig.policy
  });
  $: autoStartKey = available
    ? `${$systemSettings?.workerUrl ?? 'worker'}:ready`
    : '';
  $: dirty = active && signature !== appliedSignature;
  $: requestedResolution = $sessionConfig.camera.resolution;
  $: requestedFps = Number($sessionConfig.camera.fps);
  $: resolvedResolution = previewEvidence
    ? `${recordNumber(previewEvidence.resolved, 'width', 0)}x${recordNumber(previewEvidence.resolved, 'height', 0)}`
    : requestedResolution;
  $: resolvedFps = previewEvidence
    ? recordNumber(previewEvidence.resolved, 'fps', requestedFps)
    : requestedFps;
  $: appliedResolution = previewEvidence
    ? recordText(previewEvidence.applied, 'camera_resolution', resolvedResolution)
    : requestedResolution;
  $: appliedFps = previewEvidence
    ? recordNumber(previewEvidence.applied, 'camera_target_fps', resolvedFps)
    : requestedFps;
  $: appliedTraffic = previewEvidence
    ? recordNumber(
        previewEvidence.applied,
        'traffic_count',
        $sessionConfig.scene.trafficCount
      )
    : $sessionConfig.scene.trafficCount;
  $: appliedWalkers = previewEvidence
    ? recordNumber(
        previewEvidence.applied,
        'walker_count',
        $sessionConfig.scene.walkerCount
      )
    : $sessionConfig.scene.walkerCount;
  $: appliedMap = previewEvidence
    ? recordText(
        previewEvidence.applied,
        'map',
        recordText(previewEvidence.resolved, 'map_name', $sessionConfig.scene.mapName)
      )
    : $sessionConfig.scene.mapName;
  $: appliedWeather = previewEvidence
    ? recordText(
        previewEvidence.applied,
        'weather_preset',
        $sessionConfig.scene.weatherPreset
      )
    : $sessionConfig.scene.weatherPreset;
  $: resolutionAdjusted = Boolean(
    previewEvidence && (appliedResolution !== requestedResolution || appliedFps !== requestedFps)
  );
  $: populationAdjusted = Boolean(
    previewEvidence &&
      (appliedTraffic !== $sessionConfig.scene.trafficCount ||
        appliedWalkers !== $sessionConfig.scene.walkerCount)
  );
  $: evidenceTitle = previewEvidence
    ? `Requested: ${requestedResolution} at ${requestedFps} FPS, ${$sessionConfig.scene.trafficCount} cars, ${$sessionConfig.scene.walkerCount} walkers. Resolved: ${resolvedResolution} at ${resolvedFps} FPS. Applied: ${appliedResolution} at ${appliedFps} FPS, ${appliedTraffic} cars, ${appliedWalkers} walkers.`
    : '';
  $: streamSource = active && !driveActive
    ? `/api/garage/preview/stream.mjpg?t=${streamNonce}`
    : '';
  $: if (!available) {
    attemptedAutoStart = '';
    cancelConfigureRetry();
    cancelStreamRetry();
    if (active) detachPreview();
  }
  $: if (
    available &&
    !active &&
    !busy &&
    !configureRetryTimer &&
    attemptedAutoStart !== autoStartKey
  ) {
    attemptedAutoStart = autoStartKey;
    void configure();
  }

  onDestroy(() => {
    cancelConfigureRetry();
    cancelStreamRetry();
  });

  function cancelConfigureRetry(resetDelay = true): void {
    if (configureRetryTimer) clearTimeout(configureRetryTimer);
    configureRetryTimer = null;
    if (resetDelay) configureRetryDelay = 1000;
  }

  function scheduleConfigureRetry(): void {
    if (!available || configureRetryTimer) return;
    const delay = configureRetryDelay;
    configureRetryDelay = Math.min(configureRetryDelay * 2, 30000);
    configureRetryTimer = setTimeout(() => {
      configureRetryTimer = null;
      if (available && !active) attemptedAutoStart = '';
    }, delay);
  }

  function cancelStreamRetry(resetDelay = true): void {
    if (streamRetryTimer) clearTimeout(streamRetryTimer);
    streamRetryTimer = null;
    if (resetDelay) streamRetryDelay = 1000;
  }

  function scheduleStreamRetry(): void {
    if (!active || streamRetryTimer) return;
    const delay = streamRetryDelay;
    streamRetryDelay = Math.min(streamRetryDelay * 2, 30000);
    streamRetryTimer = setTimeout(() => {
      streamRetryTimer = null;
      if (active && available) streamNonce = Date.now();
    }, delay);
  }

  function detachPreview(): void {
    active = false;
    streamReady = false;
    appliedSignature = '';
    previewEvidence = null;
    pointer = null;
  }

  async function configure(): Promise<void> {
    if (!available || busy) return;
    cancelConfigureRetry(false);
    cancelStreamRetry();
    const requestedSignature = signature;
    const requestedSession = $sessionConfig;
    attemptedAutoStart = autoStartKey;
    busy = true;
    error = '';
    try {
      const response = await runtimeOperatorApi().configureGaragePreview(requestedSession);
      previewEvidence = response.configuration;
      active = true;
      systemSettings.update((current) =>
        current ? { ...current, workerConnected: true } : current
      );
      configureRetryDelay = 1000;
      appliedSignature = requestedSignature;
      if (response.configure_action === 'started' || response.configure_action === 'restarted') {
        streamNonce = Date.now();
        applyPreset('orbit', false);
      }
    } catch (caught) {
      error = caught instanceof Error ? caught.message : String(caught);
      try {
        const state = await runtimeOperatorApi().getGaragePreviewState();
        active = state.active === true;
        if (!active) previewEvidence = null;
      } catch {
        active = false;
        previewEvidence = null;
      }
      if (!active) scheduleConfigureRetry();
    } finally {
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
          cancelStreamRetry();
          error = '';
        }}
        onerror={() => {
          if (active) {
            error = 'The live Garage stream stopped.';
            streamReady = false;
            scheduleStreamRetry();
          }
        }}
      />
    {/if}

    <button
      type="button"
      class="garage-orbit-surface"
      aria-label="Garage orbit camera"
      onpointerdown={pointerDown}
      onpointermove={pointerMove}
      onpointerup={pointerUp}
      onpointercancel={pointerUp}
      onwheel={wheel}
    ></button>

    {#if !active}
      <div class="garage-preview-placeholder">
        <span class="garage-preview-mark">CV</span>
        <strong title="The live Garage opens automatically when the World Worker is ready.">{selectedVehicle?.label ?? 'Select a CARLA vehicle'}</strong>
      </div>
    {:else if !streamReady}
      <div class="garage-preview-placeholder compact-placeholder">
        <span class="loading-ring"></span>
        <strong>Waiting for CARLA camera…</strong>
      </div>
    {/if}

    <div class="garage-stage-hud">
      <span>{appliedMap.split('/').at(-1) ?? 'current'}</span>
      <span>{appliedWeather.replaceAll('-', ' ')}</span>
      <span class:adjusted={populationAdjusted}>{appliedTraffic} cars · {appliedWalkers} walkers</span>
      {#if previewEvidence}
        <span class:adjusted={resolutionAdjusted} title={evidenceTitle}>
          Applied {appliedResolution} · {appliedFps} FPS{resolutionAdjusted ? ` · ${requestedFps} requested` : ''}
        </span>
      {/if}
    </div>
  </div>

  <div class="garage-preview-toolbar">
    <VehiclePicker compact={true} />

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
      <button
        type="button"
        class="button primary-button"
        disabled={!active || !dirty || busy}
        title={dirty ? 'Apply the pending Garage settings' : 'Garage settings are already applied'}
        onclick={() => void configure()}
      >Apply changes</button>
      <span
        class="preview-state"
        class:updating={busy}
        aria-live="polite"
      >
        {#if busy}<span class="loading-ring" aria-hidden="true"></span>{/if}
        {#if busy}Applying…{:else if dirty}Changes pending{:else if configureRetryTimer || streamRetryTimer}Reconnecting…{:else if active}Live CARLA{:else}Waiting for bridge{/if}
      </span>
      <SessionLaunchBar compact={true} configurationPending={dirty} />
    </div>
  </div>

  {#if !$systemSettings?.workerConnected && !active}
    <p
      class="garage-preview-note"
      title="The live Garage camera is served by carla-world-worker on the CARLA computer, normally port 8766."
    >
      {$systemSettings?.connected
        ? 'CARLA online · connect the World Worker bridge for live Garage'
        : 'CARLA and the World Worker bridge are offline'}
    </p>
  {/if}
  {#if error}<p class="inline-error" role="alert">{error}</p>{/if}
</section>
