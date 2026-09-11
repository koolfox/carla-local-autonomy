<script lang="ts">
  import { onDestroy } from 'svelte';

  import type { ConfigurationEvidence, GaragePreviewOperation } from '$lib/api/operator';
  import SessionLaunchBar from '$lib/components/SessionLaunchBar.svelte';
  import VehiclePicker from '$lib/components/VehiclePicker.svelte';
  import {
    createGarageApplyQueue,
    garagePreparationStageLabel,
    garagePreparationSummary,
    garagePreviewInputError,
    garagePreviewSignature,
    type GaragePreparationProgress
  } from '$lib/domain/garagePreview';
  import {
    confirmGarageStream,
    createGarageStreamBuffer,
    failGarageStream,
    stageGarageStream
  } from '$lib/domain/garagePreviewStream';
  import type { GarageOrbitRequest } from '$lib/domain/runtime';
  import { isDriveActive } from '$lib/domain/runtime';
  import {
    refreshWorldCatalog,
    sessionConfig,
    systemSettings,
    workspaceOptions
  } from '$lib/stores/configuration';
  import { captureRuntime, garageRuntime, runtimeOperatorApi } from '$lib/stores/runtime';

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

  function lifecycleLabel(stage: string): string {
    if (stage === 'accepted') return 'Starting Garage…';
    if (stage === 'configuring') return 'Preparing CARLA…';
    if (stage === 'running') return 'Live CARLA';
    return 'Applying…';
  }

  let active = false;
  let busy = false;
  let destroyed = false;
  let configureError = '';
  let streamError = '';
  let configureRetrying = false;
  let lifecycleStage = '';
  let preparation: GaragePreparationProgress | null = null;
  let previewEvidence: ConfigurationEvidence | null = null;
  let appliedSignature = '';
  let streamNonce = 0;
  let streamBuffer = createGarageStreamBuffer();
  let streamRetryDeferred = false;
  let sequence = 0;
  let preset: CameraPreset = 'orbit';
  let yaw = presets.orbit.yaw;
  let pitch = presets.orbit.pitch;
  let distance = presets.orbit.distance;
  let pointer: { id: number; x: number; y: number; yaw: number; pitch: number } | null = null;
  let orbitInFlight = false;
  let orbitPending = false;
  let streamRetryDelay = 1000;
  let streamConnectTimer: ReturnType<typeof setTimeout> | null = null;
  let streamRetryTimer: ReturnType<typeof setTimeout> | null = null;

  function updateLifecycle(operation: GaragePreviewOperation): void {
    if (destroyed) return;
    lifecycleStage = operation.preparation?.stage ?? operation.stage;
    if (operation.preparation) preparation = operation.preparation;
  }

  const applyQueue = createGarageApplyQueue({
    // Keep only the latest edit while the new chassis/camera settles.
    vehicleSettleMs: 1000,
    apply: (session) => runtimeOperatorApi().configureGaragePreview(session, updateLifecycle),
    canApply: () => !destroyed && !isDriveActive($garageRuntime.drive)
      && $garageRuntime.action !== 'start',
    busy: (value) => {
      busy = value;
      if (!value) {
        lifecycleStage = '';
        preparation = null;
        if (streamRetryDeferred) {
          streamRetryDeferred = false;
          scheduleStreamRetry();
        }
      }
    },
    retrying: (value) => { configureRetrying = value; },
    failed: (caught) => {
      lifecycleStage = '';
      preparation = null;
      configureError = caught instanceof Error ? caught.message : String(caught);
    },
    applied: (response, requestedSignature) => {
      lifecycleStage = '';
      preparation = null;
      configureError = '';
      previewEvidence = response.configuration;
      active = true;
      appliedSignature = requestedSignature;
      systemSettings.update((current) =>
        current ? { ...current, workerConnected: true } : current
      );
      void runtimeOperatorApi().getDriveCatalog().then(refreshWorldCatalog).catch(() => {
        // Preview evidence remains authoritative; catalog refresh is convenience-only.
      });
      // Stage a replacement stream in the hidden slot. The last decoded frame
      // stays visible until the new CARLA camera has produced its first frame.
      if (response.configure_action !== 'noop') {
        cancelStreamRetry();
        refreshStream();
        if (response.configure_action === 'started' || response.configure_action === 'restarted') {
          applyPreset('orbit', false);
        }
      }
    }
  });

  $: selectedVehicle = $workspaceOptions.vehicles.find(
    (vehicle) => vehicle.id === $sessionConfig.vehicle.blueprint
  );
  $: driveActive = isDriveActive($garageRuntime.drive) || $garageRuntime.action === 'start';
  $: available = Boolean(
    $systemSettings?.workerConfigured && $sessionConfig.vehicle.blueprint && !driveActive && !$captureRuntime.holds_world
  );
  $: signature = garagePreviewSignature($sessionConfig);
  $: inputError = garagePreviewInputError($sessionConfig);
  $: dirty = available && (signature !== appliedSignature || Boolean(inputError));
  $: if (active && !busy && !dirty) configureError = '';
  $: error = inputError || configureError || streamError;
  $: applyQueue.select(
    $sessionConfig,
    available ? $systemSettings?.workerUrl ?? 'worker' : '',
    !inputError
  );
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
  $: preparationText = preparation ? garagePreparationSummary(preparation) : '';
  $: if (!available) {
    cancelStreamRetry();
    if (active) detachPreview();
  }

  onDestroy(() => {
    destroyed = true;
    applyQueue.dispose();
    cancelStreamRetry();
  });

  function cancelStreamRetry(resetDelay = true): void {
    if (streamConnectTimer) clearTimeout(streamConnectTimer);
    streamConnectTimer = null;
    if (streamRetryTimer) clearTimeout(streamRetryTimer);
    streamRetryTimer = null;
    if (resetDelay) {
      streamRetryDelay = 1000;
      streamRetryDeferred = false;
    }
  }

  function refreshStream(): void {
    if (!active || !available || destroyed) return;
    streamNonce += 1;
    streamBuffer = stageGarageStream(
      streamBuffer,
      `/api/garage/preview/stream.mjpg?t=${Date.now()}-${streamNonce}`
    );
    if (streamConnectTimer) clearTimeout(streamConnectTimer);
    streamConnectTimer = setTimeout(() => {
      streamConnectTimer = null;
      if (busy) streamRetryDeferred = true;
      else scheduleStreamRetry();
    }, 5000);
  }

  function streamLoaded(slot: number): void {
    if (destroyed || !available) return;
    streamBuffer = confirmGarageStream(streamBuffer, slot);
    if (streamConnectTimer) clearTimeout(streamConnectTimer);
    streamConnectTimer = null;
    cancelStreamRetry();
    streamError = '';
  }

  function streamFailed(slot: number): void {
    const failed = failGarageStream(streamBuffer, slot, busy);
    streamBuffer = failed.buffer;
    if (!active || !available || destroyed) return;
    if (busy) {
      streamRetryDeferred = true;
      return;
    }
    streamError = 'The live Garage stream stopped.';
    if (failed.shouldRetry) scheduleStreamRetry();
  }

  function scheduleStreamRetry(): void {
    if (busy || !active || !available || destroyed || streamRetryTimer) return;
    const delay = streamRetryDelay;
    streamRetryDelay = Math.min(2000, streamRetryDelay * 2);
    streamRetryTimer = setTimeout(() => {
      streamRetryTimer = null;
      if (active && available && !destroyed && !busy) refreshStream();
    }, delay);
  }

  function detachPreview(): void {
    active = false;
    streamBuffer = createGarageStreamBuffer();
    streamNonce = 0;
    streamRetryDeferred = false;
    appliedSignature = '';
    previewEvidence = null;
    pointer = null;
    orbitPending = false;
    lifecycleStage = '';
    preparation = null;
    configureError = '';
    streamError = '';
  }

  function retry(): void {
    if (!available || destroyed || busy || inputError) return;
    if (configureError || !active) {
      applyQueue.retry();
    } else if (streamError) {
      cancelStreamRetry();
      refreshStream();
    }
  }

  function normalizeYaw(value: number): number {
    return ((value % 360) + 360) % 360;
  }

  function clamp(value: number, minimum: number, maximum: number): number {
    return Math.min(maximum, Math.max(minimum, value));
  }

  async function sendOrbit(): Promise<void> {
    if (destroyed || !available) return;
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
      if (!destroyed && available) {
        streamError = caught instanceof Error ? caught.message : String(caught);
      }
    } finally {
      orbitInFlight = false;
      if (orbitPending && !destroyed && available) void sendOrbit();
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
  <div class="garage-preview-stage" class:live={active && streamBuffer.ready}>
    {#each streamBuffer.sources as source, slot}
      {#if source}
        <img
          src={source}
          alt={slot === streamBuffer.visible ? 'Live CARLA Garage preview' : ''}
          aria-hidden={slot !== streamBuffer.visible}
          draggable="false"
          style:opacity={slot === streamBuffer.visible ? '1' : '0'}
          style:z-index={slot === streamBuffer.visible ? '1' : '0'}
          onload={() => streamLoaded(slot)}
          onerror={() => streamFailed(slot)}
        />
      {/if}
    {/each}

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
        {#if $captureRuntime.holds_world}
          <strong>Teacher capture · {$captureRuntime.phase}</strong>
          <span>Progress and cancellation are in Research.</span>
        {:else if busy}
          <span class="loading-ring"></span>
          <strong>{garagePreparationStageLabel(lifecycleStage)}</strong>
        {:else}
          <span class="garage-preview-mark">CV</span>
          <strong title="The live Garage opens automatically when the World Worker is ready.">{selectedVehicle?.label ?? 'Select a CARLA vehicle'}</strong>
        {/if}
      </div>
    {:else if !streamBuffer.ready}
      <div class="garage-preview-placeholder compact-placeholder">
        <span class="loading-ring"></span>
        <strong>{busy ? garagePreparationStageLabel(lifecycleStage) : 'Waiting for CARLA camera…'}</strong>
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
      {#if (configureError || streamError) && !inputError && available}
        <button
          type="button"
          class="button secondary-button"
          disabled={busy}
          title="Retry the live Garage connection with the current settings"
          onclick={retry}
        >Retry</button>
      {/if}
      <span
        class="preview-state"
        class:updating={busy}
        aria-live="polite"
      >
        {#if busy}<span class="loading-ring" aria-hidden="true"></span>{/if}
        {#if $captureRuntime.holds_world}Teacher capture{:else if driveActive}Drive active{:else if busy}{garagePreparationStageLabel(lifecycleStage)}{:else if configureRetrying || streamRetryTimer}Reconnecting…{:else if error}Needs attention{:else if dirty}Syncing settings…{:else if active}Live CARLA{:else}Waiting for bridge{/if}
      </span>
      <SessionLaunchBar compact={true} configurationPending={dirty || busy} />
    </div>
  </div>

  {#if busy && preparationText}
    <p class="garage-preview-note" aria-live="polite">{preparationText}</p>
  {/if}
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
