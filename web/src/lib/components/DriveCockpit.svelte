<script lang="ts">
  import { onDestroy } from 'svelte';

  import {
    isDriveActive,
    isDriveRunning,
    recordingActive,
    speedMetresPerSecond
  } from '$lib/domain/runtime';
  import { garageRuntime } from '$lib/stores/runtime';
  import ManualControlPad from './ManualControlPad.svelte';

  let view: 'raw' | 'overlay' | 'voxel' | 'voxel_overlay' = 'raw';
  let streamReady = false;
  let streamNonce = Date.now();
  let retryTimer: ReturnType<typeof setTimeout> | null = null;
  let manualArmRequest = 0;
  let initializedViewSession = '';
  let telemetryOpen = false;

  $: drive = $garageRuntime.drive;
  $: running = isDriveRunning(drive);
  $: active = isDriveActive(drive);
  $: modelOwnsControl = String(drive.garage_mode ?? '').toLowerCase() === 'model';
  $: speedKmh = speedMetresPerSecond(drive) * 3.6;
  $: stream = drive.stream ?? {};
  $: detectorEnabled = Boolean(drive.detector?.enabled);
  $: voxel = drive.voxel;
  $: voxelEnabled = Boolean(voxel?.enabled);
  $: voxelView = view === 'voxel' || view === 'voxel_overlay';
  $: waypointLabel = voxel?.waypoint_status === 'available'
    ? 'CARLA route (teacher)'
    : voxel?.waypoint_status === 'pending'
      ? 'Route pending'
      : voxel?.waypoint_status === 'empty' ? 'No route ahead' : 'Route unavailable';
  $: if (!detectorEnabled && view === 'overlay') view = 'raw';
  $: if (!voxelEnabled && (view === 'voxel' || view === 'voxel_overlay')) view = 'raw';
  $: if (voxelView && voxel?.status !== 'running') streamReady = false;
  $: if (
    running &&
    drive.session_id &&
    initializedViewSession !== drive.session_id
  ) {
    initializedViewSession = drive.session_id;
    view = 'raw';
    streamReady = false;
    streamNonce = Date.now();
  }
  $: streamSource = running && drive.session_id && (!voxelView || voxel?.status === 'running')
    ? `/api/drive/stream.mjpg?view=${view}&session=${encodeURIComponent(drive.session_id)}&t=${streamNonce}`
    : '';
  $: viewFps = Number((view === 'overlay' ? stream.overlay_fps : stream.source_fps) || 0);

  function retryStream(): void {
    streamReady = false;
    if (retryTimer) clearTimeout(retryTimer);
    if (!running || (voxelView && voxel?.status !== 'running')) return;
    retryTimer = setTimeout(() => {
      streamNonce = Date.now();
    }, 1200);
  }

  function chooseView(next: 'raw' | 'overlay' | 'voxel' | 'voxel_overlay'): void {
    if (next === 'overlay' && !detectorEnabled) return;
    if ((next === 'voxel' || next === 'voxel_overlay') && !voxelEnabled) return;
    if (retryTimer) clearTimeout(retryTimer);
    view = next;
    streamReady = false;
    streamNonce = Date.now();
  }

  function elapsed(seconds: number | undefined): string {
    const total = Math.max(0, Number(seconds) || 0);
    const minutes = Math.floor(total / 60);
    const remainder = Math.floor(total % 60);
    return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`;
  }

  function gearLabel(value: number | undefined): string {
    if (value === -1) return 'R';
    if (value == null || value === 0) return 'N';
    return String(value);
  }

  function percent(value: number | undefined): string {
    const numeric = Number(value) || 0;
    return `${Math.round(numeric * 100)}%`;
  }

  function frameAge(): string {
    const value = view === 'overlay' ? stream.overlay_age_seconds : stream.frame_age_seconds;
    if (value == null || !Number.isFinite(Number(value))) return 'waiting';
    return `${Math.round(Number(value) * 1000)} ms`;
  }

  function armFromViewport(event: PointerEvent): void {
    if (event.target instanceof Element && event.target.closest('button')) return;
    manualArmRequest += 1;
  }

  onDestroy(() => {
    if (retryTimer) clearTimeout(retryTimer);
  });
</script>

<section id="cockpit" class="cockpit-card scroll-section" class:active={active}>
  <div class="drive-viewport-panel">
    <!-- The camera is the deliberate keyboard-driving surface. -->
    <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
    <div
      class="drive-viewport"
      role="application"
      tabindex="0"
      aria-label="Live CARLA driving camera. Click or tap to take keyboard control."
      onpointerdown={armFromViewport}
    >
      {#if streamSource}
        <img
          src={streamSource}
          alt={voxelView ? 'RGB-derived Voxel geometry with CARLA teacher route reference, advisory only' : 'Live CARLA drive camera'}
          class:ready={streamReady}
          onload={() => (streamReady = true)}
          onerror={retryStream}
        />
      {/if}

      {#if !streamReady}
        <div class="viewport-placeholder">
          {#if voxelView && running}
            {#if voxel?.status === 'failed'}
              <strong>Voxel unavailable</strong><p role="alert">{voxel.error ?? 'RGB depth inference failed.'}</p>
            {:else if voxel?.status === 'stopped'}
              <strong>Voxel stopped</strong>
            {:else}
              <span class="loading-ring"></span><strong>{voxel?.status === 'loading' ? 'Loading RGB depth model…' : 'Waiting for Voxel frame…'}</strong>
              <p>First use may download approximately 100 MB.</p>
            {/if}
            <small>RGB estimates · CARLA route is a teacher reference, not model inference</small>
          {:else if drive.status === 'starting'}
            <span class="loading-ring"></span><strong>Starting camera…</strong>
          {:else if drive.status === 'stopping'}
            <strong>Saving run…</strong>
          {:else if drive.status === 'success'}
            <strong>Run saved</strong>
          {:else if drive.status === 'failed'}
            <strong>Drive failed</strong><p>{drive.error ?? 'No additional error detail was returned.'}</p>
          {:else}
            <strong>Camera waiting</strong>
          {/if}
        </div>
      {/if}

      <div class="viewport-toolbar">
        <div class="segmented-control" aria-label="camera view">
          <button type="button" class:active={view === 'raw'} onclick={() => chooseView('raw')}>Raw</button>
          <button type="button" class:active={view === 'overlay'} disabled={!detectorEnabled} onclick={() => chooseView('overlay')}>Detections</button>
          <button type="button" class:active={view === 'voxel'} disabled={!voxelEnabled} onclick={() => chooseView('voxel')}>Voxel 3D</button>
          <button type="button" class:active={view === 'voxel_overlay'} disabled={!voxelEnabled} onclick={() => chooseView('voxel_overlay')}>Voxel overlay</button>
        </div>
        {#if voxelView}
          <span title="RGB depth estimates, not LiDAR or semantic object labels. The CARLA route is privileged teacher data, not a model prediction. Neither view controls the vehicle.">
            <span class="voxel-geometry-key">RGB geometry</span> ·
            <span class="voxel-route-key" title={voxel?.waypoint_error ?? voxel?.waypoint_source ?? 'CARLA map waypoint reference'}>{waypointLabel}</span> ·
            {voxel?.latency_ms == null ? voxel?.status ?? 'waiting' : `${Math.round(voxel.latency_ms)} ms processing`}
          </span>
        {:else}
          <span>{stream.resolution ?? 'camera'} · {viewFps.toFixed(1)} FPS · {frameAge()}</span>
        {/if}
      </div>

      <div class="drive-hud" aria-label="driving HUD">
        <div><span>Speed</span><strong>{speedKmh.toFixed(1)}<small> km/h</small></strong></div>
        <div><span>Gear</span><strong>{gearLabel(drive.telemetry?.gear)}</strong></div>
        <div><span>Elapsed</span><strong>{elapsed(drive.elapsed_seconds)}</strong></div>
      </div>

      <div class="cockpit-badges">
        <span class:ok={running} class="status-pill"><i></i>{drive.control_mode ?? drive.garage_mode ?? 'manual'}</span>
        {#if recordingActive(drive)}<span class="status-pill recording"><i></i>REC</span>{/if}
        <button
          type="button"
          class="hud-toggle"
          aria-expanded={telemetryOpen}
          onclick={() => (telemetryOpen = !telemetryOpen)}
        >HUD</button>
      </div>

      {#if telemetryOpen}
        <aside class="telemetry-panel">
          <div class="telemetry-row"><span>Run</span><strong>{drive.run_id ?? '—'}</strong></div>
          <div class="telemetry-row"><span>Control</span><strong>{drive.control_source ?? '—'}</strong></div>
          <div class="telemetry-row"><span>Throttle</span><strong>{percent(drive.telemetry?.throttle)}</strong></div>
          <div class="telemetry-row"><span>Steer</span><strong>{percent(drive.telemetry?.steer)}</strong></div>
          <div class="telemetry-row"><span>Brake</span><strong>{percent(drive.telemetry?.brake)}</strong></div>
          <div class="telemetry-row"><span>Traffic</span><strong>{drive.traffic_count_actual ?? '—'}</strong></div>
          <div class="telemetry-row"><span>Walkers</span><strong>{drive.walker_count_actual ?? '—'}</strong></div>
          <div class="telemetry-row"><span>Deadman</span><strong class:warning-text={drive.deadman_active}>{drive.deadman_active ? 'Braking' : 'Clear'}</strong></div>
          <div class="telemetry-row"><span>Frame</span><strong>{voxelView ? voxel?.source_frame ?? '—' : view === 'overlay' ? drive.overlay_frame_sequence ?? '—' : drive.raw_frame_sequence ?? '—'}</strong></div>
        </aside>
      {/if}

      {#if running && !modelOwnsControl}
        <ManualControlPad armRequest={manualArmRequest} />
      {:else if running && modelOwnsControl}
        <p class="model-control-note">Model control active · Emergency Brake remains available</p>
      {/if}

      {#if drive.error}<p class="cockpit-error" role="alert">{drive.error}</p>{/if}
    </div>
  </div>
</section>

<style>
  .viewport-toolbar {
    width: max-content;
    max-width: calc(100% - 28px);
    flex-wrap: wrap;
    justify-content: center;
    gap: 4px 12px;
  }

  .segmented-control { flex-wrap: wrap; justify-content: center; }
  .voxel-geometry-key { color: #64d9e6; }
  .voxel-route-key { color: #f092d7; }
</style>
