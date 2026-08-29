<script lang="ts">
  import { onDestroy } from 'svelte';

  import {
    driveStatusLabel,
    isDriveActive,
    isDriveRunning,
    recordingActive,
    speedMetresPerSecond
  } from '$lib/domain/runtime';
  import { garageRuntime } from '$lib/stores/runtime';
  import ManualControlPad from './ManualControlPad.svelte';

  let view: 'raw' | 'overlay' = 'raw';
  let streamReady = false;
  let streamNonce = Date.now();
  let retryTimer: ReturnType<typeof setTimeout> | null = null;

  $: drive = $garageRuntime.drive;
  $: running = isDriveRunning(drive);
  $: active = isDriveActive(drive);
  $: speedKmh = speedMetresPerSecond(drive) * 3.6;
  $: stream = drive.stream ?? {};
  $: detectorEnabled = Boolean(drive.detector?.enabled);
  $: if (!detectorEnabled && view === 'overlay') view = 'raw';
  $: streamSource = running && drive.session_id
    ? `/api/drive/stream.mjpg?view=${view}&session=${encodeURIComponent(drive.session_id)}&t=${streamNonce}`
    : '';

  function retryStream(): void {
    streamReady = false;
    if (retryTimer) clearTimeout(retryTimer);
    if (!running) return;
    retryTimer = setTimeout(() => {
      streamNonce = Date.now();
    }, 1200);
  }

  function chooseView(next: 'raw' | 'overlay'): void {
    if (next === 'overlay' && !detectorEnabled) return;
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

  onDestroy(() => {
    if (retryTimer) clearTimeout(retryTimer);
  });
</script>

<section class="cockpit-card" class:active={active}>
  <div class="cockpit-header">
    <div>
      <span class="eyebrow">Cockpit</span>
      <h2>{driveStatusLabel(drive.status)}</h2>
      <p>{drive.run_id ?? 'No active run'}</p>
    </div>
    <div class="cockpit-badges">
      <span class:ok={running} class="status-pill"><i></i>{drive.control_mode ?? drive.garage_mode ?? 'manual'}</span>
      <span class:ok={recordingActive(drive)} class="status-pill"><i></i>{recordingActive(drive) ? 'REC' : 'Recording off'}</span>
      <span class:bad={Boolean(stream.stale)} class:ok={running && !stream.stale} class="status-pill"><i></i>{stream.stale ? 'Stream stale' : 'Stream live'}</span>
    </div>
  </div>

  <div class="cockpit-layout">
    <div class="drive-viewport-panel">
      <div class="viewport-toolbar">
        <div class="segmented-control" aria-label="camera view">
          <button type="button" class:active={view === 'raw'} onclick={() => chooseView('raw')}>Raw camera</button>
          <button type="button" class:active={view === 'overlay'} disabled={!detectorEnabled} onclick={() => chooseView('overlay')}>Detections</button>
        </div>
        <span>{stream.resolution ?? 'camera'} · {Number(view === 'overlay' ? stream.overlay_fps : stream.source_fps || 0).toFixed(1)} FPS · {frameAge()}</span>
      </div>

      <div class="drive-viewport">
        {#if streamSource}
          <img
            src={streamSource}
            alt="Live CARLA drive camera"
            class:ready={streamReady}
            onload={() => (streamReady = true)}
            onerror={retryStream}
          />
        {/if}
        {#if !streamReady}
          <div class="viewport-placeholder">
            {#if drive.status === 'starting'}
              <span class="loading-ring"></span><strong>Starting camera stream…</strong>
            {:else if drive.status === 'stopping'}
              <strong>Saving the run…</strong><p>Controls are released while artifacts finalize.</p>
            {:else if drive.status === 'success'}
              <strong>Run saved</strong><p>{drive.output_path ?? 'Artifacts were finalized by the Operator.'}</p>
            {:else if drive.status === 'failed'}
              <strong>Drive failed</strong><p>{drive.error ?? 'No additional error detail was returned.'}</p>
            {:else}
              <strong>Camera waiting</strong><p>The stream appears when the Drive session reaches running.</p>
            {/if}
          </div>
        {/if}

        <div class="drive-hud">
          <div><span>Speed</span><strong>{speedKmh.toFixed(1)}<small> km/h</small></strong></div>
          <div><span>Gear</span><strong>{gearLabel(drive.telemetry?.gear)}</strong></div>
          <div><span>Elapsed</span><strong>{elapsed(drive.elapsed_seconds)}</strong></div>
        </div>
      </div>
    </div>

    <aside class="telemetry-panel">
      <div class="telemetry-row"><span>Control source</span><strong>{drive.control_source ?? '—'}</strong></div>
      <div class="telemetry-row"><span>Throttle</span><strong>{percent(drive.telemetry?.throttle)}</strong></div>
      <div class="telemetry-row"><span>Steer</span><strong>{percent(drive.telemetry?.steer)}</strong></div>
      <div class="telemetry-row"><span>Brake</span><strong>{percent(drive.telemetry?.brake)}</strong></div>
      <div class="telemetry-row"><span>Traffic</span><strong>{drive.traffic_count_actual ?? '—'}</strong></div>
      <div class="telemetry-row"><span>Walkers</span><strong>{drive.walker_count_actual ?? '—'}</strong></div>
      <div class="telemetry-row"><span>Deadman</span><strong class:warning-text={drive.deadman_active}>{drive.deadman_active ? 'Braking' : 'Clear'}</strong></div>
      <div class="telemetry-row"><span>Frame</span><strong>{view === 'overlay' ? drive.overlay_frame_sequence ?? '—' : drive.raw_frame_sequence ?? '—'}</strong></div>
    </aside>
  </div>

  {#if running}
    <ManualControlPad />
  {/if}

  {#if drive.error}<p class="inline-error" role="alert">{drive.error}</p>{/if}
  {#if drive.output_path && !running}<p class="saved-path">Saved to <code>{drive.output_path}</code></p>{/if}
</section>
