<script lang="ts">
  import { onDestroy } from 'svelte';

  import type {
    OperatorJobSnapshot,
    SituationSaveResponse,
    SituationSettings
  } from '$lib/api/operator';
  import SceneWorldFields from '$lib/components/SceneWorldFields.svelte';
  import CameraRigEditor from '$lib/components/CameraRigEditor.svelte';
  import { captureSettingsError } from '$lib/domain/capture';
  import { captureSettings } from '$lib/stores/capture';
  import { patchSessionSection, sessionConfig, systemSettings, workspaceOptions } from '$lib/stores/configuration';
  import { cancelCapture, captureRuntime, garageRuntime, runtimeOperatorApi, startCapture } from '$lib/stores/runtime';
  import { isDriveActive } from '$lib/domain/runtime';
  import { fieldValue } from '$lib/ui/events';

  export let useInGarage: () => void;

  function token(prefix: string): string {
    return `${prefix}-${new Date().toISOString().replace(/[-:.]/g, '').replace('Z', 'z').toLowerCase()}`;
  }

  let situationId = token('scene');
  let splitPlan = '';
  let planRunId = token('plan');
  let saved: SituationSaveResponse | null = null;
  let savedSituationId = '';
  let savedSignature = '';
  let job: OperatorJobSnapshot | null = null;
  let saving = false;
  let planning = false;
  let error = '';
  let disposed = false;
  let acknowledgeCapture = false;
  let captureRequest = false;

  async function capture(): Promise<void> {
    if (!acknowledgeCapture || captureRequest || $captureRuntime.active) return;
    captureRequest = true;
    error = '';
    try {
      await startCapture($sessionConfig, situation, $captureSettings.rig);
      acknowledgeCapture = false;
    } catch (caught) {
      error = caught instanceof Error ? caught.message : String(caught);
    } finally {
      captureRequest = false;
    }
  }

  async function cancel(): Promise<void> {
    try { await cancelCapture(); }
    catch (caught) { error = caught instanceof Error ? caught.message : String(caught); }
  }

  $: if (!splitPlan && $workspaceOptions.splitPlans.length) {
    splitPlan = $workspaceOptions.splitPlans[0];
  }
  $: situation = {
    situationId,
    egoSpawnIndex: $sessionConfig.route.startSpawnIndex ?? 0,
    durationSeconds: $captureSettings.durationSeconds,
    captureFps: $captureSettings.captureFps,
    repetitions: $captureSettings.repetitions
  } satisfies SituationSettings;
  $: signature = currentSignature(situation);
  $: recipeDirty = Boolean(saved && signature !== savedSignature);
  $: reproducible = Boolean(
    $sessionConfig.vehicle.blueprint &&
      $sessionConfig.scene.weatherPreset !== 'keep' &&
      ($sessionConfig.scene.mapName !== 'current' || $systemSettings?.currentMap)
  );
  $: resolved = saved?.configuration.resolved ?? {};

  function currentSignature(settings: SituationSettings): string {
    return JSON.stringify({
      currentMap: $systemSettings?.currentMap,
      identity: { seed: $sessionConfig.identity.seed },
      scene: $sessionConfig.scene,
      vehicle: $sessionConfig.vehicle,
      camera: $sessionConfig.camera,
      situation: settings
    });
  }

  async function saveRecipe(): Promise<void> {
    if (saving || !reproducible || Boolean(saved && !recipeDirty)) return;
    saving = true;
    error = '';
    job = null;
    try {
      let nextSituation = situation;
      if (saved && situation.situationId === savedSituationId) {
        const nextId = token('scene');
        situationId = nextId;
        nextSituation = { ...situation, situationId: nextId };
      }
      const response = await runtimeOperatorApi().saveSituation($sessionConfig, nextSituation);
      saved = response;
      savedSituationId = nextSituation.situationId;
      savedSignature = currentSignature(nextSituation);
      planRunId = token(`plan-${nextSituation.situationId}`);
    } catch (caught) {
      error = caught instanceof Error ? caught.message : String(caught);
    } finally {
      saving = false;
    }
  }

  async function buildPlan(): Promise<void> {
    if (!saved || recipeDirty || !splitPlan || planning) return;
    planning = true;
    error = '';
    try {
      let current = await runtimeOperatorApi().startJob('scenario_plan', {
        suite: saved.path,
        split_plan: splitPlan,
        run_id: planRunId
      });
      job = current;
      while (!disposed && !['success', 'failed', 'stopped'].includes(current.status)) {
        await new Promise((resolve) => window.setTimeout(resolve, 700));
        current = await runtimeOperatorApi().getJob(current.job_id);
        job = current;
      }
      if (current.status !== 'success') {
        error = current.error || `Scenario planning ended with status ${current.status}.`;
      }
    } catch (caught) {
      error = caught instanceof Error ? caught.message : String(caught);
    } finally {
      planning = false;
    }
  }

  onDestroy(() => {
    disposed = true;
  });
</script>

<div class="research-builder">
  <section class="builder-section">
    <div class="section-heading">
      <div>
        <span class="eyebrow">Shared scene</span>
        <h3>World and population</h3>
      </div>
      <span class="section-note">These are the same values used by Garage preview and Drive.</span>
    </div>
    <SceneWorldFields />
  </section>

  <section class="builder-section">
    <div class="section-heading">
      <div>
        <span class="eyebrow">Dataset</span>
        <h3>Capture plan</h3>
      </div>
      <span class="section-note">Save a recipe or record it with the connected World Worker.</span>
    </div>

    <div class="shared-config-summary">
      <div><span>Ego vehicle</span><strong>{$sessionConfig.vehicle.blueprint.split('.').at(-1) || 'Not selected'}</strong></div>
      <div><span>Camera</span><strong>{$sessionConfig.camera.resolution} · {$sessionConfig.camera.fov}°</strong></div>
      <div><span>Session seed</span><strong>{$sessionConfig.identity.seed}</strong></div>
    </div>

    <div class="field-grid three-columns">
      <label class="field">
        <span>Situation ID</span>
        <input bind:value={situationId} pattern="[a-z0-9-]+" />
      </label>
      <label class="field">
        <span>Ego spawn index</span>
        <input type="number" min="0" max="10000" value={$sessionConfig.route.startSpawnIndex ?? 0}
          oninput={(event) => patchSessionSection('route', { startSpawnIndex: fieldValue(event) === '' ? null : Number(fieldValue(event)) })} />
      </label>
      <label class="field">
        <span>Repetitions</span>
        <input type="number" min="1" max="32" bind:value={$captureSettings.repetitions} />
      </label>
      <label class="field">
        <span>Capture duration seconds</span>
        <input type="number" min="5" max="3600" bind:value={$captureSettings.durationSeconds} />
      </label>
      <label class="field">
        <span>Dataset capture rate</span>
        <select
          value={$captureSettings.captureFps}
          onchange={(event) =>
            ($captureSettings.captureFps = Number(fieldValue(event)) as SituationSettings['captureFps'])}
        >
          <option value="1">1 FPS</option>
          <option value="2">2 FPS</option>
          <option value="5">5 FPS</option>
          <option value="10">10 FPS</option>
        </select>
        <small>Independent from the live Garage camera FPS.</small>
      </label>
    </div>

    {#if !reproducible}
      <div class="notice warning-notice">
        <strong>Choose explicit reproducible values</strong>
        <span>Select a vehicle and a named weather preset. If the map is “current”, CARLA must report its current map.</span>
      </div>
    {/if}

    <div class="builder-actions">
      <button
        type="button"
        class="button primary-button"
        disabled={!reproducible || saving || Boolean(saved && !recipeDirty)}
        onclick={saveRecipe}
      >
        {saving ? 'Saving recipe…' : saved && !recipeDirty ? 'Recipe saved' : saved ? 'Save revised recipe' : 'Save situation recipe'}
      </button>
      <button type="button" class="button secondary-button" onclick={useInGarage}>Use these values in Garage</button>
    </div>

    {#if saved}
      <div class:warning={recipeDirty} class="evidence-strip">
        <div><span>Requested</span><strong>Shared SessionConfig</strong></div>
        <div><span>Resolved</span><strong>{String(resolved.map_name ?? '—')} · {String(resolved.weather_preset ?? '—')}</strong></div>
        <div><span>Applied</span><strong>{recipeDirty ? 'Changed · save a new recipe' : 'Offline recipe saved'}</strong></div>
      </div>
      <p class="saved-path">{saved.path}</p>
    {/if}
  </section>

  <section class="builder-section" aria-label="Teacher capture">
    <div class="section-heading">
      <h3>Record teacher dataset</h3>
      <span class="section-note">Uses these Scene and Capture plan settings. Results return here automatically.</span>
    </div>
    <CameraRigEditor disabled={$captureRuntime.active || captureRequest} />
    <label class="field">
      <span><input type="checkbox" bind:checked={acknowledgeCapture} disabled={$captureRuntime.active} />
        Reload the scene and let CARLA's BehaviorAgent drive for this capture.</span>
      <small>Garage preview pauses during collection. Images and labels are retained; review videos appear in Recordings.</small>
    </label>
    <div class="builder-actions">
      <button type="button" class="button primary-button"
        disabled={!reproducible || Boolean(captureSettingsError($captureSettings)) || !$captureRuntime.available || !acknowledgeCapture || captureRequest || $captureRuntime.active || $captureRuntime.holds_world || isDriveActive($garageRuntime.drive)}
        onclick={capture}>{captureRequest ? 'Starting capture…' : 'Record dataset'}</button>
      {#if $captureRuntime.active && $captureRuntime.holds_world}
        <button type="button" class="button secondary-button" onclick={cancel}
          disabled={$captureRuntime.cancel_requested}>{$captureRuntime.cancel_requested ? 'Cancelling…' : 'Cancel capture'}</button>
      {/if}
    </div>
    {#if captureSettingsError($captureSettings)}<p class="inline-error" role="alert">{captureSettingsError($captureSettings)}</p>{/if}
    {#if !$captureRuntime.available}<p class="section-note">Connect the normal World Worker to capture.</p>{/if}
    {#if $captureRuntime.phase !== 'idle'}
      <p role="status">{$captureRuntime.phase} · {$captureRuntime.job_id ?? ''}</p>
      {#if $captureRuntime.error}<p class="inline-error" role="alert">{$captureRuntime.error}</p>{/if}
      {#each $captureRuntime.results ?? [] as path}<p class="saved-path">{path} · available in Recordings</p>{/each}
      {#if $captureRuntime.log}<details><summary>Capture log</summary><pre>{$captureRuntime.log}</pre></details>{/if}
    {/if}
  </section>

  <section class="builder-section plan-section">
    <div class="section-heading">
      <div>
        <span class="eyebrow">Deterministic plan</span>
        <h3>Build scenario plan</h3>
      </div>
      <span class="section-note">Planning is offline and does not mutate CARLA.</span>
    </div>
    <div class="field-grid two-columns">
      <label class="field">
        <span>Split plan</span>
        <select bind:value={splitPlan}>
          {#if !$workspaceOptions.splitPlans.length}<option value="">No split plan found</option>{/if}
          {#each $workspaceOptions.splitPlans as path}<option value={path}>{path}</option>{/each}
        </select>
      </label>
      <label class="field">
        <span>Plan run ID</span>
        <input bind:value={planRunId} pattern="[a-z0-9-]+" />
      </label>
    </div>
    <div class="builder-actions">
      <button
        type="button"
        class="button primary-button"
        disabled={!saved || recipeDirty || !splitPlan || planning}
        onclick={buildPlan}
      >
        {planning ? 'Building plan…' : 'Build scenario plan'}
      </button>
    </div>
    {#if job}
      <div class:ok={job.status === 'success'} class:bad={job.status === 'failed'} class="job-result">
        <strong>{job.status === 'success' ? 'Scenario plan ready' : job.title}</strong>
        <span>{job.status}</span>
        {#if job.expected_output}<small>{job.expected_output}</small>{/if}
      </div>
    {/if}
  </section>

  {#if error}<p class="inline-error" role="alert">{error}</p>{/if}
</div>
