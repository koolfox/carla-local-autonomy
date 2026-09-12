<script lang="ts">
  import { cameraRigError, frontCamera, newCamera, presetRig, type CameraMount, type CameraView } from '$lib/domain/capture';
  import { captureSettings } from '$lib/stores/capture';
  import { sessionConfig } from '$lib/stores/configuration';
  import { fieldValue } from '$lib/ui/events';

  export let disabled = false;
  let selected = 'front';
  const mountFields: { key: keyof CameraMount; label: string; angular: boolean }[] = [
    { key: 'x', label: 'Forward (m)', angular: false },
    { key: 'y', label: 'Right (m)', angular: false },
    { key: 'z', label: 'Up (m)', angular: false },
    { key: 'yaw', label: 'Yaw (°)', angular: true },
    { key: 'pitch', label: 'Pitch (°)', angular: true },
    { key: 'roll', label: 'Roll (°)', angular: true }
  ];
  $: rig = $captureSettings.rig;
  $: current = selected === 'front' ? rig.primary_view ?? frontCamera($sessionConfig.camera.fov)
    : rig.additional_views.find((view) => view.id === selected) ?? frontCamera($sessionConfig.camera.fov);
  $: error = cameraRigError(rig);

  function update(view: CameraView): void {
    $captureSettings = { ...$captureSettings, rig: selected === 'front'
      ? { ...rig, primary_view: view }
      : { ...rig, additional_views: rig.additional_views.map((item) => item.id === selected ? { ...item, ...view } : item) } };
  }
  function add(): void {
    if (rig.additional_views.length >= 7) return;
    const ids = new Set(rig.additional_views.map((view) => view.id));
    let id = 'rear';
    if (ids.has(id)) { let index = 1; while (ids.has(`camera_${index}`)) index++; id = `camera_${index}`; }
    $captureSettings = { ...$captureSettings, rig: { ...rig, additional_views: [...rig.additional_views, newCamera(id, $sessionConfig.camera.fov)] } };
    selected = id;
  }
  function remove(): void {
    if (selected === 'front') return;
    $captureSettings = { ...$captureSettings, rig: { ...rig, additional_views: rig.additional_views.filter((view) => view.id !== selected) } };
    selected = 'front';
  }
</script>

<fieldset class="camera-editor" {disabled}>
  <legend>Capture cameras · {rig.additional_views.length + 1} RGB</legend>
  <div class="camera-toolbar">
    <label class="field"><span>Start from a layout</span>
      <select value="" onchange={(event) => {
        $captureSettings = { ...$captureSettings, rig: presetRig(fieldValue(event), $sessionConfig.camera.fov) };
        selected = 'front';
      }}>
        <option value="" disabled>Choose preset…</option>
        <option value="front">Front only</option><option value="front-rear">Front + rear</option>
        <option value="front-three">Front + left + right</option><option value="surround">Front + left + right + rear</option>
      </select>
    </label>
    <button type="button" class="button secondary-button" onclick={add} disabled={disabled || rig.additional_views.length >= 7}>Add camera</button>
  </div>
  <div class="camera-tabs" aria-label="Select camera to edit">
    {#each ['front', ...rig.additional_views.map((view) => view.id)] as id}
      <button type="button" class="button secondary-button" aria-pressed={selected === id} onclick={() => selected = id}>{id.replaceAll('_', ' ')}</button>
    {/each}
  </div>
  <div class="field-grid three-columns">
    {#each mountFields as field}
      <label class="field"><span>{field.label}</span>
        <input type="number" step={field.angular ? 1 : 0.1} min={field.angular ? -360 : undefined} max={field.angular ? 360 : undefined}
          value={current.mount[field.key]} oninput={(event) => update({ ...current, mount: { ...current.mount, [field.key]: fieldValue(event) === '' ? NaN : Number(fieldValue(event)) } })} />
      </label>
    {/each}
    <label class="field"><span>FOV (°)</span><input type="number" min="30" max="150" step="1" value={current.fov_degrees}
      oninput={(event) => update({ ...current, fov_degrees: fieldValue(event) === '' ? NaN : Number(fieldValue(event)) })} /></label>
  </div>
  <div class="camera-toolbar">
    <small>Relative to the vehicle origin. Negative X: rear, negative Y: left. Yaw 180°: rear-facing. Front is required for labels.</small>
    {#if selected !== 'front'}<button type="button" class="button secondary-button" onclick={remove}>Remove camera</button>{/if}
  </div>
  <small>Shared by Research capture and optional Drive rig recording. Changes apply to the next run, not the current live/model camera. Review angles in Recordings after a short run. More cameras cost GPU time and storage.</small>
  {#if error}<p class="inline-error" role="alert">{error}</p>{/if}
</fieldset>

<style>
  .camera-editor { border: 0; padding: 0; margin: 0; min-width: 0; display: grid; gap: 12px; }
  legend { font-weight: 600; margin-bottom: 12px; }
  .camera-toolbar { display: flex; flex-wrap: wrap; gap: 12px; align-items: end; justify-content: space-between; }
  .camera-toolbar small { flex: 1 1 240px; }
  .camera-tabs { display: flex; flex-wrap: wrap; gap: 6px; }
  .camera-tabs button { text-transform: capitalize; }
  .camera-tabs button[aria-pressed='true'] { outline: 2px solid currentColor; outline-offset: -2px; }
</style>
