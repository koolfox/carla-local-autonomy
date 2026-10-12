<script lang="ts">
  import { onMount } from 'svelte';
  import RecordingPlayer from './RecordingPlayer.svelte';
  import { runtimeOperatorApi } from '$lib/stores/runtime';

  interface Run { id: string; path: string; root_kind: string; status: string; created_at: string | null; object_type?: string; roles?: string[] }
  interface Artifact { path: string; role: string; available: boolean; downloadable: boolean; preview_kind: string; mime_type: string }
  let runs: Run[] = [];
  let selected = '';
  let artifacts: Artifact[] = [];
  let video = '';
  let videoType = '';
  let originalVideo = '';
  let preparingCopy = false;
  let copyTimer: ReturnType<typeof setTimeout> | undefined;
  let loading = false;
  let inspecting = false;
  let error = '';
  let generation = 0;
  let search = '';
  let playbackError = false;
  let onlyVideo = false;
  $: filtered = runs.filter(run => `${title(run)} ${date(run.created_at)} ${run.id} ${run.status}`.toLowerCase().includes(search.toLowerCase()) && (!onlyVideo || hasVideo(run)));
  $: current = runs.find(run => run.path === selected);
  $: recordings = artifacts.filter(file => file.preview_kind === 'video' && file.available && file.downloadable);
  function label(file: Artifact) {
    if (file.role === 'drive_camera_overlay_video') return `Detections · ${file.path.split('/').pop()?.replace(/\.overlay\.mp4$/, '').replaceAll('_', ' ')}`;
    if (file.role === 'drive_camera_video') return `Camera · ${file.path.split('/').pop()?.replace(/\.mp4$/, '').replaceAll('_', ' ')}`;
    return ({ raw_drive_video: 'Original camera', advisory_model_overlay_video: 'Model detections', rgb_voxel_review_video: 'Voxel view' } as Record<string, string>)[file.role] ?? file.role.replaceAll('_', ' ');
  }
  function hasVideo(run: Run) { return run.roles?.some(role => role.includes('video')) ?? false; }
  function title(run: Run) {
    const kind = run.object_type && run.object_type !== 'unknown' ? run.object_type : 'Saved run';
    return kind.replaceAll('_', ' ').replace(/^./, letter => letter.toUpperCase());
  }
  function date(value: string | null) {
    if (!value) return 'Date unavailable';
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
  }
  function choose(file: Artifact) {
    clearTimeout(copyTimer); preparingCopy = false;
    error = '';
    playbackError = false; videoType = file.mime_type; video = url(file.path); originalVideo = video;
  }
  async function browserCopy() {
    const source = originalVideo;
    const run = selected;
    const file = recordings.find(item => url(item.path) === source);
    if (!file) return;
    error = '';
    preparingCopy = true;
    try {
      const result = await runtimeOperatorApi().post<{status: string; id?: string}>(
        '/api/recording-preview', { object: run, path: file.path }, false, 10000
      );
      if (controller.signal.aborted || originalVideo !== source || selected !== run) return;
      if (result.status === 'ready') {
        videoType = 'video/mp4'; video = `/api/recording-preview?id=${encodeURIComponent(result.id!)}`;
        playbackError = false; preparingCopy = false;
      } else { copyTimer = setTimeout(browserCopy, 2000); }
    } catch (e) {
      if (controller.signal.aborted || originalVideo !== source || selected !== run) return;
      preparingCopy = false; error = e instanceof Error ? e.message : String(e);
    }
  }
  const controller = new AbortController();

  async function read<T>(url: string): Promise<T> {
    const response = await fetch(url, { cache: 'no-store', signal: controller.signal });
    if (!response.ok) throw new Error(`Could not load saved results (${response.status}).`);
    return response.json();
  }
  async function refresh() {
    loading = true;
    error = '';
    try {
      const data = await read<{ catalog: { research_objects: Run[] } }>('/api/bootstrap');
      runs = data.catalog.research_objects.filter(run => run.root_kind === 'runs')
        .sort((a, b) => (b.created_at ?? b.id).localeCompare(a.created_at ?? a.id));
      if (!runs.some(run => run.path === selected)) {
        selected = ''; artifacts = []; video = '';
        if (runs.length) await inspect(runs[0]);
      }
    } catch (e) {
      if (!controller.signal.aborted) error = e instanceof Error ? e.message : String(e);
    } finally { loading = false; }
  }
  async function inspect(run: Run) {
    const request = ++generation;
    selected = run.path;
    artifacts = [];
    video = '';
    originalVideo = ''; preparingCopy = false; clearTimeout(copyTimer);
    playbackError = false;
    error = '';
    inspecting = true;
    try {
      const data = await read<{ artifacts: Artifact[] }>(`/api/evidence?path=${encodeURIComponent(run.path)}`);
      if (request === generation) {
        artifacts = data.artifacts;
        const first = artifacts.find(file => file.preview_kind === 'video' && file.available && file.downloadable);
        if (first) choose(first);
      }
    } catch (e) {
      if (request === generation && !controller.signal.aborted) error = e instanceof Error ? e.message : String(e);
    } finally { if (request === generation) inspecting = false; }
  }
  function url(path: string) {
    return `/api/artifact?object=${encodeURIComponent(selected)}&path=${encodeURIComponent(path)}`;
  }
  onMount(() => { void refresh(); return () => { controller.abort(); clearTimeout(copyTimer); }; });
</script>

<section aria-label="Saved runs and recordings">
  <header><div><small>{runs.length} saved runs · This workspace</small></div><button type="button" disabled={loading} onclick={refresh}>{loading ? 'Loading…' : 'Refresh'}</button></header>
  {#if error}<p role="alert">{error}</p>{/if}
  {#if !loading && runs.length === 0}<p>No saved runs with manifests yet. Stop &amp; Save a session, then refresh.</p>{/if}
  <div class="library">
    <aside aria-label="Saved run library">
      <input type="search" aria-label="Search saved runs" placeholder="Search runs…" bind:value={search} />
      <label class="filter"><input type="checkbox" bind:checked={onlyVideo} /> With recordings only</label>
      <ul>
        {#each filtered as run (run.path)}
          <li><button class:chosen={selected === run.path} aria-pressed={selected === run.path} type="button" onclick={() => inspect(run)}>
            <strong>{title(run)}</strong><span>{date(run.created_at)}</span><small class="badge">{run.status === 'success' ? 'Completed' : run.status} · {hasVideo(run) ? 'Video available' : 'Run files'}</small>
          </button></li>
        {/each}
      </ul>
      {#if runs.length && !filtered.length}<p>No matching runs. Try another search or clear the filter.</p>{/if}
    </aside>
    <div class="detail" aria-busy={inspecting}>
      <div class="player">
        {#if video}
          {#key video}
            <RecordingPlayer src={video} type={videoType} onerror={() => playbackError = true} />
          {/key}
        {:else}
          <p role="status">{inspecting ? 'Loading recording…' : selected ? 'No playable recording in this run' : 'Select a saved run'}</p>
        {/if}
      </div>
      {#if playbackError}<p role="alert">Your browser cannot play this recording. Prepare a playable copy or download the original.</p>{/if}
      {#if playbackError || preparingCopy}
        <button type="button" disabled={preparingCopy} onclick={browserCopy}>{preparingCopy ? 'Preparing browser copy…' : 'Prepare playable copy'}</button>
        <p class="note">Creates a temporary H.264 viewing copy. Your original and its evidence hashes stay unchanged.</p>
      {/if}
      {#if current}
        <h4>{title(current)}</h4><small>{date(current.created_at)} · {current.status === 'success' ? 'Completed' : current.status}</small>
      {/if}
      {#if recordings.length}
        <div class="recording-options" aria-label="Recording tracks">
          {#each recordings as file}
            <button type="button" class:chosen={originalVideo === url(file.path)} aria-pressed={originalVideo === url(file.path)} onclick={() => choose(file)}>{label(file)}</button>
          {/each}
          <a href={originalVideo} download>Download original</a>
        </div>
      {/if}
      {#if selected && !inspecting}
        <details>
          <summary>Run files ({artifacts.length})</summary>
          <p class="note">Run ID: {current?.id}<br />Folder: {selected}</p>
          <p class="note">File availability only — integrity has not been verified here.</p>
          {#each artifacts as artifact}
            <div class="artifact">
              <span>{label(artifact)}<small>{artifact.path}</small></span>
              {#if artifact.available && artifact.downloadable}<a href={url(artifact.path)} download>Download</a>{:else}<small>Unavailable</small>{/if}
            </div>
          {/each}
        </details>
      {/if}
    </div>
  </div>
</section>

<style>
  section { display: grid; gap: 12px; }
  header, .artifact { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  header { justify-content: space-between; }
  header small { display: block; margin-top: 2px; }
  h4 { margin: 0; font-size: var(--text-md); font-weight: 600; }
  h4 { overflow-wrap: anywhere; }
  header button, .detail > button, .recording-options button, .recording-options a {
    display: inline-flex; align-items: center; min-height: var(--control-h); padding: 0 12px;
    border: 1px solid var(--line-strong); border-radius: var(--radius); background: var(--panel);
    color: var(--text); font-size: var(--text-sm); text-decoration: none; cursor: pointer;
  }
  button:hover:not(:disabled), .recording-options a:hover { background: var(--panel-2); }
  button:disabled { cursor: progress; opacity: .55; }
  .library { display: grid; grid-template-columns: minmax(180px, 1fr) minmax(0, 2fr); gap: 16px; }
  aside, .detail { min-width: 0; }
  .detail { display: grid; align-content: start; gap: 8px; }
  input[type='search'] {
    width: 100%; min-height: var(--control-h); padding: 0 10px; border: 1px solid var(--line-strong);
    border-radius: var(--radius); background: var(--panel); color: var(--text);
  }
  .filter { display: flex; align-items: center; gap: 8px; margin: 10px 0 6px; color: var(--text-2); font-size: var(--text-sm); }
  ul { list-style: none; padding: 0; margin: 0; max-height: 55vh; overflow-y: auto; border-top: 1px solid var(--line); }
  li button {
    display: grid; gap: 2px; width: 100%; padding: 9px 10px; border: 0; border-bottom: 1px solid var(--line);
    background: transparent; color: var(--text); text-align: left; cursor: pointer;
  }
  li button strong { font-size: var(--text-sm); font-weight: 600; }
  li button span { color: var(--text-2); font-size: var(--text-xs); font-variant-numeric: tabular-nums; }
  .badge { color: var(--text-2); font-size: var(--text-xs); }
  small { color: var(--text-2); font-size: var(--text-xs); }
  .chosen { background: var(--accent-soft); box-shadow: inset 2px 0 var(--accent); }
  .recording-options .chosen { border-color: var(--accent); color: var(--accent-ink); box-shadow: none; }
  span, strong { overflow-wrap: anywhere; }
  .artifact { padding: 8px 0; border-bottom: 1px solid var(--line); font-size: var(--text-sm); }
  .artifact > span:first-child { flex: 1; min-width: 10rem; }
  .artifact small { display: block; }
  .artifact a { color: var(--accent-ink); }
  .player {
    display: grid; place-items: center; aspect-ratio: 16 / 9; overflow: hidden;
    border-radius: var(--radius-artboard); background: var(--stage); color: var(--on-stage-2);
    box-shadow: var(--shadow-artboard); font-size: var(--text-sm);
  }
  .player p { padding: 16px; text-align: center; }
  .recording-options { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
  details { border-top: 1px solid var(--line); padding-top: 10px; }
  summary { cursor: pointer; font-size: var(--text-sm); font-weight: 600; }
  p { margin: 0; font-size: var(--text-sm); }
  .note { color: var(--text-2); font-size: var(--text-xs); line-height: 1.5; }
  @container inspector (max-width: 560px) {
    .library { grid-template-columns: 1fr; }
    ul { max-height: 14rem; }
  }
</style>
