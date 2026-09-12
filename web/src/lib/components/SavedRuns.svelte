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
  <header><div><h3>Recordings</h3><small>{runs.length} saved runs · This workspace</small></div><button type="button" disabled={loading} onclick={refresh}>{loading ? 'Loading…' : 'Refresh'}</button></header>
  {#if error}<p role="alert">{error}</p>{/if}
  {#if !loading && runs.length === 0}<p>No saved runs with manifests yet. Stop &amp; Save a session, then refresh.</p>{/if}
  <div class="library">
    <aside aria-label="Saved run library">
      <input type="search" aria-label="Search saved runs" placeholder="Search by date, name or status…" bind:value={search} />
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
  section { padding: .5rem 0; }
  button, input, a { font: inherit; }
  button { cursor: pointer; }
  button:disabled { cursor: wait; opacity: .65; }
  header button, .detail > button, .recording-options button, .recording-options a {
    min-height: 40px; box-sizing: border-box; padding: .55rem .8rem;
    border: 1px solid #cececa; border-radius: 4px; background: #fff;
    color: #282d30; text-decoration: none;
  }
  button:hover:not(:disabled), .recording-options a:hover { background: #f0f0ec; }
  button:focus-visible, a:focus-visible, input:focus-visible { outline: 2px solid #343a3d; outline-offset: 2px; }
  header, .artifact { display: flex; align-items: center; gap: .75rem; flex-wrap: wrap; }
  header { justify-content: space-between; }
  h3, h4 { margin: 0; }
  h4 { margin-top: .8rem; overflow-wrap: anywhere; }
  .library { display: grid; grid-template-columns: minmax(160px, 1fr) minmax(0, 2fr); gap: 1rem; margin-top: 1rem; }
  aside, .detail { min-width: 0; }
  input[type='search'] { width: 100%; box-sizing: border-box; min-height: 40px; padding: .6rem; border: 1px solid #cececa; border-radius: 4px; background: #fff; }
  .filter { display: flex; align-items: center; gap: .5rem; font-size: .85rem; margin: .75rem 0; }
  .badge { font-size: .75rem; }
  ul { list-style: none; padding: 0; margin: .5rem 0; max-height: 55vh; overflow-y: auto; }
  li button { width: 100%; text-align: left; padding: .75rem; display: grid; gap: .3rem; border: 0; border-bottom: 1px solid #dededb; border-radius: 0; background: transparent; }
  small { color: #656563; }
  .chosen { background: #eaeae6; box-shadow: inset 3px 0 #343a3d; }
  span, strong { overflow-wrap: anywhere; }
  .artifact { padding: .5rem 0; }
  .artifact > span:first-child { flex: 1; min-width: 10rem; }
  .artifact small { display: block; }
  .player { background: #15191b; color: #eee; aspect-ratio: 16 / 9; display: grid; place-items: center; }
  .player p { padding: 1rem; text-align: center; }
  .recording-options { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; margin: 1rem 0; }
  details { margin-top: 1rem; border-top: 1px solid #dededb; padding-top: .8rem; }
  summary { cursor: pointer; }
  .note { font-size: .8rem; color: #656563; }
  @media (max-width: 640px) {
    .library { grid-template-columns: 1fr; }
    ul { max-height: 10rem; }
  }
</style>
