<script lang="ts">
  import { onMount } from 'svelte';

  interface Run { id: string; path: string; root_kind: string; status: string; created_at: string | null }
  interface Artifact { path: string; role: string; available: boolean; downloadable: boolean; preview_kind: string }
  let runs: Run[] = [];
  let selected = '';
  let artifacts: Artifact[] = [];
  let video = '';
  let loading = false;
  let inspecting = false;
  let error = '';
  let generation = 0;
  let search = '';
  let playbackError = false;
  $: filtered = runs.filter(run => `${run.id} ${run.status}`.toLowerCase().includes(search.toLowerCase()));
  $: current = runs.find(run => run.path === selected);
  $: recordings = artifacts.filter(file => file.preview_kind === 'video' && file.available && file.downloadable);
  function label(file: Artifact) { return file.role.replaceAll('_', ' '); }
  function date(value: string | null) {
    if (!value) return 'Date unavailable';
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
  }
  function choose(file: Artifact) { playbackError = false; video = url(file.path); }
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
  onMount(() => { void refresh(); return () => controller.abort(); });
</script>

<section aria-label="Saved runs and recordings">
  <header><div><h3>Recordings</h3><small>{runs.length} saved runs · This workspace</small></div><button type="button" disabled={loading} onclick={refresh}>{loading ? 'Loading…' : 'Refresh'}</button></header>
  {#if error}<p role="alert">{error}</p>{/if}
  {#if !loading && runs.length === 0}<p>No saved runs with manifests yet. Stop &amp; Save a session, then refresh.</p>{/if}
  <div class="library">
    <aside aria-label="Saved run library">
      <input type="search" aria-label="Search saved runs" placeholder="Search runs or status…" bind:value={search} />
      <ul>
        {#each filtered as run (run.path)}
          <li><button class:chosen={selected === run.path} aria-pressed={selected === run.path} type="button" onclick={() => inspect(run)}>
            <strong>{run.id}</strong><span>{date(run.created_at)}</span><small>{run.status}</small>
          </button></li>
        {/each}
      </ul>
      {#if search && !filtered.length}<p>No matching runs.</p>{/if}
    </aside>
    <div class="detail" aria-busy={inspecting}>
      <div class="player">
        {#if video}
          {#key video}
            <video src={video} controls playsinline preload="metadata" aria-label="Saved recording" onerror={() => playbackError = true}>
              <track kind="captions" />
              Your browser cannot play this recording. Use Download instead.
            </video>
          {/key}
        {:else}
          <p role="status">{inspecting ? 'Loading recording…' : selected ? 'No playable recording in this run' : 'Select a saved run'}</p>
        {/if}
      </div>
      {#if playbackError}<p role="alert">This recording could not be played. Download it to open in a compatible player.</p>{/if}
      {#if current}
        <h4>{current.id}</h4><small>{date(current.created_at)} · {current.status}</small>
      {/if}
      {#if recordings.length}
        <div class="recording-options" aria-label="Recording tracks">
          {#each recordings as file}
            <button type="button" class:chosen={video === url(file.path)} aria-pressed={video === url(file.path)} onclick={() => choose(file)}>{label(file)}</button>
          {/each}
          <a href={video} download>Download video</a>
        </div>
      {/if}
      {#if selected && !inspecting}
        <details>
          <summary>Run files ({artifacts.length})</summary>
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
  header, .artifact { display: flex; align-items: center; gap: .75rem; flex-wrap: wrap; }
  header { justify-content: space-between; }
  h3, h4 { margin: 0; }
  h4 { margin-top: .8rem; overflow-wrap: anywhere; }
  .library { display: grid; grid-template-columns: minmax(160px, 1fr) minmax(0, 2fr); gap: 1rem; margin-top: 1rem; }
  aside, .detail { min-width: 0; }
  input { width: 100%; box-sizing: border-box; }
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
  video { display: block; width: 100%; height: 100%; max-height: 50vh; }
  .recording-options { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; margin: 1rem 0; }
  details { margin-top: 1rem; border-top: 1px solid #dededb; padding-top: .8rem; }
  summary { cursor: pointer; }
  .note { font-size: .8rem; color: #656563; }
  @media (max-width: 640px) {
    .library { grid-template-columns: 1fr; }
    ul { max-height: 10rem; }
  }
</style>
