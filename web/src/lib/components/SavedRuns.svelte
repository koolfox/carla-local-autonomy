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
    } catch (e) {
      if (!controller.signal.aborted) error = e instanceof Error ? e.message : String(e);
    } finally { loading = false; }
  }
  async function inspect(run: Run) {
    const request = ++generation;
    selected = run.path;
    artifacts = [];
    video = '';
    error = '';
    inspecting = true;
    try {
      const data = await read<{ artifacts: Artifact[] }>(`/api/evidence?path=${encodeURIComponent(run.path)}`);
      if (request === generation) artifacts = data.artifacts;
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
  <header><h3>Saved runs</h3><button type="button" disabled={loading} onclick={refresh}>{loading ? 'Loading…' : 'Refresh'}</button></header>
  <p>Recordings and artifacts saved by this workspace. Listing a file does not verify its contents.</p>
  {#if error}<p role="alert">{error}</p>{/if}
  {#if !loading && runs.length === 0}<p>No saved runs with manifests yet. Stop &amp; Save a session, then refresh.</p>{/if}
  <ul>
    {#each runs as run (run.path)}
      <li><button class:chosen={selected === run.path} type="button" onclick={() => inspect(run)}>
        <strong>{run.id}</strong><span>{run.status} · {run.created_at ?? 'Date unavailable'}</span>
      </button></li>
    {/each}
  </ul>
  {#if inspecting}<p role="status">Loading artifacts…</p>{/if}
  {#if selected && !inspecting}
    <h4>Files</h4>
    {#if artifacts.length === 0}<p>No registered artifacts for this run.</p>{/if}
    {#each artifacts as artifact}
      <div class="artifact">
        <span>{artifact.role} · {artifact.path}</span>
        {#if artifact.available && artifact.downloadable}
          {#if artifact.preview_kind === 'video'}<button type="button" onclick={() => video = url(artifact.path)}>Play</button>{/if}
          <a href={url(artifact.path)} download>Download</a>
        {:else}<span>Unavailable</span>{/if}
      </div>
    {/each}
  {/if}
  {#if video}
    <video src={video} controls preload="metadata" aria-label="Saved recording">
      <track kind="captions" />
      Your browser cannot play this recording. Use Download instead.
    </video>
  {/if}
</section>

<style>
  header, .artifact { display: flex; align-items: center; gap: .75rem; flex-wrap: wrap; }
  header { justify-content: space-between; }
  ul { list-style: none; padding: 0; max-height: 16rem; overflow-y: auto; }
  li button { width: 100%; text-align: left; padding: .65rem; display: grid; gap: .25rem; }
  .chosen { outline: 2px solid currentColor; outline-offset: -2px; }
  span, strong { overflow-wrap: anywhere; }
  .artifact { padding: .5rem 0; }
  .artifact > span:first-child { flex: 1; min-width: 10rem; }
  video { width: 100%; max-height: 55vh; margin-top: 1rem; background: #111; }
</style>
