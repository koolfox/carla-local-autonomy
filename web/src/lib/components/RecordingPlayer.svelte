<script lang="ts">
  import { onMount } from 'svelte';
  import 'video.js/dist/video-js.css';
  export let src: string;
  export let type: string;
  export let onerror: () => void;
  let host: HTMLDivElement;
  onMount(() => {
    let disposed = false;
    let player: import('video.js/dist/types/player').default | undefined;
    // Keep CSP strict: use bundled CSS instead of generated style elements.
    (window as Window & { VIDEOJS_NO_DYNAMIC_STYLE?: boolean }).VIDEOJS_NO_DYNAMIC_STYLE = true;
    void import('video.js').then(({ default: videojs }) => {
      if (disposed) return;
      const element = document.createElement('video');
      element.className = 'video-js vjs-big-play-centered';
      element.setAttribute('playsinline', '');
      element.setAttribute('aria-label', 'Saved recording');
      host.appendChild(element);
      player = videojs(element, {
        controls: true, autoplay: false, preload: 'metadata',
        responsive: true, experimentalSvgIcons: true,
        playbackRates: [0.25, 0.5, 1, 1.5, 2],
        sources: [{ src, type }],
        controlBar: { skipButtons: { forward: 5, backward: 5 } }
      });
      player.on('error', onerror);
    }).catch(() => { if (!disposed) onerror(); });
    return () => { disposed = true; player?.dispose(); };
  });
</script>

<div class="recording-player" bind:this={host}></div>

<style>
  .recording-player { width: 100%; height: 100%; min-width: 0; }
  .recording-player :global(.video-js) { width: 100%; height: 100%; min-height: 180px; }
</style>
