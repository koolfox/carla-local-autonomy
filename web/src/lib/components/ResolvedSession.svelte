<script lang="ts">
  import { resetSession, sessionConfig, systemSettings } from '$lib/stores/configuration';

  $: populationOwner = $systemSettings?.workerConnected
    ? 'World Worker'
    : $systemSettings?.capabilities.garage_traffic_population
      ? 'Garage PythonAPI fallback'
      : 'Unavailable';
</script>

<aside class="resolved-panel">
  <div class="resolved-header">
    <div>
      <span class="eyebrow">Resolved session</span>
      <h2>What will run</h2>
    </div>
    <button type="button" class="ghost-button" onclick={resetSession}>Reset</button>
  </div>

  <div class="resolved-summary">
    <div><span>Preset</span><strong>{$sessionConfig.experiment.preset.replaceAll('_', ' ')}</strong></div>
    <div><span>Control</span><strong>{$sessionConfig.control.mode}</strong></div>
    <div><span>Map</span><strong>{$sessionConfig.scene.mapName.split('/').at(-1)}</strong></div>
    <div>
      <span>Population</span>
      <strong>{$sessionConfig.scene.trafficCount} cars · {$sessionConfig.scene.walkerCount} walkers</strong>
    </div>
    <div><span>Population owner</span><strong>{populationOwner}</strong></div>
    <div>
      <span>Camera</span>
      <strong>{$sessionConfig.camera.resolution} · {$sessionConfig.camera.fps} FPS</strong>
    </div>
    <div>
      <span>Perception</span>
      <strong>{$sessionConfig.perception.enabled ? $sessionConfig.perception.detector.toUpperCase() : 'Off'}</strong>
    </div>
    <div><span>Recording</span><strong>{$sessionConfig.recording.video ? 'Video on' : 'Video off'}</strong></div>
  </div>

  <div class="architecture-note">
    <strong>No duplicate settings.</strong>
    <p>
      The backend resolves this one configuration onto the World Worker or Garage PythonAPI without filling both population lanes.
    </p>
  </div>
</aside>
