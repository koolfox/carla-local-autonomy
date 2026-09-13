import { browser } from '$app/environment';
import { writable } from 'svelte/store';
import { captureSettingsError, type CaptureSettings } from '$lib/domain/capture';

const key = 'carla-vision-console.capture.v1';
export const captureSettings = writable<CaptureSettings>({
  rig: { schema_version: '1.0', additional_views: [] },
  durationSeconds: 30, captureFps: 5, repetitions: 1
});
let hydrated = false;
export function hydrateCaptureSettings(): void {
  if (!browser || hydrated) return;
  try {
    const raw = localStorage.getItem(key);
    if (raw) {
      const parsed = JSON.parse(raw) as CaptureSettings;
      if (!captureSettingsError(parsed)) captureSettings.set(parsed);
    }
  } catch { /* A corrupt draft must not prevent Garage startup. */ }
  hydrated = true;
}
captureSettings.subscribe((value) => {
  if (!browser || !hydrated || captureSettingsError(value)) return;
  try { localStorage.setItem(key, JSON.stringify(value)); }
  catch { /* Capture still works when browser storage is unavailable. */ }
});
