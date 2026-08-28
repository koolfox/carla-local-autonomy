import { browser } from '$app/environment';
import { get, writable } from 'svelte/store';

import {
  applyExperimentPreset as patchExperimentPreset,
  defaultSessionConfig,
  type ExperimentPreset,
  type SessionConfig,
  type SystemSettings,
  type WorkspaceOptions
} from '$lib/domain/config';
import type { WorkspaceSnapshot } from '$lib/api/operator';

const STORAGE_KEY = 'carla-vision-console.session-config.v1';

export const systemSettings = writable<SystemSettings | null>(null);
export const workspaceOptions = writable<WorkspaceOptions>({
  maps: [],
  vehicles: [],
  weatherPresets: [],
  propPresets: [],
  checkpoints: []
});
export const operatorToken = writable('');
export const sessionConfig = writable<SessionConfig>(defaultSessionConfig());

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function readStoredConfig(): SessionConfig | null {
  if (!browser) return null;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed)) return null;
    const defaults = defaultSessionConfig();
    return {
      ...defaults,
      ...parsed,
      identity: { ...defaults.identity, ...(isRecord(parsed.identity) ? parsed.identity : {}) },
      scene: { ...defaults.scene, ...(isRecord(parsed.scene) ? parsed.scene : {}) },
      vehicle: { ...defaults.vehicle, ...(isRecord(parsed.vehicle) ? parsed.vehicle : {}) },
      route: { ...defaults.route, ...(isRecord(parsed.route) ? parsed.route : {}) },
      control: { ...defaults.control, ...(isRecord(parsed.control) ? parsed.control : {}) },
      camera: { ...defaults.camera, ...(isRecord(parsed.camera) ? parsed.camera : {}) },
      perception: {
        ...defaults.perception,
        ...(isRecord(parsed.perception) ? parsed.perception : {})
      },
      recording: {
        ...defaults.recording,
        ...(isRecord(parsed.recording) ? parsed.recording : {})
      },
      experiment: {
        ...defaults.experiment,
        ...(isRecord(parsed.experiment) ? parsed.experiment : {})
      }
    } as SessionConfig;
  } catch {
    return null;
  }
}

function persist(config: SessionConfig): void {
  if (!browser) return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
}

if (browser) {
  sessionConfig.subscribe((config) => persist(config));
}

export function hydrateWorkspace(snapshot: WorkspaceSnapshot): void {
  systemSettings.set(snapshot.system);
  workspaceOptions.set(snapshot.options);
  operatorToken.set(snapshot.token);

  const stored = readStoredConfig();
  sessionConfig.update((existing) => {
    const next = stored ?? existing;
    const firstVehicle = snapshot.options.vehicles[0]?.id ?? '';
    if (!next.vehicle.blueprint && firstVehicle) {
      next.vehicle = { ...next.vehicle, blueprint: firstVehicle };
    }
    if (!snapshot.system.visionRuntimeAvailable) {
      next.perception = { ...next.perception, enabled: false };
    }
    if (!snapshot.system.workerConnected && next.control.mode === 'autopilot') {
      next.control = { ...next.control, mode: 'manual' };
    }
    return { ...next };
  });
}

export function patchSessionSection<K extends keyof SessionConfig>(
  section: K,
  patch: Partial<SessionConfig[K]>
): void {
  sessionConfig.update((current) => ({
    ...current,
    [section]: {
      ...current[section],
      ...patch
    }
  }));
}

export function applyExperimentPreset(preset: ExperimentPreset): void {
  sessionConfig.update((current) => patchExperimentPreset(current, preset));
}

export function resetSession(): void {
  const currentSystem = get(systemSettings);
  const options = get(workspaceOptions);
  const next = defaultSessionConfig();
  next.vehicle.blueprint = options.vehicles[0]?.id ?? '';
  if (currentSystem && !currentSystem.visionRuntimeAvailable) next.perception.enabled = false;
  sessionConfig.set(next);
}
