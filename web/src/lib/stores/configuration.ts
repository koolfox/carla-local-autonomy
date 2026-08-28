import { browser } from '$app/environment';
import { get, writable } from 'svelte/store';

import {
  applyExperimentPresetDefinition,
  defaultSessionConfig,
  mergeSessionDefaults,
  type ExperimentPreset,
  type ExperimentPresetDefinition,
  type SessionConfig,
  type SystemSettings,
  type WorkspaceOptions
} from '$lib/domain/config';
import type { WorkspaceSnapshot } from '$lib/api/operator';

const STORAGE_KEY = 'carla-vision-console.session-config.v1';
let resolvedDefaults = defaultSessionConfig();

export const systemSettings = writable<SystemSettings | null>(null);
export const workspaceOptions = writable<WorkspaceOptions>({
  maps: [],
  vehicles: [],
  weatherPresets: [],
  propPresets: [],
  checkpoints: []
});
export const experimentPresets = writable<ExperimentPresetDefinition[]>([]);
export const operatorToken = writable('');
export const sessionConfig = writable<SessionConfig>(resolvedDefaults);

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function readStoredConfig(base: SessionConfig): SessionConfig | null {
  if (!browser) return null;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed)) return null;
    return {
      ...base,
      ...parsed,
      identity: { ...base.identity, ...(isRecord(parsed.identity) ? parsed.identity : {}) },
      scene: { ...base.scene, ...(isRecord(parsed.scene) ? parsed.scene : {}) },
      vehicle: { ...base.vehicle, ...(isRecord(parsed.vehicle) ? parsed.vehicle : {}) },
      route: { ...base.route, ...(isRecord(parsed.route) ? parsed.route : {}) },
      control: { ...base.control, ...(isRecord(parsed.control) ? parsed.control : {}) },
      camera: { ...base.camera, ...(isRecord(parsed.camera) ? parsed.camera : {}) },
      perception: {
        ...base.perception,
        ...(isRecord(parsed.perception) ? parsed.perception : {})
      },
      recording: {
        ...base.recording,
        ...(isRecord(parsed.recording) ? parsed.recording : {})
      },
      experiment: {
        ...base.experiment,
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
  experimentPresets.set(snapshot.experimentPresets);
  operatorToken.set(snapshot.token);

  resolvedDefaults = mergeSessionDefaults(defaultSessionConfig(), snapshot.sessionDefaults);
  const stored = readStoredConfig(resolvedDefaults);
  const next = stored ?? resolvedDefaults;
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
  sessionConfig.set({ ...next });
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
  const definition = get(experimentPresets).find((candidate) => candidate.id === preset);
  if (!definition) throw new Error(`Unknown experiment preset: ${preset}`);
  sessionConfig.update((current) => applyExperimentPresetDefinition(current, definition));
}

export function resetSession(): void {
  const currentSystem = get(systemSettings);
  const options = get(workspaceOptions);
  const next = mergeSessionDefaults(defaultSessionConfig(), resolvedDefaults);
  next.vehicle.blueprint = options.vehicles[0]?.id ?? '';
  if (currentSystem && !currentSystem.visionRuntimeAvailable) next.perception.enabled = false;
  sessionConfig.set(next);
}
