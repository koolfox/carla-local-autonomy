import type {
  ExperimentPresetDefinition,
  SessionConfig,
  SystemSettings,
  WorkspaceOptions
} from '$lib/domain/config';

export interface BootstrapPayload {
  schema_version: string;
  token: string;
  catalog: {
    workspace: string;
    defaults: {
      carla_host: string;
      carla_port: number;
      map?: string;
    };
    capabilities: Record<string, boolean>;
  };
  drive?: Record<string, unknown>;
}

export interface ConfigurationContractPayload {
  schema_version: string;
  system: {
    workspace: string;
    carlaHost: string;
    carlaPort: number;
    localOnly: boolean;
    worldWorker: {
      configured: boolean;
      url: string | null;
    };
    experimentalEnabled: boolean;
  };
  sessionDefaults: Partial<SessionConfig>;
  experimentPresets: ExperimentPresetDefinition[];
}

export interface DriveCatalogPayload {
  schema_version: string;
  connected: boolean;
  host: string;
  port: number;
  server_version: string | null;
  map: string | null;
  maps: string[];
  vehicles: Array<{ id: string; label?: string; colors?: string[] }>;
  weather_presets: Array<{ id: string; label: string }>;
  prop_presets: Array<{ id: string; label: string }>;
  capabilities: Record<string, boolean>;
  world_worker: {
    configured: boolean;
    connected: boolean;
    status?: string;
    error?: string;
  };
  vision_runtime?: {
    available?: boolean;
    missing?: string[];
  };
  policy_checkpoints?: string[];
  error?: string;
}

export interface WorkspaceSnapshot {
  token: string;
  system: SystemSettings;
  options: WorkspaceOptions;
  sessionDefaults: Partial<SessionConfig>;
  experimentPresets: ExperimentPresetDefinition[];
  driveCatalog: DriveCatalogPayload;
}

async function readJson<T>(path: string): Promise<T> {
  const response = await fetch(path, {
    headers: { Accept: 'application/json' },
    cache: 'no-store'
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message =
      payload && typeof payload === 'object' && 'error' in payload
        ? JSON.stringify(payload.error)
        : `${response.status} ${response.statusText}`;
    throw new Error(`Operator API ${path} failed: ${message}`);
  }
  return payload as T;
}

export async function loadWorkspaceSnapshot(): Promise<WorkspaceSnapshot> {
  const [bootstrap, configuration, driveCatalog] = await Promise.all([
    readJson<BootstrapPayload>('/api/bootstrap'),
    readJson<ConfigurationContractPayload>('/api/configuration'),
    readJson<DriveCatalogPayload>('/api/drive/catalog')
  ]);

  const system: SystemSettings = {
    workspace: configuration.system.workspace,
    carlaHost: configuration.system.carlaHost,
    carlaPort: configuration.system.carlaPort,
    localOnly: configuration.system.localOnly,
    connected: Boolean(driveCatalog.connected),
    serverVersion: driveCatalog.server_version ?? null,
    currentMap: driveCatalog.map ?? null,
    workerConfigured: Boolean(configuration.system.worldWorker.configured),
    workerConnected: Boolean(driveCatalog.world_worker?.connected),
    workerUrl: configuration.system.worldWorker.url,
    experimentalEnabled: Boolean(configuration.system.experimentalEnabled),
    visionRuntimeAvailable: driveCatalog.vision_runtime?.available !== false,
    capabilities: { ...driveCatalog.capabilities }
  };

  const options: WorkspaceOptions = {
    maps: Array.isArray(driveCatalog.maps) ? driveCatalog.maps : [],
    vehicles: Array.isArray(driveCatalog.vehicles) ? driveCatalog.vehicles : [],
    weatherPresets: Array.isArray(driveCatalog.weather_presets)
      ? driveCatalog.weather_presets
      : [],
    propPresets: Array.isArray(driveCatalog.prop_presets) ? driveCatalog.prop_presets : [],
    checkpoints: Array.isArray(driveCatalog.policy_checkpoints)
      ? driveCatalog.policy_checkpoints
      : []
  };

  return {
    token: bootstrap.token,
    system,
    options,
    sessionDefaults: configuration.sessionDefaults,
    experimentPresets: configuration.experimentPresets,
    driveCatalog
  };
}

export class OperatorApi {
  constructor(private readonly token: string) {}

  async post<T>(path: string, body: Record<string, unknown>): Promise<T> {
    const response = await fetch(path, {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
        'X-Operator-Token': this.token
      },
      body: JSON.stringify(body)
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const message =
        payload && typeof payload === 'object' && 'error' in payload
          ? JSON.stringify(payload.error)
          : `${response.status} ${response.statusText}`;
      throw new Error(`Operator API ${path} failed: ${message}`);
    }
    return payload as T;
  }
}
