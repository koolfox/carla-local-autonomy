import type {
  CatalogOption,
  ExperimentPresetDefinition,
  InvalidModelPackage,
  ModelPackage,
  SessionConfig,
  SystemSettings,
  WorkspaceOptions
} from '$lib/domain/config';
import type { GaragePreparationProgress } from '$lib/domain/garagePreview';
import type {
  DriveControlRequest,
  DriveState,
  GarageOrbitRequest
} from '$lib/domain/runtime';

export interface ConfigurationEvidence {
  schema_version: string;
  requested: unknown;
  resolved: Record<string, unknown>;
  applied: Record<string, unknown>;
}

export interface GaragePreviewResponse extends Record<string, unknown> {
  configuration: ConfigurationEvidence;
  applied_config?: Record<string, unknown> | null;
  configure_action?: 'noop' | 'weather' | 'updated' | 'started' | 'restarted';
}

export interface GaragePreviewOperation extends Record<string, unknown> {
  schema_version: string;
  operation_id: string;
  status: 'starting' | 'running' | 'failed';
  stage: string;
  revision: number;
  elapsed_seconds: number;
  error?: { type?: string; message?: string } | null;
  preparation?: GaragePreparationProgress | null;
  result?: Record<string, unknown>;
  resolved_config?: Record<string, unknown>;
}

export interface SituationSettings {
  situationId: string;
  egoSpawnIndex: number;
  durationSeconds: number;
  captureFps: 1 | 2 | 5 | 10;
  repetitions: number;
}

export interface SituationSaveResponse {
  status: 'saved';
  path: string;
  suite_id: string;
  recipe_id: string;
  configuration: ConfigurationEvidence;
}

export interface OperatorJobSnapshot {
  job_id: string;
  kind: string;
  title: string;
  status: string;
  expected_output?: string | null;
  expected_output_exists?: boolean;
  error?: string | null;
  [key: string]: unknown;
}

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
    weights?: string[];
    scenario_suites?: string[];
    split_plans?: string[];
  };
  drive?: DriveState;
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

export interface ModelRegistryPayload {
  schema_version: string;
  packages: ModelPackage[];
  invalid: InvalidModelPackage[];
}

export interface DriveCatalogPayload {
  schema_version: string;
  connected: boolean;
  host: string;
  port: number;
  server_version: string | null;
  map: string | null;
  maps: Array<string | { id: string; label?: string }>;
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
  driveState: DriveState;
}

interface ApiErrorPayload {
  error?: {
    type?: string;
    message?: string;
  };
}

class OperatorApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

function apiError(path: string, response: Response, payload: unknown): Error {
  const typed = payload as ApiErrorPayload | null;
  const message = typed?.error?.message || `${response.status} ${response.statusText}`;
  return new OperatorApiError(`Operator API ${path} failed: ${message}`, response.status);
}

function shortMapName(value: string): string {
  const normalized = value.trim().replace(/\/+$/, '');
  return normalized.split('/').at(-1) ?? normalized;
}

function normalizeMapOptions(raw: unknown): CatalogOption[] {
  if (!Array.isArray(raw)) return [];
  const seen = new Set<string>();
  const options: CatalogOption[] = [];
  for (const value of raw) {
    const rawId =
      typeof value === 'string'
        ? value
        : value && typeof value === 'object' && 'id' in value
          ? String(value.id)
          : '';
    const id = shortMapName(rawId);
    if (!id || seen.has(id)) continue;
    const rawLabel =
      value && typeof value === 'object' && 'label' in value
        ? String(value.label ?? '')
        : '';
    seen.add(id);
    options.push({ id, label: rawLabel.trim() || id });
  }
  return options;
}

async function readJson<T>(path: string): Promise<T> {
  const response = await fetch(path, {
    headers: { Accept: 'application/json' },
    cache: 'no-store'
  });
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) throw apiError(path, response, payload);
  return payload as T;
}

export async function loadWorkspaceSnapshot(): Promise<WorkspaceSnapshot> {
  const [bootstrap, configuration, driveCatalog, driveState, modelRegistry] = await Promise.all([
    readJson<BootstrapPayload>('/api/bootstrap'),
    readJson<ConfigurationContractPayload>('/api/configuration'),
    readJson<DriveCatalogPayload>('/api/drive/catalog'),
    readJson<DriveState>('/api/drive/state'),
    readJson<ModelRegistryPayload>('/api/models')
  ]);

  const drivingPackages = Array.isArray(modelRegistry.packages)
    ? modelRegistry.packages.filter((model) => model.role === 'driving_policy')
    : [];
  const capabilities = { ...driveCatalog.capabilities };
  capabilities.garage_model_drive = Boolean(
    configuration.system.experimentalEnabled && drivingPackages.length > 0 && driveCatalog.connected
  );

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
    capabilities
  };

  const options: WorkspaceOptions = {
    maps: normalizeMapOptions(driveCatalog.maps),
    vehicles: Array.isArray(driveCatalog.vehicles) ? driveCatalog.vehicles : [],
    weatherPresets: Array.isArray(driveCatalog.weather_presets)
      ? driveCatalog.weather_presets
      : [],
    propPresets: Array.isArray(driveCatalog.prop_presets) ? driveCatalog.prop_presets : [],
    detectorWeights: Array.isArray(bootstrap.catalog.weights) ? bootstrap.catalog.weights : [],
    checkpoints: Array.isArray(driveCatalog.policy_checkpoints)
      ? driveCatalog.policy_checkpoints
      : [],
    scenarioSuites: Array.isArray(bootstrap.catalog.scenario_suites)
      ? bootstrap.catalog.scenario_suites
      : [],
    splitPlans: Array.isArray(bootstrap.catalog.split_plans)
      ? bootstrap.catalog.split_plans
      : [],
    models: Array.isArray(modelRegistry.packages) ? modelRegistry.packages : [],
    invalidModels: Array.isArray(modelRegistry.invalid) ? modelRegistry.invalid : []
  };

  return {
    token: bootstrap.token,
    system,
    options,
    sessionDefaults: configuration.sessionDefaults,
    experimentPresets: configuration.experimentPresets,
    driveCatalog,
    driveState
  };
}

export class OperatorApi {
  constructor(private readonly token: string) {}

  getDriveState(): Promise<DriveState> {
    return readJson<DriveState>('/api/drive/state');
  }

  async post<T>(path: string, body: Record<string, unknown>, keepalive = false): Promise<T> {
    const response = await fetch(path, {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
        'X-Operator-Token': this.token
      },
      body: JSON.stringify(body),
      keepalive
    });
    const payload: unknown = await response.json().catch(() => null);
    if (!response.ok) throw apiError(path, response, payload);
    return payload as T;
  }

  startSession(session: SessionConfig): Promise<DriveState> {
    return this.post<DriveState>('/api/session/start', {
      schema_version: '1.0',
      session
    });
  }

  stopSession(sessionId: string): Promise<DriveState> {
    return this.post<DriveState>('/api/drive/stop', { session_id: sessionId });
  }

  emergencyStop(sessionId: string): Promise<DriveState> {
    return this.post<DriveState>('/api/drive/emergency-stop', { session_id: sessionId });
  }

  setDriveMode(sessionId: string, mode: 'manual' | 'autopilot'): Promise<DriveState> {
    return this.post<DriveState>('/api/drive/mode', { session_id: sessionId, mode });
  }

  sendControl(control: DriveControlRequest, keepalive = false): Promise<DriveState> {
    return this.post<DriveState>('/api/drive/control', { ...control }, keepalive);
  }

  async configureGaragePreview(
    session: SessionConfig,
    lifecycle?: (operation: GaragePreviewOperation) => void
  ): Promise<GaragePreviewResponse> {
    const startPath = '/api/garage/preview/configure/start';
    const accepted = await this.post<GaragePreviewOperation>(startPath, {
      schema_version: '1.0',
      session
    });
    lifecycle?.(accepted);

    const terminal = await this.waitForGaragePreviewOperation(accepted.operation_id, lifecycle);
    if (terminal.status !== 'running' || !terminal.result) {
      const detail = terminal.error?.message || `Garage preview ended in ${terminal.status}`;
      throw new Error(detail);
    }

    const applied = { ...terminal.result };
    return {
      ...applied,
      configuration: {
        schema_version: '1.0',
        requested: structuredClone(session),
        resolved: { ...(accepted.resolved_config ?? {}) },
        applied: { ...applied }
      }
    } as GaragePreviewResponse;
  }

  private async waitForGaragePreviewOperation(
    operationId: string,
    lifecycle?: (operation: GaragePreviewOperation) => void
  ): Promise<GaragePreviewOperation> {
    const path = `/api/garage/preview/operations/${encodeURIComponent(operationId)}/events`;
    const response = await fetch(path, {
      headers: {
        Accept: 'text/event-stream',
        'X-Operator-Token': this.token
      },
      cache: 'no-store'
    });
    if (!response.ok) {
      const payload: unknown = await response.json().catch(() => null);
      throw apiError(path, response, payload);
    }
    if (!response.body) throw new Error('Garage preview lifecycle stream has no response body');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true }).replaceAll('\r\n', '\n');
        while (true) {
          const boundary = buffer.indexOf('\n\n');
          if (boundary < 0) break;
          const block = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          const data = block
            .split('\n')
            .filter((line) => line.startsWith('data:'))
            .map((line) => line.slice(5).trimStart())
            .join('\n');
          if (!data) continue;
          const operation = JSON.parse(data) as GaragePreviewOperation;
          lifecycle?.(operation);
          if (operation.status === 'running' || operation.status === 'failed') return operation;
        }
      }
    } finally {
      reader.releaseLock();
    }
    throw new Error('Garage preview lifecycle stream ended before a terminal state');
  }

  getGaragePreviewState(): Promise<GaragePreviewResponse> {
    return readJson<GaragePreviewResponse>('/api/garage/preview/state');
  }

  stopGaragePreview(): Promise<Record<string, unknown>> {
    return this.post<Record<string, unknown>>('/api/garage/preview/stop', {});
  }

  orbitGaragePreview(request: GarageOrbitRequest): Promise<Record<string, unknown>> {
    return this.post<Record<string, unknown>>('/api/garage/preview/orbit', { ...request });
  }

  saveSituation(
    session: SessionConfig,
    situation: SituationSettings
  ): Promise<SituationSaveResponse> {
    return this.post<SituationSaveResponse>('/api/situations', {
      schema_version: '1.0',
      session,
      situation
    });
  }

  startJob(kind: string, parameters: Record<string, unknown>): Promise<OperatorJobSnapshot> {
    return this.post<OperatorJobSnapshot>('/api/jobs', {
      schema_version: '1.0',
      kind,
      parameters
    });
  }

  getJob(jobId: string): Promise<OperatorJobSnapshot> {
    return readJson<OperatorJobSnapshot>(`/api/jobs/${encodeURIComponent(jobId)}`);
  }
}
