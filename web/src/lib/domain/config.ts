export type ControlMode = 'manual' | 'autopilot' | 'behavior' | 'imitation' | 'voxel' | 'model';
export type RouteMode = 'free' | 'random_destination' | 'selected_destination';
export type DetectorKind = 'rtdetr' | 'yolo' | 'm9-hierarchical';
export type BehaviorStyle = 'cautious' | 'normal' | 'aggressive';
export type ExperimentPreset =
  | 'free_drive'
  | 'manual_handling'
  | 'autopilot_takeover'
  | 'perception_review'
  | 'traffic_stress'
  | 'adverse_weather';

export interface SystemSettings {
  workspace: string;
  carlaHost: string;
  carlaPort: number;
  localOnly: boolean;
  connected: boolean;
  serverVersion: string | null;
  currentMap: string | null;
  workerConfigured: boolean;
  workerConnected: boolean;
  workerUrl: string | null;
  experimentalEnabled: boolean;
  visionRuntimeAvailable: boolean;
  capabilities: Record<string, boolean>;
}

export interface ModelPackage {
  objectType: 'runtime_model_package';
  id: string;
  name: string;
  version: string;
  role: 'detector' | 'driving_policy' | 'scene_perception' | 'perception_guard';
  runtime: 'ultralytics' | 'python_factory' | 'torchscript_control_v1';
  artifact: string;
  expectedSha256: string | null;
  factory: string | null;
  devices: string[];
  inputs: Record<string, unknown>;
  outputs: Record<string, unknown>;
  labels: Record<string, unknown> | unknown[];
  source: string | null;
  manifestPath: string;
  requiresTrustedCode: boolean;
}

export interface InvalidModelPackage {
  path: string;
  errorType: string;
  message: string;
}

export interface CatalogOption {
  id: string;
  label: string;
}

export interface SessionConfig {
  identity: {
    runId: string;
    seed: number;
  };
  scene: {
    mapName: string;
    weatherPreset: string;
    propPreset: string;
    trafficCount: number;
    walkerCount: number;
    pedestrianCrossingFactor: number;
    speedDifferencePercent: number;
    followingDistanceMetres: number;
  };
  vehicle: {
    blueprint: string;
    color: string;
  };
  route: {
    mode: RouteMode;
    startSpawnIndex: number | null;
    destinationSpawnIndex: number | null;
  };
  control: {
    mode: ControlMode;
  };
  camera: {
    resolution: string;
    fps: number;
    fov: number;
    spectatorFollow: boolean;
  };
  perception: {
    enabled: boolean;
    voxelEnabled: boolean;
    roadEnabled: boolean;
    roadBackend: string;
    roadCheckpoint: string;
    roadDevice: string;
    signClassifier?: {
      checkpoint: string;
      ontology: string;
      confidence: number;
      crop_scale: number;
      show_rejection_status?: boolean;
    } | null;
    detector: DetectorKind;
    weights: string;
    device: string;
    imageSize: number;
    confidence: number;
  };
  recording: {
    video: boolean;
    cameraRig?: import('./capture').CaptureRig;
    cameraRigFps?: number;
    cameraPerception?: Record<string, import('./capture').CameraPerceptionMode>;
  };
  experiment: {
    preset: ExperimentPreset;
  };
  policy: {
    behavior: BehaviorStyle;
    acknowledgeAutonomy: boolean;
    acknowledgeTrustedCode: boolean;
    modelId: string;
    checkpoint: string;
    device: string;
    voxelReadinessReport: string;
    targetSpeedKmh: number;
    maxPolicyErrors: number;
    maxModelSpeedKmh: number;
    maxSteerRate: number;
  };
}

export interface ExperimentPresetPatch {
  control?: { mode?: ControlMode };
  scene?: {
    trafficCount?: number;
    walkerCount?: number;
    trafficCountMinimum?: number;
    walkerCountMinimum?: number;
    weatherPreset?: string;
    weatherPresetIfKeep?: string;
  };
  perception?: { enabled?: boolean };
  recording?: { video?: boolean };
}

export interface ExperimentPresetDefinition {
  id: ExperimentPreset;
  label: string;
  description: string;
  patch: ExperimentPresetPatch;
}

export interface SpawnPointOption {
  index: number;
  label: string;
  transform?: Record<string, unknown>;
}

export interface WorkspaceOptions {
  maps: CatalogOption[];
  spawnPointMap: string | null;
  spawnPoints: SpawnPointOption[];
  vehicles: Array<{ id: string; label?: string; colors?: string[] }>;
  weatherPresets: Array<{ id: string; label: string }>;
  propPresets: Array<{ id: string; label: string }>;
  detectorWeights: string[];
  checkpoints: string[];
  scenarioSuites: string[];
  splitPlans: string[];
  models: ModelPackage[];
  invalidModels: InvalidModelPackage[];
}

export function newRunId(): string {
  return `drive-${new Date().toISOString().replace(/[-:.]/g, '').replace('Z', 'z').toLowerCase()}`;
}

export function defaultSessionConfig(): SessionConfig {
  return {
    identity: {
      runId: newRunId(),
      seed: 7
    },
    scene: {
      mapName: 'current',
      weatherPreset: 'keep',
      propPreset: 'none',
      trafficCount: 0,
      walkerCount: 0,
      pedestrianCrossingFactor: 0.2,
      speedDifferencePercent: 12,
      followingDistanceMetres: 2
    },
    vehicle: {
      blueprint: '',
      color: ''
    },
    route: {
      mode: 'free',
      startSpawnIndex: null,
      destinationSpawnIndex: null
    },
    control: {
      mode: 'autopilot'
    },
    camera: {
      resolution: '1280x720',
      fps: 30,
      fov: 90,
      spectatorFollow: true
    },
    perception: {
      enabled: false,
      voxelEnabled: false,
      roadEnabled: false,
      roadBackend: 'segformer',
      roadCheckpoint: '',
      roadDevice: 'cpu',
      signClassifier: null,
      detector: 'rtdetr',
      weights: '',
      device: 'cpu',
      imageSize: 640,
      confidence: 0.5
    },
    recording: {
      video: true
    },
    experiment: {
      preset: 'free_drive'
    },
    policy: {
      behavior: 'normal',
      acknowledgeAutonomy: false,
      acknowledgeTrustedCode: false,
      modelId: '',
      checkpoint: '',
      device: 'cpu',
      voxelReadinessReport: '',
      targetSpeedKmh: 35,
      maxPolicyErrors: 3,
      maxModelSpeedKmh: 45,
      maxSteerRate: 2.5
    }
  };
}

export function sessionForApi(session: SessionConfig, preview = false): SessionConfig {
  const perception = { ...session.perception };
  // Preview owns the parked world, not inference. Omit disabled optional fields
  // so a freshly rebuilt UI also works while the old Operator awaits restart.
  if (preview || perception.signClassifier == null) delete perception.signClassifier;
  const recording = { ...session.recording };
  if (preview || !Object.keys(recording.cameraPerception ?? {}).length) delete recording.cameraPerception;
  return { ...session, perception, recording };
}

export function mergeSessionDefaults(
  fallback: SessionConfig,
  defaults: Partial<SessionConfig>
): SessionConfig {
  return {
    ...fallback,
    ...defaults,
    identity: { ...fallback.identity, ...(defaults.identity ?? {}), runId: fallback.identity.runId },
    scene: { ...fallback.scene, ...(defaults.scene ?? {}) },
    vehicle: { ...fallback.vehicle, ...(defaults.vehicle ?? {}) },
    route: { ...fallback.route, ...(defaults.route ?? {}) },
    control: { ...fallback.control, ...(defaults.control ?? {}) },
    camera: { ...fallback.camera, ...(defaults.camera ?? {}) },
    perception: { ...fallback.perception, ...(defaults.perception ?? {}) },
    recording: { ...fallback.recording, ...(defaults.recording ?? {}) },
    experiment: { ...fallback.experiment, ...(defaults.experiment ?? {}) },
    policy: { ...fallback.policy, ...(defaults.policy ?? {}) }
  };
}

export function applyExperimentPresetDefinition(
  current: SessionConfig,
  definition: ExperimentPresetDefinition
): SessionConfig {
  const patch = definition.patch;
  const next: SessionConfig = {
    ...current,
    scene: { ...current.scene },
    control: { ...current.control },
    perception: { ...current.perception },
    recording: { ...current.recording },
    experiment: { preset: definition.id },
    policy: { ...current.policy }
  };

  if (patch.control?.mode) next.control.mode = patch.control.mode;
  if (typeof patch.scene?.trafficCount === 'number') {
    next.scene.trafficCount = patch.scene.trafficCount;
  }
  if (typeof patch.scene?.walkerCount === 'number') {
    next.scene.walkerCount = patch.scene.walkerCount;
  }
  if (typeof patch.scene?.trafficCountMinimum === 'number') {
    next.scene.trafficCount = Math.max(next.scene.trafficCount, patch.scene.trafficCountMinimum);
  }
  if (typeof patch.scene?.walkerCountMinimum === 'number') {
    next.scene.walkerCount = Math.max(next.scene.walkerCount, patch.scene.walkerCountMinimum);
  }
  if (patch.scene?.weatherPreset) next.scene.weatherPreset = patch.scene.weatherPreset;
  if (patch.scene?.weatherPresetIfKeep && next.scene.weatherPreset === 'keep') {
    next.scene.weatherPreset = patch.scene.weatherPresetIfKeep;
  }
  if (typeof patch.perception?.enabled === 'boolean') {
    next.perception.enabled = patch.perception.enabled;
  }
  if (typeof patch.recording?.video === 'boolean') {
    next.recording.video = patch.recording.video;
  }

  return next;
}
