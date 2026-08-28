export type ControlMode = 'manual' | 'autopilot' | 'behavior' | 'imitation' | 'voxel';
export type RouteMode = 'free' | 'random_destination';
export type DetectorKind = 'rtdetr' | 'yolo';
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
  experimentalEnabled: boolean;
  visionRuntimeAvailable: boolean;
  capabilities: Record<string, boolean>;
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
    detector: DetectorKind;
    weights: string;
    device: string;
    imageSize: number;
    confidence: number;
  };
  recording: {
    video: boolean;
  };
  experiment: {
    preset: ExperimentPreset;
  };
}

export interface WorkspaceOptions {
  maps: string[];
  vehicles: Array<{ id: string; label?: string; colors?: string[] }>;
  weatherPresets: Array<{ id: string; label: string }>;
  propPresets: Array<{ id: string; label: string }>;
  checkpoints: string[];
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
      mode: 'free'
    },
    control: {
      mode: 'manual'
    },
    camera: {
      resolution: '1280x720',
      fps: 30,
      fov: 90,
      spectatorFollow: true
    },
    perception: {
      enabled: true,
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
    }
  };
}

export function applyExperimentPreset(
  current: SessionConfig,
  preset: ExperimentPreset
): SessionConfig {
  const next: SessionConfig = {
    ...current,
    scene: { ...current.scene },
    control: { ...current.control },
    perception: { ...current.perception },
    recording: { ...current.recording },
    experiment: { preset }
  };

  switch (preset) {
    case 'free_drive':
      next.control.mode = 'manual';
      next.scene.trafficCount = 0;
      next.scene.walkerCount = 0;
      break;
    case 'manual_handling':
      next.control.mode = 'manual';
      next.perception.enabled = false;
      break;
    case 'autopilot_takeover':
      next.control.mode = 'autopilot';
      next.scene.trafficCount = Math.max(8, next.scene.trafficCount);
      next.scene.walkerCount = Math.max(4, next.scene.walkerCount);
      break;
    case 'perception_review':
      next.control.mode = 'manual';
      next.perception.enabled = true;
      next.recording.video = true;
      break;
    case 'traffic_stress':
      next.scene.trafficCount = Math.max(30, next.scene.trafficCount);
      next.scene.walkerCount = Math.max(20, next.scene.walkerCount);
      break;
    case 'adverse_weather':
      if (next.scene.weatherPreset === 'keep') next.scene.weatherPreset = 'hard-rain-noon';
      next.recording.video = true;
      break;
  }

  return next;
}
