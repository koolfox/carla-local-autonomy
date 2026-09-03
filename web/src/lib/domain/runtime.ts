export type DriveStatus = 'idle' | 'starting' | 'running' | 'stopping' | 'success' | 'failed';

export interface DriveTelemetry {
  speed?: number;
  speed_mps?: number;
  gear?: number;
  throttle?: number;
  steer?: number;
  brake?: number;
}

export interface DriveStreamState {
  transport?: string;
  resolution?: string;
  target_fps?: number;
  source_fps?: number;
  frame_age_seconds?: number | null;
  stale?: boolean;
  overlay_fps?: number;
  overlay_age_seconds?: number | null;
}

export interface DriveDetectorState {
  enabled?: boolean;
  name?: string | null;
  advisory_only?: boolean;
  actuated?: boolean;
}

export interface DriveVoxelState {
  enabled: boolean;
  status: 'loading' | 'running' | 'failed' | 'stopped';
  error?: string | null;
  latency_ms?: number | null;
  source_frame?: number | null;
  waypoint_status?: 'pending' | 'available' | 'empty' | 'unavailable' | 'error';
  waypoint_source?: string | null;
  waypoint_error?: string | null;
  sequence?: number;
  actuated: false;
}

export interface DriveNavigationIntent {
  schema_version?: string;
  source_frame: {
    kind?: string;
    id: number;
    exact: true;
  };
  command: string;
  direction?: {
    forward?: number;
    right?: number;
  };
  target_point?: {
    forward_m?: number;
    right_m?: number;
  };
  distance_to_maneuver_m?: number;
  route_id?: string;
  source?: string;
  confidence?: number;
  privileged: boolean;
}

export interface DriveAutonomyState {
  enabled?: boolean;
  operator_acknowledged?: boolean;
  model_output_actuated?: boolean;
  policy_ready?: boolean;
  commands?: number;
  failsafes?: number;
  detail?: {
    navigation_intent?: unknown;
    navigation_intent_error?: unknown;
    [key: string]: unknown;
  };
  traffic_vehicle_count?: number;
  walker_count?: number;
  population_error?: string | null;
}

export interface DriveState {
  schema_version?: string;
  status: DriveStatus | string;
  session_id?: string | null;
  run_id?: string | null;
  map?: string | null;
  server_version?: string | null;
  garage_mode?: string;
  control_mode?: string;
  control_source?: string;
  deadman_active?: boolean;
  emergency_stop?: boolean;
  recording?: boolean | { active?: boolean };
  output_path?: string | null;
  error?: string | null;
  stop_reason?: string | null;
  elapsed_seconds?: number;
  telemetry?: DriveTelemetry;
  stream?: DriveStreamState;
  detector?: DriveDetectorState;
  voxel?: DriveVoxelState;
  autonomy?: DriveAutonomyState;
  traffic_count_actual?: number;
  walker_count_actual?: number;
  raw_frame_sequence?: number;
  overlay_frame_sequence?: number;
  input_age_seconds?: number | null;
  [key: string]: unknown;
}

export interface DriveControlRequest {
  session_id: string;
  sequence: number;
  throttle: number;
  steer: number;
  brake: number;
  hand_brake: boolean;
  reverse: boolean;
}

export interface GarageOrbitRequest {
  sequence: number;
  yaw: number;
  pitch: number;
  distance: number;
  preset: 'orbit' | 'front' | 'rear' | 'top' | 'cockpit';
}

export function isDriveActive(state: DriveState | null | undefined): boolean {
  return Boolean(state && ['starting', 'running', 'stopping'].includes(state.status));
}

export function isDriveRunning(state: DriveState | null | undefined): boolean {
  return state?.status === 'running';
}

export function driveStatusLabel(status: string): string {
  return (
    {
      idle: 'Ready',
      starting: 'Starting',
      running: 'Driving',
      stopping: 'Saving',
      success: 'Saved',
      failed: 'Failed'
    } as Record<string, string>
  )[status] ?? status;
}

export function speedMetresPerSecond(state: DriveState | null | undefined): number {
  const value = state?.telemetry?.speed_mps ?? state?.telemetry?.speed ?? 0;
  return Number.isFinite(Number(value)) ? Number(value) : 0;
}

export function recordingActive(state: DriveState | null | undefined): boolean {
  const value = state?.recording;
  return typeof value === 'object' && value !== null ? Boolean(value.active) : Boolean(value);
}

export function navigationIntentFromDrive(
  state: DriveState | null | undefined
): DriveNavigationIntent | null {
  const value = state?.autonomy?.detail?.navigation_intent;
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return null;
  const raw = value as Record<string, unknown>;
  const frame = raw.source_frame;
  if (typeof frame !== 'object' || frame === null || Array.isArray(frame)) return null;
  const sourceFrame = frame as Record<string, unknown>;
  const frameId = sourceFrame.id;
  if (
    typeof raw.command !== 'string' ||
    raw.command.trim() === '' ||
    typeof frameId !== 'number' ||
    !Number.isInteger(frameId) ||
    frameId < 0 ||
    sourceFrame.exact !== true ||
    typeof raw.privileged !== 'boolean'
  ) {
    return null;
  }
  return raw as unknown as DriveNavigationIntent;
}

export function navigationCommandLabel(command: string): string {
  return (
    {
      follow_lane: 'Follow lane',
      left: 'Left',
      right: 'Right',
      straight: 'Straight',
      change_lane_left: 'Change lane left',
      change_lane_right: 'Change lane right',
      stop: 'Stop'
    } as Record<string, string>
  )[command] ?? command;
}
