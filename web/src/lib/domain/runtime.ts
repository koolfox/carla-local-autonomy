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
  sequence?: number;
  actuated: false;
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
