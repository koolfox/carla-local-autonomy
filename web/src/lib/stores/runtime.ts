import { get, writable } from 'svelte/store';

import { OperatorApi, type CaptureState, type SituationSettings } from '$lib/api/operator';
import type { SessionConfig } from '$lib/domain/config';
import type { RigSelection } from '$lib/domain/capture';
import { isDriveActive, type DriveControlRequest, type DriveState } from '$lib/domain/runtime';

export type RuntimeAction = 'start' | 'stop' | 'emergency' | 'takeover' | null;

export interface GarageRuntimeState {
  drive: DriveState;
  action: RuntimeAction;
  error: string | null;
  lastUpdatedAt: number | null;
}

const idleDrive: DriveState = {
  status: 'idle',
  session_id: null,
  run_id: null,
  error: null
};

export const garageRuntime = writable<GarageRuntimeState>({
  drive: idleDrive,
  action: null,
  error: null,
  lastUpdatedAt: null
});

export const captureRuntime = writable<CaptureState>({
  phase: 'idle', active: false, holds_world: false, available: false, job_id: null, error: null
});

let api: OperatorApi | null = null;
let pollTimer: ReturnType<typeof setTimeout> | null = null;
let polling = false;

function runtimeApi(): OperatorApi {
  if (!api) throw new Error('Operator runtime is not initialized.');
  return api;
}

function setDrive(drive: DriveState): void {
  garageRuntime.update((current) => ({
    ...current,
    drive,
    error: null,
    lastUpdatedAt: Date.now()
  }));
}

function setAction(action: RuntimeAction): void {
  garageRuntime.update((current) => ({ ...current, action }));
}

function setError(error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  garageRuntime.update((current) => ({ ...current, error: message }));
}

export function initializeRuntime(token: string, initialDrive: DriveState, capture?: CaptureState): void {
  api = new OperatorApi(token);
  setDrive(initialDrive);
  if (capture) captureRuntime.set(capture);
}

export async function refreshRuntime(): Promise<void> {
  try {
    const [drive, capture] = await Promise.all([
      runtimeApi().getDriveState(), runtimeApi().getCaptureState()
    ]);
    setDrive(drive);
    captureRuntime.set(capture);
  } catch (error) {
    setError(error);
  }
}

function nextPollDelay(): number {
  return isDriveActive(get(garageRuntime).drive) || get(captureRuntime).active ? 700 : 3000;
}

async function pollOnce(): Promise<void> {
  if (!polling) return;
  await refreshRuntime();
  if (!polling) return;
  pollTimer = setTimeout(() => void pollOnce(), nextPollDelay());
}

export function beginRuntimePolling(): void {
  if (polling) return;
  polling = true;
  pollTimer = setTimeout(() => void pollOnce(), nextPollDelay());
}

export function endRuntimePolling(): void {
  polling = false;
  if (pollTimer) clearTimeout(pollTimer);
  pollTimer = null;
}

export async function startDrive(session: SessionConfig): Promise<void> {
  setAction('start');
  garageRuntime.update((current) => ({ ...current, error: null }));
  try {
    setDrive(await runtimeApi().startSession(session));
  } catch (error) {
    setError(error);
    throw error;
  } finally {
    setAction(null);
  }
}

export async function stopDrive(): Promise<void> {
  const sessionId = get(garageRuntime).drive.session_id;
  if (!sessionId) return;
  setAction('stop');
  try {
    setDrive(await runtimeApi().stopSession(sessionId));
  } catch (error) {
    setError(error);
    throw error;
  } finally {
    setAction(null);
  }
}

export async function emergencyStopDrive(): Promise<void> {
  const sessionId = get(garageRuntime).drive.session_id;
  if (!sessionId) return;
  setAction('emergency');
  try {
    setDrive(await runtimeApi().emergencyStop(sessionId));
  } catch (error) {
    setError(error);
    throw error;
  } finally {
    setAction(null);
  }
}

export async function takeManualControl(): Promise<boolean> {
  const drive = get(garageRuntime).drive;
  if (!drive.session_id || drive.status !== 'running') return false;
  if (drive.control_mode !== 'autopilot') return true;
  setAction('takeover');
  try {
    setDrive(await runtimeApi().setDriveMode(drive.session_id, 'manual'));
    return true;
  } catch (error) {
    setError(error);
    return false;
  } finally {
    setAction(null);
  }
}

export async function sendManualDriveControl(
  request: DriveControlRequest,
  keepalive = false
): Promise<void> {
  await runtimeApi().sendControl(request, keepalive);
}

export function runtimeOperatorApi(): OperatorApi {
  return runtimeApi();
}

export async function startCapture(session: SessionConfig, situation: SituationSettings, rig: RigSelection): Promise<void> {
  // Suspend auto-apply before the backend releases the preview for collection.
  captureRuntime.update((state) => ({ ...state, active: true, holds_world: true, phase: 'preparing', error: null }));
  try {
    captureRuntime.set(await runtimeApi().startCapture(session, situation, rig));
  } catch (error) {
    await refreshRuntime();
    throw error;
  }
}

export async function cancelCapture(): Promise<void> {
  captureRuntime.set(await runtimeApi().cancelCapture());
}
