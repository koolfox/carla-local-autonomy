import type { SessionConfig } from './config';

export type CameraMount = { x: number; y: number; z: number; pitch: number; yaw: number; roll: number };
export type CameraView = { mount: CameraMount; fov_degrees: number };
export type AdditionalCamera = CameraView & { id: string };
export interface CaptureRig {
  schema_version: '1.0';
  primary_view?: CameraView;
  additional_views: AdditionalCamera[];
}
export type RigSelection = 'front' | 'front-three' | CaptureRig;
export interface CaptureSettings {
  rig: CaptureRig;
  durationSeconds: number;
  captureFps: 1 | 2 | 5 | 10;
  repetitions: number;
}

export function frontCamera(fov: number): CameraView {
  return { mount: { x: 1.5, y: 0, z: 1.7, pitch: 0, yaw: 0, roll: 0 }, fov_degrees: fov };
}

export function newCamera(id: string, fov: number): AdditionalCamera {
  const view = frontCamera(fov);
  if (id === 'rear') { view.mount.x = -1.5; view.mount.yaw = 180; }
  if (id === 'front_left') view.mount.yaw = -60;
  if (id === 'front_right') view.mount.yaw = 60;
  return { id, ...view };
}

export function presetRig(preset: string, fov: number): CaptureRig {
  const names = preset === 'front-rear' ? ['rear']
    : preset === 'front-three' ? ['front_left', 'front_right']
    : preset === 'surround' ? ['front_left', 'front_right', 'rear'] : [];
  return { schema_version: '1.0', additional_views: names.map((id) => newCamera(id, fov)) };
}

export function cameraRigError(rig: CaptureRig): string | null {
  if (!rig || rig.schema_version !== '1.0' || !Array.isArray(rig.additional_views)) return 'Invalid camera rig.';
  if (rig.additional_views.length > 7) return 'Use at most 8 RGB cameras, including front.';
  const names = new Set(['front', 'front_teacher']);
  for (const view of rig.additional_views) {
    if (!view || typeof view.id !== 'string' || !/^[a-z][a-z0-9_]{0,31}$/.test(view.id) || names.has(view.id)) {
      return 'Each additional camera needs a unique ID; front is reserved.';
    }
    names.add(view.id);
  }
  const views = [...rig.additional_views, ...(rig.primary_view ? [rig.primary_view] : [])];
  for (const view of views) {
    if (!view.mount || !['x', 'y', 'z', 'pitch', 'yaw', 'roll'].every((key) => Number.isFinite(view.mount[key as keyof CameraMount]))) {
      return 'Enter a finite position and angle for every camera.';
    }
    if (['pitch', 'yaw', 'roll'].some((key) => Math.abs(view.mount[key as keyof CameraMount]) > 360)) return 'Camera angles must be between −360° and 360°.';
    if (!Number.isFinite(view.fov_degrees) || view.fov_degrees < 30 || view.fov_degrees > 150) return 'Camera FOV must be between 30° and 150°.';
  }
  return null;
}

export function captureSettingsError(settings: CaptureSettings): string | null {
  if (!settings || !Number.isInteger(settings.durationSeconds) || settings.durationSeconds < 5 || settings.durationSeconds > 3600) return 'Capture duration must be 5–3600 whole seconds.';
  if (![1, 2, 5, 10].includes(settings.captureFps)) return 'Choose 1, 2, 5 or 10 capture FPS.';
  if (!Number.isInteger(settings.repetitions) || settings.repetitions < 1 || settings.repetitions > 32) return 'Choose 1–32 episodes.';
  return cameraRigError(settings.rig);
}

export function captureSituation(session: SessionConfig, settings: CaptureSettings) {
  return {
    situationId: `scene-${Date.now()}`,
    egoSpawnIndex: session.route.startSpawnIndex ?? 0,
    durationSeconds: settings.durationSeconds,
    captureFps: settings.captureFps,
    repetitions: settings.repetitions
  };
}
