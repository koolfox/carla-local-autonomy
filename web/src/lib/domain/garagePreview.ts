import type { SessionConfig } from './config';

export interface GaragePreparationProgress {
  status: string;
  stage: string;
  requested: Record<string, unknown>;
  actual: Record<string, unknown>;
  history?: Array<Record<string, unknown>>;
  elapsed_seconds: number;
  error?: Record<string, unknown> | null;
}

function progressNumber(record: Record<string, unknown>, key: string): number | null {
  const raw = record[key];
  if (raw === null || raw === undefined || raw === '') return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

export function garagePreparationStageLabel(stage: string): string {
  const labels: Record<string, string> = {
    accepted: 'Starting Garage…',
    configuring: 'Preparing CARLA…',
    map: 'Loading CARLA map…',
    world_settings: 'Applying world settings…',
    ego: 'Spawning ego vehicle…',
    props: 'Spawning scene props…',
    traffic: 'Spawning traffic…',
    walkers: 'Spawning walkers…',
    route: 'Planning route…',
    ready: 'Scene ready',
    running: 'Live CARLA',
    failed: 'Preparation failed'
  };
  return labels[stage] ?? 'Applying Garage settings…';
}

export function garagePreparationSummary(progress: GaragePreparationProgress): string {
  const requestedTraffic = progressNumber(progress.requested, 'traffic');
  const requestedWalkers = progressNumber(progress.requested, 'walkers');
  const actualTraffic = progressNumber(progress.actual, 'traffic');
  const actualWalkers = progressNumber(progress.actual, 'walkers');
  const crossing = progressNumber(progress.actual, 'pedestrian_crossing_factor')
    ?? progressNumber(progress.requested, 'pedestrian_crossing_factor');
  const pieces = [garagePreparationStageLabel(progress.stage)];
  if (requestedTraffic !== null && actualTraffic !== null) {
    pieces.push(`${actualTraffic}/${requestedTraffic} cars`);
  }
  if (requestedWalkers !== null && actualWalkers !== null) {
    pieces.push(`${actualWalkers}/${requestedWalkers} walkers`);
  }
  if (crossing !== null) pieces.push(`crossing ${crossing.toFixed(2)}`);
  const elapsed = Number(progress.elapsed_seconds);
  if (Number.isFinite(elapsed)) pieces.push(`${elapsed.toFixed(1)} s`);
  return pieces.join(' · ');
}

/** Keep this projection aligned with build_garage_preview_request on the Operator. */
export function garagePreviewSignature(session: SessionConfig): string {
  const { scene, camera, vehicle } = session;
  const [width, height] = camera.resolution.trim().toLowerCase().split('x').map(Number);
  const profile = width === 640 && height === 384 && camera.fps <= 10
    ? 'compatibility'
    : width === 1280 && height === 720 && camera.fps >= 60
      ? 'high-refresh'
      : width === 1920 && height === 1080 ? 'detail' : 'balanced';
  return JSON.stringify({
    map_name: scene.mapName.trim().replace(/\/+$/, '').split('/').at(-1),
    weather_preset: scene.weatherPreset.trim(),
    vehicle_blueprint: vehicle.blueprint.trim(),
    color: vehicle.color.trim(),
    seed: session.identity.seed,
    traffic_count: scene.trafficCount,
    walker_count: scene.walkerCount,
    prop_preset: scene.propPreset.trim(),
    route_mode: session.route.mode,
    pedestrian_crossing_factor: scene.pedestrianCrossingFactor,
    speed_difference_percent: scene.speedDifferencePercent,
    following_distance_metres: scene.followingDistanceMetres,
    spectator_mirror: camera.spectatorFollow,
    profile,
    fov: camera.fov
  });
}

export function garagePreviewInputError(session: SessionConfig): string {
  const integer = (value: number, minimum: number, maximum: number): boolean =>
    Number.isInteger(value) && value >= minimum && value <= maximum;
  const bounded = (value: number, minimum: number, maximum: number): boolean =>
    Number.isFinite(value) && value >= minimum && value <= maximum;
  if (!integer(session.identity.seed, 0, Number.MAX_SAFE_INTEGER)) {
    return 'Seed must be a non-negative whole number.';
  }
  if (!integer(session.scene.trafficCount, 0, 250) || !integer(session.scene.walkerCount, 0, 250)) {
    return 'Traffic and walker counts must be whole numbers from 0 to 250.';
  }
  if (!bounded(session.scene.pedestrianCrossingFactor, 0, 1)
    || !bounded(session.scene.speedDifferencePercent, -100, 100)
    || !bounded(session.scene.followingDistanceMetres, 0.1, 20)) {
    return 'Complete the traffic dynamics settings within their allowed ranges.';
  }
  const [width, height] = session.camera.resolution.trim().toLowerCase().split('x').map(Number);
  if (!integer(width, 320, 3840) || !integer(height, 180, 2160)
    || !bounded(session.camera.fps, 1, 60) || !bounded(session.camera.fov, 30, 150)) {
    return 'Complete the camera settings within their allowed ranges.';
  }
  return '';
}

/** Only transport failures and explicit temporary contention are safe to retry. */
export function isTransientGarageError(error: unknown): boolean {
  if (error instanceof TypeError) return true; // fetch network failure
  if (!(error instanceof Error)) return false;
  const status = 'status' in error ? Number(error.status) : 0;
  if ([408, 429, 502, 503, 504].includes(status)) return true;
  if (status === 400 || status === 401 || status === 403 || status === 404) return false;
  return /timed? out|timeout|connection (?:refused|reset|aborted)|temporarily unavailable|\bbusy\b|in progress|failed to fetch|networkerror|network request failed/i.test(error.message);
}

interface PreviewSelection {
  session: SessionConfig;
  signature: string;
}

interface GarageApplyOptions<Result> {
  apply: (session: SessionConfig) => Promise<Result>;
  applied: (response: Result, signature: string) => void;
  failed: (error: unknown) => void;
  busy: (value: boolean) => void;
  retrying: (value: boolean) => void;
  canApply?: () => boolean;
  debounceMs?: number;
  retryDelays?: number[];
}

/** One request at a time; edits replace the pending snapshot, never append to a queue. */
export function createGarageApplyQueue<Result>(options: GarageApplyOptions<Result>) {
  const retryDelays = options.retryDelays ?? [1000, 2000, 4000];
  let desired: PreviewSelection | null = null;
  let appliedSignature = '';
  let scope = '';
  let generation = 0;
  let disposed = false;
  let inFlight = false;
  let ready = false;
  let retries = 0;
  let timer: ReturnType<typeof setTimeout> | null = null;

  function cancelTimer(): void {
    if (timer !== null) clearTimeout(timer);
    timer = null;
    ready = false;
    if (!disposed) options.retrying(false);
  }

  function schedule(delay: number, retry = false): void {
    cancelTimer();
    options.retrying(retry);
    timer = setTimeout(() => {
      timer = null;
      options.retrying(false);
      ready = true;
      void drain();
    }, delay);
  }

  async function drain(): Promise<void> {
    if (disposed || !scope || !ready || inFlight || !desired || options.canApply?.() === false) return;
    ready = false;
    if (desired.signature === appliedSignature) return;
    const request = desired;
    const revision = generation;
    inFlight = true;
    options.busy(true);
    try {
      const response = await options.apply(request.session);
      if (disposed || revision !== generation) return;
      appliedSignature = request.signature;
      retries = 0;
      options.applied(response, request.signature);
    } catch (error) {
      if (disposed || revision !== generation) return;
      options.failed(error);
      if (desired?.signature === request.signature && isTransientGarageError(error)
        && retries < retryDelays.length) {
        schedule(retryDelays[retries++], true);
      }
    } finally {
      inFlight = false;
      if (!disposed) {
        options.busy(false);
        // A newer selection may have finished its debounce during this request.
        if (ready) void drain();
      }
    }
  }

  return {
    select(session: SessionConfig, nextScope: string, valid = true): void {
      if (disposed) return;
      if (scope !== nextScope) {
        scope = nextScope;
        generation += 1;
        desired = null;
        appliedSignature = '';
        cancelTimer();
      }
      if (!scope) return;
      if (!valid) {
        desired = null;
        cancelTimer();
        return;
      }
      const signature = garagePreviewSignature(session);
      if (desired?.signature === signature) return;
      desired = { session: structuredClone(session), signature };
      retries = 0;
      schedule(options.debounceMs ?? 300);
    },
    retry(): void {
      if (disposed || !scope || !desired) return;
      cancelTimer();
      retries = 0;
      ready = true;
      void drain();
    },
    dispose(): void {
      cancelTimer();
      disposed = true;
      generation += 1;
      desired = null;
    }
  };
}
