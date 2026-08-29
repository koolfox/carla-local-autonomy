import type { SessionConfig, SystemSettings, WorkspaceOptions } from '$lib/domain/config';

export interface SessionIssue {
  id: string;
  severity: 'error' | 'warning';
  message: string;
}

function capability(system: SystemSettings, name: string): boolean {
  return Boolean(system.capabilities[name]);
}

export function validateSession(
  session: SessionConfig,
  system: SystemSettings | null,
  options: WorkspaceOptions
): SessionIssue[] {
  const issues: SessionIssue[] = [];
  if (!system) {
    return [{ id: 'system-loading', severity: 'error', message: 'Runtime status is not loaded yet.' }];
  }

  if (!system.connected && !system.workerConnected) {
    issues.push({
      id: 'carla-offline',
      severity: 'error',
      message: 'CARLA is not reachable from the Operator.'
    });
  }

  if (!session.vehicle.blueprint) {
    issues.push({
      id: 'vehicle-required',
      severity: 'error',
      message: options.vehicles.length
        ? 'Select a vehicle before starting.'
        : 'No compatible vehicle is available in the current catalog.'
    });
  }

  if (session.control.mode === 'autopilot' && (!system.workerConnected || !capability(system, 'autopilot'))) {
    issues.push({
      id: 'autopilot-unavailable',
      severity: 'error',
      message: 'Traffic Manager Autopilot requires a connected capable World Worker.'
    });
  }

  const experimentalModes = ['behavior', 'imitation', 'voxel'];
  if (experimentalModes.includes(session.control.mode)) {
    if (!system.experimentalEnabled) {
      issues.push({
        id: 'experimental-disabled',
        severity: 'error',
        message: 'This autonomy mode requires the Operator experimental feature gate.'
      });
    }
    if (!session.policy.acknowledgeAutonomy) {
      issues.push({
        id: 'autonomy-ack',
        severity: 'error',
        message: 'Autonomous actuation requires explicit operator acknowledgement.'
      });
    }
  }

  const modeCapability: Record<string, string> = {
    behavior: 'garage_behavior_drive',
    imitation: 'garage_imitation_drive',
    voxel: 'garage_voxel_drive'
  };
  const requiredCapability = modeCapability[session.control.mode];
  if (requiredCapability && !capability(system, requiredCapability)) {
    issues.push({
      id: 'mode-unavailable',
      severity: 'error',
      message: `${session.control.mode} driving is unavailable in the current runtime.`
    });
  }

  if (['imitation', 'voxel'].includes(session.control.mode) && !session.policy.checkpoint) {
    issues.push({
      id: 'checkpoint-required',
      severity: 'error',
      message: 'Select a policy checkpoint for this driving mode.'
    });
  }

  if (session.perception.enabled) {
    if (!system.visionRuntimeAvailable) {
      issues.push({
        id: 'vision-runtime',
        severity: 'error',
        message: 'Detection is enabled but the vision runtime is unavailable.'
      });
    } else if (!session.perception.weights) {
      issues.push({
        id: 'detector-weights',
        severity: 'error',
        message: 'Select detector weights or turn the detection overlay off.'
      });
    }
  }

  if (!system.workerConnected) {
    if (session.scene.mapName !== 'current') {
      issues.push({
        id: 'worker-map',
        severity: 'error',
        message: 'Map selection requires the World Worker.'
      });
    }
    if (session.route.mode !== 'free') {
      issues.push({
        id: 'worker-route',
        severity: 'error',
        message: 'Route selection requires the World Worker.'
      });
    }
    if (
      session.scene.pedestrianCrossingFactor !== 0.2 ||
      session.scene.speedDifferencePercent !== 12 ||
      session.scene.followingDistanceMetres !== 2
    ) {
      issues.push({
        id: 'worker-dynamics',
        severity: 'error',
        message: 'Custom traffic dynamics require the World Worker.'
      });
    }
  }

  if (session.scene.trafficCount > 0 && !system.workerConnected && !capability(system, 'garage_traffic_population')) {
    issues.push({
      id: 'traffic-unavailable',
      severity: 'error',
      message: 'Traffic population is unavailable in the current runtime.'
    });
  }

  if (session.scene.walkerCount > 0 && !system.workerConnected && !capability(system, 'garage_walker_population')) {
    issues.push({
      id: 'walkers-unavailable',
      severity: 'error',
      message: 'Pedestrian population is unavailable in the current runtime.'
    });
  }

  return issues;
}

export function blockingIssues(issues: SessionIssue[]): SessionIssue[] {
  return issues.filter((issue) => issue.severity === 'error');
}
