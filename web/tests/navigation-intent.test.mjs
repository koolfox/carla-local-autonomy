import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  navigationCommandLabel,
  navigationIntentFromDrive
} from '../src/lib/domain/runtime.ts';

function driveWithIntent(intent) {
  return {
    status: 'running',
    autonomy: {
      detail: {
        navigation_intent: intent
      }
    }
  };
}

test('reads exact privileged NavigationIntent from live drive detail', () => {
  const intent = {
    schema_version: '1.0',
    source_frame: { kind: 'carla_world_frame', id: 1234, exact: true },
    command: 'left',
    direction: { forward: 0.8, right: -0.6 },
    target_point: { forward_m: 14, right_m: -7 },
    distance_to_maneuver_m: 15.7,
    route_id: 'live-behavior-route-0001-d014',
    source: 'carla_global_route_planner_via_behavior_agent',
    confidence: 1,
    privileged: true
  };

  assert.deepEqual(navigationIntentFromDrive(driveWithIntent(intent)), intent);
});

test('rejects malformed or non-exact runtime route metadata', () => {
  assert.equal(navigationIntentFromDrive({ status: 'running' }), null);
  assert.equal(
    navigationIntentFromDrive(
      driveWithIntent({ source_frame: { id: 1, exact: false }, command: 'left', privileged: true })
    ),
    null
  );
  assert.equal(
    navigationIntentFromDrive(
      driveWithIntent({ source_frame: { id: -1, exact: true }, command: 'left', privileged: true })
    ),
    null
  );
  assert.equal(
    navigationIntentFromDrive(
      driveWithIntent({ source_frame: { id: 1, exact: true }, command: '', privileged: true })
    ),
    null
  );
});

test('uses stable human labels for CARLA RoadOption-derived commands', () => {
  assert.equal(navigationCommandLabel('follow_lane'), 'Follow lane');
  assert.equal(navigationCommandLabel('left'), 'Left');
  assert.equal(navigationCommandLabel('right'), 'Right');
  assert.equal(navigationCommandLabel('straight'), 'Straight');
  assert.equal(navigationCommandLabel('change_lane_left'), 'Change lane left');
  assert.equal(navigationCommandLabel('change_lane_right'), 'Change lane right');
  assert.equal(navigationCommandLabel('stop'), 'Stop');
});
