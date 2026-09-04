import assert from 'node:assert/strict';
import { test } from 'node:test';

import { defaultSessionConfig } from '../src/lib/domain/config.ts';
import { fieldNumber } from '../src/lib/ui/events.ts';
import {
  createGarageApplyQueue,
  garagePreparationStageLabel,
  garagePreparationSummary,
  garagePreviewInputError,
  garagePreviewSignature,
  isTransientGarageError
} from '../src/lib/domain/garagePreview.ts';

function session(patch = {}) {
  const value = defaultSessionConfig();
  value.vehicle.blueprint = 'vehicle.tesla.model3';
  for (const [section, changes] of Object.entries(patch)) Object.assign(value[section], changes);
  return value;
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}

async function settle() {
  await new Promise((resolve) => setImmediate(resolve));
}

function harness(t, overrides = {}) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const requests = [];
  const applied = [];
  const errors = [];
  const busy = [];
  const retrying = [];
  const queue = createGarageApplyQueue({
    apply(value) {
      const result = deferred();
      requests.push({ session: value, ...result });
      return result.promise;
    },
    applied: (response, signature) => applied.push({ response, signature }),
    failed: (error) => errors.push(error),
    busy: (value) => busy.push(value),
    retrying: (value) => retrying.push(value),
    ...overrides
  });
  t.after(() => queue.dispose());
  return { queue, requests, applied, errors, busy, retrying };
}

test('dense preparation progress is factual and human-readable', () => {
  const progress = {
    status: 'preparing',
    stage: 'walkers',
    requested: { traffic: 64, walkers: 40, pedestrian_crossing_factor: 0.45 },
    actual: { traffic: 64, walkers: 24, pedestrian_crossing_factor: null },
    elapsed_seconds: 7.25
  };
  assert.equal(garagePreparationStageLabel('walkers'), 'Spawning walkers…');
  assert.equal(
    garagePreparationSummary(progress),
    'Spawning walkers… · 64/64 cars · 24/40 walkers · crossing 0.45 · 7.3 s'
  );
});


test('signature includes only the effective parked preview contract', () => {
  const baseline = session();
  const driveOnly = session({
    identity: { runId: 'next-run' },
    control: { mode: 'model' },
    perception: {
      enabled: true, voxelEnabled: true, detector: 'yolo', weights: 'models/custom.pt',
      device: 'mps', imageSize: 1280, confidence: 0.25
    },
    recording: { video: false },
    experiment: { preset: 'perception_review' },
    policy: {
      behavior: 'aggressive', acknowledgeAutonomy: true, acknowledgeTrustedCode: true,
      modelId: 'custom-model', checkpoint: 'model.pt', device: 'cuda',
      voxelReadinessReport: 'report.json', targetSpeedKmh: 50, maxPolicyErrors: 5,
      maxModelSpeedKmh: 60, maxSteerRate: 3
    }
  });
  assert.equal(garagePreviewSignature(driveOnly), garagePreviewSignature(baseline));
  const fields = {
    identity: { seed: 8 },
    scene: {
      mapName: 'Town04', weatherPreset: 'wet-night', propPreset: 'roadworks',
      trafficCount: 10, walkerCount: 15, pedestrianCrossingFactor: 0.5,
      speedDifferencePercent: 20, followingDistanceMetres: 4
    },
    vehicle: { blueprint: 'vehicle.audi.a2', color: '255,0,0' },
    route: { mode: 'selected_destination', startSpawnIndex: 2, destinationSpawnIndex: 7 },
    camera: { resolution: '1920x1080', fps: 60, fov: 100, spectatorFollow: false }
  };
  for (const [section, values] of Object.entries(fields)) {
    for (const [field, value] of Object.entries(values)) {
      assert.notEqual(
        garagePreviewSignature(session({ [section]: { [field]: value } })),
        garagePreviewSignature(baseline),
        `${section}.${field} must reach the preview`
      );
    }
  }
});

test('signature canonicalizes map paths and the bounded camera profile', () => {
  assert.equal(
    garagePreviewSignature(session({ scene: { mapName: ' /Game/Carla/Maps/Town04/ ' } })),
    garagePreviewSignature(session({ scene: { mapName: 'Town04' } }))
  );
  assert.equal(
    garagePreviewSignature(session({ camera: { resolution: '1600x900', fps: 45 } })),
    garagePreviewSignature(session())
  );
  assert.equal(
    JSON.parse(garagePreviewSignature(session({ camera: { resolution: '640x384', fps: 10 } }))).profile,
    'compatibility'
  );
});

test('incomplete numeric inputs are not treated as settled valid settings', () => {
  assert.equal(garagePreviewInputError(session()), '');
  for (const patch of [
    { identity: { seed: NaN } }, { scene: { trafficCount: undefined } },
    { scene: { walkerCount: 251 } }, { scene: { followingDistanceMetres: 0 } },
    { camera: { fov: NaN } }, { camera: { fps: 0 } }
  ]) assert.notEqual(garagePreviewInputError(session(patch)), '');
});

test('clearing a numeric field never silently applies zero to the Garage population', () => {
  const event = (value) => ({ currentTarget: { value } });
  const cleared = fieldNumber(event(''));
  assert.equal(Number.isNaN(cleared), true);
  assert.notEqual(garagePreviewInputError(session({ scene: { trafficCount: cleared } })), '');
  assert.equal(Number.isNaN(fieldNumber(event('  '))), true);
  assert.equal(fieldNumber(event('0')), 0);
  assert.equal(fieldNumber(event(' 25 ')), 25);
});

test('debounces rapid edits and sends a stable snapshot of only the latest selection', async (t) => {
  const h = harness(t);
  h.queue.select(session(), 'worker');
  t.mock.timers.tick(299);
  assert.equal(h.requests.length, 0);
  const latest = session({ scene: { trafficCount: 20 } });
  h.queue.select(latest, 'worker');
  latest.scene.trafficCount = 99;
  t.mock.timers.tick(299);
  assert.equal(h.requests.length, 0);
  t.mock.timers.tick(1);
  assert.equal(h.requests.length, 1);
  assert.equal(h.requests[0].session.scene.trafficCount, 20);
  h.requests[0].resolve('live');
  await settle();
  h.queue.select(session({ scene: { trafficCount: 20 }, perception: { voxelEnabled: true } }), 'worker');
  t.mock.timers.tick(10000);
  assert.equal(h.requests.length, 1, 'Drive-only edits must not dirty the Garage');
});

test('keeps one request in flight and follows it with only the newest settings', async (t) => {
  const h = harness(t);
  h.queue.select(session(), 'worker');
  t.mock.timers.tick(300);
  h.queue.select(session({ scene: { trafficCount: 10 } }), 'worker');
  t.mock.timers.tick(100);
  h.queue.select(session({ scene: { trafficCount: 20 } }), 'worker');
  t.mock.timers.tick(300);
  assert.equal(h.requests.length, 1);
  h.requests[0].resolve('first');
  await settle();
  assert.equal(h.requests.length, 2);
  assert.equal(h.requests[1].session.scene.trafficCount, 20);
  h.requests[1].resolve('latest');
  await settle();
  assert.deepEqual(h.applied.map((value) => value.response), ['first', 'latest']);
  assert.deepEqual(h.busy, [true, false, true, false]);
});

test('returning to the in-flight selection does not issue a duplicate configure', async (t) => {
  const h = harness(t);
  h.queue.select(session(), 'worker');
  t.mock.timers.tick(300);
  h.queue.select(session({ scene: { trafficCount: 20 } }), 'worker');
  h.queue.select(session(), 'worker');
  t.mock.timers.tick(300);
  h.requests[0].resolve('live');
  await settle();
  assert.equal(h.requests.length, 1);
});

test('an invalid intermediate edit cancels pending work without resetting the applied scene', async (t) => {
  const h = harness(t);
  h.queue.select(session(), 'worker');
  t.mock.timers.tick(300);
  h.requests[0].resolve('live');
  await settle();
  h.queue.select(session({ scene: { trafficCount: 20 } }), 'worker');
  h.queue.select(session({ scene: { trafficCount: undefined } }), 'worker', false);
  t.mock.timers.tick(300);
  h.queue.select(session(), 'worker');
  t.mock.timers.tick(300);
  assert.equal(h.requests.length, 1);
});

test('Drive suspension discards queued edits and ignores the old response', async (t) => {
  const h = harness(t);
  h.queue.select(session(), 'worker');
  t.mock.timers.tick(300);
  h.queue.select(session({ scene: { trafficCount: 20 } }), 'worker');
  h.queue.select(session(), '');
  t.mock.timers.tick(300);
  h.requests[0].resolve('obsolete');
  await settle();
  assert.equal(h.applied.length, 0);
  assert.equal(h.requests.length, 1);
  h.queue.select(session({ scene: { trafficCount: 30 } }), 'worker');
  t.mock.timers.tick(300);
  assert.equal(h.requests.length, 2);
  assert.equal(h.requests[1].session.scene.trafficCount, 30);
});

test('unmount suppresses all async result callbacks and retries', async (t) => {
  const h = harness(t);
  h.queue.select(session(), 'worker');
  t.mock.timers.tick(300);
  h.queue.dispose();
  h.requests[0].reject(new TypeError('Failed to fetch'));
  await settle();
  t.mock.timers.tick(10000);
  assert.deepEqual(h.errors, []);
  assert.deepEqual(h.applied, []);
  assert.deepEqual(h.busy, [true]);
  assert.equal(h.requests.length, 1);
});

test('transient reconnection uses bounded backoff; manual retry remains available', async (t) => {
  const h = harness(t, { retryDelays: [1000, 2000] });
  h.queue.select(session(), 'worker');
  t.mock.timers.tick(300);
  h.requests[0].reject(new TypeError('Failed to fetch'));
  await settle();
  t.mock.timers.tick(999);
  assert.equal(h.requests.length, 1);
  t.mock.timers.tick(1);
  h.requests[1].reject(new Error('Worker is busy'));
  await settle();
  t.mock.timers.tick(2000);
  h.requests[2].reject(new Error('connection refused'));
  await settle();
  t.mock.timers.tick(100000);
  assert.equal(h.requests.length, 3);
  h.queue.retry();
  assert.equal(h.requests.length, 4);
});

test('permanent configuration errors do not retry, but changed settings still apply', async (t) => {
  const h = harness(t);
  h.queue.select(session(), 'worker');
  t.mock.timers.tick(300);
  h.requests[0].reject(Object.assign(new Error('invalid vehicle'), { status: 400 }));
  await settle();
  t.mock.timers.tick(100000);
  assert.equal(h.requests.length, 1);
  h.queue.select(session({ vehicle: { blueprint: 'vehicle.audi.a2' } }), 'worker');
  t.mock.timers.tick(300);
  assert.equal(h.requests.length, 2);
});

test('late failure of an obsolete selection does not retry it ahead of newer settings', async (t) => {
  const h = harness(t);
  h.queue.select(session(), 'worker');
  t.mock.timers.tick(300);
  h.queue.select(session({ scene: { trafficCount: 20 } }), 'worker');
  t.mock.timers.tick(300);
  h.requests[0].reject(new TypeError('Failed to fetch'));
  await settle();
  assert.equal(h.requests.length, 2);
  assert.equal(h.requests[1].session.scene.trafficCount, 20);
  assert.equal(h.retrying.includes(true), false);
});

test('Drive starting immediately prevents a queued request before reactive teardown', (t) => {
  let allowed = true;
  const h = harness(t, { canApply: () => allowed });
  h.queue.select(session(), 'worker');
  allowed = false;
  t.mock.timers.tick(300);
  assert.equal(h.requests.length, 0);
  h.queue.select(session(), '');
  t.mock.timers.tick(100000);
  assert.equal(h.requests.length, 0);
});

test('retry classification distinguishes configuration errors from network contention', () => {
  assert.equal(isTransientGarageError(new Error('World Worker is busy')), true);
  assert.equal(isTransientGarageError(new Error('connection refused')), true);
  assert.equal(isTransientGarageError(Object.assign(new Error('proxy'), { status: 503 })), true);
  assert.equal(isTransientGarageError(Object.assign(new Error('busy is not a map'), { status: 400 })), false);
  assert.equal(isTransientGarageError(Object.assign(new Error('invalid token'), { status: 403 })), false);
  assert.equal(isTransientGarageError(new Error('unsupported camera profile')), false);
});
