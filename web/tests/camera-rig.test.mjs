import assert from 'node:assert/strict';
import { test } from 'node:test';
import { presetRig, cameraRigError, driveRecordingSettings } from '../src/lib/domain/capture.ts';
import { sessionForApi } from '../src/lib/domain/config.ts';

test('rear preset faces backwards and customized angles persist in Drive request', () => {
  const rig = presetRig('front-rear', 90);
  assert.equal(rig.additional_views[0].mount.yaw, 180);
  rig.additional_views[0].mount.pitch = -12;
  const settings = { rig, recordDuringDrive: true, captureFps: 5 };
  const recording = driveRecordingSettings({ recording: { video: true } }, settings);
  assert.equal(recording.cameraRig.additional_views[0].mount.pitch, -12);
  rig.additional_views[0].mount.pitch = 0;
  assert.equal(recording.cameraRig.additional_views[0].mount.pitch, -12);
});

test('normal Drive does not acquire extra cameras unless explicitly selected', () => {
  const settings = { rig: presetRig('surround', 90), captureFps: 5 };
  assert.deepEqual(driveRecordingSettings({ recording: { video: true } }, settings), { video: true });
  assert.deepEqual(driveRecordingSettings({ recording: { video: false } }, { ...settings, recordDuringDrive: true }), { video: false });
});

test('duplicate camera IDs and invalid angles cannot reach Start session', () => {
  const rig = presetRig('front-rear', 90);
  rig.additional_views.push(structuredClone(rig.additional_views[0]));
  assert.ok(cameraRigError(rig));
  assert.throws(() => driveRecordingSettings({ recording: { video: true } }, { rig, recordDuringDrive: true, captureFps: 5 }));
  rig.additional_views.pop();
  rig.additional_views[0].mount.yaw = NaN;
  assert.ok(cameraRigError(rig));
});

test('camera perception is opt-in, validated and snapshotted independently of camera geometry', () => {
  const session = { recording: { video: true }, perception: { enabled: true, detector: 'm9-hierarchical', signClassifier: { checkpoint: 'sign.pt' } } };
  const settings = { recordDuringDrive: true, rig: presetRig('front-rear', 90), captureFps: 5,
    perceptionViews: { rear: 'signs', front: 'detections' } };
  const recording = driveRecordingSettings(session, settings);
  assert.deepEqual(recording.cameraPerception, settings.perceptionViews);
  assert.equal(recording.cameraRig.additional_views[0].perception, undefined);
  settings.perceptionViews.rear = 'detections';
  assert.equal(recording.cameraPerception.rear, 'signs');
  assert.throws(() => driveRecordingSettings(session, { ...settings, perceptionViews: { missing: 'detections' } }), /existing rig cameras/);
  assert.throws(() => driveRecordingSettings(session, { ...settings, perceptionViews: { rear: 'invalid' } }));
  assert.throws(() => driveRecordingSettings({ ...session, perception: { enabled: false } }, settings), /Enable Detection/);
  assert.throws(() => driveRecordingSettings({ ...session, perception: { enabled: true, detector: 'yolo' } },
    { ...settings, perceptionViews: { rear: 'signs' } }), /M9/);
  assert.equal(driveRecordingSettings(session, { ...settings, perceptionViews: {} }).cameraPerception, undefined);
});

test('per-camera inference settings go to session start, never to parked Garage preview', () => {
  const session = { perception: { enabled: true }, recording: { video: true, cameraPerception: { rear: 'detections' } } };
  assert.deepEqual(sessionForApi(session).recording.cameraPerception, { rear: 'detections' });
  assert.equal(sessionForApi(session, true).recording.cameraPerception, undefined);
  assert.deepEqual(session.recording.cameraPerception, { rear: 'detections' });
});
