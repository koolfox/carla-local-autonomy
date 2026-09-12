import assert from 'node:assert/strict';
import { test } from 'node:test';
import { presetRig, cameraRigError, driveRecordingSettings } from '../src/lib/domain/capture.ts';

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
