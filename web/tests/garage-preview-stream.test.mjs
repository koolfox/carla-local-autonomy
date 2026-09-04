import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  confirmGarageStream,
  createGarageStreamBuffer,
  failGarageStream,
  stageGarageStream
} from '../src/lib/domain/garagePreviewStream.ts';

test('replacement stream is double-buffered until its first frame is confirmed', () => {
  let buffer = createGarageStreamBuffer();
  buffer = stageGarageStream(buffer, '/stream?t=first');
  assert.deepEqual(buffer.sources, ['/stream?t=first', '']);
  assert.equal(buffer.pending, 0);
  assert.equal(buffer.ready, false);

  buffer = confirmGarageStream(buffer, 0);
  assert.equal(buffer.visible, 0);
  assert.equal(buffer.ready, true);

  buffer = stageGarageStream(buffer, '/stream?t=second');
  assert.deepEqual(buffer.sources, ['/stream?t=first', '/stream?t=second']);
  assert.equal(buffer.visible, 0, 'the last good image remains the visible slot');
  assert.equal(buffer.pending, 1);

  buffer = confirmGarageStream(buffer, 1);
  assert.deepEqual(buffer.sources, ['', '/stream?t=second']);
  assert.equal(buffer.visible, 1);
  assert.equal(buffer.ready, true);
});

test('visible stream failure preserves the last decoded frame during reconfiguration', () => {
  let buffer = createGarageStreamBuffer();
  buffer = confirmGarageStream(stageGarageStream(buffer, '/stream?t=first'), 0);

  const duringConfigure = failGarageStream(buffer, 0, true);
  assert.equal(duringConfigure.preservedLastFrame, true);
  assert.equal(duringConfigure.shouldRetry, false);
  assert.equal(duringConfigure.buffer.ready, true);
  assert.equal(duringConfigure.buffer.sources[0], '/stream?t=first');

  const afterConfigure = failGarageStream(duringConfigure.buffer, 0, false);
  assert.equal(afterConfigure.preservedLastFrame, true);
  assert.equal(afterConfigure.shouldRetry, true);
  assert.equal(afterConfigure.buffer.sources[0], '/stream?t=first');
});

test('failed candidate never replaces the visible last-good stream', () => {
  let buffer = createGarageStreamBuffer();
  buffer = confirmGarageStream(stageGarageStream(buffer, '/stream?t=first'), 0);
  buffer = stageGarageStream(buffer, '/stream?t=candidate');

  const failed = failGarageStream(buffer, 1, false);
  assert.equal(failed.preservedLastFrame, false);
  assert.equal(failed.shouldRetry, true);
  assert.deepEqual(failed.buffer.sources, ['/stream?t=first', '']);
  assert.equal(failed.buffer.visible, 0);
  assert.equal(failed.buffer.ready, true);
});
