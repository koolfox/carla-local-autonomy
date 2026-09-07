import assert from 'node:assert/strict';
import { test } from 'node:test';
import { paintLabel } from '../src/lib/ui/paint.ts';

test('CARLA RGB values become readable names and icons', () => {
  assert.equal(paintLabel('0,0,0'), '⚫ Black');
  assert.equal(paintLabel('255, 255, 255'), '⚪ White');
  assert.equal(paintLabel('30,90,220'), '🔵 Blue');
  assert.equal(paintLabel('#1e5adc'), '🔵 Blue');
});

test('malformed colors do not expose technical values as labels', () => {
  for (const value of ['999,0,0', 'invalid', '', '#xyz']) {
    assert.equal(paintLabel(value), '◌ Custom paint');
  }
});
