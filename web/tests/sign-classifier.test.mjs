import assert from 'node:assert/strict';
import { test } from 'node:test';
import { defaultSessionConfig, mergeSessionDefaults, sessionForApi } from '../src/lib/domain/config.ts';

test('DeiT is opt-in and old saved session defaults remain compatible', () => {
  assert.equal(defaultSessionConfig().perception.signClassifier, null);
  const old = defaultSessionConfig();
  delete old.perception.signClassifier;
  assert.equal(mergeSessionDefaults(defaultSessionConfig(), old).perception.signClassifier, null);
});

test('sign settings travel in the same perception session configuration', () => {
  const config = defaultSessionConfig();
  config.perception.detector = 'm9-hierarchical';
  const signClassifier = {
    checkpoint: 'models/deit64/deit64_stageB_blocks10_11_best.pt',
    ontology: 'models/deit64/ontology_final_64.csv',
    confidence: 0.8, crop_scale: 4
  };
  config.perception.signClassifier = signClassifier;
  const restored = mergeSessionDefaults(defaultSessionConfig(), JSON.parse(JSON.stringify(config)));
  assert.deepEqual(restored.perception.signClassifier, signClassifier);
  assert.equal(restored.perception.detector, 'm9-hierarchical');
  assert.deepEqual(restored.vehicle, config.vehicle);
  assert.deepEqual(restored.control, config.control);
});

test('preview omits inference stage; session start preserves enabled stage without mutating selections', () => {
  const config = defaultSessionConfig();
  assert.equal('signClassifier' in sessionForApi(config).perception, false);
  config.perception.signClassifier = { checkpoint: 'model.pt', ontology: 'labels.csv', confidence: 0.7, crop_scale: 4 };
  assert.equal('signClassifier' in sessionForApi(config, true).perception, false);
  assert.deepEqual(sessionForApi(config).perception.signClassifier, config.perception.signClassifier);
  assert.ok(config.perception.signClassifier);
});
