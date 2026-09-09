import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import ts from 'typescript';

const source = readFileSync(new URL('../src/lib/api/operator.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 }
});
const { OperatorApi } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);

test('dropped lifecycle stream recovers the same operation without repeating POST', async (t) => {
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (path, options) => {
    calls.push([path, options.method]);
    if (options.method === 'POST') return Response.json({ operation_id: 'op1', status: 'starting' });
    if (path.endsWith('/events')) throw new TypeError('connection lost');
    assert.equal(path, '/api/garage/preview/operations/op1');
    return Response.json({ operation_id: 'op1', status: 'running', result: { vehicle_id: 123 } });
  });
  const result = await new OperatorApi('test').configureGaragePreview({});
  assert.equal(result.vehicle_id, 123);
  assert.equal(calls.filter(([, method]) => method === 'POST').length, 1);
});
