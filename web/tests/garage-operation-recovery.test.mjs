import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import ts from 'typescript';

const source = readFileSync(new URL('../src/lib/api/operator.ts', import.meta.url), 'utf8')
  .replaceAll("'$lib/domain/config'", JSON.stringify(new URL('../src/lib/domain/config.ts', import.meta.url).href));
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

test('status timeout retries the read, not the vehicle replacement', async (t) => {
  let posts = 0;
  let reads = 0;
  t.mock.method(globalThis, 'fetch', async (path, options) => {
    if (options.method === 'POST') {
      posts++;
      return Response.json({ operation_id: 'op2', status: 'starting' });
    }
    if (path.endsWith('/events')) throw new TypeError('disconnected');
    assert.equal(path, '/api/garage/preview/operations/op2');
    if (++reads === 1) throw new DOMException('signal timed out', 'TimeoutError');
    return Response.json({ status: 'running', result: { vehicle_id: 456 } });
  });
  const result = await new OperatorApi('test').configureGaragePreview({});
  assert.equal(result.vehicle_id, 456);
  assert.equal(posts, 1);
  assert.equal(reads, 2);
});
