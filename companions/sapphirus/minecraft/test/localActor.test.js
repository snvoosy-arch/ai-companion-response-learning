const test = require('node:test');
const assert = require('node:assert/strict');
const { createLocalActor } = require('../src/localActor');
test('remote endpoint rejected without network', () => {
  assert.throws(() => createLocalActor({ endpoint: 'https://example.com/v1', model: 'test' }), /local endpoint/);
});
test('single request preserves model reply', async () => {
  let calls = 0;
  const actor = createLocalActor({ endpoint: 'http://127.0.0.1:8001/v1', model: 'test', fetchImpl: async () => {
    calls++;
    return new Response(JSON.stringify({ model: 'test', choices: [{ message: { content: '{"action":"silence","speech":"","tool_calls":[],"memory_candidates":[]}' } }] }));
  } });
  assert.equal((await actor({}, new AbortController().signal)).action, 'silence');
  assert.equal(calls, 1);
});
test('HTTP failure is not retried', async () => {
  let calls = 0;
  const actor = createLocalActor({ endpoint: 'http://127.0.0.1:8001/v1', model: 'test', fetchImpl: async () => { calls++; return new Response('', { status: 503 }); } });
  await assert.rejects(actor({}, new AbortController().signal), /503/);
  assert.equal(calls, 1);
});
