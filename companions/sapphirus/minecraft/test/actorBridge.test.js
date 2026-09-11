const test = require('node:test');
const assert = require('node:assert/strict');
const { createActorBridge } = require('../src/actorBridge');
const turn = (action, speech = '', tool_calls = []) => ({ action, speech, tool_calls, memory_candidates: [] });
const go = (name = 'owner') => turn('use_tool', '', [{ name: 'minecraft_goto', arguments: { player: name } }]);
function harness(actor, options = {}) {
  const moves = [];
  const bridge = createActorBridge({ owner: 'owner', actor, observe: () => ({
    self: { x: 0, y: 64, z: 0 }, players: [{ name: 'owner', position: { x: 6, y: 64, z: 0 } }]
  }), move: async (position) => { moves.push(position); return { arrived: true }; }, stop: () => {}, ...options });
  return { bridge, moves };
}
test('only model tool choice starts movement, then returns evidence to same actor', async () => {
  const inputs = [];
  const { bridge, moves } = harness(async input => { inputs.push(input); return input.phase === 'decide' ? go() : turn('reply', '도착했어'); });
  const result = await bridge.handle('owner', '사피루스 나한테 와');
  assert.equal(result.status, 'completed');
  assert.equal(moves.length, 1);
  assert.equal(inputs[1].tool_result.arrived, true);
});
test('reply and silence do not become fallback movement', async () => {
  for (const choice of [turn('reply', '지금은 기다릴게'), turn('silence')]) {
    const { bridge, moves } = harness(async () => choice);
    await bridge.handle('owner', '사피루스 와');
    assert.equal(moves.length, 0);
  }
});
test('other players and unaddressed chat do not invoke model', async () => {
  let calls = 0;
  const { bridge } = harness(async () => { calls++; return go(); });
  await bridge.handle('other', '사피루스 와');
  await bridge.handle('owner', '그냥 채팅');
  assert.equal(calls, 0);
});
test('malformed output and different player never move', async () => {
  for (const output of [{ action: 'goto' }, go('other')]) {
    const { bridge, moves } = harness(async () => output);
    assert.equal((await bridge.handle('owner', '사피루스 와')).status, 'failed');
    assert.equal(moves.length, 0);
  }
});
test('uses refreshed observed position after model decision', async () => {
  let x = 4;
  const { bridge, moves } = harness(async input => { x = 9; return input.phase === 'decide' ? go() : turn('silence'); },
    { observe: () => ({ self: { x: 0, y: 64, z: 0 }, players: [{ name: 'owner', position: { x, y: 64, z: 0 } }] }) });
  await bridge.handle('owner', '사피루스 와');
  assert.equal(moves[0].x, 9);
});
test('second tool request is refused without additional movement', async () => {
  const { bridge, moves } = harness(async () => go());
  const result = await bridge.handle('owner', '사피루스 와');
  assert.equal(result.status, 'failed');
  assert.equal(moves.length, 1);
});
test('session request budget never triggers automatic retries', async () => {
  let calls = 0;
  const { bridge } = harness(async () => { calls++; throw new Error('unavailable'); }, { maxRequests: 1 });
  await bridge.handle('owner', '사피루스 와');
  assert.equal((await bridge.handle('owner', '사피루스 와')).status, 'budget_exhausted');
  assert.equal(calls, 1);
});
test('cancellation blocks a late model response', async () => {
  let release;
  const { bridge, moves } = harness(() => new Promise(resolve => { release = resolve; }));
  const pending = bridge.handle('owner', '사피루스 와');
  await Promise.resolve();
  bridge.cancel();
  release(go());
  assert.equal((await pending).status, 'failed');
  assert.equal(moves.length, 0);
});

test('simultaneous messages cannot start parallel model work', async () => {
  let release;
  const { bridge } = harness(() => new Promise(resolve => { release = resolve; }));
  const first = bridge.handle('owner', '사피루스 와');
  await Promise.resolve();
  assert.equal((await bridge.handle('owner', '사피루스 와')).status, 'busy');
  release(turn('silence'));
  await first;
});

test('task deadline stops stalled generation without movement', async () => {
  let stops = 0;
  const { bridge, moves } = harness(() => new Promise(() => {}), { timeoutMs: 15, stop: () => { stops++; } });
  assert.equal((await bridge.handle('owner', '사피루스 와')).status, 'failed');
  assert.ok(stops > 0);
  assert.equal(moves.length, 0);
});

test('target disappearing during inference prevents movement', async () => {
  let players = [{ name: 'owner', position: { x: 6, y: 64, z: 0 } }];
  const { bridge, moves } = harness(async () => { players = []; return go(); }, {
    observe: () => ({ self: { x: 0, y: 64, z: 0 }, players })
  });
  assert.equal((await bridge.handle('owner', '사피루스 와')).status, 'failed');
  assert.equal(moves.length, 0);
});
