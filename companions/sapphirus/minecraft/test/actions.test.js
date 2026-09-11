const test = require('node:test');
const assert = require('node:assert/strict');
const { executeAction, buildCollectResult } = require('../src/actions');

test('inventory gain, not mining message, determines collection', () => {
  assert.equal(buildCollectResult('oak_log', 3, 3).collected, false);
  assert.equal(buildCollectResult('oak_log', 3, 6).delta, 3);
});

test('unsupported actions fail instead of choosing a fallback', async () => {
  await assert.rejects(executeAction({}, { action: 'attack_nearest', parameters: {} }), /Unsupported/);
});

test('mining requires a selected position', async () => {
  await assert.rejects(executeAction({}, { action: 'mine_block', parameters: { block_name: 'oak_log' } }), /position/);
});

test('missing selected block never searches for an alternative', async () => {
  const bot = { blockAt: () => null, inventory: { items: () => [] },
    findBlocks: () => { throw new Error('must not search'); } };
  await assert.rejects(executeAction(bot, { action: 'mine_block', parameters: {
    block_name: 'oak_log', target_position: { x: 1, y: 64, z: 1 }
  } }), /Could not find/);
});

test('invalid position fails before any game operation', async () => {
  await assert.rejects(executeAction({}, { action: 'goto', parameters: { x: '1', y: 64, z: 0 } }), /coordinate/);
});
