const test = require('node:test');
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const path = require('node:path');
const host = path.join(__dirname, '../src/actorHost.js');

test('host requires an explicit owner and port before any connection', () => {
  const script = `
    const Module = require('node:module');
    const load = Module._load;
    Module._load = function(name, ...args) {
      if (name === 'mineflayer') throw new Error('Live dependency reached before validation');
      return load.call(this, name, ...args);
    };
    process.argv = ['node', '--connect', '--allow-model-requests', '--model', 'test'];
    try { require(${JSON.stringify(host)}).main(); }
    catch (error) { console.error(error.message); process.exitCode = 1; }
  `;
  const result = spawnSync(process.execPath, ['-e', script], { encoding: 'utf8', timeout: 3000 });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /Requires.*--owner.*--port/);
});

test('help exits without connecting', () => {
  const result = spawnSync(process.execPath, [host, '--help'], { encoding: 'utf8', timeout: 3000 });
  assert.equal(result.status, 0);
  assert.match(result.stdout, /--owner PLAYER_NAME --port PORT/);
});
