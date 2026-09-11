const { parseArgs } = require('node:util');
const { createActorBridge } = require('./actorBridge');
const { createLocalActor } = require('./localActor');
const { executeAction, createDefaultMovements } = require('./actions');

function main() {
  const { values } = parseArgs({ options: {
    connect: { type: 'boolean', default: false },
    'allow-model-requests': { type: 'boolean', default: false },
    owner: { type: 'string' },
    model: { type: 'string' },
    endpoint: { type: 'string', default: 'http://127.0.0.1:8001/v1' },
    port: { type: 'string' },
    'max-requests': { type: 'string', default: '10' },
    help: { type: 'boolean', default: false }
  } });
  if (values.help) {
    console.log('node src/actorHost.js --connect --allow-model-requests --model MODEL_ID --owner PLAYER_NAME --port PORT [--endpoint http://127.0.0.1:8001/v1] [--max-requests 10]');
    return;
  }
  if (!values.connect || !values['allow-model-requests'] || !values.model || !values.owner || !values.port) {
    throw new Error('Requires --connect --allow-model-requests --model --owner --port; model server must already be running');
  }
  const port = Number(values.port), maxRequests = Number(values['max-requests']);
  if (!Number.isInteger(port) || port < 1024 || port > 65535 || !Number.isInteger(maxRequests) ||
      maxRequests < 1 || maxRequests > 100 || !/^[A-Za-z0-9_]{1,16}$/.test(values.owner)) throw new Error('Invalid host arguments');
  const actor = createLocalActor({ endpoint: values.endpoint, model: values.model,
    apiKey: process.env.SAPPHIRUS_MC_API_KEY || '' });
  const mineflayer = require('mineflayer');
  const { pathfinder } = require('mineflayer-pathfinder');
  const bot = mineflayer.createBot({ host: '127.0.0.1', port, username: 'WhiteBot',
    version: '26.1', auth: 'offline', respawn: false, hideErrors: true });
  bot.loadPlugin(pathfinder);
  let ready = false;
  const stop = () => { bot.pathfinder.setGoal(null); bot.clearControlStates(); };
  const position = vector => vector ? { x: vector.x, y: vector.y, z: vector.z } : null;
  const observe = () => ({ self: position(bot.entity?.position), health: bot.health, food: bot.food,
    players: Object.values(bot.players).filter(p => p.username === values.owner && p.entity)
      .map(p => ({ name: p.username, position: position(p.entity.position) })) });
  const bridge = createActorBridge({ owner: values.owner, actor, observe, stop, maxRequests,
    move: async (target, signal) => {
      signal.throwIfAborted();
      const result = await executeAction(bot, { action: 'goto', parameters: { ...target, range: 3 } });
      signal.throwIfAborted();
      const current = bot.players[values.owner]?.entity?.position;
      const distance = current ? bot.entity.position.distanceTo(current) : null;
      return { arrived: distance !== null && distance <= 4, distance, position: position(bot.entity.position), result };
    }
  });
  const connectTimer = setTimeout(() => { if (!ready) bot.end('connection timeout'); }, 30000);
  bot.on('spawn', () => {
    clearTimeout(connectTimer);
    bot.pathfinder.setMovements(createDefaultMovements(bot));
    ready = true;
    console.log(JSON.stringify({ event: 'ready', owner: values.owner, maxRequests }));
  });
  bot.on('chat', (sender, message) => {
    if (!ready || sender === bot.username) return;
    // Abort is an external owner control, not a generated action.
    if (sender === values.owner && message.trim() === '!sapphirus stop') {
      bridge.cancel(); stop(); return;
    }
    bridge.handle(sender, message).then(result => {
      if (result.status !== 'ignored') console.log(JSON.stringify(result));
    }).catch(error => console.error(JSON.stringify({ error: error.message })));
  });
  bot.on('death', () => { ready = false; bridge.cancel(); });
  bot.on('error', error => { ready = false; bridge.cancel(); console.error(error.message); });
  bot.on('end', reason => { ready = false; clearTimeout(connectTimer); bridge.cancel(); console.log(JSON.stringify({ event: 'disconnected', reason })); });
  // Successful tasks do not disconnect the bot. No auto reconnect or model load.
}
if (require.main === module) {
  try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
}
module.exports = { main };
