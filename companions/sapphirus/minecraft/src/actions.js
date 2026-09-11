const primitives = require('./primitives');

function requirePosition(position) {
  if (!position || !['x', 'y', 'z'].every(k =>
    typeof position[k] === 'number' && Number.isFinite(position[k]) && Number.isInteger(position[k]))) {
    throw new Error('Explicit integer position coordinates are required');
  }
}

async function executeAction(bot, decision) {
  const allowed = ['goto', 'mine_block', 'recover_drop', 'stop'];
  if (!decision || !allowed.includes(decision.action)) throw new Error('Unsupported action');
  const parameters = structuredClone(decision.parameters || {});
  if (decision.action === 'goto') {
    requirePosition(parameters);
    if (parameters.purpose) throw new Error('Use recover_drop for collection');
  }
  if (['mine_block', 'recover_drop'].includes(decision.action)) {
    requirePosition(parameters.target_position);
    const name = decision.action === 'mine_block' ? parameters.block_name : parameters.item_name;
    if (!['oak_log', 'birch_log', 'spruce_log', 'jungle_log', 'acacia_log',
      'dark_oak_log', 'mangrove_log'].includes(name)) throw new Error('Only log collection is enabled');
  }
  const methods = { goto: 'actionGoto', mine_block: 'actionMineBlock',
    recover_drop: 'actionRecoverDrop', stop: 'actionStop' };
  const message = await primitives[methods[decision.action]](bot, parameters);
  return { action: decision.action, message, parameters };
}

module.exports = { executeAction, createDefaultMovements: primitives.createDefaultMovements,
  buildCollectResult: primitives.buildCollectResult };
