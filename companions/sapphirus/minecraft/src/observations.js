const { Movements, goals } = require('mineflayer-pathfinder');
const { Vec3 } = require('vec3');

const { GoalNear } = goals;

function summarizeInventory(bot) {
  return bot.inventory
    .items()
    .sort((left, right) => right.count - left.count)
    .map((item) => ({
      name: item.name,
      count: item.count,
    }));
}

function summarizeNearbyPlayers(bot, config) {
  const ignored = new Set((config.ignoredPlayerUsernames || []).map((name) => String(name || '').toLowerCase()));
  return Object.values(bot.players)
    .filter((player) => {
      if (!player || !player.entity || player.username === bot.username) {
        return false;
      }
      return !ignored.has(String(player.username || '').toLowerCase());
    })
    .map((player) => ({
      username: player.username,
      distance: distanceFromBot(bot, player.entity.position),
      position: formatPosition(player.entity.position),
    }))
    .sort((left, right) => left.distance - right.distance)
    .slice(0, 6);
}

function summarizeNearbyEntities(bot) {
  return Object.values(bot.entities)
    .filter((entity) => entity && entity.type !== 'player')
    .map((entity) => ({
      name: entity.name || entity.displayName || entity.kind || null,
      type: entity.type,
      kind: entity.kind || null,
      distance: distanceFromBot(bot, entity.position),
      position: formatPosition(entity.position),
    }))
    .sort((left, right) => left.distance - right.distance)
    .slice(0, 8);
}

function summarizeInterestingBlocks(bot, blockNames) {
  const seen = [];

  for (const blockName of blockNames) {
    const matchedBlocks = findReachableBlocksByName(bot, blockName, 24, 6);
    if (!matchedBlocks.length) {
      continue;
    }

    seen.push({
      name: matchedBlocks[0].name,
      position: formatPosition(matchedBlocks[0].position),
      distance: distanceFromBot(bot, matchedBlocks[0].position),
    });
  }

  return seen
    .sort((left, right) => left.distance - right.distance)
    .slice(0, 10);
}

function findBlocksByName(bot, blockName, maxDistance, count) {
  const block = bot.registry.blocksByName[blockName];
  if (!block) {
    return [];
  }

  const positions = bot.findBlocks({
    matching: block.id,
    maxDistance,
    count,
  });

  return positions
    .map((position) => bot.blockAt(position))
    .filter(Boolean);
}

function findReachableBlocksByName(bot, blockName, maxDistance, count) {
  return findBlocksByName(bot, blockName, maxDistance, Math.max(count * 2, count))
    .map((block) => {
      const recovery = findRecoveryPosition(bot, block.position);
      const distance = distanceFromBot(bot, block.position);
      return {
        block,
        recovery,
        distance,
      };
    })
    .filter((entry) => entry.recovery)
    .sort((left, right) => left.distance - right.distance)
    .slice(0, count)
    .map((entry) => entry.block);
}

function findRecoveryPosition(bot, position) {
  const offsets = [
    new Vec3(1, 0, 0),
    new Vec3(-1, 0, 0),
    new Vec3(0, 0, 1),
    new Vec3(0, 0, -1),
    new Vec3(1, 0, 1),
    new Vec3(-1, 0, 1),
    new Vec3(1, 0, -1),
    new Vec3(-1, 0, -1),
    new Vec3(0, -1, 0),
    new Vec3(1, -1, 0),
    new Vec3(-1, -1, 0),
    new Vec3(0, -1, 1),
    new Vec3(0, -1, -1),
    new Vec3(0, -2, 0),
  ];

  for (const offset of offsets) {
    const candidate = position.plus(offset);
    if (!isStandableRecoveryPosition(bot, candidate)) {
      continue;
    }
    const probe = evaluateGoalReachability(bot, candidate);
    if (!probe.blocked && probe.status === 'success') {
      return candidate;
    }
  }

  return null;
}

function isStandableRecoveryPosition(bot, position) {
  if (!bot || !position) {
    return false;
  }

  const feet = bot.blockAt(position);
  const head = bot.blockAt(position.offset(0, 1, 0));
  const below = bot.blockAt(position.offset(0, -1, 0));

  const feetPassable = !feet || feet.boundingBox !== 'block';
  const headPassable = !head || head.boundingBox !== 'block';
  const supportSolid = Boolean(below && below.boundingBox === 'block');

  return feetPassable && headPassable && supportSolid;
}

function evaluateGoalReachability(bot, position, timeoutMs = 900) {
  try {
    const movements = bot.pathfinder?.movements || createProbeMovements(bot);
    const goal = new GoalNear(position.x, position.y, position.z, 1);
    const result = bot.pathfinder?.getPathTo
      ? bot.pathfinder.getPathTo(movements, goal, timeoutMs)
      : null;

    if (!result) {
      return {
        status: 'unknown',
        blocked: false,
      };
    }

    const pathLength = Array.isArray(result.path) ? result.path.length : Number.POSITIVE_INFINITY;
    const status = String(result.status || 'unknown');
    return {
      status,
      blocked: status === 'noPath' || pathLength === 0,
    };
  } catch {
    return {
      status: 'probe_error',
      blocked: false,
    };
  }
}

function createProbeMovements(bot) {
  const movements = new Movements(bot);
  movements.canOpenDoors = true;
  movements.allow1by1towers = false;
  return movements;
}

function distanceFromBot(bot, position) {
  if (!bot.entity || !position) {
    return Number.POSITIVE_INFINITY;
  }
  return Number(bot.entity.position.distanceTo(position).toFixed(1));
}

function formatPosition(position) {
  if (!position) {
    return null;
  }
  return {
    x: Math.floor(position.x),
    y: Math.floor(position.y),
    z: Math.floor(position.z),
  };
}

module.exports = {
  summarizeInventory,
  summarizeNearbyPlayers,
  summarizeNearbyEntities,
  summarizeInterestingBlocks,
  findBlocksByName,
  distanceFromBot,
  formatPosition,
};
