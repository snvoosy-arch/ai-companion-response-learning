// Extracted from minecraft-bot; target substitution removed.
const { Movements, goals } = require('mineflayer-pathfinder');
const { Vec3 } = require('vec3');

const { GoalNear } = goals;
const LOG_TO_PLANK = {
  oak_log: 'oak_planks',
  birch_log: 'birch_planks',
  spruce_log: 'spruce_planks',
  jungle_log: 'jungle_planks',
  acacia_log: 'acacia_planks',
  dark_oak_log: 'dark_oak_planks',
  mangrove_log: 'mangrove_planks',
};
const WOOD_BLOCK_NAMES = new Set(Object.keys(LOG_TO_PLANK));
const ORE_BLOCK_NAMES = new Set(['coal_ore', 'iron_ore', 'copper_ore']);
const BLOCK_DROP_ITEMS = {
  oak_log: ['oak_log'],
  birch_log: ['birch_log'],
  spruce_log: ['spruce_log'],
  jungle_log: ['jungle_log'],
  acacia_log: ['acacia_log'],
  dark_oak_log: ['dark_oak_log'],
  mangrove_log: ['mangrove_log'],
  coal_ore: ['coal'],
  iron_ore: ['raw_iron'],
  copper_ore: ['raw_copper'],
  deepslate_coal_ore: ['coal'],
  deepslate_iron_ore: ['raw_iron'],
  deepslate_copper_ore: ['raw_copper'],
};

function createDefaultMovements(bot) {
  const movements = new Movements(bot);
  movements.canOpenDoors = true;
  movements.allow1by1towers = false;
  movements.canDig = false;
  movements.allowFreeMotion = false;
  movements.allowParkour = false;
  movements.scafoldingBlocks = [];
  return movements;
}

async function actionGoto(bot, parameters) {
  const x = roundCoordinate(parameters.x);
  const y = roundCoordinate(parameters.y);
  const z = roundCoordinate(parameters.z);
  const range = clampNumber(parameters.range, 1, 1, 4);
  const goal = new GoalNear(x, y, z, range);
  const reachability = evaluateGoalReachability(bot, goal);
  const isCollectPurpose = String(parameters.purpose || '').toLowerCase() === 'collect_recent_drop';

  if (reachability.blocked) {
    if (isCollectPurpose) {
      const collectItemName = normalizeMinecraftName(parameters.collect_item_name);
      const inventoryBefore = collectItemName
        ? countInventoryItemsForCollection(bot.inventory?.items?.() || [], collectItemName)
        : 0;
      parameters.collect_result = buildCollectResult(collectItemName, inventoryBefore, inventoryBefore);
      return `Could not reach the recent drop near (${x}, ${y}, ${z}).`;
    }
    throw new Error(`No path near (${x}, ${y}, ${z}).`);
  }

  try {
    await withTimeout(
      bot.pathfinder.goto(goal),
      computeGotoTimeoutMs(reachability, bot.entity?.position?.distanceTo(new Vec3(x, y, z)) || 0),
      () => bot.pathfinder.setGoal(null),
      `Pathfinding to (${x}, ${y}, ${z}) timed out.`
    );
  } catch (error) {
    if (isCollectPurpose) {
      const collectItemName = normalizeMinecraftName(parameters.collect_item_name);
      const inventoryBefore = collectItemName
        ? countInventoryItemsForCollection(bot.inventory?.items?.() || [], collectItemName)
        : 0;
      parameters.collect_result = buildCollectResult(collectItemName, inventoryBefore, inventoryBefore);
      return `Tried to recover the recent drop near (${x}, ${y}, ${z}) but could not reach it.`;
    }
    throw error;
  }

  if (isCollectPurpose) {
    const collectItemName = normalizeMinecraftName(parameters.collect_item_name);
    const inventoryBefore = collectItemName
      ? countInventoryItemsForCollection(bot.inventory?.items?.() || [], collectItemName)
      : 0;
    await delay(500);
    const collectResult = await collectDropsNearPosition(
      bot,
      new Vec3(x, y, z),
      collectItemName,
      inventoryBefore
    );
    parameters.collect_result = collectResult;
    if (collectResult?.collected) {
      return `Moved near (${x}, ${y}, ${z}) and recovered ${collectResult.item_name} x${collectResult.delta}.`;
    }
    return `Moved near (${x}, ${y}, ${z}) but did not recover ${collectResult?.item_name || 'the drop'} yet.`;
  }

  return `Moved near (${x}, ${y}, ${z}).`;
}

async function actionMineBlock(bot, parameters) {
  const requestedBlockName = normalizeMinecraftName(parameters.block_name);
  let blockName = requestedBlockName;
  if (!blockName) {
    throw new Error('block_name is required for mine_block.');
  }

  const maxDistance = clampNumber(parameters.max_distance, 16, 4, 32);
  const hintedPosition = normalizeTargetPosition(parameters.target_position);
  let candidateBlocks = findCandidateMineTargets(bot, blockName, maxDistance, 8, hintedPosition);
  let inventoryBefore = countInventoryItemsForCollection(bot.inventory?.items?.() || [], blockName);

  if (!candidateBlocks.length) {
    throw new Error(`Could not find nearby block: ${requestedBlockName}`);
  }
  let lastError = null;

  for (const candidate of candidateBlocks) {
    let targetBlock = bot.blockAt(candidate.block.position);
    const recoveryPosition = candidate.recoveryPosition || findRecoveryPosition(bot, targetBlock.position);
    if (!recoveryPosition) {
      lastError = new Error(`No safe recovery spot for ${blockName} right now.`);
      continue;
    }

    const recoveryReachability =
      candidate.recoveryProbe ||
      evaluateGoalReachability(
        bot,
        new GoalNear(recoveryPosition.x, recoveryPosition.y, recoveryPosition.z, 1)
      );
    parameters.resolved_position = {
      x: Math.round(recoveryPosition.x),
      y: Math.round(recoveryPosition.y),
      z: Math.round(recoveryPosition.z),
    };
    parameters.recovery_probe_status = recoveryReachability.status;
    parameters.recovery_probe_path = Number.isFinite(recoveryReachability.pathLength)
      ? recoveryReachability.pathLength
      : null;
    if (recoveryReachability.blocked) {
      lastError = new Error(`No recovery path for ${blockName} right now.`);
      continue;
    }
    const closeEnough =
      bot.entity &&
      targetBlock?.position &&
      bot.entity.position.distanceTo(targetBlock.position) <= 1.8;

    if (!closeEnough) {
      try {
        await withTimeout(
          bot.pathfinder.goto(new GoalNear(recoveryPosition.x, recoveryPosition.y, recoveryPosition.z, 1)),
          computeGotoTimeoutMs(
            recoveryReachability,
            bot.entity?.position?.distanceTo(recoveryPosition) || maxDistance
          ),
          () => bot.pathfinder.setGoal(null),
          `Pathfinding to ${blockName} timed out.`
        );
      } catch (error) {
        lastError = error;
        continue;
      }
    }

    targetBlock = bot.blockAt(candidate.block.position) || targetBlock;
    if (!targetBlock || targetBlock.name !== blockName || !bot.canDigBlock(targetBlock)) {
      lastError = new Error(`Cannot dig ${blockName} right now.`);
      continue;
    }

    try {
      await withTimeout(
        bot.dig(targetBlock, true),
        7000,
        () => {
          try {
            bot.stopDigging();
          } catch {}
        },
        `Digging ${blockName} timed out.`
      );
      let collectResult = await collectDropsNearPosition(bot, targetBlock.position, blockName, inventoryBefore);

      if (!collectResult?.collected && recoveryPosition) {
        try {
          await withTimeout(
            bot.pathfinder.goto(new GoalNear(recoveryPosition.x, recoveryPosition.y, recoveryPosition.z, 1)),
            5000,
            () => bot.pathfinder.setGoal(null),
            `Pathfinding to recover ${blockName} drops timed out.`
          );
          collectResult = await collectDropsNearPosition(
            bot,
            recoveryPosition,
            blockName,
            inventoryBefore
          );
        } catch {}
      }

      parameters.collect_result = collectResult;
      parameters.pickup_pending = !collectResult?.collected && recoveryReachability.status === 'success';
      if (!collectResult?.collected && collectResult?.last_seen_drop_position) {
        parameters.resolved_position = collectResult.last_seen_drop_position;
      }
      if (collectResult?.collected) {
        return `Mined ${blockName} and recovered x${collectResult.delta}.`;
      }
      return `Mined ${blockName}, but pickup is still pending.`;
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError || new Error(`Could not reach a diggable ${blockName}.`);
}

async function actionRecoverDrop(bot, parameters) {
  const itemName = normalizeMinecraftName(parameters.item_name || parameters.collect_item_name);
  const anchorPosition =
    normalizeTargetPosition(parameters.target_position) ||
    normalizeTargetPosition(parameters.position) ||
    (bot.entity
      ? {
          x: Math.round(Number(bot.entity.position.x)),
          y: Math.round(Number(bot.entity.position.y)),
          z: Math.round(Number(bot.entity.position.z)),
        }
      : null);

  if (!anchorPosition) {
    throw new Error('recover_drop requires a visible anchor position.');
  }

  const inventoryBefore = itemName
    ? countInventoryItemsForCollection(bot.inventory?.items?.() || [], itemName)
    : 0;

  const collectResult = await collectDropsNearPosition(
    bot,
    new Vec3(anchorPosition.x, anchorPosition.y, anchorPosition.z),
    itemName,
    inventoryBefore
  );

  parameters.collect_result = collectResult;
  parameters.pickup_pending = !collectResult?.collected;
  parameters.resolved_position = collectResult?.last_seen_drop_position || anchorPosition;

  if (collectResult?.collected) {
    return `Recovered ${collectResult.item_name} x${collectResult.delta}.`;
  }

  return `Tried to recover ${collectResult?.item_name || itemName || 'the recent drop'}, but pickup is still pending.`;
}

function actionStop(bot) {
  bot.clearControlStates();
  bot.pathfinder.setGoal(null);
  return 'Stopped the current movement and control state.';
}

async function collectDropsNearPosition(bot, position, blockName, inventoryBefore = 0) {
  if (!bot?.entity || !position) {
    return buildCollectResult(blockName, inventoryBefore, inventoryBefore);
  }

  let lastSeenDropPosition = null;

  for (let attempt = 0; attempt < 4; attempt += 1) {
    await delay(attempt === 0 ? 300 : 650);

    const scanAnchor = positionFromTarget(lastSeenDropPosition) || position;
    const nearbyDropEntities = mergeDropEntities(
      findNearbyDropEntities(bot, scanAnchor, lastSeenDropPosition ? 5 : 10),
      scanAnchor === position ? [] : findNearbyDropEntities(bot, position, 10)
    );
    if (nearbyDropEntities.length) {
      lastSeenDropPosition = normalizeTargetPosition(nearbyDropEntities[0].position) || lastSeenDropPosition;
    }
    const recoveryTargets = nearbyDropEntities.length
      ? nearbyDropEntities.flatMap((entity) =>
          findRecoveryPositions(bot, entity.position, 6).map((recoveryPosition) => ({
            recoveryPosition,
            source: entity.position,
          }))
        )
      : [
          ...(lastSeenDropPosition
            ? findRecoveryPositions(bot, scanAnchor, 5).map((recoveryPosition) => ({
                recoveryPosition,
                source: scanAnchor,
              }))
            : []),
          ...findRecoveryPositions(bot, position, 8).map((recoveryPosition) => ({
            recoveryPosition,
            source: position,
          })),
        ];

    for (const recoveryTarget of recoveryTargets) {
      try {
        await withTimeout(
          bot.pathfinder.goto(
            new GoalNear(
              recoveryTarget.recoveryPosition.x,
              recoveryTarget.recoveryPosition.y,
              recoveryTarget.recoveryPosition.z,
              0
            )
          ),
          nearbyDropEntities.length ? 4500 : 3500,
          () => bot.pathfinder.setGoal(null),
          nearbyDropEntities.length
            ? 'Pathfinding to dropped item timed out.'
            : 'Pathfinding to mined block drops timed out.'
        );
      } catch {
        continue;
      }

      await delay(350);
      const sweepCenter =
        positionFromTarget(lastSeenDropPosition) ||
        recoveryTarget.source ||
        recoveryTarget.recoveryPosition;
      const tightSweepResult = await attemptClosePickupSweep(
        bot,
        sweepCenter,
        blockName,
        inventoryBefore
      );
      if (tightSweepResult.collected) {
        if (lastSeenDropPosition) {
          tightSweepResult.last_seen_drop_position = lastSeenDropPosition;
        }
        return tightSweepResult;
      }
    }
  }

  const sweepResult = await sweepNearbyDropRecovery(bot, position, blockName, inventoryBefore);
  if (sweepResult.collected) {
    if (lastSeenDropPosition) {
      sweepResult.last_seen_drop_position = lastSeenDropPosition;
    }
    return sweepResult;
  }

  const inventoryAfter = countInventoryItemsForCollection(bot.inventory?.items?.() || [], blockName);
  const result = buildCollectResult(blockName, inventoryBefore, inventoryAfter);
  if (lastSeenDropPosition) {
    result.last_seen_drop_position = lastSeenDropPosition;
  }
  return result;
}

async function attemptClosePickupSweep(bot, center, blockName, inventoryBefore) {
  const sweepCenter = positionFromTarget(center);
  if (!sweepCenter) {
    return buildCollectResult(
      blockName,
      inventoryBefore,
      countInventoryItemsForCollection(bot.inventory?.items?.() || [], blockName)
    );
  }

  const sweepTargets = [
    sweepCenter,
    ...findRecoveryPositions(bot, sweepCenter, 6).slice(0, 6),
  ];
  const seen = new Set();

  for (const target of sweepTargets) {
    const key = `${Math.round(target.x)},${Math.round(target.y)},${Math.round(target.z)}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);

    const distance = bot.entity?.position?.distanceTo(target) || Number.POSITIVE_INFINITY;
    if (distance > 1.25) {
      const reachability = evaluateGoalReachability(bot, new GoalNear(target.x, target.y, target.z, 1));
      if (reachability.blocked) {
        continue;
      }
      try {
        await withTimeout(
          bot.pathfinder.goto(new GoalNear(target.x, target.y, target.z, 0)),
          1600,
          () => bot.pathfinder.setGoal(null),
          'Pathfinding during close pickup sweep timed out.'
        );
      } catch {
        continue;
      }
    }

    await delay(250);
    const inventoryAfter = countInventoryItemsForCollection(bot.inventory?.items?.() || [], blockName);
    if (inventoryAfter > inventoryBefore) {
      return buildCollectResult(blockName, inventoryBefore, inventoryAfter);
    }
  }

  const movementSweepResult = await attemptPickupMovementSweep(bot, blockName, inventoryBefore);
  if (movementSweepResult.collected) {
    return movementSweepResult;
  }

  return buildCollectResult(
    blockName,
    inventoryBefore,
    countInventoryItemsForCollection(bot.inventory?.items?.() || [], blockName)
  );
}

async function attemptPickupMovementSweep(bot, blockName, inventoryBefore) {
  const movements = [
    ['forward'],
    ['left'],
    ['right'],
    ['back'],
    ['jump', 'forward'],
  ];

  for (const combo of movements) {
    clearMovementControls(bot);
    for (const control of combo) {
      bot.setControlState(control, true);
    }
    await delay(combo.includes('jump') ? 320 : 220);
    clearMovementControls(bot);
    await delay(140);

    const inventoryAfter = countInventoryItemsForCollection(bot.inventory?.items?.() || [], blockName);
    if (inventoryAfter > inventoryBefore) {
      return buildCollectResult(blockName, inventoryBefore, inventoryAfter);
    }
  }

  clearMovementControls(bot);
  return buildCollectResult(
    blockName,
    inventoryBefore,
    countInventoryItemsForCollection(bot.inventory?.items?.() || [], blockName)
  );
}

function clearMovementControls(bot) {
  for (const control of ['forward', 'back', 'left', 'right', 'jump', 'sprint']) {
    bot.setControlState(control, false);
  }
}

function findNearbyDropEntities(bot, position, maxDistance = 10) {
  return Object.values(bot.entities || {})
    .filter((entity) => entity && entity.type === 'object' && entity.position)
    .filter((entity) => entity.position.distanceTo(position) <= maxDistance)
    .sort((left, right) => left.position.distanceTo(position) - right.position.distanceTo(position))
    .slice(0, 5);
}

function mergeDropEntities(primary, secondary) {
  const merged = [];
  const seen = new Set();
  for (const entity of [...(primary || []), ...(secondary || [])]) {
    const key = String(entity?.id ?? `${entity?.position?.x}:${entity?.position?.y}:${entity?.position?.z}`);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    merged.push(entity);
  }
  return merged;
}

function positionFromTarget(target) {
  if (!target) {
    return null;
  }
  if (target instanceof Vec3) {
    return target;
  }
  const x = Number(target.x);
  const y = Number(target.y);
  const z = Number(target.z);
  if (![x, y, z].every(Number.isFinite)) {
    return null;
  }
  return new Vec3(x, y, z);
}

function buildCollectResult(blockName, before, after) {
  const normalized = normalizeMinecraftName(blockName);
  const collectNames = getCollectibleItemNames(normalized);
  const safeBefore = Number.isFinite(before) ? before : 0;
  const safeAfter = Number.isFinite(after) ? after : safeBefore;
  return {
    item_name: collectNames[0] || normalized,
    source_block_name: normalized,
    collectible_item_names: collectNames,
    before: safeBefore,
    after: safeAfter,
    delta: Math.max(safeAfter - safeBefore, 0),
    collected: safeAfter > safeBefore,
  };
}

async function sweepNearbyDropRecovery(bot, position, blockName, inventoryBefore) {
  const sweepOffsets = [
    new Vec3(0, 0, 0),
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
  ];

  for (const offset of sweepOffsets) {
    const candidate = position.plus(offset);
    const reachability = evaluateGoalReachability(bot, new GoalNear(candidate.x, candidate.y, candidate.z, 1));
    if (reachability.blocked) {
      continue;
    }

    try {
      await withTimeout(
        bot.pathfinder.goto(new GoalNear(candidate.x, candidate.y, candidate.z, 0)),
        2500,
        () => bot.pathfinder.setGoal(null),
        'Pathfinding during drop sweep timed out.'
      );
    } catch {
      continue;
    }

    await delay(500);
    const inventoryAfter = countInventoryItemsForCollection(bot.inventory?.items?.() || [], blockName);
    if (inventoryAfter > inventoryBefore) {
      return buildCollectResult(blockName, inventoryBefore, inventoryAfter);
    }
  }

  return buildCollectResult(
    blockName,
    inventoryBefore,
    countInventoryItemsForCollection(bot.inventory?.items?.() || [], blockName)
  );
}

function countInventoryItemsForCollection(items, itemOrBlockName) {
  const collectibleNames = getCollectibleItemNames(itemOrBlockName);
  if (!collectibleNames.length) {
    return 0;
  }

  const acceptable = new Set(collectibleNames);
  return items.reduce((total, item) => {
    const candidate = normalizeMinecraftName(item?.name);
    if (!acceptable.has(candidate)) {
      return total;
    }
    return total + Math.max(item?.count || 1, 1);
  }, 0);
}

function getCollectibleItemNames(itemOrBlockName) {
  const normalized = normalizeMinecraftName(itemOrBlockName);
  if (!normalized) {
    return [];
  }
  return BLOCK_DROP_ITEMS[normalized] || [normalized];
}

function findRecoveryPosition(bot, position) {
  const candidates = findRecoveryPositions(bot, position, 1);
  return candidates[0] || null;
}

function findRecoveryPositions(bot, position, limit = 1) {
  if (!bot?.entity || !position) {
    return [];
  }

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
    new Vec3(1, -1, 1),
    new Vec3(-1, -1, 1),
    new Vec3(1, -1, -1),
    new Vec3(-1, -1, -1),
    new Vec3(0, -2, 0),
    new Vec3(1, -2, 0),
    new Vec3(-1, -2, 0),
    new Vec3(0, -2, 1),
    new Vec3(0, -2, -1),
  ];

  offsets.sort((left, right) => {
    const leftWeight = Math.abs(left.x) + Math.abs(left.y) + Math.abs(left.z);
    const rightWeight = Math.abs(right.x) + Math.abs(right.y) + Math.abs(right.z);
    return leftWeight - rightWeight;
  });

  const candidates = offsets
    .map((offset) => position.plus(offset))
    .map((candidate) => ({
      position: candidate,
      standable: isStandableRecoveryPosition(bot, candidate),
      probe: evaluateGoalReachability(bot, new GoalNear(candidate.x, candidate.y, candidate.z, 1)),
      distance: bot.entity.position.distanceTo(candidate),
    }))
    .filter((entry) => entry.standable && !entry.probe.blocked && entry.probe.status === 'success');

  candidates.sort((left, right) => {
    if (left.standable !== right.standable) {
      return left.standable ? -1 : 1;
    }
    if (left.probe.status === 'success' && right.probe.status !== 'success') {
      return -1;
    }
    if (right.probe.status === 'success' && left.probe.status !== 'success') {
      return 1;
    }
    if (left.probe.pathLength !== right.probe.pathLength) {
      return left.probe.pathLength - right.probe.pathLength;
    }
    return left.distance - right.distance;
  });

  return candidates.slice(0, Math.max(limit, 1)).map((entry) => entry.position);
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

function findCandidateMineTargets(bot, blockName, maxDistance, count, hintedPosition = null) {
  const hintedBlock = hintedPosition
    ? bot.blockAt(new Vec3(hintedPosition.x, hintedPosition.y, hintedPosition.z))
    : null;
  const hintedMatches =
    hintedBlock && normalizeMinecraftName(hintedBlock.name) === normalizeMinecraftName(blockName)
      ? [hintedBlock]
      : [];

  return hintedMatches
    .map((block) => {
      const blockProbe = evaluateGoalReachability(
        bot,
        new GoalNear(block.position.x, block.position.y, block.position.z, 2)
      );
      const recoveryPosition = findRecoveryPosition(bot, block.position);
      const recoveryProbe = recoveryPosition
        ? evaluateGoalReachability(bot, new GoalNear(recoveryPosition.x, recoveryPosition.y, recoveryPosition.z, 1))
        : { status: 'no_recovery', pathLength: Number.POSITIVE_INFINITY, blocked: true };
      const recoveryDistance = recoveryPosition
        ? bot.entity.position.distanceTo(recoveryPosition)
        : Number.POSITIVE_INFINITY;

      return {
        block,
        blockProbe,
        recoveryPosition,
        recoveryProbe,
        recoveryDistance,
      };
    })
    .filter(
      (entry) =>
        entry.recoveryPosition &&
        !entry.recoveryProbe.blocked &&
        entry.recoveryProbe.status === 'success'
    )
    .sort((left, right) => {
      if (left.recoveryProbe.status === 'success' && right.recoveryProbe.status !== 'success') {
        return -1;
      }
      if (right.recoveryProbe.status === 'success' && left.recoveryProbe.status !== 'success') {
        return 1;
      }
      if (left.recoveryProbe.pathLength !== right.recoveryProbe.pathLength) {
        return left.recoveryProbe.pathLength - right.recoveryProbe.pathLength;
      }
      if (left.recoveryDistance !== right.recoveryDistance) {
        return left.recoveryDistance - right.recoveryDistance;
      }
      return left.block.position.distanceTo(bot.entity.position) - right.block.position.distanceTo(bot.entity.position);
    });
}

function normalizeTargetPosition(position) {
  const x = Number(position?.x);
  const y = Number(position?.y);
  const z = Number(position?.z);
  if (![x, y, z].every(Number.isFinite)) {
    return null;
  }
  return {
    x: Math.round(x),
    y: Math.round(y),
    z: Math.round(z),
  };
}

function evaluateGoalReachability(bot, goal, timeoutMs = 1200) {
  try {
    const movements = bot.pathfinder?.movements || createDefaultMovements(bot);
    const result = bot.pathfinder?.getPathTo
      ? bot.pathfinder.getPathTo(movements, goal, timeoutMs)
      : null;

    if (!result) {
      return {
        status: 'unknown',
        pathLength: Number.POSITIVE_INFINITY,
        blocked: false,
      };
    }

    const pathLength = Array.isArray(result.path) ? result.path.length : Number.POSITIVE_INFINITY;
    const status = String(result.status || 'unknown');
    const blocked = status === 'noPath' || pathLength === 0;

    return {
      status,
      pathLength,
      blocked,
    };
  } catch {
    return {
      status: 'probe_error',
      pathLength: Number.POSITIVE_INFINITY,
      blocked: false,
    };
  }
}

function computeGotoTimeoutMs(reachability, linearDistance) {
  const pathLength = Number.isFinite(reachability?.pathLength)
    ? reachability.pathLength
    : Math.max(Math.ceil(Number(linearDistance) || 0), 1);
  const base = 2500 + pathLength * 350;
  return Math.max(4500, Math.min(18000, Math.round(base)));
}

function normalizeMinecraftName(value) {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/\s+/g, '_');
}

function roundCoordinate(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    throw new Error(`Invalid coordinate: ${value}`);
  }
  return Math.round(numeric);
}

function clampNumber(value, fallback, min, max) {
  const numeric = Number(value);
  const resolved = Number.isFinite(numeric) ? numeric : fallback;
  return Math.max(min, Math.min(max, resolved));
}

function delay(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function withTimeout(promise, timeoutMs, onTimeout, message) {
  let timer = null;
  return Promise.race([
    Promise.resolve(promise).finally(() => {
      if (timer) {
        clearTimeout(timer);
      }
    }),
    new Promise((_, reject) => {
      timer = setTimeout(() => {
        try {
          if (typeof onTimeout === 'function') {
            onTimeout();
          }
        } catch {}
        reject(new Error(message || `Timed out after ${timeoutMs}ms.`));
      }, timeoutMs);
    }),
  ]);
}

module.exports = { actionGoto, actionMineBlock, actionRecoverDrop, actionStop, buildCollectResult, createDefaultMovements };
