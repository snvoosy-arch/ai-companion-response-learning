function exact(value, fields) {
  return value && typeof value === 'object' && !Array.isArray(value) &&
    Object.keys(value).sort().join(',') === [...fields].sort().join(',');
}

function validateTurn(value, owner, allowTool) {
  if (!exact(value, ['action', 'speech', 'tool_calls', 'memory_candidates']) ||
      typeof value.speech !== 'string' || value.speech.length > 2000 ||
      !Array.isArray(value.tool_calls) || !Array.isArray(value.memory_candidates) || value.memory_candidates.length) {
    throw new Error('Invalid Actor contract');
  }
  if (value.action === 'reply' && value.speech.trim() && !value.tool_calls.length) return;
  if (value.action === 'silence' && value.speech === '' && !value.tool_calls.length) return;
  const call = value.tool_calls[0];
  if (allowTool && value.action === 'use_tool' && value.speech === '' && value.tool_calls.length === 1 &&
      exact(call, ['name', 'arguments']) && call.name === 'minecraft_goto' &&
      exact(call.arguments, ['player']) && call.arguments.player === owner) return;
  throw new Error('Invalid or unauthorized Actor action');
}

function createActorBridge({ owner, actor, observe, move, stop, maxRequests = 10, timeoutMs = 60000 }) {
  if (!owner || !Number.isInteger(maxRequests) || maxRequests < 1 ||
      !Number.isFinite(timeoutMs) || timeoutMs < 1) throw new Error('Invalid bridge configuration');
  let active = null, requests = 0;
  function cancel() { if (active) { active.abort(); stop(); } }
  async function handle(sender, message) {
    if (sender !== owner || typeof message !== 'string' || message.length > 1000 ||
        !/^사피루스(?:[\s,，:야아]|$)/u.test(message.trim())) return { status: 'ignored' };
    if (active) return { status: 'busy' };
    if (requests >= maxRequests) return { status: 'budget_exhausted' };
    const controller = new AbortController();
    active = controller;
    const timer = setTimeout(() => { controller.abort(); stop(); }, timeoutMs);
    const trace = [];
    function checked() { controller.signal.throwIfAborted(); }
    async function bounded(promise) {
      checked();
      let listener;
      try {
        return await Promise.race([promise, new Promise((_, reject) => {
          listener = () => reject(new Error('Task cancelled or timed out'));
          controller.signal.addEventListener('abort', listener, { once: true });
        })]);
      } finally { controller.signal.removeEventListener('abort', listener); }
    }
    async function ask(phase, tool_result = null) {
      checked();
      if (requests >= maxRequests) throw new Error('Model request budget exhausted');
      requests++;
      const input = { phase, requester: owner, message, observation: observe(), tool_result,
        allowed_tools: phase === 'decide' ? ['minecraft_goto'] : [] };
      trace.push({ kind: 'input', value: structuredClone(input) });
      const raw = await bounded(Promise.resolve().then(() => { checked(); return actor(input, controller.signal); }));
      checked();
      trace.push({ kind: 'actor', raw });
      validateTurn(raw, owner, phase === 'decide');
      return raw;
    }
    try {
      let choice = await ask('decide');
      if (choice.action === 'use_tool') {
        checked();
        const state = observe();
        const target = state.players.find(player => player.name === owner)?.position;
        if (!target || !state.self || !['x', 'y', 'z'].every(k => Number.isFinite(target[k]) && Number.isFinite(state.self[k]))) {
          throw new Error('Requested player is not currently visible');
        }
        const distance = Math.hypot(...['x', 'y', 'z'].map(k => target[k] - state.self[k]));
        if (distance > 32) throw new Error('Player outside 32-block movement scope');
        const position = Object.fromEntries(['x', 'y', 'z'].map(k => [k, Math.round(target[k])]));
        let result;
        try { result = await bounded(Promise.resolve().then(() => { checked(); return move(position, controller.signal); })); }
        catch (error) { checked(); result = { arrived: false, error: error.message }; }
        checked();
        trace.push({ kind: 'tool', name: 'minecraft_goto', position, result });
        choice = await ask('after_tool', result);
      }
      return { status: 'completed', action: choice.action, speech: choice.speech,
        answer_status: 'unreviewed', requests, trace };
    } catch (error) {
      stop();
      return { status: 'failed', error: error.message, requests, trace };
    } finally { clearTimeout(timer); active = null; }
  }
  return { handle, cancel };
}

module.exports = { createActorBridge, validateTurn };
