(function (root, factory) {
  'use strict';
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.TraceModel = factory();
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const fail = message => { throw new Error(message); };
  const firstEos = (ids, stops) => { const i = ids.findIndex(id => stops.includes(id)); return i < 0 ? null : i; };
  function validate(data) {
    if (!data || data.schema !== 'rubrum.diffusion_trace.v1' || !Array.isArray(data.cases) || !data.cases.length) fail('Trace 데이터 형식 오류');
    const seen = new Set();
    for (const c of data.cases) {
      if (!c.id || seen.has(c.id)) fail('중복 사례');
      seen.add(c.id);
      if (!Number.isInteger(c.canvas_length) || c.canvas_length < 1 || c.canvas_length > 256) fail('Canvas 길이 오류');
      if (!Array.isArray(c.stop_ids) || !c.stop_ids.length || !c.stop_ids.every(Number.isInteger)) fail('종료 토큰 오류');
      const checkIds = ids => {
        if (!Array.isArray(ids) || ids.length !== c.canvas_length || !ids.every(id => Number.isInteger(id) && id >= 0 && id < 151936 && c.vocabulary?.[id])) fail('토큰 배열 오류');
      };
      if (!Array.isArray(c.steps) || c.steps.length < 2) fail('복원 단계 없음');
      c.steps.forEach((s, j) => {
        checkIds(s.ids);
        if (s.step !== j || s.eos_index !== firstEos(s.ids, c.stop_ids)) fail('단계·EOS 불일치');
        if (j === 0 ? s.mean_entropy !== null : !Number.isFinite(s.mean_entropy) || s.mean_entropy < 0) fail('Entropy 오류');
        const changes = j ? s.ids.flatMap((id, i) => id !== c.steps[j-1].ids[i] ? [i] : []) : [];
        if (JSON.stringify(s.changed) !== JSON.stringify(changes)) fail('변경 위치 불일치');
      });
      checkIds(c.final.ids);
      if (c.final.eos_index !== firstEos(c.final.ids, c.stop_ids) || typeof c.final.reply !== 'string') fail('최종 출력 오류');
    }
    return data;
  }
  function stepAt(c, index) { return c.steps[Math.max(0, Math.min(c.steps.length - 1, Math.trunc(Number(index)) || 0))]; }
  function visiblePositions(length, full) { return Array.from({length: full ? length : Math.min(32, length)}, (_, i) => i); }
  return {validate, stepAt, visiblePositions};
});
