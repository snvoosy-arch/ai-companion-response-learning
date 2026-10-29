(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.SapphirusComparison = api;
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';
  const phases = {accepted:'접수',running:'진행 중',completed:'완료',deferred:'보류',partial:'부분 확인',failed:'실패',cancelled:'중단',clarify:'확인 질문',chat:'일반 대화'};
  function validate(data) {
    if (data.schema_version !== 'sapphirus.portfolio.comparison.v1' ||
        data.scope.kind !== 'recorded_model_outputs' || data.scope.spent !== true ||
        data.scope.synthetic_evaluation !== true || data.scope.actual_game_success_evaluated !== false ||
        !Array.isArray(data.cases) || !data.cases.length || data.scope.selected_cases !== data.cases.length) {
      throw new Error('지원하지 않는 기록 또는 공개 범위');
    }
    const ids = new Set();
    for (const c of data.cases) {
      if (ids.has(c.id) || !['improved','remaining_error'].includes(c.theme) ||
          typeof c.input.message !== 'string' || !phases[c.input.game_context.phase]) throw new Error('잘못된 사례');
      ids.add(c.id);
      for (const key of ['baseline','current']) {
        const o = c.outputs[key];
        if (typeof o.text !== 'string' || typeof o.note !== 'string' ||
            typeof o.grounded !== 'boolean' || typeof o.normal_eos !== 'boolean' ||
            ![0,1,2].includes(o.language_quality) ||
            o.usable !== (o.grounded && o.language_quality >= 1 && o.normal_eos)) throw new Error('잘못된 답변 판정');
      }
    }
    return data;
  }
  function filterCases(data, theme) {
    if (!['all','improved','remaining_error'].includes(theme)) throw new Error('잘못된 필터');
    return data.cases.filter(c => theme === 'all' || c.theme === theme);
  }
  function getCase(data, id) {
    const item = data.cases.find(c => c.id === id);
    if (!item) throw new Error('사례 없음');
    return item;
  }
  function orderedOutputs(item, reverse) {
    return (reverse ? ['current','baseline'] : ['baseline','current']).map(key => ({key,output:item.outputs[key]}));
  }
  function factRows(context) {
    const facts=context.facts || {};
    const rows = [['진행 상태', phases[context.phase] || context.phase],
      ['확인된 최소 수량', `${context.verified_count}개 · 실제 총량과 구분`],
      ['완료 확인', context.completion_verified ? '외부 확인됨' : '아직 확인되지 않음']];
    if ('food_level' in facts) rows.push(['허기 수치', facts.food_level === null ? '미확인' : `${facts.food_level} / 20`]);
    if (facts.inventory) rows.push(['현재 재고', Object.entries(facts.inventory).map(([k,v])=>`${k}: ${v}`).join(' · ')]);
    if (facts.current_step) rows.push(['관측 단계', facts.current_step]);
    if (context.reason_codes.length) rows.push(['외부 사유', context.reason_codes.join(', ')]);
    return rows;
  }
  return {validate,filterCases,getCase,orderedOutputs,factRows};
});
