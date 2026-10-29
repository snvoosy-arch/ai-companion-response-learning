/* Saved traces only: no model calls, telemetry, imports or network requests. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const model = window.TraceModel;
  let data;
  try { data = model.validate(window.RUBRUM_TRACE_DATA); }
  catch (error) {
    $('error').hidden = false; $('error').textContent = 'Trace를 읽지 못했습니다. 파일 구성을 확인하세요. ' + error.message;
    document.querySelectorAll('button, input, select').forEach(el => { el.disabled = true; });
    return;
  }
  let caseIndex = 0, index = data.cases[0].steps.length - 1, timer = null, selected = null;
  const currentCase = () => data.cases[caseIndex];
  function pause() {
    if (timer) clearInterval(timer);
    timer = null; $('play').setAttribute('aria-pressed', 'false');
    $('play').textContent = index === currentCase().steps.length - 1 ? '▶ 처음부터 재생' : '▶ 재생';
  }
  function svg(tag, attrs, text) {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function chart(c) {
    const el = $('chart'); el.replaceChildren();
    const width = 580, left = 36, top = 14, height = 60;
    const maximum = Math.max(...c.steps.slice(1).map(s => s.mean_entropy), 0.00001);
    const x = i => left + i / (c.steps.length - 1) * width;
    const y = value => top + height - value / maximum * height;
    for (const value of [0, maximum]) {
      el.append(svg('line', {x1:left,y1:y(value),x2:left+width,y2:y(value),stroke:'#dedcd6','stroke-width':1}));
      el.append(svg('text', {x:0,y:y(value)+3}, value ? value.toFixed(2) : '0'));
    }
    el.append(svg('polyline', {points:c.steps.slice(1).map(s => `${x(s.step)},${y(s.mean_entropy)}`).join(' '),fill:'none',stroke:'#8f2938','stroke-width':1.8}));
    el.append(svg('line', {x1:x(index),x2:x(index),y1:top-4,y2:top+height,stroke:'#8f2938','stroke-dasharray':'3 3',opacity:.45}));
    if (index > 0) el.append(svg('circle',{cx:x(index),cy:y(c.steps[index].mean_entropy),r:3.5,fill:'#8f2938'}));
    el.append(svg('text',{x:left,y:94}, '1'));
    el.append(svg('text',{x:left+width,y:94,'text-anchor':'end'}, 'STEP '+(c.steps.length-1)));
  }
  function tokenDetail(position) {
    const c = currentCase(), s = model.stepAt(c, index), id = s.ids[position], token = c.vocabulary[id];
    $('token-detail').textContent = `위치 ${position} / ID ${id} / raw: ${token.piece} / decode: ${token.decoded}`;
  }
  function render() {
    const c = currentCase(), s = model.stepAt(c, index);
    index = s.step;
    $('step-range').max = c.steps.length - 1; $('step-range').value = index;
    $('step-range').setAttribute('aria-valuetext', `${index} / ${c.steps.length-1} 단계`);
    $('step-number').textContent = index; $('step-total').textContent = ' / '+(c.steps.length-1);
    $('previous').disabled = index === 0; $('next').disabled = index === c.steps.length - 1;
    $('draft-text').textContent = s.prefix_text || (s.eos_index === 0 ? '(첫 위치가 종료 토큰)' : '(빈 텍스트)');
    $('draft-caption').textContent = index === 0 ? '초기 무작위 노이즈 · 처음 32개 토큰 해독' : (s.eos_index === null ? '원시 canvas · 처음 32개 토큰 해독' : '원시 canvas · 첫 종료 토큰 앞까지 해독');
    $('changed-count').textContent = index ? s.changed.length+' / '+c.canvas_length : '—';
    $('entropy-value').textContent = s.mean_entropy === null ? '측정 없음' : s.mean_entropy.toFixed(6);
    $('eos-value').textContent = s.eos_index === null ? '없음' : s.eos_index;
    $('token-grid').replaceChildren();
    for (const position of model.visiblePositions(c.canvas_length, $('full-canvas').checked)) {
      const id = s.ids[position], token = c.vocabulary[id], end = c.stop_ids.includes(id);
      const button = document.createElement('button'); button.type = 'button'; button.className = 'token';
      if (s.changed.includes(position)) button.classList.add('changed');
      if (end) button.classList.add('end');
      if (selected === position) button.classList.add('selected');
      button.setAttribute('aria-label',`위치 ${position}, 토큰 ${id}, ${token.piece}`);
      button.title = `위치 ${position} · ID ${id} · ${token.piece}`;
      const number = document.createElement('span'); number.className = 'index'; number.textContent = position;
      const label = document.createElement('span'); label.textContent = id === 151645 ? 'TURN END' : id === 151643 ? 'END / FILL' : (token.decoded.replaceAll(' ', '␠') || '∅');
      button.append(number,label);
      button.addEventListener('click', () => {
        selected = position;
        $('token-grid').querySelectorAll('.selected').forEach(el => el.classList.remove('selected'));
        button.classList.add('selected'); tokenDetail(position);
      });
      $('token-grid').append(button);
    }
    if (selected !== null) tokenDetail(selected);
    $('stop-note').textContent = index === 0 ? '초기 노이즈에는 모델 예측이 없어 entropy를 표시하지 않습니다.' :
      `Argmax 예측의 턴 종료 위치: ${s.prediction_eos_index ?? '없음'} · 현재 canvas의 종료 위치와 다를 수 있습니다. ${s.stop_reason ? '이 단계의 중단 사유: '+s.stop_reason+'.' : '복원 진행 중.'}`;
    chart(c);
    if (!timer) pause();
  }
  function loadCase() {
    const c = currentCase();
    $('case-title').textContent = c.title; $('case-id').textContent = c.id;
    $('user-text').textContent = '“'+c.user+'”';
    $('game-state').textContent = JSON.stringify(c.context,null,2);
    $('final-reply').textContent = '“'+c.final.reply+'”';
    $('verdict').textContent = c.label; $('verdict').className = 'verdict '+c.verdict;
    $('case-note').textContent = c.note;
    $('provenance').textContent = JSON.stringify({model:data.model,run_id:data.run_id,adapter_sha256:data.adapter_sha256,sc_sha256:data.sc_sha256,case_id:c.id,...c.source},null,2);
    $('token-detail').textContent = '토큰을 선택해 세부 정보를 확인하세요.';
    render();
  }
  $('case-select').addEventListener('change', () => { pause(); caseIndex = Number($('case-select').value); selected = null; index = currentCase().steps.length-1; loadCase(); });
  $('step-range').addEventListener('input', () => { pause(); index = Number($('step-range').value); render(); });
  $('previous').addEventListener('click', () => { pause(); index--; render(); });
  $('next').addEventListener('click', () => { pause(); index++; render(); });
  $('full-canvas').addEventListener('change', render);
  $('play').addEventListener('click', () => {
    if (timer) { pause(); return; }
    if (index === currentCase().steps.length-1) index = 0;
    timer = setInterval(() => { index++; render(); if (index >= currentCase().steps.length-1) pause(); }, 650);
    $('play').textContent = 'Ⅱ 일시 정지'; $('play').setAttribute('aria-pressed','true'); render();
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
  window.addEventListener('pagehide', pause);
  loadCase();
})();
