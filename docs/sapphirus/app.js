(function () {
  'use strict';
  const m=window.SapphirusComparison, d=window.SAPPHIRUS_COMPARISON;
  const el=id=>document.getElementById(id);
  function node(tag, text, cls) {
    const n=document.createElement(tag);
    if (text !== undefined) n.textContent=text;
    if (cls) n.className=cls;
    return n;
  }
  try { m.validate(d); } catch (error) { el('viewer-status').textContent='기록을 표시할 수 없어. 공개 JSON과 파일 구성을 확인해 줘.'; return; }
  let visible=m.filterCases(d,'all'), index=0, reverse=false;
  function render() {
    const item=m.getCase(d,visible[index].id);
    el('case-title').textContent=item.title;
    el('case-id').textContent=item.id;
    el('case-position').textContent=`${index+1} / ${visible.length}`;
    el('case-picker').value=item.id;
    el('user-message').textContent=item.input.message;
    el('raw-input').textContent=JSON.stringify(item.input,null,2);
    el('prev-case').disabled=index === 0;
    el('next-case').disabled=index === visible.length-1;
    el('facts-list').replaceChildren(...m.factRows(item.input.game_context).flatMap(([k,v])=>[node('dt',k),node('dd',v)]));
    el('answers').replaceChildren(...m.orderedOutputs(item,reverse).map(({key,output:o})=>{
      const card=node('article',undefined,`answer ${key}`);
      const header=node('div',undefined,'answer-heading');
      header.append(node('span',key==='current'?'CURRENT CHAT':'PREVIOUS CHAT','eyebrow'),node('span',o.usable?'사용 가능 판정':'사용 불가 판정',o.usable?'verdict pass':'verdict fail'));
      const grades=node('div',undefined,'grades');
      grades.append(node('span',o.grounded?'근거 충족':'근거 오류',o.grounded?'grounded':'unsupported'),node('span',`표현 품질 ${o.language_quality} / 2`),node('span',o.normal_eos?'정상 종료':'종료 미확인'));
      card.append(header,node('h4',d.models[key].label),node('blockquote',o.text || '(빈 답변)','response'),grades,node('p',o.note,'review-note'));
      return card;
    }));
    el('viewer-status').textContent=`선택: ${item.title}`;
  }
  function options() {
    el('case-picker').replaceChildren(...visible.map(c=>{const o=node('option',c.title);o.value=c.id;return o;}));
  }
  el('case-filter').value='all';
  el('case-filter').addEventListener('change',()=>{
    visible=m.filterCases(d,el('case-filter').value);index=0;options();render();
  });
  el('case-picker').addEventListener('change',()=>{
    index=visible.findIndex(c=>c.id===el('case-picker').value);render();
  });
  el('prev-case').addEventListener('click',()=>{if(index>0){index--;render();}});
  el('next-case').addEventListener('click',()=>{if(index<visible.length-1){index++;render();}});
  el('swap-models').addEventListener('click',()=>{reverse=!reverse;render();});
  options();render();
})();
