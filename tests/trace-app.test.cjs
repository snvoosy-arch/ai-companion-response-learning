const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

// A small DOM harness for app events. The range enforces the HTML constraint:
// assigning value above its current max clamps immediately, even if max changes later.
class Element {
  constructor(id) {
    this.id=id; this.children=[]; this.listeners={}; this.attrs={}; this.textContent='';
    this.checked=false; this.disabled=false; this.hidden=false; this._max=14; this._value=14;
    this.classList={add(){},remove(){}};
  }
  get max(){return this._max;}
  set max(value){this._max=Number(value); this.value=this._value;}
  get value(){return this._value;}
  set value(value){this._value=this.id==='step-range' ? Math.max(0,Math.min(this._max,Number(value))) : value;}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(...nodes){this.children=[...nodes];}
  setAttribute(key,value){this.attrs[key]=value;}
  addEventListener(event,handler){this.listeners[event]=handler;}
  querySelectorAll(){return [];}
  emit(event){this.listeners[event]?.();}
}
function boot() {
  const elements=new Map();
  const get=id=>{if(!elements.has(id)) elements.set(id,new Element(id)); return elements.get(id);};
  const doc={getElementById:get,createElement:()=>new Element(),createElementNS:()=>new Element(),
    querySelectorAll:()=>[],addEventListener(){}};
  const data=JSON.parse(fs.readFileSync(path.join(__dirname,'../docs/diffusion-trace/data/traces.json'),'utf8'));
  let timer=null;
  const context={document:doc,window:{TraceModel:require('../docs/diffusion-trace/trace-model.js'),RUBRUM_TRACE_DATA:data,addEventListener(){}},setInterval:callback=>{timer=callback;return 1;},clearInterval(){timer=null;}};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../docs/diffusion-trace/app.js'),'utf8'),context);
  return {get,data,tick:()=>timer?.()};
}

test('switching to a longer trace keeps range, text and accessibility on the same step',()=>{
  const {get}=boot();
  get('case-select').value='1'; get('case-select').emit('change');
  assert.equal(Number(get('step-number').textContent),22);
  assert.equal(get('step-range').max,22);
  assert.equal(Number(get('step-range').value),22);
  assert.equal(get('step-range').attrs['aria-valuetext'],'22 / 22 단계');
  assert.equal(get('next').disabled,true);
});

test('range navigation and whole-canvas toggle show real corresponding frames',()=>{
  const {get,data}=boot();
  get('step-range').value=0; get('step-range').emit('input');
  assert.equal(get('draft-text').textContent,data.cases[0].steps[0].prefix_text);
  assert.equal(get('entropy-value').textContent,'측정 없음');
  assert.equal(get('previous').disabled,true);
  assert.equal(get('token-grid').children.length,32);
  get('full-canvas').checked=true; get('full-canvas').emit('change');
  assert.equal(get('token-grid').children.length,256);
  get('next').emit('click');
  assert.equal(get('draft-text').textContent,data.cases[0].steps[1].prefix_text);
});

test('playback restarts, pauses on case change, and stops on the final recorded step',()=>{
  const {get,tick}=boot();
  get('play').emit('click');
  assert.equal(get('step-number').textContent,0);
  assert.equal(get('play').attrs['aria-pressed'],'true');
  tick(); assert.equal(get('step-number').textContent,1);
  get('case-select').value='1'; get('case-select').emit('change');
  assert.equal(get('play').attrs['aria-pressed'],'false');
  tick(); assert.equal(get('step-number').textContent,22);
  get('play').emit('click');
  for(let i=0;i<25;i++) tick();
  assert.equal(get('step-number').textContent,22);
  assert.equal(get('play').attrs['aria-pressed'],'false');
});
