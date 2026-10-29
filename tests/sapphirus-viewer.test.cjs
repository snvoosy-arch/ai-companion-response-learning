const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.join(__dirname, '..');
const modelPath = path.join(root, 'docs/sapphirus/comparison-model.js');
const appPath = path.join(root, 'docs/sapphirus/app.js');
const load = () => { assert.ok(fs.existsSync(modelPath), 'comparison model missing'); return require(modelPath); };
const data = () => JSON.parse(fs.readFileSync(path.join(root, 'docs/sapphirus/data/comparison.json'), 'utf8'));

test('filters preserve identity and swapping changes only presentation order', () => {
  const m = load(), d = data(); m.validate(d);
  const failures = m.filterCases(d, 'remaining_error');
  assert.equal(failures.length, 2);
  assert.equal(m.getCase(d, failures[1].id).id, failures[1].id);
  const before = m.orderedOutputs(failures[0], false);
  const after = m.orderedOutputs(failures[0], true);
  assert.deepEqual(before.map(x => x.key), ['baseline', 'current']);
  assert.equal(after[0].output.text, failures[0].outputs.current.text);
  assert.deepEqual(before[1].output, after[0].output);
  assert.throws(() => m.getCase(d, 'missing'));
});

test('rejects duplicate cases, inconsistent grades and fabricated live scope', () => {
  const m = load();
  for (const mutate of [d => d.cases.push(d.cases[0]),
    d => d.cases[0].outputs.current.usable = false,
    d => d.scope.kind = 'live', d => d.scope.spent = false,
    d => d.cases[0].outputs.current.language_quality = 3]) {
    const d = data(); mutate(d); assert.throws(() => m.validate(d));
  }
});

test('facts display unknown separately and never turn a lower bound into total', () => {
  const m = load();
  const rows = m.factRows({phase:'running', verified_count:0, completion_verified:false,
    reason_codes:[], facts:{food_level:null, inventory:{bread:2}}});
  assert.ok(rows.some(([k,v]) => k === '완료 확인' && v === '아직 확인되지 않음'));
  assert.ok(rows.some(([k,v]) => k === '확인된 최소 수량' && v === '0개 · 실제 총량과 구분'));
  assert.ok(rows.some(([k,v]) => k === '허기 수치' && v === '미확인'));
});

function dom() {
  class Element {
    constructor(tag='div') { this.tagName=tag; this.children=[]; this.listeners={}; this.textContent=''; this.value=''; this.disabled=false; this.dataset={}; }
    append(...xs) { this.children.push(...xs); }
    replaceChildren(...xs) { this.children=xs; }
    addEventListener(name, fn) { this.listeners[name]=fn; }
    set innerHTML(value) { throw new Error('unsafe HTML insertion: '+value); }
    setAttribute(key,value) { this[key]=value; }
  }
  const nodes=new Map();
  return {nodes, document:{getElementById(id){if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);},createElement(tag){return new Element(tag);}}};
}

test('actual app preserves labels through filter/swap and inserts saved output as text', () => {
  const m=load(); assert.ok(fs.existsSync(appPath),'viewer app missing');
  const d=data(), {nodes,document}=dom();
  const hostile='<img src=x onerror=alert(1)>';
  d.cases[0].outputs.current.text=hostile;
  vm.runInNewContext(fs.readFileSync(appPath,'utf8'), {
    document, window:{SapphirusComparison:m,SAPPHIRUS_COMPARISON:d}, console,
  });
  const text=node=>node.textContent+' '+node.children.map(text).join(' ');
  assert.ok(text(nodes.get('answers')).includes(hostile));
  assert.ok(text(nodes.get('answers')).indexOf(d.models.baseline.label)<text(nodes.get('answers')).indexOf(d.models.current.label));
  nodes.get('swap-models').listeners.click();
  assert.ok(text(nodes.get('answers')).indexOf(d.models.current.label)<text(nodes.get('answers')).indexOf(d.models.baseline.label));
  nodes.get('case-filter').value='remaining_error';
  nodes.get('case-filter').listeners.change();
  assert.equal(nodes.get('case-picker').children.length,2);
  assert.equal(nodes.get('case-position').textContent,'1 / 2');
  nodes.get('next-case').listeners.click();
  assert.equal(nodes.get('case-title').textContent,d.cases[5].title);
  assert.equal(nodes.get('next-case').disabled,true);
});
