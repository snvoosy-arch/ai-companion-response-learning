const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const modelPath = path.join(__dirname, '../docs/diffusion-trace/trace-model.js');
const load = () => { assert.ok(fs.existsSync(modelPath), 'trace model missing'); return require(modelPath); };
const fixture = () => ({schema:'rubrum.diffusion_trace.v1', cases:[{id:'test', canvas_length:3,
  stop_ids:[9], vocabulary:{1:{piece:'a'},2:{piece:'b'},9:{piece:'<end>'}},
  steps:[{step:0,ids:[2,2,2],changed:[],mean_entropy:null,eos_index:null},
    {step:1,ids:[1,9,2],changed:[0,1],mean_entropy:0.2,eos_index:1}],
  final:{ids:[1,9,9],reply:'a',eos_index:1}}]});

test('validates real trace navigation boundaries and raw/final distinction', () => {
  const m=load(), data=fixture(); m.validate(data);
  assert.equal(m.stepAt(data.cases[0], -8).step,0);
  assert.equal(m.stepAt(data.cases[0], 100).step,1);
  assert.deepEqual(m.stepAt(data.cases[0],1).ids,[1,9,2]);
  assert.deepEqual(m.visiblePositions(256,false).length,32);
  assert.equal(m.visiblePositions(256,true).at(-1),255);
});
test('rejects invalid canvas, EOS, entropy and changed-position claims', () => {
  const m=load();
  for (const mutate of [d=>d.cases[0].steps[1].ids.pop(),
    d=>d.cases[0].steps[1].eos_index=0,
    d=>d.cases[0].steps[1].mean_entropy=NaN,
    d=>d.cases[0].steps[1].changed=[0],
    d=>d.cases[0].steps[0].mean_entropy=0,
    d=>d.cases[0].steps[1].ids[0]=-1]) {
    const d=fixture(); mutate(d); assert.throws(()=>m.validate(d));
  }
});
test('rejects unrecognized schema and empty data', () => {
  const m=load(); assert.throws(()=>m.validate({schema:'wrong',cases:[]}));
});

test('validates every saved real frame without conflating argmax and sampled EOS', () => {
  const m=load(), data=JSON.parse(fs.readFileSync(path.join(__dirname,'../docs/diffusion-trace/data/traces.json'),'utf8'));
  m.validate(data);
  assert.deepEqual(data.cases.map(c=>c.steps.length-1),[14,22]);
  assert.equal(data.cases[0].final.reply,'세 개 나왔어.');
  assert.equal(data.cases[1].context.observed.received,null);
  assert.equal(data.cases[1].verdict,'unsupported');
  assert.ok(data.cases[0].steps.some(s=>s.step>0 && s.prediction_eos_index!==s.eos_index));
});
