import assert from 'node:assert/strict';
import {train,predict,evaluate} from './model.mjs';
import {training,heldout} from './fixtures.mjs';
const m=train(training),e=evaluate(m,heldout);
assert.equal(e.correct,8);assert.equal(e.total,8);
assert.equal(predict(m,'quasar astronomy nebula').decision,'REVIEW');
assert.equal(predict(m,'').decision,'REVIEW');
assert.equal(predict(m,'password login account').suggested,'Access');
assert.throws(()=>evaluate(m,[training[0]]),/disjoint/);
assert.throws(()=>train([...training,training[0]]),/unique/);
assert.throws(()=>train(training.map(r=>({...r,label:'one'}))),/distinct/);
assert.throws(()=>predict(m,'hello',2),/threshold/);
for(const r of heldout){const p=predict(m,r.text);assert.ok(Math.abs(p.scores.reduce((a,x)=>a+x.score,0)-1)<1e-10);}
// Independent two-class hand calculation: P(a|A)=3/5, P(a|B)=1/5; equal priors => 0.75.
const tiny=train([{id:'1',label:'A',text:'aa aa'},{id:'2',label:'A',text:'bb'},{id:'3',label:'B',text:'bb bb'},{id:'4',label:'B',text:'bb'}]);
assert.ok(Math.abs(predict(tiny,'aa').scores[0].score-0.75)<1e-10);
console.log(JSON.stringify({tests:'passed',syntheticHeldout:e},null,2));
