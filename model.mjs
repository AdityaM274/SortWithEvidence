export function tokens(text) {
  if(typeof text!=='string'||text.length>10000) throw Error('Text must be a string of at most 10,000 characters.');
  return text.toLowerCase().match(/[a-z]{2,}/g)||[];
}
export function train(rows) {
  if(!Array.isArray(rows)||rows.length<4||rows.length>500) throw Error('Use 4–500 labelled examples.');
  const classes=new Map(),vocab=new Set(),ids=new Set();
  for(const r of rows){
    if(!r||typeof r.id!=='string'||!r.id.trim()||ids.has(r.id)||typeof r.label!=='string'||!r.label.trim()||r.label.length>40)throw Error('Each example needs a unique ID and a label of 1–40 characters.');
    ids.add(r.id); const words=tokens(r.text);if(!words.length)throw Error('Training examples need English words.');
    if(!classes.has(r.label))classes.set(r.label,{docs:0,total:0,counts:new Map()});
    const c=classes.get(r.label);c.docs++;for(const w of words){vocab.add(w);c.total++;c.counts.set(w,(c.counts.get(w)||0)+1);}
  }
  if(classes.size<2||classes.size>12)throw Error('Use 2–12 distinct labels.');
  return {classes,vocab,n:rows.length,ids};
}
export function predict(model,text,threshold=0.8){
  if(!Number.isFinite(threshold)||threshold<0.5||threshold>0.99)throw Error('Review threshold must be 0.50–0.99.');
  const all=tokens(text),known=all.filter(w=>model.vocab.has(w));
  const scores=[...model.classes].map(([label,c])=>({label,log:Math.log(c.docs/model.n)+known.reduce((s,w)=>s+Math.log(((c.counts.get(w)||0)+1)/(c.total+model.vocab.size)),0)}));
  const max=Math.max(...scores.map(x=>x.log)),den=scores.reduce((s,x)=>s+Math.exp(x.log-max),0);
  scores.forEach(x=>x.score=Math.exp(x.log-max)/den);scores.sort((a,b)=>b.score-a.score||a.label.localeCompare(b.label));
  const coverage=all.length?known.length/all.length:0;
  const reason=!known.length?'No known vocabulary':coverage<0.35?'Low vocabulary coverage':scores[0].score<threshold?'Score below review threshold':null;
  const top=model.classes.get(scores[0].label),second=model.classes.get(scores[1].label);
  const evidence=[...new Set(known)].map(word=>({word,logRatio:Math.log(((top.counts.get(word)||0)+1)/(top.total+model.vocab.size))-Math.log(((second.counts.get(word)||0)+1)/(second.total+model.vocab.size))})).sort((a,b)=>b.logRatio-a.logRatio).slice(0,6);
  return {suggested:scores[0].label,decision:reason?'REVIEW':scores[0].label,reason,coverage,scores:scores.map(({label,score})=>({label,score})),evidence};
}
export function evaluate(model,test,threshold=0.8){
  if(!Array.isArray(test)||!test.length)throw Error('Need held-out test examples.');
  const ids=new Set(),rows=test.map(r=>{
    if(!r||typeof r.id!=='string'||ids.has(r.id)||model.ids.has(r.id)||!model.classes.has(r.label))throw Error('Test IDs must be unique, disjoint from training, and use known labels.');
    ids.add(r.id);const p=predict(model,r.text,threshold);return {id:r.id,expected:r.label,...p,correct:p.suggested===r.label};
  });
  const accepted=rows.filter(r=>r.decision!=='REVIEW');
  return {total:rows.length,correct:rows.filter(r=>r.correct).length,accepted:accepted.length,acceptedCorrect:accepted.filter(r=>r.correct).length,rows};
}
