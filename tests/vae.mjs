import {readFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
import {operatorVisual,validateModel,valueAt,numel,moduleView} from '../dist/silicondevine.js';
let checks=0;
for(const model of JSON.parse(await readFile('.qa/vae-cases.json','utf8'))){
 validateModel(model);const tensors=new Map(model.tensors.map(t=>[t.id,t]));
 const overview=moduleView(model);const sample=overview.find(n=>n.op==='gaussian_sample');assert.ok(sample,model.name);
 assert.equal(sample.attrs.children.length,5);assert.ok(moduleView(model,'',true).some(n=>n.op==='standard_normal'));
 for(const n of model.nodes){const visual=operatorVisual(n.op);assert.equal(visual.detail,'exact',n.op);
  for(const id of n.outputs){const out=tensors.get(id);
   for(let i=0;i<Math.min(numel(out.shape),out.data?.values.length??0);i++){
    const deps=visual.dependencies(n,out,i,tensors);for(const d of deps)assert.ok(d.index>=0&&d.index<numel(tensors.get(d.tensor).shape));
    const values=deps.map(d=>valueAt(tensors.get(d.tensor),d.index));let expected;
    if(n.op==='standard_normal'){assert.equal(deps.length,0);assert.ok(n.attrs.stochastic);continue;}
    if(n.op==='linear'||n.op.startsWith('conv'))expected=deps.reduce((s,d)=>s+valueAt(tensors.get(d.tensor),d.index)*(d.weight??1),0);
    else if(n.op==='multiply')expected=values[0]*values[1];
    else if(n.op==='add')expected=values[0]+values[1];
    else if(visual.curve)expected=visual.curve(values[0],n.attrs);
    else expected=values[0];
    if(Number.isFinite(expected)){assert.ok(Math.abs(expected-valueAt(out,i))<2e-5,`${model.name}/${n.op}/${i}: ${expected} vs ${valueAt(out,i)}`);checks++;}
   }
  }
 }
 const u=sample.attrs.gaussian;const heads=[u.mean,u.logvar].map(id=>model.nodes.find(n=>n.outputs.includes(id)));assert.ok(heads.every(n=>n.attrs.gaussianHead));
}
console.log(`3 VAE graphs: ${checks} finite scalar checks; sampling, heads and hierarchical detail verified.`);
