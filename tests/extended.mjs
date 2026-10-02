import {readFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const {operatorVisual,validateModel,valueAt}=await import('../.qa/core.mjs');
let checked=0;
for(const model of JSON.parse(await readFile('.qa/extended.json','utf8'))){
 validateModel(model);const tensors=new Map(model.tensors.map(t=>[t.id,t]));
 for(const node of model.nodes){const visual=operatorVisual(node.op),out=tensors.get(node.outputs[0]);assert.equal(visual.detail,'exact',`${model.name}: ${node.source}`);
  for(let index=0;index<out.data.values.length;index+=Math.max(1,Math.floor(out.data.values.length/100))){const deps=visual.dependencies(node,out,index,tensors);assert.ok(deps.length,`${model.name} ${node.id}: empty dependencies`);for(const d of deps)assert.ok(d.index>=0&&d.index<tensors.get(d.tensor).data.values.length,`${model.name}: invalid coordinate ${d.index}`);
   const numeric=deps.filter(d=>d.weight!==undefined);let expected;
   if(node.op==='concat')expected=valueAt(tensors.get(deps[0].tensor),deps[0].index);
   else expected=numeric.reduce((sum,d)=>sum+valueAt(tensors.get(d.tensor),d.index)*d.weight,0);
   const actual=valueAt(out,index);assert.ok(Math.abs(expected-actual)<5e-5,`${model.name} ${node.op} #${index}: ${expected} vs ${actual}`);checked++;
  }
 }
}
console.log(`${checked} extended PyTorch scalar outputs agree.`);
