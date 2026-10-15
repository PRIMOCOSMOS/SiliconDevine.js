import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {operatorVisual,validateModel} from '../dist/silicondevine.js';
const g=JSON.parse(await readFile('.qa/logic.json','utf8'));
validateModel(g);
const tensors=new Map(g.tensors.map(t=>[t.id,t]));
for(const n of g.nodes){
 const v=operatorVisual(n.op),out=tensors.get(n.outputs[0]);
 assert.equal(v.detail,'exact');
 const count=out.shape.reduce((a,b)=>a*b,1);
 for(let i=0;i<count;i++){
  const deps=v.dependencies(n,out,i,tensors);
  if(['any','all'].includes(n.op)){
   assert.deepEqual(deps.map(d=>d.index),[i*3,i*3+1,i*3+2]);
   const values=deps.map(d=>Boolean(tensors.get(d.tensor).data.values[d.index]));
   assert.equal(Number(n.op==='any'?values.some(Boolean):values.every(Boolean)),Number(out.data.values[i]));
  }else if(n.op==='ne')assert.deepEqual(deps.map(d=>d.index),[i]);
  else assert.deepEqual(deps,[]);
 }
}
console.log('Logic reduction coordinates and PyTorch boolean outputs verified.');
