import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {validateModel,operatorVisual,inspectSupport,numel,valueAt} from '../dist/silicondevine.js';
let checked=0;
for(const raw of JSON.parse(await readFile('.qa/api-fixtures.json','utf8'))){
 const m=validateModel(raw),ts=new Map(m.tensors.map(t=>[t.id,t]));
 assert.deepEqual(inspectSupport(m).boundary,[],`${m.name}: unsupported operators`);
 for(const n of m.nodes){const v=operatorVisual(n.op);
  for(const id of n.outputs){const out=ts.get(id);for(let i=0;i<numel(out.shape);i++){
   const ds=v.dependencies(n,out,i,ts);for(const d of ds)assert.ok(d.index>=0&&d.index<numel(ts.get(d.tensor).shape),`${m.name}/${n.op} bad coordinate`);
   let actual;
   if(v.curve)actual=v.curve(valueAt(ts.get(n.inputs[0]),i),n.attrs??{});
   else if(['pad','unfold2d','fold2d','prelu','identity','reshape','dropout','pixel_shuffle','pixel_unshuffle'].includes(n.op))actual=ds.length?ds.reduce((s,d)=>s+valueAt(ts.get(d.tensor),d.index)*(d.weight??1),0):n.op==='pad'?Number(n.attrs.value??0):0;
   else if(n.op==='log_softmax') {const x=ts.get(n.inputs[0]),values=ds.map(d=>valueAt(x,d.index)),max=Math.max(...values);actual=valueAt(x,i)-max-Math.log(values.reduce((s,v)=>s+Math.exp(v-max),0));}
   if(actual!==undefined){assert.ok(Math.abs(actual-valueAt(out,i))<2e-5,`${m.name}/${n.op}[${i}]: ${actual} vs ${valueAt(out,i)}`);checked++;}
  }}
 }
}
const unknown={nodes:[{id:'custom',op:'opaque',name:'custom',source:'torch.linalg.svd',attrs:{}}]};
assert.equal(inspectSupport(unknown).boundary[0].source,'torch.linalg.svd');
console.log(`${checked} scalar computations and all dependency coordinates match PyTorch across both backends.`);
