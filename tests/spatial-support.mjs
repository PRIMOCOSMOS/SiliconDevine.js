import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {validateModel,operatorVisual,inspectSupport,numel,valueAt} from '../dist/silicondevine.js';
let checked=0;
for(const raw of JSON.parse(await readFile('.qa/spatial-fixtures.json','utf8'))){
 const m=validateModel(raw),ts=new Map(m.tensors.map(t=>[t.id,t]));assert.deepEqual(inspectSupport(m).boundary,[],m.name);
 for(const n of m.nodes){const v=operatorVisual(n.op);for(const oid of n.outputs){const out=ts.get(oid);for(let i=0;i<numel(out.shape);i++){
  const ds=v.dependencies(n,out,i,ts);for(const d of ds)assert.ok(d.index>=0&&d.index<numel(ts.get(d.tensor).shape),`${m.name} invalid coordinate`);
  const input=ds.filter(d=>d.tensor===n.inputs[0]);let result;
  if(['interpolate','grid_sample2d','sum','gather','index_select','identity','reshape'].includes(n.op))result=input.reduce((s,d)=>s+valueAt(ts.get(d.tensor),d.index)*(d.weight??1),0);
  else if(n.op==='amax'||n.op==='amin')result=(n.op==='amax'?Math.max:Math.min)(...ds.map(d=>valueAt(ts.get(d.tensor),d.index)));
  if(result!==undefined){assert.ok(Math.abs(result-valueAt(out,i))<2e-5,`${m.name}/${n.op}[${i}]: ${result} vs ${valueAt(out,i)}`);checked++;}
 }}}
}
console.log(`${checked} interpolation, grid sampling, reduction and index results match PyTorch.`);
