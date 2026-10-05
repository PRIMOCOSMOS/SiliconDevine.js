import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {validateModel,operatorVisual,valueAt,numel} from '../dist/silicondevine.js';
let checked=0;const report={};
for(const family of ['deepseek_v3','glm45','minimax_m1']){
 const m=validateModel(JSON.parse(await readFile(`demo/public/models/${family}.sd.json`,'utf8')));report[family]={};
 for(const [key,g] of Object.entries(m.architecture.mechanisms)){
  const ts=new Map(g.tensors.map(t=>[t.id,t]));let links=0;
  for(const n of g.nodes){const visual=operatorVisual(n.op);assert.equal(visual.detail,'exact',`${key}/${n.op} needs implementation`);
   for(const oid of n.outputs){const out=ts.get(oid);for(let i=0;i<numel(out.shape);i++){
    const deps=visual.dependencies(n,out,i,ts);links+=deps.length;
    for(const d of deps){assert.ok(ts.has(d.tensor),`${key} missing ${d.tensor}`);assert.ok(Number.isInteger(d.index)&&d.index>=0&&d.index<numel(ts.get(d.tensor).shape),`${key}/${n.id} ${d.tensor}[${d.index}]`);}
    if(['linear','matmul','kernel_mix','dynamic_conv2d','expert_combine','state_update'].includes(n.op)){
     const actual=valueAt(out,i),expected=deps.filter(d=>n.op!=='matmul'||d.tensor===n.inputs[0]).reduce((s,d)=>s+valueAt(ts.get(d.tensor),d.index)*(d.weight??1),0);
     assert.ok(Math.abs(actual-expected)<2e-5,`${family}/${key}/${n.id}[${i}] expected ${expected}, got ${actual}`);checked++;
    }
   }}
  }assert.ok(links>0);report[family][key]={nodes:g.nodes.length,links};
 }
}
for(const name of ['dynamic_conv','conditional_attention']){
 const g=validateModel(JSON.parse(await readFile(`demo/public/models/${name}.sd.json`,'utf8'))),ts=new Map(g.tensors.map(t=>[t.id,t]));
 for(const n of g.nodes.filter(n=>['kernel_mix','dynamic_conv2d'].includes(n.op))){const out=ts.get(n.outputs[0]);for(let i=0;i<numel(out.shape);i++){const deps=operatorVisual(n.op).dependencies(n,out,i,ts);const result=deps.reduce((s,d)=>s+valueAt(ts.get(d.tensor),d.index)*(d.weight??1),0);assert.ok(Math.abs(result-valueAt(out,i))<1e-5);checked++;}}
}
console.log(JSON.stringify({checked,report},null,2));

