import {readFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const {operatorVisual,validateModel,valueAt,coordinates,numel,moduleView}=await import('../.qa/core.mjs');
let checked=0,graphs=0;
for(const model of [...JSON.parse(await readFile('.qa/llm-cases.json','utf8')),...JSON.parse(await readFile('.qa/set-cases.json','utf8')),...JSON.parse(await readFile('.qa/official-cases.json','utf8'))]){
 validateModel(model);graphs++;const tensors=new Map(model.tensors.map(t=>[t.id,t]));
 const overview=moduleView(model);assert.ok(overview.length<=model.nodes.length);
 for(const scope of ['',...new Set(model.nodes.map(n=>n.group))])for(const n of moduleView(model,scope)){if(n.op==='layout')for(const id of n.outputs){const out=tensors.get(id);for(let i=0;i<numel(out.shape);i++){const d=operatorVisual('layout').dependencies(n,out,i,tensors);assert.equal(d.length,1);assert.equal(valueAt(tensors.get(d[0].tensor),d[0].index),valueAt(out,i));}}}
 for(const n of model.nodes){const v=operatorVisual(n.op);assert.equal(v.detail,'exact',`${model.name}: ${n.op}`);
  for(const oid of n.outputs){const out=tensors.get(oid);
   for(let i=0;i<numel(out.shape);i++){const deps=v.dependencies(n,out,i,tensors);for(const d of deps)assert.ok(Number.isInteger(d.index)&&d.index>=0&&d.index<numel(tensors.get(d.tensor).shape),`${n.op}: invalid coordinate ${d.index}`);
    const values=deps.map(d=>valueAt(tensors.get(d.tensor),d.index));let expected;
    if(['linear','matmul','embedding','mean'].includes(n.op))expected=deps.filter(d=>d.weight!==undefined).reduce((sum,d)=>sum+valueAt(tensors.get(d.tensor),d.index)*d.weight,0);
    else if(n.op==='divide')expected=values[0]/values[1];
    else if(n.op==='power')expected=values[0]**values[1];
    else if(n.op==='multiply')expected=values[0]*values[1];
    else if(n.op==='add')expected=values[0]+values[1]*Number(n.attrs.alpha??1);
    else if(n.op==='subtract')expected=values[0]-values[1]*Number(n.attrs.alpha??1);
    else if(n.op==='arange')expected=Number(n.attrs.start)+i*Number(n.attrs.step);
    else if(v.curve)expected=v.curve(values[0],n.attrs);
    else if(n.op==='softmax') {const max=Math.max(...values),exp=values.map(x=>Math.exp(x-max)),c=coordinates(i,out.shape);expected=max===-Infinity&&n.attrs.safe?0:exp[c.at(-1)]/exp.reduce((a,b)=>a+b,0);}
    else if(n.op==='layernorm'||n.op==='rmsnorm') {const x=values.slice(0,(n.attrs.normalized_shape??[out.shape.at(-1)]).reduce((a,b)=>a*b,1)),mean=n.op==='rmsnorm'?0:x.reduce((a,b)=>a+b,0)/x.length,variance=x.reduce((a,b)=>a+(b-mean)**2,0)/x.length,gamma=n.parameters.weight?valueAt(tensors.get(n.parameters.weight),i%x.length):1,bias=n.parameters.bias?valueAt(tensors.get(n.parameters.bias),i%x.length):0;expected=(valueAt(tensors.get(n.inputs[0]),i)-mean)/Math.sqrt(variance+Number(n.attrs.eps??1e-5))*gamma+bias;}
    else if(n.op==='attention_mask'){const c=coordinates(i,out.shape);expected=n.attrs.causal&&c.at(-1)>c.at(-2)?-Infinity:!deps.some(d=>d.tensor===n.inputs[0])?-Infinity:values[0]+(n.attrs.mask_kind==='additive'?(values[1]??0):0);}
    else expected=values[0];
    const actual=valueAt(out,i);assert.ok(expected===actual||Math.abs(expected-actual)<2e-5,`${model.name} ${n.id}/${n.op} #${i}: ${expected} != ${actual}`);checked++;
   }
  }
 }
}
console.log(`${graphs} LLM graphs: ${checked} scalar computations and all dependency coordinates verified.`);
