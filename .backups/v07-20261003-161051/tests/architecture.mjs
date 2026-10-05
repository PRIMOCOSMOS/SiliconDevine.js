import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {validateModel,aggregatePartition,defineArchitecture} from '../dist/silicondevine.js';
for(const family of ['deepseek_v3','glm45','minimax_m1']){
 const m=JSON.parse(await readFile(`demo/public/models/${family}.sd.json`,'utf8'));validateModel(m);
 const composed=defineArchitecture(m.name,m.architecture.scopes);assert.equal(composed.architecture.parameterCount,m.architecture.parameterCount);
 const cyclic=structuredClone(m);cyclic.architecture.scopes.root.nodes[0].attrs.scopeRef='root';assert.throws(()=>validateModel(cyclic),/循环/);
 const missing=structuredClone(m);missing.architecture.scopes.root.nodes[0].attrs.scopeRef='missing';assert.throws(()=>validateModel(missing),/不存在/);
 const falseValues=structuredClone(m);falseValues.architecture.scopes.root.tensors[0].data={offset:0,values:[1]};assert.throws(()=>validateModel(falseValues),/无数值/);
}
const t={shape:[8,7168],dtype:'logical',role:'parameter',id:'w',representation:'aggregate'};
const covered=new Set();for(let i=0;i<18;i++){const [[a,b],[c,d]]=aggregatePartition(t,i);for(let r=a;r<b;r++)for(let k=c;k<d;k++){const key=r*7168+k;assert.ok(!covered.has(key));covered.add(key);}}
assert.equal(covered.size,8*7168);assert.throws(()=>aggregatePartition(t,18));console.log('Architecture validation, cycles, missing scopes and complete disjoint partitions passed.');
