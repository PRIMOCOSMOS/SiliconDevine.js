import {build} from 'esbuild';
import {mkdir,readFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
await mkdir('.qa',{recursive:true});
await build({entryPoints:['src/index.ts'],outfile:'.qa/core.mjs',bundle:true,platform:'node',format:'esm',packages:'external'});
await build({entryPoints:['src/core/layout.ts'],outfile:'.qa/layout.mjs',bundle:true,platform:'node',format:'esm',packages:'external'});
const {defineModel,compileDescription,validateModel,valueAt,operatorVisual,coordinates,flatIndex}=await import('../.qa/core.mjs');
const {tensorWindow,layoutModel}=await import('../.qa/layout.mjs');
const m=defineModel('test',42).input('x',[2,8]).linear('a',12).activation('relu').linear('b',4).build();
const recipe=compileDescription(JSON.parse(await readFile('examples/recipe.json','utf8')));assert.equal(recipe.nodes.length,3);
assert.throws(()=>defineModel('bad').input('x',[-1,3]),/正整数/);
assert.throws(()=>defineModel('bad').input('x',[2,3],[1]),/完整/);
assert.equal(m.nodes.length,3);assert.equal(m.tensors.find(t=>t.id==='b.out').data.values.length,8);
for(const n of m.nodes.filter(n=>n.op==='linear')){const map=new Map(m.tensors.map(t=>[t.id,t])),out=map.get(n.outputs[0]);for(let i=0;i<out.data.values.length;i++){const deps=operatorVisual(n.op).dependencies(n,out,i,map),sum=deps.reduce((s,d)=>s+valueAt(map.get(d.tensor),d.index)*d.weight,0);assert.ok(Math.abs(sum-valueAt(out,i))<1e-10);}}
const bad=structuredClone(m);bad.nodes[0].inputs=['missing'];assert.throws(()=>validateModel(bad),/缺失张量/);
const cycle=structuredClone(m);cycle.nodes[0].inputs=['b.out'];assert.throws(()=>validateModel(cycle),/环/);
const duplicate=structuredClone(m);duplicate.tensors.push(duplicate.tensors[0]);assert.throws(()=>validateModel(duplicate),/重复/);
const missing={id:'unknown',shape:[2,6,8],dtype:'float32',role:'input'};assert.ok(Number.isNaN(valueAt(missing,0)));
for(const shape of [[6,8],[1,2,3,4,5],[2,8],[48]]){const w=tensorWindow({...missing,shape},8192);assert.equal(new Set(w.positions.map(p=>p.join(','))).size,w.indices.length);assert.equal(w.indices.length,shape.reduce((a,b)=>a*b,1)>32&&shape.length===1?32:shape.reduce((a,b)=>a*b,1));for(const i of w.indices)assert.equal(flatIndex(coordinates(i,shape),shape),i);}
const volume=tensorWindow({...missing,shape:[1,1,4,4,4]},128);assert.equal(new Set(volume.positions.map(p=>p[1])).size,4);
const planes=layoutModel(m);for(const n of m.nodes.filter(n=>n.parameters)){const out=planes.get(n.outputs[0]);for(const id of Object.values(n.parameters))assert.equal(planes.get(id).center[0],out.center[0]);}
let checked=0;
for(const example of ['mlp','conv2d','conv3d','attention']){
  const graph=validateModel(JSON.parse(await readFile(`demo/public/models/${example}.sd.json`,'utf8'))),tensors=new Map(graph.tensors.map(t=>[t.id,t]));
  const layout=layoutModel(graph);for(const p of layout.values())assert.equal(new Set(p.positions.map(v=>v.join(','))).size,p.positions.length);
  for(const n of graph.nodes){const out=tensors.get(n.outputs[0]),v=operatorVisual(n.op);if(['linear','conv1d','conv2d','conv3d'].includes(n.op))for(let i=0;i<Math.min(out.data?.values.length??0,300);i++){const terms=v.dependencies(n,out,i,tensors);assert.ok(terms.length);const sum=terms.reduce((s,d)=>s+valueAt(tensors.get(d.tensor),d.index)*d.weight,0);assert.ok(Math.abs(sum-valueAt(out,i))<2e-5,`${example} ${n.id} ${i}: ${sum} / ${valueAt(out,i)}`);checked++;}
    if(['transpose','permute'].includes(n.op))for(let i=0;i<out.data.values.length;i++){const d=v.dependencies(n,out,i,tensors)[0];assert.equal(valueAt(tensors.get(d.tensor),d.index),valueAt(out,i));checked++;}
    if(n.op==='matmul')for(let i=0;i<out.data.values.length;i++){const terms=v.dependencies(n,out,i,tensors).filter(d=>d.weight!==undefined),sum=terms.reduce((s,d)=>s+valueAt(tensors.get(d.tensor),d.index)*d.weight,0);assert.ok(Math.abs(sum-valueAt(out,i))<1e-5);checked++;}
  }
}
console.log(`Core verified: ${checked} PyTorch scalar results, graph validation, exact coordinate windows, 3D volume depth and parameter alignment.`);
