import {build} from 'esbuild';
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
await build({entryPoints:['src/index.ts'],bundle:true,platform:'node',format:'esm',outfile:'.qa/semantic-library.mjs',external:['three','three/*']});
const {moduleView,operatorVisual,topologicalNodes,valueAt}=await import('../.qa/semantic-library.mjs');
let mappings=0;
for(const family of ['deepseek_v3','glm45','minimax_m1']){
 const model=JSON.parse(await fs.readFile(`demo/public/models/${family}.source.json`,'utf8')),t=new Map(model.tensors.map(v=>[v.id,v]));
 const raw=moduleView(model,'decoder.self_attn',true),nodes=moduleView(model,'decoder.self_attn');
 assert.ok(nodes.length<raw.length*.65);topologicalNodes({...model,nodes});
 assert.ok(nodes.some(n=>n.op==='semantic'&&n.attrs.motionKind==='softmax'));
 assert.ok(nodes.some(n=>n.op==='semantic'&&n.attrs.motionKind==='rope'));
 for(const n of nodes.filter(n=>['semantic','layout'].includes(n.op))){
  assert.ok(n.attrs.description.length>20);
  const children=n.attrs.children,outputs=new Set(children.flatMap(c=>c.outputs));
  assert.ok(n.outputs.every(id=>outputs.has(id)));
  for(const id of n.outputs){const out=t.get(id);for(let i=0;i<Math.min(16,out.data?.values.length??0);i++){
   const deps=operatorVisual(n.op).dependencies(n,out,i,t);assert.ok(deps.length>0,`${family} ${n.name} ${id}`);
   assert.ok(deps.every(d=>n.inputs.includes(d.tensor)),`${family} ${n.name} must reach boundary inputs`);
   if(n.op==='layout')for(const d of deps){assert.equal(valueAt(out,i),valueAt(t.get(d.tensor),d.index));mappings++;}
  }}
 }
}
console.log(`Semantic views preserve boundaries; ${mappings} coordinate/value mappings verified.`);
