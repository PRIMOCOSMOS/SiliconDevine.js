import {build} from 'esbuild';
import {readFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
await build({entryPoints:['src/index.ts'],bundle:true,platform:'node',format:'esm',outfile:'.qa/source-library.mjs',external:['three','three/*']});
const {validateModel,operatorVisual,valueAt,numel,moduleView}=await import('../.qa/source-library.mjs');
let checked=0;
for(const name of ['deepseek_v3','glm45','minimax_m1','minimax_m1_lightning']){
 const g=validateModel(JSON.parse(await readFile(`demo/public/models/${name}.source.json`,'utf8'))),t=new Map(g.tensors.map(t=>[t.id,t]));
 assert.equal(g.producer.backend,'pytorch-execution');
 const block=moduleView(g,'decoder');assert.ok(block.some(n=>n.attrs?.modulePath==='decoder.self_attn'));
 for(const n of g.nodes){
  const vis=operatorVisual(n.op);assert.equal(vis.detail,'exact',`${name}/${n.id}/${n.op}`);
  for(const id of n.outputs){const o=t.get(id);for(let i=0;i<numel(o.shape);i++){
   const ds=vis.dependencies(n,o,i,t);
   for(const d of ds)assert.ok(t.has(d.tensor)&&d.index>=0&&d.index<numel(t.get(d.tensor).shape),`${name}/${n.id}: ${JSON.stringify(d)}`);
   if(['linear','matmul','routing_index','routing_index_add','routing_unbind'].includes(n.op)){
    const expected=ds.filter(d=>n.op!=='matmul'||d.tensor===n.inputs[0]).reduce((s,d)=>s+valueAt(t.get(d.tensor),d.index)*(d.weight??1),0);
    assert.ok(Math.abs(expected-valueAt(o,i))<2e-4,`${name}/${n.id}[${i}]: ${expected} != ${valueAt(o,i)}`);checked++;
   }
  }}
 }
 // A real residual consumes a produced submodule value and its earlier input.
 const producers=new Map(g.nodes.flatMap(n=>n.outputs.map(o=>[o,n])));
 assert.ok(g.nodes.some(n=>n.op==='add'&&n.inputs.some(i=>producers.get(i)?.group.startsWith('decoder.self_attn'))));
}
console.log(`Source composition: ${checked} scalar computations; real module boundaries and coordinate dependencies passed.`);
