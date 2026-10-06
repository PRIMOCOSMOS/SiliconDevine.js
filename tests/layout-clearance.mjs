import {build} from 'esbuild';
import {readFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
await build({entryPoints:['src/core/layout.ts'],outfile:'.qa/layout-clearance.mjs',bundle:true,platform:'node',format:'esm'});
await build({entryPoints:['src/core/hierarchy.ts'],outfile:'.qa/layout-hierarchy.mjs',bundle:true,platform:'node',format:'esm'});
const {layoutModel}=await import('../.qa/layout-clearance.mjs');
const {moduleView}=await import('../.qa/layout-hierarchy.mjs');
let cases=0,pairs=0;
function check(model,nodes,label){
 const ids=new Set(nodes.flatMap(n=>[...n.inputs,...n.outputs,...Object.values(n.parameters??{})]));
 const made=new Set(nodes.flatMap(n=>n.outputs));
 const view={...model,nodes,tensors:model.tensors.filter(t=>ids.has(t.id)),inputs:[...ids].filter(id=>!made.has(id))};
 for(const limit of [32,128,512]){
  const planes=[...layoutModel(view,limit).values()];
  for(let i=0;i<planes.length;i++)for(let j=i+1;j<planes.length;j++){
   const a=planes[i],b=planes[j];pairs++;
   const overlap=Math.abs(a.center[0]-b.center[0])<(a.width+b.width)/2&&Math.abs(a.center[2]-b.center[2])<(a.depth+b.depth)/2&&a.center[1]-.14<b.center[1]+b.height&&a.center[1]+a.height>b.center[1]-.14;
   assert.ok(!overlap,`${label}: ${a.tensor.id} overlaps ${b.tensor.id}`);
  }
  for(const n of nodes.filter(n=>n.op==='batchnorm'))for(const id of Object.values(n.parameters??{})){
   const p=planes.find(p=>p.tensor.id===id);assert.equal(p.owner,n.id);assert.ok(p.center[1]<0,'BatchNorm buffers must not pile up at the input origin');
  }
  cases++;
 }
}
for(const name of ['layout-residual-fx','layout-residual-export','layout-user']){
 const model=JSON.parse(await readFile(`.qa/${name}.json`,'utf8'));
 for(const scope of name==='layout-user'?['','res_block1','res_block2','attention','ffn']:[''])for(const detail of [false,true])check(model,moduleView(model,scope,detail),`${name}/${scope}/${detail}`);
}
for(const name of ['mlp','conv3d','attention','deepseek_v3.source','glm45.source','minimax_m1.source']){
 const model=JSON.parse(await readFile(`demo/public/models/${name}.json`.replace(/(?<!source)\.json$/,'.sd.json'),'utf8'));
 check(model,moduleView(model,name.includes('source')?'decoder.self_attn':''),name);
}
console.log(`${cases} coordinate-window layouts; ${pairs} tensor-pair clearance checks passed.`);
