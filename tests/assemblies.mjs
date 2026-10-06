import {build} from 'esbuild';
import {readFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
await build({entryPoints:['src/core/assemblies.ts'],outfile:'.qa/assemblies.mjs',bundle:true,platform:'node',format:'esm'});
const {modelAssemblies}=await import('../.qa/assemblies.mjs');
let groups=0;
for(const file of ['demo/public/models/mlp.sd.json','.qa/layout-user.json','demo/public/models/deepseek_v3.source.json','demo/public/models/glm45.source.json','demo/public/models/minimax_m1.source.json']){
 const model=JSON.parse(await readFile(file,'utf8')),before=JSON.stringify(model),units=modelAssemblies(model),seen=[];
 for(const a of units){
  groups++;seen.push(...a.nodes);
  for(const id of a.nodes){const n=model.nodes.find(n=>n.id===id);assert.ok(n);
   for(const t of [...n.inputs,...n.outputs,...Object.values(n.parameters??{})])assert.ok(a.related.includes(t),`${n.id}: missing ${t}`);
   for(const t of Object.values(n.parameters??{}))assert.ok(a.owned.includes(t));
  }
  assert.ok(a.inputs.every(id=>!a.owned.includes(id)),'Shared data inputs are boundary references, not exclusive property');
 }
 assert.equal(new Set(seen).size,model.nodes.length);assert.equal(seen.length,model.nodes.length,'One functional owner per visible operator');
 assert.equal(JSON.stringify(model),before,'Visual aggregation must not mutate computation');
 if(file.includes('mlp'))assert.ok(units.filter(a=>a.nodes.some(id=>model.nodes.find(n=>n.id===id).op==='linear')).every(a=>a.related.length===4));
 if(file.includes('source'))assert.ok(units.some(a=>a.title.includes('注意力分配')&&a.nodes.length>1),'Raw attention scale/mask/softmax must share a functional frame');
}
console.log(`${groups} functional assemblies: complete membership, unique operator ownership and unchanged captured graphs.`);
